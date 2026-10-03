"""Development-only profile of one training run: where does train() spend its time?

Wraps the submission's data augmentation and optimizer steps with synchronised timers
(forward + backward is the remainder), then lists the most expensive GPU kernels from
torch.profiler. Syncing adds a little overhead, so read the shares, not absolute totals.

    python -m dev.profile --submission airbench_muon
"""

import argparse
import importlib
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from benchmark.api import BuildContext
from benchmark.data import load_split
from benchmark.worker import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--params", default="{}")
    parser.add_argument(
        "--variants", default="",
        help='JSON list of {"name", "params"}: timed back to back on the same GPU',
    )
    parser.add_argument("--repeats", type=int, default=3, help="Untimed train() runs per variant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch._inductor.config.triton.unique_kernel_names = True  # name fused kernels by their ops
    os.environ["CIFAR100_SUBMISSION_DIR"] = str(Path("submissions", args.submission).resolve())
    module = importlib.import_module("benchmark._submission")
    train_data = load_split(args.data_root, train=True)
    variants = json.loads(args.variants) if args.variants else [
        {"name": "params", "params": json.loads(args.params)}
    ]
    lines = [f"GPU: {torch.cuda.get_device_name(device)}", ""]
    summary = []

    for index, variant in enumerate(variants):
        torch._dynamo.reset()  # fresh compile per variant (cache limits are per code object)
        t = time.perf_counter()
        state = module.build(BuildContext(device=device, parameters=variant["params"]))
        build_time = time.perf_counter() - t

        def run(timed: bool, state=state):
            """One full prepare + train, optionally with per-component synchronised timers."""
            seed_everything(args.seed)
            module.prepare(state, train_data, args.seed)
            times = defaultdict(float)
            if timed:
                def wrap(fn, key):
                    def wrapped(*a, **k):
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        out = fn(*a, **k)
                        torch.cuda.synchronize()
                        times[key] += time.perf_counter() - t
                        return out
                    return wrapped
                original_epoch_images = module.epoch_images
                module.epoch_images = wrap(original_epoch_images, "data augmentation")
                state.sgd.step = wrap(state.sgd.step, "SGD step (head, biases)")
                state.filter_opt.step = wrap(state.filter_opt.step, "filter optimizer step")
            torch.cuda.synchronize()
            t = time.perf_counter()
            module.train(state)
            torch.cuda.synchronize()
            total = time.perf_counter() - t
            if timed:
                module.epoch_images = original_epoch_images
            return total, times

        totals = sorted(run(timed=False)[0] for _ in range(args.repeats))
        summary.append((variant["name"], totals, build_time))
        lines.append(f"[{variant['name']}] params={json.dumps(variant['params'])}")
        lines.append(f"  build {build_time:.0f} s; train() runs: "
                     + ", ".join(f"{x:.3f}" for x in totals) + f" s (min {totals[0]:.3f})")
        print("\n".join(lines[-2:]), flush=True)  # survive a later crash

        if index == len(variants) - 1:
            timed_total, times = run(timed=True)
            times["forward + backward + loss (remainder)"] = timed_total - sum(times.values())
            lines.append(f"  component shares (synchronised timers, total {timed_total:.3f} s):")
            for key, value in sorted(times.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {key:40s} {value:7.3f} s  {value / timed_total:6.1%}")

            seed_everything(args.seed)
            module.prepare(state, train_data, args.seed)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                module.train(state)
                torch.cuda.synchronize()
            events = prof.key_averages()

            def device_us(e):
                return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)

            def category(name):
                low = name.lower()
                if low.startswith("triton"):
                    return "fused elementwise (triton)"
                if "adaptivemax" in low:
                    return "adaptive max-pool backward"
                if any(k in low for k in ("cudnn", "xmma", "implicit_gemm", "conv")):
                    return "convolution (cudnn)"
                if "gemm" in low or "cublas" in low:
                    return "matmul (Muon Newton-Schulz)"
                return "other"

            grand = sum(device_us(e) for e in events)
            by_cat = defaultdict(float)
            for e in events:
                by_cat[category(e.key)] += device_us(e)
            lines.append(f"  GPU kernel time {grand / 1e6:.3f} s by category:")
            for cat, us in sorted(by_cat.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {cat:32s} {us / 1e6:7.3f} s  {us / grand:6.1%}")
            triton = sorted((e for e in events if e.key.lower().startswith("triton")),
                            key=device_us, reverse=True)[:20]
            lines.append("  top fused (triton) kernels:")
            for e in triton:
                lines.append(f"    {device_us(e) / 1e3:8.1f} ms  x{e.count:<5d} {e.key[:110]}")
        lines.append("")
        del state
        torch.cuda.empty_cache()

    base = summary[0][1][0]
    lines.append("Summary (min train() time, same GPU):")
    for name, totals, build_time in summary:
        lines.append(f"  {name:30s} {totals[0]:.3f} s  {totals[0] / base - 1:+.1%}  "
                     f"(build {build_time:.0f} s)")

    report = "\n".join(lines)
    print(report)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = args.results_root / "profiles" / args.submission / f"{run_id}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n")
    print(f"Results: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
