"""Development-only: train each variant once, then check where non-finite values appear.

Separates "training diverged" (non-finite weights) from "evaluation produced non-finite
logits" (compiled vs uncompiled forward, at full and final eval batch sizes).

    python -m dev.check_finite --submission airbench_muon --variants-file v.json
"""

import argparse
import importlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmark.api import BuildContext
from benchmark.data import load_split
from benchmark.worker import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--variants", required=True, help='JSON list of {"name", "params"}')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()

    device = torch.device("cuda")
    os.environ["CIFAR100_SUBMISSION_DIR"] = str(Path("submissions", args.submission).resolve())
    module = importlib.import_module("benchmark._submission")
    train_data = load_split(args.data_root, train=True)
    test = load_split(args.data_root, train=False)
    lines = [f"GPU: {torch.cuda.get_device_name(device)}"]

    for variant in json.loads(args.variants):
        torch._dynamo.reset()
        state = module.build(BuildContext(device=device, parameters=variant["params"]))
        seed_everything(args.seed)
        module.prepare(state, train_data, args.seed)
        model = module.train(state)
        bad = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
               if t.is_floating_point() and not torch.isfinite(t).all()]
        lines.append(f"[{variant['name']}] non-finite params/buffers after training: "
                     f"{bad or 'none'}")
        model.eval()
        with torch.inference_mode():
            for label, fn in (("compiled", model), ("uncompiled", model.forward)):
                for n in (1024, 784):
                    x = test.images[:n].to(device).float().div_(255)
                    logits = fn(x)
                    nonfinite = (~torch.isfinite(logits)).any(dim=1).sum().item()
                    acc = (logits.argmax(1).cpu() == test.labels[:n]).float().mean().item()
                    lines.append(f"  {label:10s} batch {n:4d}: rows with non-finite logits "
                                 f"{nonfinite:4d}/{n}, accuracy on those images {acc:.3f}")
        del state, model
        torch.cuda.empty_cache()

    report = "\n".join(lines)
    print(report)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = args.results_root / "profiles" / args.submission / f"finite-{run_id}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n")
    print(f"Results: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
