"""Development-only accuracy-vs-time curve for one training run.

Trains a submission once and measures test accuracy after every `--eval-every`
epochs, with the clock paused during evaluation. This answers "how many seconds of
training does this recipe need to reach X%?" in a single run, which the official
harness (final accuracy only) cannot.

This is NOT a score. Test accuracy here is used only to compare recipes during
development, which the rules allow. The submission itself never sees the test set:
evaluation happens through a dev-only `state.epoch_callback`, which is None under
the official harness.

    python -m dev.curve --submission baseline_resnet9 --seed 0 --params '{"epochs": 20}'
"""

import argparse
import importlib
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmark.api import BuildContext
from benchmark.data import load_split
from benchmark.evaluate import accuracy, predict
from benchmark.worker import seed_everything


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, help="Folder name under submissions/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params", default="{}", help="JSON object passed to build()")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--target", type=float, default=0.75)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()

    params = json.loads(args.params)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    os.environ["CIFAR100_SUBMISSION_DIR"] = str(Path("submissions", args.submission).resolve())
    module = importlib.import_module("benchmark._submission")

    train_data = load_split(args.data_root, train=True)
    test_data = load_split(args.data_root, train=False)

    state = module.build(BuildContext(device=device, parameters=params))
    seed_everything(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    points = []
    clock = {"elapsed": 0.0, "start": None}

    def on_epoch(epoch: int, step: int) -> None:
        sync(device)
        clock["elapsed"] += time.perf_counter() - clock["start"]
        predictions, _ = predict(state.model, test_data.images, device, 1024)
        acc = accuracy(predictions, test_data.labels)
        points.append({"epoch": epoch, "step": step, "time": clock["elapsed"], "accuracy": acc})
        print(f"epoch {epoch:3d}  step {step:6d}  {clock['elapsed']:7.2f}s  acc {acc:.4f}")
        state.model.train()
        sync(device)
        clock["start"] = time.perf_counter()

    def callback(epoch: int, step: int) -> None:
        if epoch % args.eval_every == 0:
            on_epoch(epoch, step)

    sync(device)
    clock["start"] = time.perf_counter()
    module.prepare(state, train_data, args.seed)
    state.epoch_callback = callback
    model = module.train(state)
    sync(device)
    clock["elapsed"] += time.perf_counter() - clock["start"]

    if not points or points[-1]["step"] != getattr(state, "steps", None):
        predictions, _ = predict(model, test_data.images, device, 1024)
        points.append(
            {
                "epoch": None,
                "step": getattr(state, "steps", None),
                "time": clock["elapsed"],
                "accuracy": accuracy(predictions, test_data.labels),
            }
        )

    reached = next((p for p in points if p["accuracy"] >= args.target), None)
    cfg = getattr(state, "cfg", params)
    steps = getattr(state, "steps", None)
    batch_size = cfg.get("batch_size")
    result = {
        "kind": "curve",
        "submission": args.submission,
        "seed": args.seed,
        "parameters": params,
        "resolved_config": cfg,
        "target": args.target,
        "time_to_target": reached["time"] if reached else None,
        "epoch_to_target": reached["epoch"] if reached else None,
        "final_accuracy": points[-1]["accuracy"],
        "training_time": clock["elapsed"],
        "steps": steps,
        "throughput_img_per_s": (
            steps * batch_size / clock["elapsed"] if steps and batch_size else None
        ),
        "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "points": points,
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = args.results_root / "curves" / args.submission / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "points"}, indent=2))
    print(f"Results: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
