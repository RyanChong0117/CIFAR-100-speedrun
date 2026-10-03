"""Collect every harness run and dev curve under results/ into results/results.csv.

One row per experiment. Harness runs (results/<submission>/<run_id>/) carry the
score-relevant numbers: mean prepare+train time and mean accuracy over n seeds.
Curve runs (results/curves/...) add time-to-target, throughput and peak memory.

Put "experiment_name" and "hypothesis" in --params; they land in the CSV.

    python -m dev.collect_results            # writes results/results.csv, prints the best run
"""

import argparse
import csv
import json
from pathlib import Path

KNOBS = [
    "epochs",
    "batch_size",
    "width",
    "depth",
    "optimizer",
    "lr",
    "weight_decay",
    "momentum",
    "label_smoothing",
    "augmentation",
    "resolution_schedule",
]
COLUMNS = [
    "kind",
    "experiment_name",
    "submission",
    "run_id",
    "n_trials",
    "seeds",
    *KNOBS,
    "mean_accuracy",
    "accuracy_std",
    "mean_time",
    "time_std",
    "qualified",
    "time_to_target",
    "final_accuracy",
    "steps",
    "throughput_img_per_s",
    "peak_memory_gb",
    "gpu",
    "parameters",
    "hypothesis",
]


def knob_values(resolved: dict, params: dict) -> dict:
    return {k: json.dumps(v) if isinstance(v, list | dict) else v
            for k in KNOBS if (v := params.get(k, resolved.get(k))) is not None}


def harness_rows(results: Path):
    for summary_path in sorted(results.glob("*/*/summary.json")):
        if summary_path.parts[-3] == "curves":
            continue
        run = summary_path.parent
        summary = json.loads(summary_path.read_text())
        config_path = run / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        params = config.get("parameters", {})
        devices = config.get("environment", {}).get("cuda_devices") or []
        yield {
            "kind": "harness",
            "experiment_name": params.get("experiment_name", ""),
            "submission": run.parent.name,
            "run_id": run.name,
            "n_trials": summary.get("number_of_trials"),
            "seeds": json.dumps(config.get("seeds", [])),
            **knob_values({}, params),
            "mean_accuracy": summary.get("mean_accuracy"),
            "accuracy_std": summary.get("accuracy_std"),
            "mean_time": summary.get("mean_training_time"),
            "time_std": summary.get("training_time_std"),
            "qualified": summary.get("qualified"),
            "gpu": ";".join(devices),
            "parameters": json.dumps(params, sort_keys=True),
            "hypothesis": params.get("hypothesis", ""),
        }


def curve_rows(results: Path):
    for path in sorted((results / "curves").glob("*/*.json")):
        curve = json.loads(path.read_text())
        params = curve.get("parameters", {})
        yield {
            "kind": "curve",
            "experiment_name": params.get("experiment_name", ""),
            "submission": curve["submission"],
            "run_id": path.stem,
            "n_trials": 1,
            "seeds": json.dumps([curve["seed"]]),
            **knob_values(curve.get("resolved_config", {}), params),
            "mean_accuracy": curve["final_accuracy"],
            "mean_time": curve["training_time"],
            "time_to_target": curve["time_to_target"],
            "final_accuracy": curve["final_accuracy"],
            "steps": curve["steps"],
            "throughput_img_per_s": curve["throughput_img_per_s"],
            "peak_memory_gb": curve["peak_memory_gb"],
            "gpu": curve["gpu"],
            "parameters": json.dumps(params, sort_keys=True),
            "hypothesis": params.get("hypothesis", ""),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--target", type=float, default=0.75)
    parser.add_argument(
        "--min-trials", type=int, default=10,
        help="Seeds a run needs to count as best (2-seed means are too noisy near the target)",
    )
    args = parser.parse_args()

    rows = [*harness_rows(args.results_root), *curve_rows(args.results_root)]
    rows.sort(key=lambda r: r["run_id"])
    out = args.results_root / "results.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")

    # The score is mean time of a complete multi-seed run whose mean accuracy clears target.
    qualifying = [
        r for r in rows
        if r["kind"] == "harness" and r["mean_accuracy"] is not None
        and r["mean_accuracy"] >= args.target and (r["n_trials"] or 0) >= args.min_trials
    ]
    if qualifying:
        best = min(qualifying, key=lambda r: r["mean_time"])
        print(
            f"Best harness run >= {args.target:.0%}: {best['submission']} {best['run_id']} "
            f"({best['experiment_name'] or 'unnamed'}) "
            f"{best['mean_time']:.2f}s, acc {best['mean_accuracy']:.4f} "
            f"over {best['n_trials']} seeds"
        )
    else:
        print(f"No harness run with >= {args.min_trials} seeds has reached {args.target:.0%} yet.")


if __name__ == "__main__":
    main()
