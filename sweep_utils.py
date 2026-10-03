"""Development sweep orchestration and reporting; never imported by the recipe."""

import csv
import json
import math
import statistics
import subprocess
from pathlib import Path

from submissions.it_compiles.architecture_config import ARCHITECTURE_KEYS, resolve_architecture
from submissions.it_compiles.lr_schedule import SCHEDULE_DEFAULTS, resolve_schedule

BASELINE = {
    "epochs": 60,
    "width": 64,
    "batch_size": 512,
    "warmup_epochs": 5,
    "cutout": 8,
    "label_smoothing": 0.05,
    "lr": 0.2,
    "weight_decay": 5e-4,
    "momentum": 0.9,
}
PHASE1_PROBES = [
    {},
    {"lr": 0.1},
    {"lr": 0.3},
    {"warmup_epochs": 0},
    {"warmup_epochs": 2},
    {"label_smoothing": 0.0},
    {"label_smoothing": 0.1},
    {"cutout": 0},
    {"cutout": 12},
    {"weight_decay": 2.5e-4},
    {"weight_decay": 1e-3},
    {"momentum": 0.85},
    {"momentum": 0.95},
    {"batch_size": 256},
    {"batch_size": 1024},
]
ACCURACY_TARGET = 0.75
SAFETY_TARGET = 0.755
REFERENCE_SECONDS = 81.4  # User-reported observation, not a sweep measurement.
REQUIRED_GPU = "NVIDIA A100 80GB PCIe"
SWEEP_GPUS = {REQUIRED_GPU, "NVIDIA A100-PCIE-80GB", "NVIDIA A100-SXM4-80GB"}


def require_sweep_gpu(names):
    """Accept one A100 80GB PCIe or SXM GPU for development screening."""
    if len(names) != 1 or names[0] not in SWEEP_GPUS:
        raise RuntimeError(
            f"Requested one A100 80GB PCIe or SXM GPU; provider supplied {names}. "
            "No training started."
        )


def resolve_parameters(parameters):
    """Expand overrides so recorded configurations always contain effective LR."""
    if not isinstance(parameters, list) or not parameters:
        raise ValueError("Provide a nonempty JSON list of parameter dictionaries")
    resolved = []
    for overrides in parameters:
        if not isinstance(overrides, dict):
            raise ValueError("Every configuration must be a parameter dictionary")
        unknown = overrides.keys() - BASELINE.keys() - ARCHITECTURE_KEYS - SCHEDULE_DEFAULTS.keys()
        if unknown:
            raise ValueError(f"Unknown recipe parameters: {sorted(unknown)}")
        params = BASELINE | {"epochs": 30} | overrides
        for key in ("epochs", "width", "batch_size", "warmup_epochs", "cutout"):
            if type(params[key]) is not int:
                raise ValueError(f"{key} must be an integer")
        if min(params[key] for key in ("epochs", "width", "batch_size")) < 1:
            raise ValueError("epochs, width, and batch_size must be positive")
        if params["warmup_epochs"] < 0 or not 0 <= params["cutout"] <= 32:
            raise ValueError("warmup_epochs >= 0 and cutout in [0, 32] required")
        # Match submission.build's default batch-dependent LR for custom lists.
        # The built-in phase-1 list passes explicit LR=0.2 for both batch probes.
        if "lr" not in overrides:
            params["lr"] = 0.2 * params["batch_size"] / 512
        for key in ("lr", "weight_decay", "momentum", "label_smoothing"):
            value = params[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        if params["lr"] <= 0 or params["weight_decay"] < 0:
            raise ValueError("lr > 0 and weight_decay >= 0 required")
        if not 0 <= params["momentum"] < 1 or not 0 <= params["label_smoothing"] < 1:
            raise ValueError("momentum and label_smoothing must be in [0, 1)")
        # Keep legacy recipe records unchanged when no architecture keys were
        # supplied; architecture sweeps record the complete effective structure.
        if ARCHITECTURE_KEYS.intersection(overrides):
            params.update(resolve_architecture(params))
        if SCHEDULE_DEFAULTS.keys() & overrides.keys():
            params.update(resolve_schedule(params))
        resolved.append(params)
    return resolved


def phase1_parameters():
    return [BASELINE | probe for probe in PHASE1_PROBES]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _empty_row(index, parameters, seeds):
    return {
        "configuration": f"config-{index:03d}",
        "parameters": parameters,
        "seeds": seeds,
        "seed_results": [],
        "completed_trials": 0,
        "metrics_trial_count": 0,
        "all_trials_completed": False,
        "mean_accuracy": None,
        "min_accuracy": None,
        "max_accuracy": None,
        "accuracy_std": None,
        "mean_prepare_time": None,
        "mean_train_time": None,
        "mean_total_timed_time": None,
        "total_timed_time_std": None,
        "mean_evaluation_time": None,
        "max_evaluation_time": None,
        "runtime_change_percent": None,
        "all_seeds_above_75": False,
        "screening_status": "incomplete",
        "benchmark_exit_code": None,
        "benchmark_process_succeeded": False,
        "benchmark_result_directory": None,
        "cuda_devices": [],
        "competition_gpu_match": None,
        "error": None,
    }


def collect_row(index, parameters, seeds, results_root, exit_code, error=None):
    """Read only organizer-produced result files, never dataset files or labels."""
    row = _empty_row(index, parameters, seeds)
    row["benchmark_exit_code"] = exit_code
    row["benchmark_process_succeeded"] = exit_code == 0
    row["error"] = error
    runs = sorted((Path(results_root) / "it_compiles").glob("*"))
    if len(runs) != 1:
        row["error"] = error or f"Expected one benchmark result directory; found {len(runs)}"
        return row
    result_dir = runs[0]
    row["benchmark_result_directory"] = str(result_dir)
    config_path = result_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        row["cuda_devices"] = config.get("environment", {}).get("cuda_devices", [])
        row["competition_gpu_match"] = row["cuda_devices"] == [REQUIRED_GPU]
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    trials_path = result_dir / "trials.jsonl"
    trials = (
        [json.loads(line) for line in trials_path.read_text().splitlines() if line.strip()]
        if trials_path.exists()
        else []
    )
    fields = (
        "seed",
        "status",
        "accuracy",
        "prepare_time",
        "train_time",
        "total_timed_time",
        "evaluation_time",
        "failure_reason",
    )
    row["seed_results"] = [{key: trial[key] for key in fields if key in trial} for trial in trials]
    successful = [trial for trial in trials if trial.get("status") == "ok"]
    row["completed_trials"] = len(successful)
    row["metrics_trial_count"] = len(successful)
    row["all_trials_completed"] = bool(
        summary.get("complete")
        and len(successful) == len(seeds)
        and len(trials) == len(seeds)
        and [trial.get("seed") for trial in trials] == seeds
        and summary.get("requested_trials") == len(seeds)
    )
    row["error"] = error or summary.get("run_error")
    if (
        not (row["all_trials_completed"] and row["benchmark_process_succeeded"])
        and not row["error"]
    ):
        row["error"] = "Incomplete trial set, seed mismatch, or benchmark failure"
    # Retain partial means for diagnostics, explicitly mark the trial count and
    # incompleteness, and never rank an incomplete run as qualifying.
    for source, target in (
        ("accuracy", "mean_accuracy"),
        ("prepare_time", "mean_prepare_time"),
        ("train_time", "mean_train_time"),
        ("total_timed_time", "mean_total_timed_time"),
        ("evaluation_time", "mean_evaluation_time"),
    ):
        values = [trial[source] for trial in successful]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Invalid benchmark measurements for {source}")
        row[target] = statistics.mean(values) if values else None
        if source in ("accuracy", "total_timed_time"):
            key = "accuracy_std" if source == "accuracy" else "total_timed_time_std"
            row[key] = statistics.stdev(values) if len(values) > 1 else None
    if successful:
        accuracies = [trial["accuracy"] for trial in successful]
        if any(value > 1 for value in accuracies):
            raise ValueError("Accuracy must be a fraction in [0, 1]")
        row["min_accuracy"], row["max_accuracy"] = min(accuracies), max(accuracies)
        row["max_evaluation_time"] = max(trial["evaluation_time"] for trial in successful)
        row["runtime_change_percent"] = (row["mean_total_timed_time"] / REFERENCE_SECONDS - 1) * 100
        row["all_seeds_above_75"] = (
            row["all_trials_completed"] and row["min_accuracy"] >= ACCURACY_TARGET
        )
    if row["all_trials_completed"] and row["benchmark_process_succeeded"]:
        row["screening_status"] = (
            "safe-screen"
            if row["mean_accuracy"] >= SAFETY_TARGET
            else "marginal"
            if row["mean_accuracy"] >= ACCURACY_TARGET
            else "non-qualifying"
        )
    return row


def ranked_rows(rows):
    def key(row):
        tier = {"safe-screen": 0, "marginal": 1, "non-qualifying": 2, "incomplete": 3}[
            row["screening_status"]
        ]
        runtime = row["mean_total_timed_time"]
        accuracy = row["mean_accuracy"]
        return (
            tier,
            runtime if tier == 0 else -(accuracy or 0),
            runtime if runtime is not None else math.inf,
            row["configuration"],
        )

    return sorted(rows, key=key)


def save_summaries(directory, rows):
    ranked = ranked_rows(rows)
    write_json(Path(directory) / "summary.json", ranked)
    if not ranked:
        return
    path = Path(directory) / "summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=ranked[0].keys())
        writer.writeheader()
        for row in ranked:
            writer.writerow(
                {
                    key: json.dumps(value, allow_nan=False)
                    if isinstance(value, dict | list)
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def print_ranking(rows):
    print("\nPROVISIONAL SCREENING — no final winner; validate shortlisted recipes on fresh seeds.")
    print("Safe-screen: completed mean >=75.5%; marginal: >=75%; other runs cannot qualify.")
    devices = sorted({name for row in rows for name in row.get("cuda_devices", [])})
    print(f"Observed CUDA devices: {devices or 'unknown'}; competition device: {REQUIRED_GPU}.")
    if any(row.get("competition_gpu_match") is False for row in rows):
        print("HARDWARE MISMATCH: these are development results, not PCIe benchmark measurements.")
    print(
        "Rank  Config      Status          Done  Mean%   Min%    Max%    SD(pp)  Prep(s)  "
        "Train(s)  Total(s)  vs81.4%"
    )

    def number(value, multiplier=1):
        return "--" if value is None else f"{value * multiplier:.2f}"

    for rank, row in enumerate(ranked_rows(rows), 1):
        print(
            f"{rank:>4}  {row['configuration']:<11} {row['screening_status']:<15} "
            f"{row['completed_trials']:>2}/{len(row['seeds']):<2} "
            f"{number(row['mean_accuracy'], 100):>6} "
            f"{number(row['min_accuracy'], 100):>7} "
            f"{number(row['max_accuracy'], 100):>7} "
            f"{number(row['accuracy_std'], 100):>7} "
            f"{number(row['mean_prepare_time']):>8} "
            f"{number(row['mean_train_time']):>9} "
            f"{number(row['mean_total_timed_time']):>9} "
            f"{number(row['runtime_change_percent']):>8}"
        )
        print(f"      GPU={row.get('cuda_devices', [])}")
        print(f"      parameters={json.dumps(row['parameters'], sort_keys=True)}")
    print(
        "Partial means use successful trials only; incomplete configurations are ineligible.",
        flush=True,
    )


def run_sweep(parameters, directory, *, n=3, seed=0, repo_root="/app", checkpoint=lambda: None):
    """One blocking benchmark process at a time; persist failures and keep going."""
    parameters = resolve_parameters(parameters)
    if type(n) is not int or n < 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("n must be positive and seed must be an unsigned 32-bit integer")
    if seed + n > 2**32:
        raise ValueError("Requested seed range exceeds unsigned 32-bit integers")
    seeds = list(range(seed, seed + n))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(
        directory / "request.json",
        {
            "parameters": parameters,
            "n": n,
            "seeds": seeds,
            "accuracy_target": ACCURACY_TARGET,
            "safety_target": SAFETY_TARGET,
            "reference_seconds_user_reported": REFERENCE_SECONDS,
            "note": "Development screening only; no automatic winner or recipe changes.",
        },
    )
    rows = []
    save_summaries(directory, rows)
    checkpoint()
    for index, params in enumerate(parameters, 1):
        config_dir = directory / f"config-{index:03d}"
        config_dir.mkdir()
        results_root = config_dir / "benchmark"
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "benchmark.run",
            "--submission",
            "it_compiles",
            "--n",
            str(n),
            "--seed",
            str(seed),
            "--no-accuracy-target",
            "--params",
            json.dumps(params, allow_nan=False),
            "--results-root",
            str(results_root),
        ]
        write_json(config_dir / "command.json", command)
        write_json(config_dir / "parameters.json", params)
        print(
            f"\nStarting {index}/{len(parameters)}: {json.dumps(params, sort_keys=True)}",
            flush=True,
        )
        exit_code, error = None, None
        try:
            # The harness enforces its normal build/trial/evaluation deadlines.
            # No retries, concurrent GPU work, or benchmark modifications.
            with (config_dir / "stdout.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            exit_code = result.returncode
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        try:
            row = collect_row(index, params, seeds, results_root, exit_code, error)
        except Exception as exc:
            row = _empty_row(index, params, seeds)
            row["benchmark_exit_code"] = exit_code
            row["error"] = f"Result parsing failed: {type(exc).__name__}: {exc}"
        rows.append(row)
        write_json(config_dir / "result.json", row)
        save_summaries(directory, rows)
        # The callback runs only between benchmark invocations, outside trials.
        checkpoint()
        log_path = config_dir / "stdout.log"
        if log_path.exists():
            print(log_path.read_text(encoding="utf-8"), flush=True)
        print(
            f"Finished {row['configuration']}: {row['screening_status']}; " f"error={row['error']}",
            flush=True,
        )
    print_ranking(rows)
    return rows
