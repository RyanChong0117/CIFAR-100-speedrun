"""Fixed ResNet11/30-epoch recipe screening, validation, and a gated width comparison."""

from pathlib import Path

from submissions.it_compiles.architecture_config import resolve_architecture
from submissions.it_compiles.lr_schedule import SCHEDULE_DEFAULTS
from sweep_utils import (
    BASELINE,
    _empty_row,
    print_ranking,
    resolve_parameters,
    save_summaries,
    write_json,
)

PROMOTION_ACCURACY = 0.747
RELIABLE_ACCURACY = 0.755
STANDARD_MODEL = resolve_architecture({"architecture": "resnet11", "width": 64})
SHORT_BASE = BASELINE | SCHEDULE_DEFAULTS | STANDARD_MODEL | {"epochs": 30, "warmup_epochs": 1}
RAPID_PROBES = [
    ("previous-control", {"warmup_epochs": 5}),
    ("one-epoch-anchor", {}),
    ("no-warmup", {"warmup_epochs": 0}),
    ("two-epoch-warmup", {"warmup_epochs": 2}),
    ("lower-lr", {"lr": 0.12}),
    ("higher-lr", {"lr": 0.28}),
    ("no-smoothing", {"label_smoothing": 0.0}),
    ("no-cutout", {"cutout": 0}),
    ("higher-decay", {"weight_decay": 0.001}),
    ("one-cycle", {"lr_schedule": "one_cycle"}),
    ("one-cycle-higher-lr", {"lr_schedule": "one_cycle", "lr": 0.28}),
    (
        "combined-short-recipe",
        {
            "lr_schedule": "one_cycle",
            "warmup_epochs": 2,
            "label_smoothing": 0.0,
            "cutout": 4,
            "weight_decay": 0.001,
        },
    ),
]


def rapid_parameters():
    return resolve_rapid_parameters([SHORT_BASE | probe for _, probe in RAPID_PROBES])


def resolve_rapid_parameters(parameters):
    if (
        not isinstance(parameters, list)
        or not parameters
        or any(not isinstance(params, dict) for params in parameters)
    ):
        raise ValueError("Provide a nonempty list of recipe parameter dictionaries")
    configs = resolve_parameters([SHORT_BASE | overrides for overrides in parameters])
    for params in configs:
        for key, value in STANDARD_MODEL.items():
            if params[key] != value:
                raise ValueError(f"Architecture is frozen: {key} must be {value}")
        for key in ("epochs", "batch_size", "width", "momentum"):
            if params[key] != SHORT_BASE[key]:
                raise ValueError(f"Recipe study holds {key}={SHORT_BASE[key]} fixed")
        if params["warmup_epochs"] not in (0, 1, 2, 5):
            raise ValueError("Use warmup 0-2 epochs, or 5 for the historical control")
    return configs


def completed(row, count):
    return (
        row["all_trials_completed"]
        and row["benchmark_process_succeeded"]
        and row["metrics_trial_count"] == count
        and row["mean_accuracy"] is not None
    )


def reliable(row):
    # Three seeds are provisional evidence, not official qualification. Require
    # both the requested mean margin and no below-target validation trial.
    return (
        completed(row, 3)
        and row["mean_accuracy"] >= RELIABLE_ACCURACY
        and row["min_accuracy"] is not None
        and row["min_accuracy"] >= 0.75
    )


def run_rapid_stages(
    parameters,
    directory,
    *,
    execute_stage,
    execute_comparison,
    checkpoint=lambda: None,
    screen_seed=0,
    validation_seed=100,
    comparison_seed=200,
):
    """Parallel stages use fresh seeds; the paired width comparison uses one GPU."""
    parameters = resolve_rapid_parameters(parameters)
    ranges = []
    for seed, count in ((screen_seed, 1), (validation_seed, 3), (comparison_seed, 3)):
        if type(seed) is not int or seed < 0 or seed + count > 2**32:
            raise ValueError("Use unsigned 32-bit seed ranges")
        requested = set(range(seed, seed + count))
        if any(requested & previous for previous in ranges):
            raise ValueError("Screening, validation, and comparison seeds must be disjoint")
        ranges.append(requested)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(
        directory / "request.json",
        {
            "parameters": parameters,
            "screen_seeds": [screen_seed],
            "validation_seeds": list(range(validation_seed, validation_seed + 3)),
            "comparison_seeds": list(range(comparison_seed, comparison_seed + 3)),
            "promotion_accuracy": PROMOTION_ACCURACY,
            "reliable_mean_accuracy": RELIABLE_ACCURACY,
            "reliable_min_accuracy": 0.75,
            "note": "Development screening; architecture and epoch budget fixed; no final winner.",
        },
    )
    checkpoint()
    rows = []

    def append(job, result):
        if isinstance(result, Exception):
            result = _empty_row(
                1, job["parameters"], list(range(job["seed"], job["seed"] + job["n"]))
            ) | {"error": f"Remote job failed: {type(result).__name__}: {result}"}
        row = result | {"configuration": job["configuration"], "stage": job["stage"]}
        row["promoted"] = (
            job["stage"] == "screen"
            and completed(row, 1)
            and row["mean_accuracy"] >= PROMOTION_ACCURACY
        )
        row["reliability_gate_passed"] = job["stage"] == "validation" and reliable(row)
        rows.append(row)
        save_summaries(directory, rows)
        checkpoint()
        print(
            f"{row['stage']} {row['configuration']}: accuracy={row['mean_accuracy']}, "
            f"prepare+train={row['mean_total_timed_time']}, promoted={row['promoted']}, "
            f"reliable={row['reliability_gate_passed']}, GPU={row['cuda_devices']}, "
            f"error={row['error']}",
            flush=True,
        )
        return row

    def stage(jobs):
        return [append(job, result) for job, result in zip(jobs, execute_stage(jobs), strict=True)]

    screens = stage(
        [
            {
                "configuration": f"recipe-{index:03d}",
                "stage": "screen",
                "parameters": params,
                "seed": screen_seed,
                "n": 1,
                "directory": str(directory / "screen" / f"recipe-{index:03d}"),
            }
            for index, params in enumerate(parameters, 1)
        ]
    )
    promoted = [row for row in screens if row["promoted"]]
    write_json(
        directory / "promotion.json",
        {
            "threshold": PROMOTION_ACCURACY,
            "configurations": [row["configuration"] for row in promoted],
            "validation_seeds": list(range(validation_seed, validation_seed + 3)),
        },
    )
    checkpoint()
    validations = (
        stage(
            [
                {
                    "configuration": row["configuration"],
                    "stage": "validation",
                    "parameters": row["parameters"],
                    "seed": validation_seed,
                    "n": 3,
                    "directory": str(directory / "validation" / row["configuration"]),
                }
                for row in promoted
            ]
        )
        if promoted
        else []
    )
    eligible = [row for row in validations if row["reliability_gate_passed"]]
    selected = min(eligible, key=lambda row: row["mean_total_timed_time"]) if eligible else None
    write_json(
        directory / "selection.json",
        {
            "selected_configuration": selected["configuration"] if selected else None,
            "parameters": selected["parameters"] if selected else None,
            "eligible_configurations": [row["configuration"] for row in eligible],
            "selection_rule": "Lowest observed timed runtime among provisional gate passes.",
            "note": "Fresh paired measurements required; GPU variant times are not equivalent.",
            "reason": None
            if selected
            else "No three-seed recipe met mean >=75.5% and minimum >=75%",
        },
    )
    checkpoint()
    comparison = []
    if selected:
        standard = selected["parameters"]
        narrow = standard | {"stage_widths": [64, 128, 256, 384]}
        job = {
            "configuration": selected["configuration"],
            "parameters": [standard, narrow],
            "directory": str(directory / "comparison"),
            "n": 3,
            "seed": comparison_seed,
        }
        result = execute_comparison(job)
        results = [result, result] if isinstance(result, Exception) else result
        for params, stage_name, result in zip(
            (standard, narrow),
            ("comparison-standard", "comparison-width384"),
            results,
            strict=True,
        ):
            comparison.append(
                append(
                    job | {"parameters": params, "stage": stage_name},
                    result,
                )
            )
        write_json(
            directory / "width_comparison.json",
            {
                "standard": comparison[0],
                "width384": comparison[1],
                "same_device_variant": bool(comparison[0]["cuda_devices"])
                and comparison[0]["cuda_devices"] == comparison[1]["cuda_devices"],
                "same_remote_gpu_allocation": True,
                "accuracy_difference_pp": (
                    (comparison[1]["mean_accuracy"] - comparison[0]["mean_accuracy"]) * 100
                    if all(completed(row, 3) for row in comparison)
                    else None
                ),
                "timed_runtime_difference_seconds": (
                    comparison[1]["mean_total_timed_time"] - comparison[0]["mean_total_timed_time"]
                    if all(completed(row, 3) for row in comparison)
                    else None
                ),
            },
        )
        checkpoint()
    else:
        print("Reliability gate not met; width-384 comparison skipped.", flush=True)
    for stage_name in ("screen", "validation", "comparison-standard", "comparison-width384"):
        chosen = [row for row in rows if row["stage"] == stage_name]
        if chosen:
            print(f"\n{stage_name.upper()} RESULTS", flush=True)
            print_ranking(chosen)
    write_json(
        directory / "completion.json",
        {
            "workflow_completed": True,
            "screen_configurations": len(parameters),
            "promoted_configurations": len(promoted),
            "reliable_configurations": len(eligible),
            "width_comparison_executed": bool(comparison),
            "all_benchmarks_completed": all(completed(row, len(row["seeds"])) for row in rows),
        },
    )
    checkpoint()
    return rows
