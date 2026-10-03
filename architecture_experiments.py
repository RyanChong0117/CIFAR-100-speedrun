"""Architecture-only screening and fresh-seed promotion, without dataset access."""

from pathlib import Path

from sweep_utils import (
    BASELINE,
    _empty_row,
    print_ranking,
    resolve_parameters,
    save_summaries,
    write_json,
)

PROMOTION_ACCURACY = 0.745
TRAINING_RECIPE = BASELINE | {"epochs": 30}
ARCHITECTURE_PROBES = [
    ("control", {"architecture": "resnet9"}),
    ("middle-residual", {"architecture": "resnet11"}),
    ("deeper", {"architecture": "resnet15"}),
    ("deeper-average-head", {"architecture": "resnet15", "global_pool": "avg"}),
    ("narrow-head", {"architecture": "resnet11", "stage_widths": [64, 128, 256, 384]}),
    (
        "narrow-head-average",
        {"architecture": "resnet11", "stage_widths": [64, 128, 256, 384], "global_pool": "avg"},
    ),
    ("stride-downsample", {"architecture": "resnet11", "downsampling": "stride"}),
    ("average-downsample", {"architecture": "resnet11", "downsampling": "avgpool"}),
]


def architecture_parameters():
    return resolve_architecture_parameters(
        [TRAINING_RECIPE | probe for _, probe in ARCHITECTURE_PROBES]
    )


def resolve_architecture_parameters(parameters):
    # Supply the architecture key even for a custom control, to record effective
    # widths/depth/pooling rather than only a model nickname.
    if not isinstance(parameters, list) or not parameters:
        raise ValueError("Provide a nonempty list of architecture parameter dictionaries")
    if any(not isinstance(params, dict) for params in parameters):
        raise ValueError("Every architecture must be a parameter dictionary")
    resolved = resolve_parameters([{"architecture": "resnet9"} | params for params in parameters])
    for params in resolved:
        for key, value in TRAINING_RECIPE.items():
            if key != "width" and params[key] != value:
                raise ValueError(f"Architecture study holds {key}={value} fixed")
    return resolved


def eligible_for_promotion(row):
    return (
        row["all_trials_completed"]
        and row["benchmark_process_succeeded"]
        and row["metrics_trial_count"] == 1
        and row["mean_accuracy"] is not None
        and row["mean_accuracy"] >= PROMOTION_ACCURACY
    )


def run_architecture_stages(
    parameters,
    directory,
    *,
    execute_stage,
    checkpoint=lambda: None,
    screen_seed=0,
    validation_seed=100,
):
    """execute_stage maps independent jobs in parallel and returns results in input order."""
    parameters = resolve_architecture_parameters(parameters)
    for seed, count in ((screen_seed, 1), (validation_seed, 3)):
        if type(seed) is not int or not 0 <= seed or seed + count > 2**32:
            raise ValueError("Seeds must describe unsigned 32-bit seed ranges")
    if screen_seed in range(validation_seed, validation_seed + 3):
        raise ValueError("Validation must use three seeds different from the screening seed")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(
        directory / "request.json",
        {
            "parameters": parameters,
            "screen_seeds": [screen_seed],
            "validation_seeds": list(range(validation_seed, validation_seed + 3)),
            "promotion_accuracy": PROMOTION_ACCURACY,
            "note": "Parallel independent single-GPU development runs; no automatic winner.",
        },
    )
    checkpoint()
    rows = []

    def stage(jobs):
        for job, result in zip(jobs, execute_stage(jobs), strict=True):
            if isinstance(result, Exception):
                result = _empty_row(
                    1, job["parameters"], list(range(job["seed"], job["seed"] + job["n"]))
                ) | {"error": f"Remote job failed: {type(result).__name__}: {result}"}
            row = result | {"configuration": job["configuration"], "stage": job["stage"]}
            row["promoted"] = eligible_for_promotion(row) if job["stage"] == "screen" else False
            rows.append(row)
            save_summaries(directory, rows)
            checkpoint()
            print(
                f"{row['stage']} {row['configuration']}: accuracy={row['mean_accuracy']}, "
                f"prepare+train={row['mean_total_timed_time']}, promoted={row['promoted']}, "
                f"error={row['error']}",
                flush=True,
            )

    stage(
        [
            {
                "configuration": f"arch-{index:03d}",
                "stage": "screen",
                "parameters": params,
                "seed": screen_seed,
                "n": 1,
                "directory": str(directory / "screen" / f"arch-{index:03d}"),
            }
            for index, params in enumerate(parameters, 1)
        ]
    )
    promoted = [row for row in rows if row["promoted"]]
    write_json(
        directory / "promotion.json",
        {
            "threshold": PROMOTION_ACCURACY,
            "configurations": [row["configuration"] for row in promoted],
            "validation_seeds": list(range(validation_seed, validation_seed + 3)),
        },
    )
    checkpoint()
    if promoted:
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
    else:
        print("No completed architecture reached 74.5%; no validation jobs launched.", flush=True)
    for name in ("screen", "validation"):
        selected = [row for row in rows if row["stage"] == name]
        if selected:
            print(f"\n{name.upper()} RESULTS", flush=True)
            print_ranking(selected)
    write_json(
        directory / "completion.json",
        {
            "workflow_completed": True,
            "screen_configurations": len(parameters),
            "promoted_configurations": len(promoted),
            "all_benchmarks_completed": all(
                row["all_trials_completed"] and row["benchmark_process_succeeded"] for row in rows
            ),
        },
    )
    checkpoint()
    return rows
