"""Synthetic orchestration fixtures, not measured training results."""

import json

import pytest

from architecture_experiments import (
    TRAINING_RECIPE,
    architecture_parameters,
    eligible_for_promotion,
    resolve_architecture_parameters,
    run_architecture_stages,
)
from submissions.it_compiles.architecture_config import resolve_architecture
from sweep_utils import _empty_row, resolve_parameters


def test_control_and_independent_architecture_settings():
    assert resolve_architecture({}) == {
        "architecture": "resnet11",
        "stage_widths": [64, 128, 256, 512],
        "residual_blocks": [1, 1, 1],
        "global_pool": "max",
        "downsampling": "maxpool",
    }
    result = resolve_architecture(
        {
            "architecture": "resnet15",
            "stage_widths": [48, 112, 240, 384],
            "residual_blocks": [0, 3, 2],
            "global_pool": "avg",
            "downsampling": "stride",
        }
    )
    assert result["stage_widths"] == [48, 112, 240, 384]
    assert result["residual_blocks"] == [0, 3, 2]
    assert resolve_architecture({"width": 32})["stage_widths"] == [32, 64, 128, 256]
    assert resolve_parameters([{}])[0]["epochs"] == 30


@pytest.mark.parametrize(
    "parameters",
    [
        {"architecture": "unknown"},
        {"architecture": []},
        {"width": 1.5},
        {"stage_widths": [64, 128, 256]},
        {"stage_widths": [64, 128, 256, 0]},
        {"stage_widths": [64, 128, 256, True]},
        {"residual_blocks": [1, -1, 1]},
        {"residual_blocks": [1, 0]},
        {"residual_blocks": [1, 0, 1.5]},
        {"global_pool": "median"},
        {"downsampling": "blur"},
    ],
)
def test_invalid_architectures(parameters):
    with pytest.raises(ValueError):
        resolve_architecture(parameters)


def test_eight_unique_architectures_keep_recipe_fixed():
    configs = architecture_parameters()
    assert len(configs) == 8
    assert len({json.dumps(config, sort_keys=True) for config in configs}) == 8
    for config in configs:
        assert all(config[key] == value for key, value in TRAINING_RECIPE.items())
    with pytest.raises(ValueError, match="holds epochs=30"):
        resolve_architecture_parameters([{"epochs": 60}])
    with pytest.raises(ValueError, match="holds lr=0.2"):
        resolve_architecture_parameters([{"lr": 0.1}])


def measured_fixture(job, accuracy):
    # Arbitrary fixture values only, never presented as benchmark observations.
    return _empty_row(1, job["parameters"], list(range(job["seed"], job["seed"] + job["n"]))) | {
        "all_trials_completed": True,
        "benchmark_process_succeeded": True,
        "completed_trials": job["n"],
        "metrics_trial_count": job["n"],
        "mean_accuracy": accuracy,
        "mean_prepare_time": 1,
        "mean_train_time": 9,
        "mean_total_timed_time": 10,
        "screening_status": "non-qualifying",
    }


def test_parallel_stage_promotes_threshold_and_uses_three_fresh_seeds(tmp_path):
    batches, checkpoints = [], []

    def execute(jobs):
        batches.append(jobs)
        for index, job in enumerate(jobs):
            if job["stage"] == "validation":
                assert job["n"] == 3 and job["seed"] == 100
                yield measured_fixture(job, 0.746)
            elif index == 3:
                yield RuntimeError("synthetic remote failure")
            else:
                yield measured_fixture(job, [0.745, 0.7449, 0.76][index])

    def checkpoint():
        checkpoints.append(True)

    rows = run_architecture_stages(
        architecture_parameters()[:4],
        tmp_path / "run",
        execute_stage=execute,
        checkpoint=checkpoint,
    )
    assert len(batches) == 2 and [len(batch) for batch in batches] == [4, 2]
    assert [row["configuration"] for row in rows if row["promoted"]] == ["arch-001", "arch-003"]
    assert batches[1][0]["parameters"] == batches[0][0]["parameters"]
    assert batches[1][1]["parameters"] == batches[0][2]["parameters"]
    assert rows[3]["error"] and not rows[3]["promoted"]
    assert len(rows) == 6
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert len(summary) == 6 and (tmp_path / "run" / "summary.csv").exists()
    assert all(row["stage"] in ("screen", "validation") for row in summary)
    assert len(checkpoints) == 9
    assert not json.loads((tmp_path / "run" / "completion.json").read_text())[
        "all_benchmarks_completed"
    ]


def test_partial_high_accuracy_is_ineligible_and_no_validation_if_none_promoted(tmp_path):
    jobs_seen = []

    def execute(jobs):
        jobs_seen.extend(jobs)
        row = measured_fixture(jobs[0], 0.9) | {"all_trials_completed": False}
        assert not eligible_for_promotion(row)
        yield row

    rows = run_architecture_stages(
        architecture_parameters()[:1],
        tmp_path / "run",
        execute_stage=execute,
    )
    assert len(jobs_seen) == len(rows) == 1 and not rows[0]["promoted"]
    assert json.loads((tmp_path / "run" / "promotion.json").read_text())["configurations"] == []


def test_overlapping_validation_seeds_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="different"):
        run_architecture_stages(
            architecture_parameters(),
            tmp_path / "run",
            execute_stage=lambda jobs: [],
            screen_seed=101,
            validation_seed=100,
        )
