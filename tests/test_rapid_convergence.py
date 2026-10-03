"""Pure schedule and synthetic workflow checks; no GPU measurements."""

import json
import math

import pytest

from rapid_convergence_experiments import (
    SHORT_BASE,
    rapid_parameters,
    reliable,
    resolve_rapid_parameters,
    run_rapid_stages,
)
from submissions.it_compiles.lr_schedule import learning_rates, resolve_schedule
from sweep_utils import _empty_row, resolve_parameters


@pytest.mark.parametrize("warmup", [0, 98, 196, 490])
def test_cosine_endpoints_length_and_ramp(warmup):
    rates = learning_rates(2940, warmup, 0.2)
    assert len(rates) == 2940 and all(math.isfinite(rate) and rate > 0 for rate in rates)
    assert max(rates) == pytest.approx(0.2)
    assert rates[-1] == pytest.approx(0.0002)
    if warmup:
        assert rates[0] == pytest.approx(0.2 * (0.1 + 0.9 / warmup))
        assert all(a <= b for a, b in zip(rates[: warmup - 1], rates[1:warmup]))
    assert all(a >= b for a, b in zip(rates[warmup:-1], rates[warmup + 1 :]))


@pytest.mark.parametrize("warmup", [0, 1, 98, 196])
def test_one_cycle_has_one_peak_and_reaches_minimum(warmup):
    rates = learning_rates(2940, warmup, 0.2, {"lr_schedule": "one_cycle"})
    assert len(rates) == 2940 and max(rates) == pytest.approx(0.2)
    assert rates[-1] == pytest.approx(0.2 / 25 / 10000)
    assert rates[0] == pytest.approx(0.2 / 25 if warmup > 1 else 0.2)
    assert all(math.isfinite(rate) and rate > 0 for rate in rates)


@pytest.mark.parametrize(
    "params",
    [
        {"lr_schedule": "unknown"},
        {"one_cycle_div_factor": 0},
        {"one_cycle_final_div_factor": float("nan")},
        {"one_cycle_div_factor": True},
    ],
)
def test_invalid_schedule_options_are_rejected(params):
    with pytest.raises(ValueError):
        resolve_schedule(params)
    with pytest.raises(ValueError):
        resolve_parameters([params])


def test_twelve_probes_change_only_recipe_and_hold_budget():
    configs = rapid_parameters()
    assert len(configs) == 12
    assert len({json.dumps(params, sort_keys=True) for params in configs}) == 12
    for params in configs:
        assert params["epochs"] == 30 and params["batch_size"] == 512 and params["momentum"] == 0.9
        assert params["architecture"] == "resnet11" and params["stage_widths"] == [
            64,
            128,
            256,
            512,
        ]
        assert params["residual_blocks"] == [1, 1, 1]
        assert params["downsampling"] == "maxpool" and params["global_pool"] == "max"
    assert configs[0]["warmup_epochs"] == 5
    assert {params["warmup_epochs"] for params in configs[1:]} == {0, 1, 2}
    assert configs[1] == SHORT_BASE
    for index in (2, 3, 4, 5, 6, 7, 8, 9):
        assert sum(configs[index][key] != SHORT_BASE[key] for key in SHORT_BASE) == 1
    assert configs[10] == configs[5] | {"lr_schedule": "one_cycle"}


@pytest.mark.parametrize(
    "params",
    [
        {"architecture": "resnet15"},
        {"stage_widths": [64, 128, 256, 384]},
        {"global_pool": "avg"},
        {"downsampling": "stride"},
        {"epochs": 60},
        {"batch_size": 256},
        {"momentum": 0.95},
    ],
)
def test_frozen_architecture_and_fixed_budget(params):
    with pytest.raises(ValueError):
        resolve_rapid_parameters([params])


def fixture_row(job, mean=0.756, minimum=0.752):
    return _empty_row(1, job["parameters"], list(range(job["seed"], job["seed"] + job["n"]))) | {
        "all_trials_completed": True,
        "benchmark_process_succeeded": True,
        "completed_trials": job["n"],
        "metrics_trial_count": job["n"],
        "mean_accuracy": mean,
        "min_accuracy": minimum,
        "mean_total_timed_time": 10,
        "screening_status": "safe-screen",
        "cuda_devices": ["NVIDIA A100 80GB PCIe"],
    }


def test_promotion_fresh_validation_and_exact_paired_transfer(tmp_path):
    batches, comparisons = [], []

    def execute(jobs):
        batches.append(jobs)
        for index, job in enumerate(jobs):
            if job["stage"] == "screen":
                yield fixture_row(job, [0.747, 0.7469, 0.76][index])
            else:
                assert job["n"] == 3 and job["seed"] == 100
                yield fixture_row(job, 0.756, 0.753)

    def compare(job):
        comparisons.append(job)
        standard, narrow = job["parameters"]
        assert job["n"] == 3 and job["seed"] == 200
        assert narrow == standard | {"stage_widths": [64, 128, 256, 384]}
        return [fixture_row(job | {"parameters": params}) for params in job["parameters"]]

    rows = run_rapid_stages(
        rapid_parameters()[:3],
        tmp_path / "run",
        execute_stage=execute,
        execute_comparison=compare,
    )
    assert [len(batch) for batch in batches] == [3, 2]
    assert len(comparisons) == 1 and len(rows) == 7
    assert [row["configuration"] for row in rows if row["promoted"]] == ["recipe-001", "recipe-003"]
    assert len(json.loads((tmp_path / "run" / "summary.json").read_text())) == 7
    assert json.loads((tmp_path / "run" / "completion.json").read_text())[
        "width_comparison_executed"
    ]


def test_lucky_partial_or_low_minimum_does_not_trigger_width_comparison(tmp_path):
    batches = []

    def execute(jobs):
        batches.append(jobs)
        for index, job in enumerate(jobs):
            row = fixture_row(job)
            if job["stage"] == "screen" and index == 0:
                row["all_trials_completed"] = False
            if job["stage"] == "validation":
                row["min_accuracy"] = 0.749
                assert not reliable(row)
            yield row

    def forbidden(job):
        raise AssertionError("Width comparison must not run without reliable validation")

    rows = run_rapid_stages(
        rapid_parameters()[:2],
        tmp_path / "run",
        execute_stage=execute,
        execute_comparison=forbidden,
    )
    assert [len(batch) for batch in batches] == [2, 1] and len(rows) == 3
    assert not any(row["reliability_gate_passed"] for row in rows)
    assert json.loads((tmp_path / "run" / "selection.json").read_text())["parameters"] is None


def test_remote_failures_are_recorded_and_other_configurations_continue(tmp_path):
    def execute(jobs):
        for index, job in enumerate(jobs):
            yield RuntimeError("fixture remote failure") if index == 0 else fixture_row(job, 0.74)

    rows = run_rapid_stages(
        rapid_parameters()[:2],
        tmp_path / "run",
        execute_stage=execute,
        execute_comparison=lambda job: None,
    )
    assert len(rows) == 2 and rows[0]["error"] and rows[1]["all_trials_completed"]


def test_seed_overlap_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        run_rapid_stages(
            rapid_parameters(),
            tmp_path / "run",
            execute_stage=lambda jobs: [],
            execute_comparison=lambda job: [],
            validation_seed=200,
        )
