"""Synthetic result fixtures test orchestration only; no measured GPU results."""

import csv
import json
import math
import subprocess

import pytest

from sweep_utils import (
    BASELINE,
    collect_row,
    phase1_parameters,
    ranked_rows,
    require_sweep_gpu,
    resolve_parameters,
    run_sweep,
    save_summaries,
)


def fixture_results(root, accuracies=(0.755, 0.756, 0.757), *, complete=True, seeds=(0, 1, 2)):
    directory = root / "it_compiles" / "fixture-run"
    directory.mkdir(parents=True)
    trials = [
        {
            "seed": seed,
            "status": "ok",
            "accuracy": accuracy,
            "prepare_time": index + 1.0,
            "train_time": 80.0,
            "total_timed_time": index + 81.0,
            "evaluation_time": 0.2,
        }
        for index, (seed, accuracy) in enumerate(zip(seeds, accuracies, strict=False))
    ]
    if not complete:
        trials.append({"seed": seeds[-1], "status": "error", "failure_reason": "fixture failure"})
    (directory / "trials.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trials))
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "requested_trials": 3,
                "complete": complete,
                "qualified": None,
                "run_error": None if complete else "fixture failure",
            }
        )
    )
    return directory


def test_phase1_changes_one_factor_and_keeps_lr_explicit():
    configs = resolve_parameters(phase1_parameters())
    assert len(configs) == 15
    assert configs[0] == BASELINE
    for config in configs[1:]:
        assert config["epochs"] == 60 and config["width"] == 64
        assert sum(value != BASELINE[key] for key, value in config.items()) == 1
    assert all(config["lr"] == 0.2 for config in configs[-2:])
    assert resolve_parameters([{"batch_size": 1024}])[0]["lr"] == 0.4
    assert resolve_parameters([{}])[0]["momentum"] == 0.9


def test_gpu_guard_accepts_pcie_and_sxm_but_rejects_other_devices():
    for name in ("NVIDIA A100 80GB PCIe", "NVIDIA A100-PCIE-80GB", "NVIDIA A100-SXM4-80GB"):
        require_sweep_gpu([name])
    for names in (
        [],
        ["NVIDIA A100-SXM4-40GB"],
        ["NVIDIA H100 80GB HBM3"],
        ["NVIDIA A100 80GB PCIe"] * 2,
    ):
        with pytest.raises(RuntimeError, match="No training started"):
            require_sweep_gpu(names)


def test_report_preserves_actual_gpu_name(tmp_path):
    directory = fixture_results(tmp_path)
    (directory / "config.json").write_text(
        json.dumps({"environment": {"cuda_devices": ["NVIDIA A100-SXM4-80GB"]}})
    )
    row = collect_row(1, BASELINE, [0, 1, 2], tmp_path, 0)
    assert row["cuda_devices"] == ["NVIDIA A100-SXM4-80GB"]
    assert row["competition_gpu_match"] is False


@pytest.mark.parametrize(
    "parameters",
    [
        [],
        {},
        [None],
        [{"unknown": 1}],
        [{"momentum": -0.1}],
        [{"momentum": 1}],
        [{"momentum": math.nan}],
        [{"lr": math.inf}],
        [{"batch_size": "512"}],
        [{"epochs": 0}],
        [{"cutout": 33}],
        [{"label_smoothing": 1}],
    ],
)
def test_invalid_configurations_are_rejected(parameters):
    with pytest.raises(ValueError):
        resolve_parameters(parameters)


def test_reports_accuracy_and_separate_prepare_train_times(tmp_path):
    fixture_results(tmp_path)
    row = collect_row(1, BASELINE, [0, 1, 2], tmp_path, 0)
    assert row["all_trials_completed"]
    assert row["screening_status"] == "safe-screen"
    assert row["mean_accuracy"] == pytest.approx(0.756)
    assert row["accuracy_std"] == pytest.approx(0.001)
    assert row["min_accuracy"] == 0.755 and row["max_accuracy"] == 0.757
    assert row["mean_prepare_time"] == 2
    assert row["mean_train_time"] == 80
    assert row["mean_total_timed_time"] == 82
    assert [trial["accuracy"] for trial in row["seed_results"]] == [0.755, 0.756, 0.757]
    save_summaries(tmp_path, [row])
    assert json.loads((tmp_path / "summary.json").read_text())[0] == row
    with (tmp_path / "summary.csv").open(newline="") as file:
        result = next(csv.DictReader(file))
    assert json.loads(result["parameters"]) == BASELINE
    assert json.loads(result["seed_results"]) == row["seed_results"]


def test_incomplete_or_wrong_seed_runs_cannot_qualify(tmp_path):
    partial = tmp_path / "partial"
    fixture_results(partial, (0.9, 0.9), complete=False)
    row = collect_row(1, BASELINE, [0, 1, 2], partial, 1)
    assert row["mean_accuracy"] == 0.9  # Explicit partial diagnostic, not a qualifying mean.
    assert row["metrics_trial_count"] == 2 and not row["all_trials_completed"]
    assert row["screening_status"] == "incomplete"
    assert len(row["seed_results"]) == 3 and row["seed_results"][-1]["status"] == "error"
    mismatched = tmp_path / "wrong-seeds"
    fixture_results(mismatched, seeds=(1, 2, 3))
    assert collect_row(2, BASELINE, [0, 1, 2], mismatched, 0)["screening_status"] == "incomplete"


def test_completed_trials_survive_interrupted_sweep_collection(tmp_path):
    fixture_results(tmp_path)
    row = collect_row(1, BASELINE, [0, 1, 2], tmp_path, None, "Sweep stopped by user")
    assert row["all_trials_completed"]
    assert not row["benchmark_process_succeeded"]
    assert row["benchmark_exit_code"] is None
    assert row["mean_accuracy"] == pytest.approx(0.756)
    assert row["screening_status"] == "incomplete"


def test_safe_screens_rank_by_speed_and_failed_runs_stay_last():
    def row(name, status, accuracy, runtime):
        return {
            "configuration": name,
            "screening_status": status,
            "mean_accuracy": accuracy,
            "mean_total_timed_time": runtime,
        }

    rows = [
        row("high-accuracy-slow", "safe-screen", 0.80, 100),
        row("below-threshold-fast", "non-qualifying", 0.749, 40),
        row("safe-fast", "safe-screen", 0.756, 80),
        row("incomplete", "incomplete", 0.90, 30),
        row("marginal-fast", "marginal", 0.752, 60),
    ]
    assert [r["configuration"] for r in ranked_rows(rows)] == [
        "safe-fast",
        "high-accuracy-slow",
        "marginal-fast",
        "below-threshold-fast",
        "incomplete",
    ]


@pytest.mark.parametrize("failure", ["launch", "benchmark", "malformed"])
def test_sweep_continues_after_failure_and_commits_each_result(tmp_path, monkeypatch, failure):
    calls, checkpoints = [], []
    directory = tmp_path / "sweep"

    def fake_run(command, *, cwd, stdout, stderr, check):
        calls.append(command)
        assert cwd == "/app" and check is False and stderr == subprocess.STDOUT
        assert command[command.index("--n") + 1] == "3"
        assert command[command.index("--seed") + 1] == "0"
        assert "--no-accuracy-target" in command
        assert json.loads(command[command.index("--params") + 1])["epochs"] == 60
        stdout.write("synthetic test log\n")
        if len(calls) == 1:
            if failure == "launch":
                raise OSError("synthetic subprocess launch failure")
            result_dir = fixture_results(
                type(directory)(command[command.index("--results-root") + 1]),
                (0.9, 0.9),
                complete=False,
            )
            if failure == "malformed":
                (result_dir / "summary.json").write_text("invalid JSON")
            return subprocess.CompletedProcess(command, 1)
        fixture_results(type(directory)(command[command.index("--results-root") + 1]))
        return subprocess.CompletedProcess(command, 0)

    def checkpoint():
        summary = json.loads((directory / "summary.json").read_text())
        checkpoints.append(len(summary))

    monkeypatch.setattr("sweep_utils.subprocess.run", fake_run)
    rows = run_sweep([BASELINE, BASELINE | {"lr": 0.1}], directory, checkpoint=checkpoint)
    assert len(calls) == 2 and checkpoints == [0, 1, 2]
    assert rows[0]["screening_status"] == "incomplete"
    assert rows[0]["error"]
    assert rows[1]["all_trials_completed"]
    assert (directory / "config-001" / "stdout.log").exists()
    assert (directory / "config-002" / "benchmark" / "it_compiles" / "fixture-run").exists()
    assert len(json.loads((directory / "summary.json").read_text())) == 2
