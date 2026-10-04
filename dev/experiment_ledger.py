"""Append-only experiment history, hardware-separated frontiers, and immutable reports.

No torch import, model execution, benchmark modification, or cloud access is needed.
Historical source snapshots are parsed as Python syntax, never imported. Unknown
historical parameters remain explicitly incomplete instead of using today's defaults.

    python -m dev.experiment_ledger import --results-root results \
        --ledger results/overnight/ledger.jsonl --report-dir results/overnight/reports
    python -m dev.experiment_ledger status EXPERIMENT_ID --promotion-status rejected \
        --conclusion "Slower than the matched control" --ledger PATH

Python API: Ledger(path).import_results(root), .record(record),
.annotate(experiment_id, **decision), .records(), .write_reports(directory).
normalize_run(run_dir, metadata=None) accepts an optional complete parameter dict
and experiment metadata; experiment.json beside config.json supplies the same data.
All accuracy values are fractions, times seconds, SD uses the sample convention.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

RC3_RUN = "20261003T223840Z-034593f6"
RC3_COMMIT = "edca55e"
SCHEMA_VERSION = 1
META_KEYS = {"experiment_name", "hypothesis", "parent_experiment", "stage"}
ARCH_KEYS = ("architecture", "widths", "width", "depth", "stage_depths", "stage_residuals",
             "stage_widths",
             "residual_blocks", "global_pool", "downsampling")
OPT_KEYS = ("optimizer", "muon_lr", "muon_momentum", "bias_lr", "head_lr",
            "sgd_momentum", "weight_decay", "lr", "momentum", "ns_steps",
            "ns_iterations", "lr_warmup_frac", "lr_hold_frac", "lr_schedule",
            "compiled_muon", "batched_muon")
# Upper 5% Student-t critical values. Larger df use a conservative lower-df row.
T95 = (None, 6.3138, 2.9200, 2.3534, 2.1319, 2.0150, 1.9432, 1.8946,
       1.8595, 1.8331, 1.8125, 1.7959, 1.7823, 1.7709, 1.7613, 1.7531,
       1.7459, 1.7396, 1.7341, 1.7291, 1.7247, 1.7207, 1.7171, 1.7139,
       1.7109, 1.7081, 1.7056, 1.7033, 1.7011, 1.6991, 1.6973)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def finite(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def clean(value):
    """Preserve invalid numeric results as null; trials/status still expose failures."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    return value


def summarize_trials(trials: list[dict], requested_trials: int | None = None) -> dict:
    """Keep every attempted trial, including errors; never fabricate missing metrics."""
    requested = len(trials) if requested_trials is None else requested_trials
    successful = [t for t in trials if t.get("status") == "ok" and finite(t.get("accuracy"))]
    accuracies = [t["accuracy"] for t in successful]
    average = statistics.mean(accuracies) if accuracies else None
    sd = statistics.stdev(accuracies) if len(accuracies) > 1 else None
    lower = None
    if sd is not None:
        lower = average - T95[min(len(accuracies) - 1, 30)] * sd / math.sqrt(len(accuracies))
    result = {
        "requested_trials": requested, "trial_count": len(trials),
        "successful_trials": len(successful), "failed_trials": len(trials) - len(successful),
        "missing_trials": max(0, requested - len(trials)),
        "individual_accuracies": [t.get("accuracy") for t in trials],
        "mean_accuracy": average, "accuracy_std": sd,
        "accuracy_variance": sd * sd if sd is not None else None,
        "min_accuracy": min(accuracies) if accuracies else None,
        "max_accuracy": max(accuracies) if accuracies else None,
        "accuracy_lower_95_one_sided": lower,
        "confidence_method": "Student-t, df capped at 30, one-sided 95%; exploratory, "
                             "not corrected for adaptive selection or repeated peeking",
        "metrics_population": "Successful finite-accuracy trials; all failures retained",
        "complete": len(successful) == len(trials) == requested and requested > 0,
    }
    for out, field in (("prepare", "prepare_time"), ("train", "train_time"),
                       ("prepare_train", "total_timed_time")):
        values = []
        for trial in trials:
            value = trial.get(field)
            if field == "total_timed_time" and value is None:
                if finite(trial.get("prepare_time")) and finite(trial.get("train_time")):
                    value = trial["prepare_time"] + trial["train_time"]
            if finite(value):
                values.append(value)
        result[f"mean_{out}_time"] = statistics.mean(values) if values else None
        result[f"{out}_time_std"] = statistics.stdev(values) if len(values) > 1 else None
        result[f"{out}_time_count"] = len(values)
    n = len(successful)
    result["evidence_stage_by_count"] = 4 if n >= 40 else 3 if n >= 20 else 2 if n >= 8 else 1
    result["status"] = "complete" if result["complete"] else "incomplete_or_failed"
    return result


def source_defaults(path: Path) -> tuple[dict, str]:
    """Read literal DEFAULTS and literal params.get defaults without executing source."""
    if not path.exists():
        return {}, "missing_source"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEFAULTS" for target in node.targets
        ):
            try:
                return ast.literal_eval(node.value), "saved_source_DEFAULTS"
            except (ValueError, TypeError):
                pass
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and len(node.args) >= 2:
                base = node.func.value
                if isinstance(base, ast.Name) and base.id in {"params", "parameters"}:
                    try:
                        key, value = ast.literal_eval(node.args[0]), ast.literal_eval(node.args[1])
                        if isinstance(key, str):
                            defaults[key] = value
                    except (ValueError, TypeError):
                        pass
    return defaults, "partial_literal_defaults" if defaults else "unresolved_defaults"


def _gpu_names(config: dict, summary: dict, trials: list[dict]) -> list[str]:
    names = set(config.get("environment", {}).get("cuda_devices", []))
    names.update(config.get("cuda_devices", []))
    for obj in (config, summary):
        if isinstance(obj.get("gpu"), str):
            names.add(obj["gpu"])
    for trial in trials:
        names.update(t["name"] for t in trial.get("telemetry", []) if t.get("name"))
    return sorted(names)


def accuracy_evidence(record: dict) -> str:
    """A descriptive evidence label, never an automatic change to the submission."""
    if not record.get("complete"):
        return "incomplete_or_failed"
    n, mean, lower = (record.get(key) for key in
                      ("successful_trials", "mean_accuracy", "accuracy_lower_95_one_sided"))
    if not finite(mean):
        return "missing_accuracy"
    if mean <= 0.75:
        return "observed_mean_below_threshold"
    if not n or n < 10:
        return "screen_only"
    if not finite(lower) or lower <= 0.75:
        return "mean_above_threshold_uncertain"
    if n >= 40 and record.get("fresh_seed_validation"):
        return "fresh_40_seed_accuracy_evidence"
    return "appears_accuracy_safe_10plus_seeds"


def normalize_run(run_dir: Path | str, metadata: dict | None = None) -> dict:
    run = Path(run_dir)
    config = read_json(run / "config.json", {})
    summary = read_json(run / "summary.json", {})
    meta = read_json(run / "experiment.json", {})
    if not meta and run.parent.parent.name == "runs":
        # A container can be preempted before the worker writes experiment.json.
        # Its pre-launch metadata was checkpointed outside the harness directory.
        meta = read_json(run.parent.parent.parent / f"{run.parent.name}.json", {})
    meta = meta | (metadata or {})
    if not isinstance(summary, dict):
        raise ValueError(f"Not an individual run: {run}")
    trials_path = run / "trials.jsonl"
    trials = [clean(json.loads(line)) for line in trials_path.read_text(encoding="utf-8-sig")
              .splitlines() if line.strip()] if trials_path.exists() else []
    saved_source = run / "source" / "submission.py"
    if not saved_source.exists():
        saved_source = run / "airbench_reference.py"
    defaults, resolution = source_defaults(saved_source)
    params = defaults | config.get("resolved_config", {}) | config.get("parameters", {})
    params.update(meta.get("parameters", {}))
    source_hashes = dict(config.get("submission_sha256", {}))
    if not source_hashes:
        source_paths = sorted((run / "source").rglob("*.py"))
        if not source_paths:
            source_paths = sorted(run.glob("*.py"))
        source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    run_id = config.get("run_id") or summary.get("run_id")
    if not run_id:
        # Performance-study directories have named variants but no harness run ID.
        run_id = f"{run.parent.name}/{run.name}"
    n_requested = config.get("n_trials", summary.get("requested_trials",
                              summary.get("number_of_trials", len(trials))))
    stats = summarize_trials(trials, n_requested)
    warnings = []
    if resolution not in {"saved_source_DEFAULTS"} and not config.get("resolved_config"):
        warnings.append("Historical parameter defaults are not completely recoverable")
    if summary.get("mean_accuracy") is not None and stats["mean_accuracy"] is not None:
        if abs(summary["mean_accuracy"] - stats["mean_accuracy"]) > 1e-8:
            warnings.append("Raw-trial mean accuracy differs from saved summary")
    if summary.get("run_error") or summary.get("complete") is False:
        stats["complete"] = False
        stats["status"] = "incomplete_or_failed"
    if not (run / "summary.json").exists():
        stats["complete"] = False
        stats["status"] = "incomplete_or_failed"
        warnings.append("Harness summary missing; interrupted evidence cannot qualify")
    if not trials:
        # Saved aggregate metrics remain visible, but cannot satisfy raw-evidence gates.
        stats.update({"mean_accuracy": summary.get("mean_accuracy"),
                      "accuracy_std": summary.get("accuracy_std"),
                      "mean_prepare_time": summary.get("mean_prepare_time"),
                      "mean_train_time": summary.get("mean_train_time"),
                      "mean_prepare_train_time": summary.get("mean_training_time")})
        warnings.append("No individual trial records available")
    observed_steps = sorted({t["steps"] for t in trials if t.get("steps") is not None})
    steps, step_basis = None, "unknown"
    if len(observed_steps) == 1:
        steps, step_basis = observed_steps[0], "observed_trial_output"
    elif "total_steps" in meta:
        steps, step_basis = meta["total_steps"], "experiment_metadata"
    elif "widths" in params and "muon_lr" in params and "epochs" in params:
        steps = math.ceil(params["epochs"] * (50000 // params["batch_size"]))
        step_basis = "source formula ceil(epochs * floor(50000 / batch_size)); inferred"
    gpu_models = _gpu_names(config, summary, trials)
    nvidia_smi = meta.get("nvidia_smi", meta.get("nvidia_smi_gpu_name",
                                             meta.get("nvidia_smi_name")))
    if not gpu_models and isinstance(nvidia_smi, str):
        gpu_models = sorted(set(nvidia_smi.strip().splitlines()))
    variant = config.get("variant", summary.get("variant", {}))
    parameters = {k: v for k, v in params.items() if k not in META_KEYS}
    protected = run_id == RC3_RUN
    if protected:
        warnings.append("Protected 7.0665 s control was measured on SXM4; official target is PCIe")
    record = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": meta.get("experiment_id", meta.get("id", run_id)), "run_id": run_id,
        "parent_experiment": meta.get("parent_experiment", meta.get("parent",
                                      params.get("parent_experiment"))),
        "experiment_name": meta.get("experiment_name", params.get("experiment_name",
                                    variant.get("name", run.name))),
        "hypothesis": meta.get("hypothesis", params.get("hypothesis", "Historical import")),
        "source_hash": meta.get("source_hash") or (
            digest(source_hashes) if source_hashes else None),
        "source_file_hashes": source_hashes,
        "git_commit": meta.get("git_commit", meta.get("source_commit", config.get("git_commit"))),
        "protected_control_commit": RC3_COMMIT if protected else None,
        "parameters": parameters,
        "parameter_resolution": "explicit_complete" if meta.get("parameters_complete")
                                else "saved_resolved_config" if config.get("resolved_config")
                                else resolution,
        "architecture": {k: parameters[k] for k in ARCH_KEYS if k in parameters},
        "optimizer": {k: parameters[k] for k in OPT_KEYS if k in parameters},
        "total_epochs": params.get("epochs"), "total_optimizer_steps": steps,
        "step_count_basis": step_basis, "observed_step_counts": observed_steps,
        "training_examples": 50000, "seeds": [t.get("seed") for t in trials],
        "requested_seeds": config.get("seeds", []),
        "trials": trials, **stats,
        "gpu_models": gpu_models,
        "gpu_model": gpu_models[0] if len(gpu_models) == 1 else "; ".join(gpu_models) or "unknown",
        "nvidia_smi": nvidia_smi,
        "nvidia_smi_command": "nvidia-smi --query-gpu=name --format=csv,noheader"
                              if nvidia_smi else None,
        "gpu_uuids": sorted({t["uuid"] for trial in trials
                             for t in trial.get("telemetry", []) if t.get("uuid")}),
        "official": bool(config.get("official", summary.get("official", False))),
        "hardware_target": "NVIDIA A100 80GB PCIe",
        "protected_control": protected,
        "instrumented": bool(meta.get("instrumented", variant.get("profile", False))),
        "stage": meta.get("stage", "historical"),
        "campaign": meta.get("campaign", "overnight" if meta else "historical"),
        "fresh_seed_validation": meta.get("fresh_seed_validation"),
        "promotion_status": meta.get("promotion_status", "protected_control" if protected
                                     else "historical_unreviewed"),
        "conclusion": meta.get("conclusion", "Protected RC3 development baseline" if protected
                               else "Imported historical evidence; not automatically promoted"),
        "result_paths": [run.as_posix()], "warnings": warnings,
        "raw_summary": summary, "raw_config": config, "raw_metadata": meta,
    }
    if protected:
        record["fresh_seed_validation"] = True
        record["stage"] = 4
    record["accuracy_evidence"] = accuracy_evidence(record)
    return clean(record)


def pareto_frontier(records: list[dict], min_trials: int = 2) -> list[dict]:
    """No cross-hardware dominance; keep reliability, component time, and sample count.

    Descriptive frontier only, not an automatic qualification/promotion decision.
    Profiles and incomplete/failed runs never dominate completed benchmarks.
    """
    axes = (("mean_accuracy", 1), ("accuracy_std", -1), ("min_accuracy", 1),
            ("mean_prepare_time", -1), ("mean_train_time", -1),
            ("mean_prepare_train_time", -1), ("successful_trials", 1))
    eligible = [r for r in records if r.get("complete") and not r.get("instrumented")
                and r.get("successful_trials", 0) >= min_trials
                and len(r.get("gpu_models", [])) == 1
                and all(finite(r.get(key)) for key, _ in axes)]
    def dominates(left, right):
        if left["gpu_model"] != right["gpu_model"]:
            return False
        # Historical source families remain visible without eliminating new campaign points.
        if (left.get("campaign", "historical") != right.get("campaign", "historical")
                and not left.get("protected_control") and not right.get("protected_control")):
            return False
        comparisons = [(left[key] - right[key]) * direction for key, direction in axes]
        return all(x >= 0 for x in comparisons) and any(x > 0 for x in comparisons)
    return [r for r in eligible if not any(dominates(other, r) for other in eligible)]


class Ledger:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def _events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8")
                .splitlines() if line.strip()]

    def records(self) -> list[dict]:
        records = {}
        decisions = defaultdict(dict)
        for event in self._events():
            key = event["experiment_id"]
            if event["event_type"] == "record":
                records[key] = event["record"]
            elif event["event_type"] == "decision":
                decisions[key].update(event["decision"])
        for key, decision in decisions.items():
            if key in records:
                records[key].update(decision)
        for record in records.values():
            record["accuracy_evidence"] = accuracy_evidence(record)
        return list(records.values())

    def _append(self, event: dict) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        # Fail rather than interleave writes or delete another process's lock.
        lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            content_hash = digest(event)
            if any(e.get("content_hash") == content_hash for e in self._events()):
                return False
            event = event | {"timestamp_utc": utc_now(), "content_hash": content_hash}
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return True
        finally:
            os.close(lock)
            lock_path.unlink()

    def record(self, record: dict) -> bool:
        value = clean(record)
        if not value.get("experiment_id"):
            raise ValueError("experiment_id is required")
        parent_id = value.get("parent_experiment")
        known = {r["experiment_id"]: r for r in self.records()}
        if parent_id in known:
            ancestor_seeds, seen = set(), set()
            while parent_id in known and parent_id not in seen:
                seen.add(parent_id)
                ancestor_seeds.update(known[parent_id].get("seeds", []))
                parent_id = known[parent_id].get("parent_experiment")
            overlap = sorted(set(value.get("seeds", [])) & ancestor_seeds - {None})
            value["seed_overlap_with_ancestors"] = overlap
            if overlap and value.get("fresh_seed_validation"):
                value["fresh_seed_validation"] = False
                value.setdefault("warnings", []).append(
                    "Claimed fresh validation reuses seeds from a recorded ancestor")
        value["accuracy_evidence"] = accuracy_evidence(value)
        return self._append({"event_type": "record", "experiment_id": value["experiment_id"],
                             "record": value})

    def annotate(self, experiment_id: str, **decision) -> bool:
        allowed = {"promotion_status", "conclusion", "parent_experiment", "hypothesis", "stage",
                   "fresh_seed_validation", "seed_overlap_with_parent", "paired_control",
                   "validation_notes"}
        if not set(decision) <= allowed:
            raise ValueError(f"Decision cannot alter measured fields: {set(decision) - allowed}")
        if experiment_id not in {r["experiment_id"] for r in self.records()}:
            raise KeyError(experiment_id)
        return self._append({"event_type": "decision", "experiment_id": experiment_id,
                             "decision": clean(decision)})

    def import_results(self, root: Path | str) -> dict:
        root = Path(root)
        found, by_identity, warnings = [], {}, []
        for path in sorted(root.rglob("config.json")):
            try:
                record = normalize_run(path.parent)
                # Ignore location: copied downloads with the same run/source/seeds are one run.
                identity = digest([record["run_id"], record["source_hash"],
                                   record["parameters"], record["requested_seeds"]])
                if identity in by_identity:
                    old = by_identity[identity]
                    aliases = sorted(set(old["result_paths"] + record["result_paths"]))
                    old_trials = {t.get("seed", t.get("trial")): t for t in old["trials"]}
                    conflicts = [t.get("seed") for t in record["trials"]
                                 if t.get("seed", t.get("trial")) in old_trials
                                 and old_trials[t.get("seed", t.get("trial"))] != t]
                    if record["trial_count"] > old["trial_count"]:
                        by_identity[identity] = record
                    by_identity[identity]["result_paths"] = aliases
                    if conflicts:
                        by_identity[identity]["warnings"].append(
                            f"Copied trial records conflict for seeds {conflicts}; inspect aliases")
                        warnings.append({"path": path.as_posix(), "error": "Conflicting copies"})
                else:
                    by_identity[identity] = record
            except (ValueError, KeyError, TypeError, OSError, SyntaxError) as error:
                warnings.append({"path": path.as_posix(), "error": str(error)})
        found.extend(by_identity.values())
        ids = defaultdict(list)
        for record in found:
            ids[record["experiment_id"]].append(record)
        for collisions in ids.values():
            if len(collisions) > 1:
                for record in collisions:
                    suffix = digest([record["run_id"], record["source_hash"],
                                     record["parameters"], record["requested_seeds"]])[:12]
                    record["experiment_id"] += "-" + suffix
                    record["warnings"].append("Run ID reused across distinct configurations")
        known_runs = {r["run_id"] for r in found}
        # Retain launch/build failures and partially downloaded results with no harness config.
        for path in sorted(root.rglob("result.json")):
            data = read_json(path, {})
            if not isinstance(data, dict) or "seed_results" not in data:
                continue
            run_id = str(data.get("benchmark_result_directory", "")).rstrip("/").split("/")[-1]
            if run_id and run_id in known_runs:
                continue
            params = data.get("parameters", {})
            key = run_id or "historical-failure-" + digest([params, data.get("seeds"),
                                                          data.get("error")])[:16]
            trials = clean(data.get("seed_results", []))
            models = data.get("cuda_devices", [])
            record = {
                "schema_version": SCHEMA_VERSION, "experiment_id": key, "run_id": key,
                "parent_experiment": None, "experiment_name": data.get("configuration", key),
                "hypothesis": params.get("hypothesis", "Historical result without source/config"),
                "source_hash": None, "source_file_hashes": {}, "git_commit": None,
                "parameters": params, "parameter_resolution": "partial_result_only",
                "architecture": {k: params[k] for k in ARCH_KEYS if k in params},
                "optimizer": {k: params[k] for k in OPT_KEYS if k in params},
                "total_epochs": params.get("epochs"), "total_optimizer_steps": None,
                "step_count_basis": "unknown", "trials": trials,
                "seeds": [t.get("seed") for t in trials], "requested_seeds": data.get("seeds", []),
                **summarize_trials(trials, len(data.get("seeds", []))),
                "gpu_models": models, "gpu_model": "; ".join(models) or "unknown",
                "protected_control": False, "instrumented": False,
                "stage": "historical", "fresh_seed_validation": None,
                "promotion_status": "historical_unreviewed",
                "conclusion": "Historical partial result; source provenance unavailable",
                "result_paths": [path.parent.as_posix()], "raw_result": data,
                "warnings": ["Source provenance and full configuration unavailable"],
            }
            if data.get("error") or not data.get("all_trials_completed", False):
                record["complete"] = False
                record["status"] = "incomplete_or_failed"
            found.append(record)
            known_runs.add(key)
        added = sum(self.record(record) for record in found)
        return {"discovered_runs": len(found), "appended_snapshots": added,
                "total_experiments": len(self.records()), "import_warnings": warnings}

    def write_reports(self, directory: Path | str) -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:6]
        records = self.records()
        frontier = pareto_frontier(records)
        payload = {"generated_at_utc": utc_now(), "schema_version": SCHEMA_VERSION,
                   "total_experiments": len(records), "records": records,
                   "pareto_frontier_ids": [r["experiment_id"] for r in frontier],
                   "frontier_policy": "Exact GPU model only; maximize accuracy/min/count, "
                                      "minimize SD/prepare/train/total; no automatic promotion"}
        json_path, md_path = directory / (stem + ".json"), directory / (stem + ".md")
        with json_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        frontier_ids = {r["experiment_id"] for r in frontier}
        lines = ["# Experiment leaderboard", "", f"Generated {payload['generated_at_utc']}.", "",
                 "RC3 is protected at edca55e: 75.2955%, 7.066516 s, 40/40, "
                 "NVIDIA A100-SXM4-80GB. Official hardware target is A100 80GB PCIe.", "",
                 "Timings are separated by exact GPU model. All attempts/failures remain in the "
                 "JSON ledger. L95 is an exploratory one-sided Student-t bound, not a guarantee "
                 "after adaptive selection. Stage count alone does not prove fresh-seed "
                 "validation. Historical source families are marked separately.",
                 "", f"Unique experiments: {len(records)}. Pareto points: {len(frontier)}."]
        groups = defaultdict(list)
        for record in records:
            groups[record.get("gpu_model", "unknown")].append(record)
        def number(value, scale=1, places=4):
            return f"{value * scale:.{places}f}" if finite(value) else "?"
        for gpu, rows in sorted(groups.items()):
            lines.extend(["", f"## {gpu}", "", "| Experiment | n ok/attempted/requested | "
                          "Accuracy % | SD pp | Min % | L95 % | Prepare s | Train s | Total s | "
                          "Steps | Frontier | Evidence | Status |", "|---|---:|---:|---:|---:|"
                          "---:|---:|---:|---:|---:|---|---|---|"])
            rows.sort(key=lambda r: r.get("mean_prepare_train_time")
                      if finite(r.get("mean_prepare_train_time")) else math.inf)
            for row in rows:
                key = row["experiment_id"]
                metrics = [number(row.get(k), 100) for k in
                           ("mean_accuracy", "accuracy_std", "min_accuracy",
                            "accuracy_lower_95_one_sided")]
                metrics += [number(row.get(k)) for k in
                            ("mean_prepare_time", "mean_train_time", "mean_prepare_train_time")]
                count = f"{row['successful_trials']}/{row['trial_count']}/{row['requested_trials']}"
                label = str(row.get("experiment_name", key)).replace("|", "/")
                status = row.get("promotion_status", "unreviewed")
                if row.get("campaign", "historical") == "historical":
                    status += "; historical"
                if not row.get("complete"):
                    status += "; incomplete/failed"
                if row.get("instrumented"):
                    status += "; instrumented"
                lines.append(f"| {label} (`{key}`) | {count} | " + " | ".join(metrics)
                             + f" | {row.get('total_optimizer_steps') or '?'} | "
                             + ("yes" if key in frontier_ids else "")
                             + f" | {row.get('accuracy_evidence', '?')} | {status} |")
        with md_path.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return {"json": str(json_path), "markdown": str(md_path),
                "experiments": len(records), "frontier_points": len(frontier)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("import", "report", "status"))
    parser.add_argument("experiment_id", nargs="?")
    parser.add_argument("--ledger", type=Path, default=Path("results/overnight/ledger.jsonl"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--report-dir", type=Path, default=Path("results/overnight/reports"))
    parser.add_argument("--promotion-status")
    parser.add_argument("--conclusion")
    args = parser.parse_args()
    ledger = Ledger(args.ledger)
    if args.command == "import":
        print(json.dumps(ledger.import_results(args.results_root), indent=2))
    elif args.command == "status":
        if not args.experiment_id or not args.promotion_status or not args.conclusion:
            parser.error("status needs EXPERIMENT_ID, --promotion-status and --conclusion")
        ledger.annotate(args.experiment_id, promotion_status=args.promotion_status,
                        conclusion=args.conclusion)
    print(json.dumps(ledger.write_reports(args.report_dir), indent=2))


if __name__ == "__main__":
    main()
