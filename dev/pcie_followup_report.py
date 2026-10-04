"""Report a preplanned PCIe cohort recovered after container preemption.

Underlying harness runs remain separate. This derived summary is not a fabricated
harness result or a submission promotion, and is not double-counted in the ledger.
"""

import argparse
import hashlib
import json
from pathlib import Path

from dev.experiment_ledger import Ledger, summarize_trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-id", required=True)
    args = parser.parse_args()
    ledger = Ledger(Path("results/overnight/ledger.jsonl"))
    ledger.import_results(Path("results"))
    records = {r["experiment_id"]: r for r in ledger.records()}
    original = records["w6_firstdepth_6_3_tuned_10"]
    segments = [records["pcie_original_provisional_40_a2"], records[args.recovery_id]]
    root = Path("results/overnight/rc3-pcie-followup-20261004")
    interruption_path = root / "batches/pcie-original-provisional-attempt2/interruption.json"
    interruption = json.loads(interruption_path.read_text())
    if segments[0]["complete"] or segments[0]["seeds"] != list(range(11000, 11037)):
        raise ValueError("Expected preserved interrupted 37-trial run")
    if not segments[1]["complete"] or segments[1]["seeds"] != list(range(11037, 11040)):
        raise ValueError("Recovery must complete exactly the three missing planned seeds")
    def learning_params(segment):
        return {k: v for k, v in segment["parameters"].items()
                if k not in {"experiment_name", "hypothesis"}}
    parameters = learning_params(segments[0])
    for segment in segments:
        if segment["gpu_models"] != ["NVIDIA A100 80GB PCIe"]:
            raise ValueError("Hardware does not match requested PCIe variant")
        if segment["source_file_hashes"] != original["source_file_hashes"]:
            raise ValueError("Source differs from exact original archived candidate")
        if learning_params(segment) != parameters or segment.get("seed_overlap_with_ancestors"):
            raise ValueError("Configuration changed or seeds overlap validation ancestors")
        for result_path in segment["result_paths"]:
            for filename, expected in original["source_file_hashes"].items():
                actual = hashlib.sha256((Path(result_path) / "source" / filename).read_bytes())
                if actual.hexdigest() != expected:
                    raise ValueError("Retained source bytes do not match archive")
    trials = [t for segment in segments for t in segment["trials"]]
    if [t["seed"] for t in trials] != list(range(11000, 11040)):
        raise ValueError("Cohort must contain all forty planned unique seeds in order")
    result = summarize_trials(trials, 40)
    if not result["complete"] or result["prepare_train_time_count"] != 40:
        raise ValueError("All forty successful accuracy and timing observations required")
    result.update(
        record_kind="derived_preplanned_validation_cohort",
        experiment_id="pcie_original_provisional_planned40_cohort",
        segment_ids=[s["experiment_id"] for s in segments],
        seeds=[t["seed"] for t in trials], trials=trials, parameters=parameters,
        source_file_hashes=original["source_file_hashes"],
        gpu_model="NVIDIA A100 80GB PCIe",
        gpu_uuids=sorted({u for s in segments for u in s["gpu_uuids"]}),
        total_optimizer_steps=segments[0]["total_optimizer_steps"],
        step_count_basis=segments[0]["step_count_basis"],
        uninterrupted_harness_validation=False,
    )
    supported = result["accuracy_lower_95_one_sided"] > 0.75
    conclusion = (
        "The observed mean and exploratory one-sided 95% lower bound both exceed 75%."
        if supported else "The PCIe results do not establish a true mean accuracy above 75%."
    )
    for segment in segments:
        ledger.annotate(
            segment["experiment_id"], promotion_status="pcie_validation_only",
            conclusion=conclusion + " RC3 remains unchanged.",
            validation_cohort=result["experiment_id"],
            validation_notes="Derived preplanned cohort includes all 37 preserved results plus "
            "the three missing seeds recovered from fresh state. Original harness run remains "
            "incomplete; no completed seed replay; no duplicate aggregate ledger trial count.",
        )
    control = records["pcie_rc3_control_before_a2"]
    count_above = sum(a >= 0.75 for a in result["individual_accuracies"])
    payload = dict(
        result=result, segments=segments, original_provisional=original,
        controls=[control], interruption=interruption,
        infrastructure_notes="Container preempted after 37 recorded trials. The next "
        "attempt's outcome was not recorded. Missing seeds 11037-11039 were recovered from "
        "fresh state on another verified PCIe allocation; all observed results retained.",
        individual_seeds_at_least_75=count_above,
        conservative_lower95_above_75=supported,
        mean_above_75=result["mean_accuracy"] > 0.75,
        source_exact=True, default_promoted=False, conclusion=conclusion,
        below_requested_7_0665=result["mean_prepare_train_time"] < 7.0665159039,
        leaderboard=ledger.write_reports(Path("results/overnight/reports")),
    )
    lines = [
        "# Provisional candidate: forty planned PCIe seeds", "", conclusion, "",
        f"- Completed observations: 40/40; seeds 11000-11039; GPU {result['gpu_model']}.",
        f"- Mean accuracy: {100*result['mean_accuracy']:.5f}%.",
        "- Exploratory conservative one-sided 95% lower confidence bound: "
        f"{100*result['accuracy_lower_95_one_sided']:.5f}%.",
        f"- Sample SD: {100*result['accuracy_std']:.5f} percentage points.",
        f"- Minimum / maximum: {100*result['min_accuracy']:.2f}% / "
        f"{100*result['max_accuracy']:.2f}%.",
        f"- Individual seeds at least 75%: {count_above}/40.",
        f"- Mean prepare: {result['mean_prepare_time']:.6f} seconds.",
        f"- Mean train: {result['mean_train_time']:.6f} seconds.",
        f"- Mean prepare + train: {result['mean_prepare_train_time']:.6f} seconds.",
        "- Original container preempted after 37 observations. Three missing seeds were "
        "recovered from fresh state on a second verified PCIe allocation. This is a derived "
        "preplanned cohort, not an uninterrupted forty-trial harness qualification run.",
        "- Every observed result is retained. The lost attempt's unrecorded outcome is "
        "unknown; no completed seed was replayed. Original interrupted run stays incomplete.",
        "- Original-container RC3 before control: three seeds, mean total "
        f"{control['mean_prepare_train_time']:.6f} seconds. After control was not reached; "
        "a complete matched before/after timing comparison is unavailable.",
        "- Protected RC3's 7.066516-second reference was measured on SXM4. Comparing "
        "this PCIe cohort with that reference cannot establish a small runtime gain.",
        "- Source hashes match the original archived provisional recipe byte-for-byte.",
        "- Configuration: 6.3 epochs; 158 source-derived steps; batch 2000; widths "
        "[128,512,512]; depths [2,3,3]; Muon LR 0.25; head LR 3.25; fast reset. "
        "Complete parameters, per-seed observations and provenance are in adjacent JSON.",
        "- RC3 remains the default. No replacement is promoted.", "", "## Segments", "",
    ]
    for s in segments:
        lines.append(f"- {s['experiment_id']}: {s['trial_count']} observations, "
                     f"mean accuracy {100*s['mean_accuracy']:.5f}%, total "
                     f"{s['mean_prepare_train_time']:.6f} seconds; "
                     f"raw path {s['result_paths'][0]}.")
    body = chr(10).join(lines) + chr(10)
    output = root / "pcie40-report"
    for suffix, content in [(".json", json.dumps(payload, indent=2) + chr(10)), (".md", body)]:
        with output.with_suffix(suffix).open("x", encoding="utf-8") as file:
            file.write(content)
    with Path("dev/PROVISIONAL_PCIE_40_RESULT.md").open("x", encoding="utf-8") as file:
        file.write(body)
    print(body)


if __name__ == "__main__":
    main()
