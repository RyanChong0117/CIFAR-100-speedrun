"""Report the requested PCIe forty-trial validation and same-container controls."""

import argparse
import json
import statistics
from pathlib import Path

from dev.experiment_ledger import Ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    ledger = Ledger(Path("results/overnight/ledger.jsonl"))
    ledger.import_results(Path("results"))
    records = {r["experiment_id"]: r for r in ledger.records()}
    result = records[args.experiment_id]
    original = records["w6_firstdepth_6_3_tuned_10"]
    suffix = args.experiment_id.rsplit("_", 1)[1]
    controls = [records[f"pcie_rc3_control_{phase}_{suffix}"] for phase in ["before", "after"]]
    if not result["complete"] or result["trial_count"] != 40:
        raise ValueError("Forty complete candidate trials are required")
    if result["gpu_models"] != ["NVIDIA A100 80GB PCIe"]:
        raise ValueError("Hardware does not match requested PCIe variant")
    if result["source_file_hashes"] != original["source_file_hashes"]:
        raise ValueError("Candidate source differs from the exact archived provisional recipe")
    if result.get("seed_overlap_with_ancestors"):
        raise ValueError("Candidate seeds overlap validation ancestors")
    for control in controls:
        if (
            not control["complete"]
            or control["gpu_models"] != result["gpu_models"]
            or control["raw_metadata"]["batch_id"] != result["raw_metadata"]["batch_id"]
        ):
            raise ValueError("Same-container complete controls are required")
    control_total = statistics.mean(
        t["prepare_time"] + t["train_time"] for r in controls for t in r["trials"]
    )
    count_above = sum(a >= 0.75 for a in result["individual_accuracies"])
    supported = result["accuracy_lower_95_one_sided"] > 0.75
    conclusion = (
        "Fresh PCIe forty-trial mean and conservative one-sided 95% lower bound " "both exceed 75%."
        if supported
        else "Fresh PCIe evidence does not establish a true mean above 75%."
    )
    ledger.annotate(
        result["experiment_id"],
        promotion_status="pcie_validation_only",
        conclusion=conclusion + " RC3 remains unchanged.",
        validation_notes="Exact archived source verified; all forty seeds retained; "
        "same-container RC3 before/after controls retained.",
    )
    reports = ledger.write_reports(Path("results/overnight/reports"))
    payload = dict(
        result=result,
        controls=controls,
        original_provisional=original,
        individual_seeds_at_least_75=count_above,
        conservative_lower95_above_75=supported,
        mean_above_75=result["mean_accuracy"] > 0.75,
        rc3_same_container_mean_total=control_total,
        time_saved_vs_same_container_rc3=control_total - result["mean_prepare_train_time"],
        below_requested_7_0665=result["mean_prepare_train_time"] < 7.0665159039,
        source_exact=True,
        default_promoted=False,
        conclusion=conclusion,
        leaderboard=reports,
    )
    output = Path("results/overnight/rc3-pcie-followup-20261004/pcie40-report")
    body = (
        "\n".join(
            [
                "# Provisional candidate: forty fresh PCIe trials",
                "",
                conclusion,
                "",
                f"- GPU: {result['gpu_model']}.",
                f"- Completed: {result['trial_count']}/40; seeds {result['requested_seeds'][0]}-"
                f"{result['requested_seeds'][-1]}.",
                f"- Mean accuracy: {100*result['mean_accuracy']:.5f}%.",
                '- Conservative one-sided 95% lower confidence bound: '
                f"{100*result['accuracy_lower_95_one_sided']:.5f}%.",
                f"- Sample SD: {100*result['accuracy_std']:.5f} percentage points.",
                f"- Minimum / maximum: {100*result['min_accuracy']:.2f}% / "
                f"{100*result['max_accuracy']:.2f}%.",
                f"- Individual seeds at least 75%: {count_above}/40.",
                f"- Mean prepare: {result['mean_prepare_time']:.6f} seconds.",
                f"- Mean train: {result['mean_train_time']:.6f} seconds.",
                f"- Mean prepare + train: {result['mean_prepare_train_time']:.6f} seconds.",
                '- Same-container RC3 total mean over six control trials: '
                f'{control_total:.6f} seconds.',
                f"- RC3 before / after: {controls[0]['mean_prepare_train_time']:.6f} / "
                f"{controls[1]['mean_prepare_train_time']:.6f} seconds.",
                "- Source hashes match the original provisional recipe exactly; "
                "configuration remains "
                "6.3 epochs, 158 source-derived steps, batch 2000, widths [128,512,512], "
                "stage depths [2,3,3], Muon LR 0.25, head LR 3.25, exact fast reset.",
                "- All trials and hardware allocation audits retained. RC3 remains the default.",
                "",
                f"Raw candidate result: `{result['result_paths'][0]}`",
                "",
            ]
        )
        + "\n"
    )
    for suffix, content in [(".json", json.dumps(payload, indent=2) + "\n"), (".md", body)]:
        with output.with_suffix(suffix).open("x", encoding="utf-8") as file:
            file.write(content)
    print(body)


if __name__ == "__main__":
    main()
