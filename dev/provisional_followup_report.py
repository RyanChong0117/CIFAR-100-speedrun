"""Summarize the explicitly requested archived-recipe forty-trial check."""

import json
from pathlib import Path

from dev.experiment_ledger import Ledger


def main():
    ledger = Ledger(Path("results/overnight/ledger.jsonl"))
    ledger.import_results(Path("results"))
    rows = ledger.records()
    result = next(r for r in rows if r["experiment_id"] == "followup_original_provisional_40")
    parent = next(r for r in rows if r["experiment_id"] == "w6_firstdepth_6_3_tuned_10")
    if not result["complete"] or result["trial_count"] != 40:
        raise ValueError("Forty complete trials are required; incomplete evidence stays in ledger")
    if result["source_file_hashes"] != parent["source_file_hashes"]:
        raise ValueError("Validation source differs from the original provisional recipe")
    seeds = result["requested_seeds"]
    if seeds != list(range(10000, 10040)) or result.get("seed_overlap_with_ancestors"):
        raise ValueError("Validation seed set is not the planned fresh set")
    count_above = sum(a >= 0.75 for a in result["individual_accuracies"])
    supported = result["accuracy_lower_95_one_sided"] > 0.75
    conclusion = (
        "Fresh 40-trial mean and conservative one-sided 95% lower bound both exceed 75%."
        if supported
        else "Fresh 40-trial evidence does not establish a true mean above 75%."
    )
    ledger.annotate(
        result["experiment_id"],
        promotion_status="accuracy_validation_only",
        conclusion=conclusion + " RC3 unchanged; no submission replacement.",
        validation_notes="Exact archived source hashes match provisional 10; "
        "all 40 seeds retained; individual 75% passes reported separately.",
    )
    reports = ledger.write_reports(Path("results/overnight/reports"))
    payload = dict(
        result=result,
        original_provisional=parent,
        individual_seeds_at_least_75=count_above,
        mean_above_75=result["mean_accuracy"] > 0.75,
        conservative_lower95_above_75=supported,
        source_exact=True,
        default_promoted=False,
        conclusion=conclusion,
        leaderboard=reports,
    )
    output = Path("results/overnight/rc3-followup-20261004/accuracy40-report")
    for suffix, content in [
        (".json", json.dumps(payload, indent=2) + "\n"),
        (
            ".md",
            "\n".join(
                [
                    "# Provisional candidate: forty fresh trials",
                    "",
                    conclusion,
                    "",
                    f"- Completed: {result['trial_count']}/40; seeds 10000-10039.",
                    f"- Mean accuracy: {100*result['mean_accuracy']:.5f}%.",
                    '- Conservative one-sided 95% lower confidence bound: '
                    f"{100*result['accuracy_lower_95_one_sided']:.5f}%.",
                    f"- Sample SD: {100*result['accuracy_std']:.5f} percentage points.",
                    f"- Minimum / maximum: {100*result['min_accuracy']:.2f}% / "
                    f"{100*result['max_accuracy']:.2f}%.",
                    f"- Individual seeds at least 75%: {count_above}/40.",
                    f"- Prepare / train / total mean: {result['mean_prepare_time']:.6f} / "
                    f"{result['mean_train_time']:.6f} / "
                    f"{result['mean_prepare_train_time']:.6f} seconds.",
                    f"- GPU: {result['gpu_model']}.",
                    "- Configuration: 6.3 epochs; 158 source-derived steps; batch 2000; "
                    "widths [128,512,512]; stage depths [2,3,3]; Muon LR 0.25; head LR 3.25; "
                    "exact fast reset; remaining parameters recorded in adjacent JSON.",
                    "- Original archived source hashes verified byte-for-byte.",
                    "- All trials retained. Mean qualification and individual seed "
                    "qualification are distinct. RC3 remains the default.",
                    "",
                    f"Raw result: `{result['result_paths'][0]}`",
                    "",
                ]
            )
            + "\n",
        ),
    ]:
        with output.with_suffix(suffix).open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(
        json.dumps(
            {
                k: payload[k]
                for k in [
                    "mean_above_75",
                    "conservative_lower95_above_75",
                    "individual_seeds_at_least_75",
                    "conclusion",
                ]
            },
            indent=2,
        )
    )
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
