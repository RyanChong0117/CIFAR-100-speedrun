"""Freeze the final targeted averaging/batched-Muon screen before launch."""

import json
import subprocess
from pathlib import Path

from dev.experiment_ledger import Ledger


def main():
    ledger = Ledger(Path("results/overnight/ledger.jsonl"))
    decisions = {
        "w10_bs1750_6_1_10": ("rejected", "Fresh10 mean74.811%; no advancement unchanged."),
        "w10_bs1750_6_0_aligned": ("rejected", "Mean74.6067%; aligning schedule lost accuracy."),
        "w10_bs1750_6_0_scaled_muon": (
            "held",
            "Three-seed75.0367% is insufficient; paired BN screen only.",
        ),
        "w10_bs1750_6_2_area": (
            "held",
            "Three-seed75.10%; thin accuracy evidence; no large validation.",
        ),
        "w10_bs1900_6_1_aligned": (
            "held",
            "Three-seed74.9833%; first-stage BN short-budget screen takes priority.",
        ),
        "w10_bs1900_6_2": ("held", "Three-seed75.0267%; insufficient accuracy margin."),
        "w10_batched_muon_rc3": (
            "held",
            "About52ms faster than preceding control; accuracy75.1233%, insufficient margin.",
        ),
        "w10_firstshort_batched_muon": (
            "promoted_to_10",
            "About47ms faster;75.1767% screen warrants limited fresh10, not replacement.",
        ),
    }
    for ident, (status, conclusion) in decisions.items():
        ledger.annotate(ident, promotion_status=status, conclusion=conclusion)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    template = dict(
        session_id="rc3-20261004",
        deadline_unix=1791086518,
        source_commit=commit,
        protected_source_sha256="9ecb6c57cab5f8a63e0edc51ff8970dd858f6d12406099e6c42b10953a810ec2",
    )
    first = dict(stage_depths=[2, 3, 3], fast_reset=True)
    short = first | dict(epochs=6.3, head_lr=3.25, muon_lr=0.25)
    small = first | dict(
        batch_size=1750,
        epochs=6.0,
        head_lr=3.4285714285714284,
        bias_lr=0.060571428571428575,
        muon_lr=0.21,
    )
    requests = []
    screens = [
        ("1750-bn", small, "w10_bs1750_6_0_scaled_muon"),
        ("first61-bn", short | dict(epochs=6.1), "w7_firstdepth_6_3_tuned_20"),
        ("first62-bn", short | dict(epochs=6.2), "w7_firstdepth_6_3_tuned_20"),
    ]
    for name, params, parent in screens:
        experiments = []
        for suffix, momentum in [("control", 0.6), ("bn65", 0.65), ("bn75", 0.75)]:
            experiments.append(
                dict(
                    id=f'w11_{name.replace("-", "_")}_{suffix}',
                    params=params | dict(bn_momentum=momentum),
                    n=3,
                    seed=2000,
                    parent_experiment=parent,
                    stage=1,
                    fresh_seed_validation=False,
                    hypothesis=("Paired training-only BN running-statistics averaging; "
                                "current-batch training unchanged, accuracy screened "
                                "by unchanged harness"),
                )
            )
        requests.append(template | dict(batch_id=f"w11-{name}", experiments=experiments))
    requests.append(
        template
        | dict(
            batch_id="w11-batched-muon-validation",
            experiments=[
                dict(
                    id="w11_muon_control_before",
                    params=short,
                    n=3,
                    seed=7100,
                    parent_experiment="w7_firstdepth_6_3_tuned_20",
                    stage=1,
                    fresh_seed_validation=False,
                    hypothesis="Matched shorter-model NS control",
                ),
                dict(
                    id="w11_firstshort_batched_muon_10",
                    params=short | dict(batched_muon=True),
                    n=10,
                    seed=7100,
                    parent_experiment="w10_firstshort_batched_muon",
                    stage=2,
                    fresh_seed_validation=True,
                    hypothesis="Fresh10 test of measured47ms batched-Muon gain",
                ),
                dict(
                    id="w11_muon_control_after",
                    params=short,
                    n=3,
                    seed=7100,
                    parent_experiment="w7_firstdepth_6_3_tuned_20",
                    stage=1,
                    fresh_seed_validation=False,
                    hypothesis="After control, all warming evidence retained",
                ),
            ],
        )
    )
    Path("dev/sweeps/overnight_wave11.json").write_text(json.dumps(requests, indent=2) + "\n")
    rows = ledger.records()
    complete = [
        r
        for r in rows
        if r.get("campaign") == "rc3-overnight" and r.get("complete") and not r.get("instrumented")
    ]
    active = Path("results/overnight/rc3-20261004/active.json")
    state = json.loads(active.read_text())
    state.update(
        active_wave=10,
        exec_session_id=None,
        modal_app_id=None,
        completed_waves=list(range(1, 11)),
        completed_benchmark_experiments=len(complete),
        completed_benchmark_trials=sum(r["trial_count"] for r in complete),
        next_action=("Wave10 complete. Frozen wave11 targeted BN screens and fresh10 "
                     "batchedMuon; fixed deadline unchanged."),
    )
    active.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
