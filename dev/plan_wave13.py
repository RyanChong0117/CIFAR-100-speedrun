"""Last local screens and fresh checks chosen from the completed averaging evidence."""

import json
import subprocess
from pathlib import Path

from dev.experiment_ledger import Ledger


def main():
    ledger = Ledger(Path("results/overnight/ledger.jsonl"))
    ledger.import_results(Path("results"))
    rows = {r["experiment_id"]: r for r in ledger.records()}
    parents = ["w12_1750_bn80", "w12_1750_bn65_smooth35", "w11_first62_bn_control"]
    for ident in parents:
        if not rows[ident]["complete"] or rows[ident]["mean_accuracy"] < 0.75:
            raise ValueError(f"Candidate no longer supports limited validation: {ident}")
    for ident in ["w12_1750_bn65_10", "w12_1750_bn75_10"]:
        ledger.annotate(
            ident,
            promotion_status="rejected",
            conclusion="Fresh10 too close to/below75%; no advancement unchanged.",
        )
    ledger.annotate(
        "w11_first62_bn_control",
        promotion_status="promoted_to_10",
        conclusion="Bounded exception: existing firstshort80 evidence plus "
        "three fewer steps and near75.15% screen justify limited10.",
    )
    for ident in parents[:2]:
        ledger.annotate(
            ident,
            promotion_status="promoted_to_10",
            conclusion="Screen around75.2% and large speed gain justify fresh10 only.",
        )
    template = json.loads(Path("dev/sweeps/overnight_wave12.json").read_text())[0]
    template = {k: v for k, v in template.items() if k not in ["batch_id", "experiments"]}
    template["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    requests = []
    for label, ident in zip(["bn80", "smooth35", "first62"], parents, strict=True):
        params = rows[ident]["parameters"]
        experiments = [
            dict(
                id=f"w13_{label}_10",
                params=params,
                n=10,
                seed=7300,
                parent_experiment=ident,
                stage=2,
                fresh_seed_validation=True,
                hypothesis="Fresh10 of selected fast screen; all seeds retained",
            )
        ]
        requests.append(
            template | dict(batch_id=f"w13-{label}-validation", experiments=experiments)
        )
    short = dict(stage_depths=[2, 3, 3], fast_reset=True, epochs=6.2, head_lr=3.25, muon_lr=0.25)
    experiments = []
    for label, params in [
        ("control", short),
        ("lr", short | dict(head_lr=3.5, muon_lr=0.26)),
        ("lr_hold", short | dict(head_lr=3.5, muon_lr=0.26, lr_hold_frac=0.45)),
    ]:
        experiments.append(
            dict(
                id=f"w13_first62_{label}",
                params=params,
                n=3,
                seed=2300,
                parent_experiment="w11_first62_bn_control",
                stage=1,
                fresh_seed_validation=False,
                hypothesis="Small LR/schedule compensation for three removed steps",
            )
        )
    requests.append(template | dict(batch_id="w13-first62-local", experiments=experiments))
    Path("dev/sweeps/overnight_wave13.json").write_text(json.dumps(requests, indent=2) + "\n")
    ledger.write_reports(Path("results/overnight/reports"))


if __name__ == "__main__":
    main()
