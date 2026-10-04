"""Fresh validation of the remaining fast BN branch and small local retuning."""

import json
import subprocess
from pathlib import Path

from dev.experiment_ledger import Ledger


def main():
    ledger = Ledger(Path("results/overnight/ledger.jsonl"))
    ledger.import_results(Path("results"))
    for r in ledger.records():
        ident = r["experiment_id"]
        if not ident.startswith("w11_") or "control" in ident:
            continue
        if ident in ["w11_1750_bn_bn65", "w11_1750_bn_bn75"]:
            status, conclusion = (
                "promoted_to_10",
                "Thin three-seed accuracy, exceptional speed; limited fresh10.",
            )
        else:
            status, conclusion = (
                "rejected",
                "No persuasive accuracy/runtime margin for further validation.",
            )
        ledger.annotate(ident, promotion_status=status, conclusion=conclusion)
    template = json.loads(Path("dev/sweeps/overnight_wave11.json").read_text())[0]
    template = {k: v for k, v in template.items() if k not in ["batch_id", "experiments"]}
    template["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    small = dict(
        stage_depths=[2, 3, 3],
        fast_reset=True,
        batch_size=1750,
        epochs=6.0,
        head_lr=3.4285714285714284,
        bias_lr=0.060571428571428575,
        muon_lr=0.21,
    )

    def experiment(ident, params, parent, n=3, seed=2200, stage=1, hypothesis="Paired screen"):
        return dict(
            id=ident,
            params=params,
            parent_experiment=parent,
            n=n,
            seed=seed,
            stage=stage,
            fresh_seed_validation=stage > 1,
            hypothesis=hypothesis,
        )

    requests = []
    for label, decay in [("65", 0.65), ("75", 0.75)]:
        params = small | dict(bn_momentum=decay)
        parent = f"w11_1750_bn_bn{label}"
        requests.append(
            template
            | dict(
                batch_id=f"w12-bn{label}-validation",
                experiments=[
                    experiment(
                        f"w12_bn{label}_control_before", small | dict(bn_momentum=0.6), parent
                    ),
                    experiment(
                        f"w12_1750_bn{label}_10",
                        params,
                        parent,
                        n=10,
                        seed=7200,
                        stage=2,
                        hypothesis="Fresh10 validation of fast averaging; fresh seeds",
                    ),
                    experiment(
                        f"w12_bn{label}_control_after", small | dict(bn_momentum=0.6), parent
                    ),
                ],
            )
        )
    experiments = [
        experiment("w12_smooth_control", small | dict(bn_momentum=0.65), "w11_1750_bn_bn65")
    ]
    for label, value in [("25", 0.25), ("35", 0.35)]:
        experiments.append(
            experiment(
                f"w12_1750_bn65_smooth{label}",
                small | dict(bn_momentum=0.65, label_smoothing=value),
                "w11_1750_bn_bn65",
                hypothesis=("Limited smoothing compensation for shorter sample budget"),
            )
        )
    requests.append(template | dict(batch_id="w12-smoothing-screen", experiments=experiments))
    experiments = [
        experiment("w12_avg_control", small | dict(bn_momentum=0.75), "w11_1750_bn_bn75")
    ]
    for label, value in [("80", 0.8), ("85", 0.85)]:
        experiments.append(
            experiment(
                f"w12_1750_bn{label}",
                small | dict(bn_momentum=value),
                "w11_1750_bn_bn75",
                hypothesis="Test nearby longer running-statistics window",
            )
        )
    requests.append(template | dict(batch_id="w12-averaging-screen", experiments=experiments))
    Path("dev/sweeps/overnight_wave12.json").write_text(json.dumps(requests, indent=2) + "\n")
    ledger.write_reports(Path("results/overnight/reports"))


if __name__ == "__main__":
    main()
