"""Write an immutable evidence-based campaign checkpoint or closeout report."""

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from dev.experiment_ledger import RC3_RUN, Ledger, finite, summarize_trials
from dev.overnight_gate import evaluate


def describe(r):
    if not r:
        return "None has reached this evidence gate."
    return (f"`{r['experiment_id']}`: {r['mean_accuracy'] * 100:.4f}% mean, "
            f"SD {r['accuracy_std'] * 100:.4f} pp, worst {r['min_accuracy'] * 100:.2f}%, "
            f"prepare {r['mean_prepare_time']:.6f} s, train {r['mean_train_time']:.6f} s, "
            f"total {r['mean_prepare_train_time']:.6f} s; {r['trial_count']} trials, "
            f"{r['total_optimizer_steps']} steps, {r['gpu_model']}.")


def write_report(reason, finished=False):
    ledger = Ledger(Path("results/overnight/ledger.jsonl"))
    records = ledger.records()
    reports = ledger.write_reports(Path("results/overnight/reports"))
    session = Path("results/overnight/rc3-20261004")
    new = [r for r in records if r.get("campaign") == "rc3-20261004"
           and not r.get("instrumented")]
    complete = [r for r in new if r.get("complete") and finite(r.get("mean_accuracy"))]
    candidates = [r for r in complete if "control" not in r['experiment_id']]
    safe = [r for r in candidates if r['trial_count'] >= 10
            and finite(r.get('accuracy_lower_95_one_sided'))
            and r['accuracy_lower_95_one_sided'] > .75]
    validated = [r for r in safe if r['trial_count'] >= 40
                 and r.get('fresh_seed_validation')]
    finalists = [r for r in candidates if r['trial_count'] == 40
                 and r.get('fresh_seed_validation')]
    gates = [evaluate(r, records) for r in finalists]
    cohorts = defaultdict(list)
    for r in finalists:
        cohorts[json.dumps([r['parameters'], r['gpu_models']], sort_keys=True)].append(r)
    pooled = []
    for rows in cohorts.values():
        if len(rows) < 2:
            continue
        trials = [t for r in rows for t in r['trials']]
        seeds = [t['seed'] for t in trials]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Replicate cohort has overlapping seeds; do not pool')
        pooled.append(dict(experiment_ids=[r['experiment_id'] for r in rows],
                           parameters=rows[0]['parameters'], gpu_models=rows[0]['gpu_models'],
                           statistics=summarize_trials(trials, len(trials)),
                           kind='descriptive_posthoc_pool_not_a_new_experiment'))
    rc3 = next(r for r in records if r['run_id'] == RC3_RUN)
    def fastest(rows):
        return min(rows, key=lambda r: r['mean_prepare_train_time']) if rows else None
    best, best_safe, best40 = fastest(candidates), fastest(safe), fastest(validated)
    by_gpu = defaultdict(list)
    for r in safe:
        by_gpu[r['gpu_model']].append(r)
    profile = json.loads((session / "profile-wave1.json").read_text())
    cuda = profile['summary']['cuda_seconds']
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')
    output = session / 'reports' / (stamp + ("-closeout" if finished else "-checkpoint"))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(reason=reason, finished=finished, default_promoted=False,
                   protected_control=rc3, fastest_tested=best,
                   fastest_appears_accuracy_safe=best_safe,
                   fresh40=best40, all_fresh40=finalists, replacement_gates=gates,
                   all_replication_cohorts=pooled,
                   strongest_accuracy20=max((r for r in candidates if r['trial_count'] >= 20),
                                            key=lambda r: r['mean_accuracy'], default=None),
                   leaderboard=reports,
                   complete_new_experiments=len(complete),
                   incomplete_new_experiments=len(new)-len(complete),
                   observed_new_trials=sum(r['trial_count'] for r in new),
                   missing_new_trials=sum(r['missing_trials'] for r in new))
    lines = [f"# RC3 {'closeout' if finished else 'checkpoint'} — {stamp}", "", reason, "",
             "RC3 remains the default. Candidate labels describe evidence, not promotion. "
             "Accuracy bounds are exploratory one-sided 95% Student-t bounds and are not "
             "corrected for adaptive candidate selection. All bad seeds and failures remain.", "",
             "1. **RC3 control.** " + describe(rc3) + " Protected tag: "
             "`rc3-control-20261004`, commit `edca55e7f0fb19eaca0c3521ac622cf9e37490d8`. "
             "The 7.0665 s result was SXM4; official target is PCIe.", "",
             "2. **Fastest candidate tested.** " + describe(best)
             + " This can be accuracy-unsafe and is not a replacement.", "",
             "3. **Fastest candidate with preliminary accuracy safety.** " + describe(best_safe)
             + " Fresh 40-seed PCIe validation remains required unless separately recorded.", ""]
    lines += ["4. **Best 10+ seed evidence by exact hardware.**"]
    lines += ["", *[f"- {describe(fastest(rows))}" for rows in by_gpu.values()], "",
              "5. **Best fresh 40-seed candidate.** " + describe(best40)
              + (" RC3 retains the protected existing 40-seed result." if not best40 else ""), "",
              "All fresh forty-seed replications and read-only replacement checks:", ""]
    for r, gate in zip(finalists, gates, strict=True):
        failures = [name for name, passed in gate['checks'].items() if not passed]
        lines += [f"- {describe(r)} Replacement eligible: {gate['promotable']}; "
                  f"failed checks: {', '.join(failures) or 'none'}."]
    for cohort in pooled:
        stats = cohort['statistics']
        lines += [f"- Descriptive pool of {len(cohort['experiment_ids'])} same-configuration "
                  f"replications: {stats['trial_count']} distinct fresh seeds, mean "
                  f"{100*stats['mean_accuracy']:.4f}%, SD {100*stats['accuracy_std']:.4f} pp, "
                  f"worst {100*stats['min_accuracy']:.2f}%, total "
                  f"{stats['mean_prepare_train_time']:.6f} s. This is not a new experiment."]
    lines += ["", "All replications of a configuration must be reviewed together. "
              "Neither a fast host nor a higher-accuracy replication can be selected alone.", "",
              f"6. **Complete Pareto frontier.** Immutable leaderboard `{reports['markdown']}` "
              f"and machine-readable report `{reports['json']}` contain every historical and "
              "new frontier point, separated by exact GPU. Axes: mean/minimum accuracy and "
              "count maximized; accuracy SD, prepare/train/total minimized. This descriptive "
              "frontier alone does not qualify a candidate.", "",
              f"7. **Experiment totals.** {len(complete)} complete new benchmark experiments, "
              f"{len(new)-len(complete)} incomplete; {payload['observed_new_trials']} observed "
              f"trials, {payload['missing_new_trials']} requested trials missing. One diagnostic "
              "profile experiment contains three train-only trials. Historical imports are "
              "not new experiments. Live/unstarted requests are not counted as performed.", "",
              "8. **Major failed ideas.** NS=2 lost about 1.5 pp with little speed benefit; "
              "no NS=1 real-data search followed. Larger batches through 3125 remained below "
              "qualification after limited LR/schedule compensation. Vectorized/compiled "
              "crop did not improve measured totals. Translate=0 lost accuracy. Reducing last "
              "stage depth lost substantial accuracy. Middle-stage depth needed extra epochs "
              "and LR compensation; retaining its skip gave weaker fresh 10-seed evidence. "
              "A lucky three-seed 6.3+reset result did not repeat over ten. PCIe warming and "
              "clock reduction raised sustained runtime; every warm result was retained. "
              "Smaller batch1750 at6.1 epochs averaged74.811% over fresh10 despite a "
              "promising screen. Compiled pre-Muon fusion saved only5ms in the shorter "
              "model and gave weaker accuracy. Two identical first-stage6.3 frozen "
              "forty-seed replications failed the combined accuracy/runtime replacement "
              "gate. Longer BN windows and smoothing screens produced encouraging "
              "three-seed means but failed fresh10: BN0.65=74.925%, BN0.75=75.041%, "
              "BN0.80=74.986%, smoothing0.35=75.000%. First-stage6.2 fresh10 averaged "
              "75.087% with an accuracy bound below75%; stronger LR/hold retuning lost "
              "accuracy. Batched Muon fresh10 averaged75.182% with its lower bound "
              "essentially75%, without a clear sustained PCIe win. Later structural "
              "decisions remain in the ledger.", "",
              "9. **RC3 profile.** Three synchronized CUDA-event diagnostic trials on SXM4. "
              "Instrumentation includes event/host dispatch overhead; these totals cannot "
              "establish a submission speedup. NS is nested in Muon and crop in augmentation.", "",
              "| Region | Mean seconds |", "|---|---:|"]
    keys = ['prepare', 'reset', 'transfer_cast', 'normalize', 'whitening', 'initial_flip',
            'padding', 'optimizers', 'train', 'augmentation', 'batch_crop', 'batch_gather',
            'forward', 'backward', 'sgd', 'muon', 'newton_schulz', 'loss', 'train_misc']
    lines += [f"| {key} | {cuda[key]['mean']:.6f} |" for key in keys]
    share = (cuda['forward']['mean']+cuda['backward']['mean'])/cuda['train']['mean']*100
    lines += ["", f"10. **Largest bottleneck.** Forward/backward compute: {share:.2f}% of "
              "profiled training. Whitening is about 1.7 ms and crop about 27 ms, so reducing "
              "those offers little remaining benefit. Exact reset saves roughly 50–65 ms.", "",
              "11. **Recommended next optimization.** Keep RC3 as the submission. The "
              "first-stage depth reduction has the strongest capacity evidence; further "
              "shortening needs accuracy recovery as well as sustained PCIe speed. Advance "
              "BN averaging or batched Muon only when their fresh-seed evidence warrants it. "
              "Use matched sustained PCIe controls and avoid broad grids without new evidence.", "",
              "12. **Reproduction.** Complete parameter dictionaries, source hashes, commit, "
              "seeds, architecture/optimizer and raw result paths for the fastest screen and "
              "accuracy-safe candidate are embedded in the adjacent JSON report. Frozen source "
              "is under each raw run's `source/`; helper files are required. The unchanged "
              "harness reproduces a run with `python -m benchmark.run --submission-path "
              "<frozen-recipe> --params <complete-parameter-json> --n <trial-count> --seed "
              "<first-seed>`. RC3 source/config is at the protected tag and "
              "`dev/controls/rc3.json`. Hardware, software and sustained thermal conditions "
              "must match for small timing comparisons. The byte-exact frozen forty-seed "
              "recipe is in `results/overnight/rc3-20261004/exports/firstshort40-exact/`; "
              "`verification.json` checks both source hashes. Its parameters are embedded "
              "in defaults, so reproduce that packaged artifact without training overrides. "
              "Earlier historical source downloads may normalize newlines; benchmark hashes "
              "refer to original remote bytes, and those historical copies were preserved. "
              "The earlier fastest10-seed recipe is byte-exact in "
              "`results/overnight/rc3-20261004/exports/firstshort10-exact/`, with its full "
              "`parameters.json` and original seeds2600-2609. Optimizer step counts are "
              "source-derived deterministic counts where labelled inferred; the diagnostic "
              "control separately observed163 updates."]
    json_path = output.parent / (output.name + '.json')
    md_path = output.parent / (output.name + '.md')
    with json_path.open('x', encoding='utf-8') as file:
        file.write(json.dumps(payload, indent=2)+'\n')
    with md_path.open('x', encoding='utf-8') as file:
        file.write('\n'.join(lines)+'\n')
    return str(md_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reason', required=True)
    parser.add_argument('--finished', action='store_true')
    args = parser.parse_args()
    print(write_report(args.reason, args.finished))
