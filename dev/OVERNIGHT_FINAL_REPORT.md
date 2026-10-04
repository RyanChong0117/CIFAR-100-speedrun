# RC3 closeout — 20261004T033256.302827Z

Search closed after the latest independent fresh10 validations and structural screens plateaued under accuracy/hardware constraints. All GPU jobs completed; empty Modal container inventory. Ended within the original five-hour deadline. RC3 remains the unchanged default; no candidate met the replacement gate.

RC3 remains the default. Candidate labels describe evidence, not promotion. Accuracy bounds are exploratory one-sided 95% Student-t bounds and are not corrected for adaptive candidate selection. All bad seeds and failures remain.

1. **RC3 control.** `20261003T223840Z-034593f6`: 75.2955% mean, SD 0.2286 pp, worst 74.86%, prepare 0.099430 s, train 6.967086 s, total 7.066516 s; 40 trials, 163 steps, NVIDIA A100-SXM4-80GB. Protected tag: `rc3-control-20261004`, commit `edca55e7f0fb19eaca0c3521ac622cf9e37490d8`. The 7.0665 s result was SXM4; official target is PCIe.

2. **Fastest candidate tested.** `w8_firstdepth_bs1600`: 74.3267% mean, SD 0.3398 pp, worst 73.95%, prepare 0.050305 s, train 5.550864 s, total 5.601169 s; 3 trials, 171 steps, NVIDIA A100-SXM4-80GB. This can be accuracy-unsafe and is not a replacement.

3. **Fastest candidate with preliminary accuracy safety.** `w6_firstdepth_6_3_tuned_10`: 75.1730% mean, SD 0.2099 pp, worst 74.78%, prepare 0.024842 s, train 6.238688 s, total 6.263530 s; 10 trials, 158 steps, NVIDIA A100-SXM4-80GB. Fresh 40-seed PCIe validation remains required unless separately recorded.

4. **Best 10+ seed evidence by exact hardware.**

- `w9_firstdepth_6_3_tuned_40_pcie`: 75.0883% mean, SD 0.2402 pp, worst 74.60%, prepare 0.027136 s, train 7.021074 s, total 7.048210 s; 40 trials, 158 steps, NVIDIA A100 80GB PCIe.
- `w6_firstdepth_6_3_tuned_10`: 75.1730% mean, SD 0.2099 pp, worst 74.78%, prepare 0.024842 s, train 6.238688 s, total 6.263530 s; 10 trials, 158 steps, NVIDIA A100-SXM4-80GB.

Strongest mean-accuracy result with at least20 fresh candidate trials: `w6_depth_2_3_3_20`: 75.3590% mean, SD 0.2098 pp, worst 74.92%, prepare 0.095295 s, train 6.437073 s, total 6.532368 s; 20 trials, 163 steps, NVIDIA A100-SXM4-80GB. This is not a forty-seed replacement.

5. **Best fresh 40-seed candidate.** `w9_firstdepth_6_3_tuned_40_pcie`: 75.0883% mean, SD 0.2402 pp, worst 74.60%, prepare 0.027136 s, train 7.021074 s, total 7.048210 s; 40 trials, 158 steps, NVIDIA A100 80GB PCIe.

All fresh forty-seed replications and read-only replacement checks:

- `w9_firstdepth_6_3_tuned_40_any_a100`: 75.1598% mean, SD 0.2452 pp, worst 74.66%, prepare 0.022551 s, train 7.319892 s, total 7.342443 s; 40 trials, 158 steps, NVIDIA A100 80GB PCIe. Replacement eligible: False; failed checks: runtime_improves_clearly.
- `w9_firstdepth_6_3_tuned_40_pcie`: 75.0883% mean, SD 0.2402 pp, worst 74.60%, prepare 0.027136 s, train 7.021074 s, total 7.048210 s; 40 trials, 158 steps, NVIDIA A100 80GB PCIe. Replacement eligible: False; failed checks: accuracy_bound_has_margin, runtime_improves_clearly.
- Descriptive pool of 2 same-configuration replications: 80 distinct fresh seeds, mean 75.1240%, SD 0.2438 pp, worst 74.60%, total 7.195326 s. This is not a new experiment.

All replications of a configuration must be reviewed together. Neither a fast host nor a higher-accuracy replication can be selected alone.

6. **Complete Pareto frontier.** Immutable leaderboard `results\overnight\reports\20261004T033256.147376Z-1daae8.md` and machine-readable report `results\overnight\reports\20261004T033256.147376Z-1daae8.json` contain every historical and new frontier point, separated by exact GPU. Axes: mean/minimum accuracy and count maximized; accuracy SD, prepare/train/total minimized. This descriptive frontier alone does not qualify a candidate.

7. **Experiment totals.** 191 complete new benchmark experiments, 1 incomplete; 1105 observed trials, 10 requested trials missing. One diagnostic profile experiment contains three train-only trials. Historical imports are not new experiments. Live/unstarted requests are not counted as performed.

8. **Major failed ideas.** NS=2 lost about 1.5 pp with little speed benefit; no NS=1 real-data search followed. Larger batches through 3125 remained below qualification after limited LR/schedule compensation. Vectorized/compiled crop did not improve measured totals. Translate=0 lost accuracy. Reducing last stage depth lost substantial accuracy. Middle-stage depth needed extra epochs and LR compensation; retaining its skip gave weaker fresh 10-seed evidence. A lucky three-seed 6.3+reset result did not repeat over ten. PCIe warming and clock reduction raised sustained runtime; every warm result was retained. Smaller batch1750 at6.1 epochs averaged74.811% over fresh10 despite a promising screen. Compiled pre-Muon fusion saved only5ms in the shorter model and gave weaker accuracy. Two identical first-stage6.3 frozen forty-seed replications failed the combined accuracy/runtime replacement gate. Longer BN windows and smoothing screens produced encouraging three-seed means but failed fresh10: BN0.65=74.925%, BN0.75=75.041%, BN0.80=74.986%, smoothing0.35=75.000%. First-stage6.2 fresh10 averaged 75.087% with an accuracy bound below75%; stronger LR/hold retuning lost accuracy. Batched Muon fresh10 averaged75.182% with its lower bound essentially75%, without a clear sustained PCIe win. Later structural screens retaining the first-stage skip did not recover sufficient accuracy at6.1/6.2 epochs;6.3 remained marginal. Full decisions remain in the ledger.

9. **RC3 profile.** Three synchronized CUDA-event diagnostic trials on SXM4. Instrumentation includes event/host dispatch overhead; these totals cannot establish a submission speedup. NS is nested in Muon and crop in augmentation.

| Region | Mean seconds |
|---|---:|
| prepare | 0.113174 |
| reset | 0.067454 |
| transfer_cast | 0.036318 |
| normalize | 0.003009 |
| whitening | 0.001706 |
| initial_flip | 0.001598 |
| padding | 0.002325 |
| optimizers | 0.000466 |
| train | 7.013227 |
| augmentation | 0.029305 |
| batch_crop | 0.027282 |
| batch_gather | 0.012749 |
| forward | 2.169966 |
| backward | 4.368658 |
| sgd | 0.007646 |
| muon | 0.406661 |
| newton_schulz | 0.292249 |
| loss | 0.012264 |
| train_misc | 0.005977 |

10. **Largest bottleneck.** Forward/backward compute: 93.23% of profiled training. Whitening is about 1.7 ms and crop about 27 ms, so reducing those offers little remaining benefit. Exact reset saves roughly 50–65 ms.

11. **Recommended next optimization.** Keep RC3 as the submission. Obtain a development-only kernel/shape trace of forward and backward on PCIe, then target the largest convolution or layout-copy cost with a change that preserves learning behavior. The6.5-epoch first-stage depth reduction has the strongest accuracy evidence as a development reference. Further shortening and scalar retuning have plateaued; do not expand a blind grid. Use matched sustained PCIe controls for runtime and the full fresh-seed funnel for any candidate.

12. **Reproduction.** Complete parameter dictionaries, source hashes, commit, seeds, architecture/optimizer and raw result paths for the fastest screen and accuracy-safe candidate are embedded in the adjacent JSON report. Frozen source is under each raw run's `source/`; helper files are required. The unchanged harness reproduces a run with `python -m benchmark.run --submission-path <frozen-recipe> --params <complete-parameter-json> --n <trial-count> --seed <first-seed>`. RC3 source/config is at the protected tag and `dev/controls/rc3.json`. Hardware, software and sustained thermal conditions must match for small timing comparisons. The byte-exact frozen forty-seed recipe is in `results/overnight/rc3-20261004/exports/firstshort40-exact/`; `verification.json` checks both source hashes. Its parameters are embedded in defaults, so reproduce that packaged artifact without training overrides. Earlier historical source downloads may normalize newlines; benchmark hashes refer to original remote bytes, and those historical copies were preserved. The earlier fastest10-seed recipe is byte-exact in `results/overnight/rc3-20261004/exports/firstshort10-exact/`, with its full `parameters.json` and original seeds2600-2609. Optimizer step counts are source-derived deterministic counts where labelled inferred; the diagnostic control separately observed163 updates.


Artifact links

- [Full machine-readable closeout](../results/overnight/rc3-20261004/reports/20261004T033256.302827Z-closeout.json)
- [Complete leaderboard and Pareto frontier](../results/overnight/reports/20261004T033256.147376Z-1daae8.md)
- [Append-only experiment ledger](../results/overnight/ledger.jsonl)
- [Verified original recipe source](../results/overnight/rc3-20261004/exports/firstshort10-exact/submission.py) and [required helper](../results/overnight/rc3-20261004/exports/firstshort10-exact/overnight_variants.py)
- [Strongest20 complete parameters](../results/overnight/rc3-20261004/exports/firstshort10-exact/strongest20-parameters.json) and [reproduction metadata](../results/overnight/rc3-20261004/exports/firstshort10-exact/strongest20-reproduction.json)
- [Frozen forty-trial finalist source](../results/overnight/rc3-20261004/exports/firstshort40-exact/submission.py)

Reproduce the strongest20 result using the verified original source directory and required helper above, the complete strongest20 parameter dictionary, twenty trials, first seed3600, and the recorded pinned2.4.0+cu124 environment on an A100-SXM4-80GB. This is an experimental reference, not a qualified submission. The frozen forty-trial artifact uses embedded training defaults and empty benchmark parameter overrides; its first seed is5000.

Completion accounting:191 completed new benchmark experiments /1105 observed trials; one interrupted build with zero observed trials and ten missing planned trials. Three separate incompatible-hardware allocation audits retain50 unstarted planned trials; those are not benchmark observations. The synchronized profile has three separate diagnostic trials. All GPU jobs stopped; Modal container inventory was empty. RC3/default/benchmark files match the protected revision.
