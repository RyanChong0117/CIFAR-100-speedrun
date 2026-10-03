# RC3 session closeout — 2026-10-04

The session stopped because Codex reported that the usage limit was exhausted,
matching the user's explicit stopping condition. There was no separate Modal
spending cap. No new GPU experiments were launched in this overnight session.
The work below prepares the experiment loop; it does not establish an improvement.

RC3 remains the default submission. The `RC3` branch and annotated tag
`rc3-control-20261004` both resolve to
`edca55e7f0fb19eaca0c3521ac622cf9e37490d8`.
Development work is on `overnight/rc3-20261004` and remains uncommitted.
The production submission, best configuration and `benchmark/` match RC3 exactly.
Modal's container inventory returned an empty list during closeout.

1. **RC3 control:** 40/40 completed, mean accuracy 75.2955%, sample SD
   0.22863 percentage points, worst accuracy 74.86%, mean prepare 0.099430 s,
   mean train 6.967086 s, mean total 7.066516 s; 6.5 epochs and 163 updates.
   Actual recorded GPU was **NVIDIA A100-SXM4-80GB**, while the official target
   is **NVIDIA A100 80GB PCIe**. This distinction prevents treating differences
   between these hardware variants as algorithmic gains.
2. **Fastest candidate tested this session:** none. RC3 is the fastest measured
   result in the imported history, but it is the protected control.
3. **Fastest candidate that appears accuracy-safe:** no new candidate. RC3
   retains its existing 40-seed accuracy evidence. Its historical 10-seed PCIe
   screen measured 75.3740% and 8.091955 s.
4. **Best relevant 10+ seed result:** RC3's existing 40-seed result above. The
   10-seed PCIe screen has sample SD 0.2227 percentage points and minimum 75.01%.
5. **Best 40-seed validated result:** RC3; no replacement was validated or promoted.
6. **Complete current Pareto frontier:** the immutable
   [leaderboard](../results/overnight/reports/20261003T231306.995395Z-f2d1f7.md)
   and [machine-readable report](../results/overnight/reports/20261003T231306.995395Z-f2d1f7.json)
   contain all 53 imported experiments and 20 frontier points. Hardware variants
   are separated. Historical architectures and instrumented results are labelled;
   they must not be mistaken for new RC3 candidates. The append-only source is
   [ledger.jsonl](../results/overnight/ledger.jsonl).
7. **Total new experiments performed:** zero, zero new benchmark trials, and zero
   new profiling runs. Importing historical records does not count as running
   experiments. All 53 historical runs imported without parse warnings.
8. **Failed ideas from prior evidence:** batched Muon at 7 epochs measured
   8.611548 s versus 8.601023 s for its matched PCIe control; fused preparation
   measured about 8.642 s. Neither established a useful runtime improvement.
   Earlier compiled cropping required synthetic warmup through the actual
   preparation path: guessing its memory layout caused timed recompilation.
   The new helper follows that path, but still needs pinned CUDA validation.
9. **RC3 profile breakdown:** a new synchronized component profile is not yet
   available. Only the existing uninstrumented RC3 phase means above are
   confirmed. Prior development profiling measured weight reset around 53.73 ms,
   normalization 3.03 ms and whitening 1.69 ms; those are historical diagnostic
   measurements, not a fresh component breakdown of the 7.066516 s control.
10. **Largest known bottleneck:** training is 98.6% of RC3's measured total.
    The division among forward, backward, Muon and launch overhead remains
    unmeasured for this control. Whitening is a low-priority hypothesis based on
    earlier evidence.
11. **Recommended next optimization:** first run the new synthetic CUDA checks
    and RC3 profiler. Then screen 6.4/6.3/6.2 epochs with paired control seeds,
    alongside NS=2 and a limited larger-batch learning-rate search. Test exact
    vectorized weight reset as a separate implementation experiment. Promote
    through fresh 10-, 20- and 40-seed stages only when the evidence supports it.
12. **Exact source/config for the best measured result:** the protected tag above,
    `submissions/airbench_muon/submission.py`, and `dev/configs/best.json`.
    The original result is
    `results/airbench_muon/20261003T223840Z-034593f6/` and used seeds 10–49.
    `dev/controls/rc3.json` records complete parameters and source hash.
    Reproduce with the unchanged harness and those parameters on the recorded
    SXM4 hardware; a PCIe result is a separate validation, not an identical timing
    reproduction. `dev/sweeps/compression_6_5_confirm.json` retains the original
    confirmation request.

## Prepared development tooling

- `dev/rc3_profile.py`: train-data-only CUDA-event profiling, with preparation,
  augmentation, gathering, forward, loss, backward, SGD, Muon, nested NS and
  miscellaneous intervals. Instrumentation is absent from the submitted model.
- `dev/overnight_variants.py`: opt-in NS count, exact vectorized reset,
  vectorized/compiled crop and batched Muon, applied only to frozen experiment
  copies. These are experimental implementations, not validated speedups.
- `dev/overnight_checks.py`: synthetic pixel, RNG, layout, reset and optimizer
  checks. Pinned PyTorch 2.4 CUDA checks remain outstanding. Supplemental CPU
  checks used a different torch version and are not authoritative for A100.
- `dev/overnight_worker.py` and `overnight_modal.py`: bounded sequential jobs in
  independent containers, maximum four containers, hardware recording, frozen
  source hashes, retained logs/results, deadline and failure stops, and an
  explicit PCIe gate for final validation.
- `dev/experiment_ledger.py`: append-only result history and hardware-separated
  frontiers, preserving failed attempts and every recorded seed.

The new launcher has not been exercised end to end. No experiment request or
deadline was committed for unattended execution. Before resuming, confirm the
usage allowance, inspect this worktree, run pinned CUDA checks, and construct a
fresh bounded request. Do not launch final validation or modify the submission
merely because the tooling exists.
