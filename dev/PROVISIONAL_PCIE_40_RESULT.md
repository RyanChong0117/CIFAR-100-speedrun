# Provisional candidate: forty planned PCIe seeds

The PCIe results do not establish a true mean accuracy above 75%.

- Completed observations: 40/40; seeds 11000-11039; GPU NVIDIA A100 80GB PCIe.
- Mean accuracy: 75.02625%.
- Exploratory conservative one-sided 95% lower confidence bound: 74.96819%.
- Sample SD: 0.21635 percentage points.
- Minimum / maximum: 74.53% / 75.49%.
- Individual seeds at least 75%: 21/40.
- Mean prepare: 0.021989 seconds.
- Mean train: 7.017311 seconds.
- Mean prepare + train: 7.039300 seconds.
- Original container preempted after 37 observations. Three missing seeds were recovered from fresh state on a second verified PCIe allocation. This is a derived preplanned cohort, not an uninterrupted forty-trial harness qualification run.
- Every observed result is retained. The lost attempt's unrecorded outcome is unknown; no completed seed was replayed. Original interrupted run stays incomplete.
- Original-container RC3 before control: three seeds, mean total 7.767568 seconds. After control was not reached; a complete matched before/after timing comparison is unavailable.
- Protected RC3's 7.066516-second reference was measured on SXM4. Comparing this PCIe cohort with that reference cannot establish a small runtime gain.
- Source hashes match the original archived provisional recipe byte-for-byte.
- Configuration: 6.3 epochs; 158 source-derived steps; batch 2000; widths [128,512,512]; depths [2,3,3]; Muon LR 0.25; head LR 3.25; fast reset. Complete parameters, per-seed observations and provenance are in adjacent JSON.
- RC3 remains the default. No replacement is promoted.

## Segments

- pcie_original_provisional_40_a2: 37 observations, mean accuracy 75.03811%, total 7.047228 seconds; raw path results/overnight/rc3-pcie-followup-20261004/batches/pcie-original-provisional-attempt2/runs/pcie_original_provisional_40_a2/20261004T101333Z-cc5d45fd.
- pcie_original_provisional_recovery3_a3: 3 observations, mean accuracy 74.88000%, total 6.941528 seconds; raw path results/overnight/rc3-pcie-followup-20261004/batches/pcie-original-provisional-recovery-attempt3/runs/pcie_original_provisional_recovery3_a3/20261004T102658Z-0971dcde.
