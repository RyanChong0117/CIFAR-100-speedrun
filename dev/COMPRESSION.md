# Compression experiments

## 40-seed confirmation and current default

**6.5 epochs is now the production default on RC3.** Confirmation used 40 new
seeds, **10-49**, separate from screening seeds 0-9. All 40 trials succeeded and
their mean accuracy passed the 75% development target:

| Metric | Confirmation |
| --- | ---: |
| Mean accuracy | **75.2955%** |
| Accuracy SD | 0.2286 percentage points |
| Mean prepare + train | **7.066516 s** |
| Mean prepare | 0.099430 s |
| Mean train | 6.967086 s |
| Slowest full test inference | 1.038345 s |
| Build | 261.812 s |

This allocation used an **A100-SXM4-80GB**, PyTorch 2.4.0+cu124 and four PyTorch
CPU threads. It is a development confirmation, not an official PCIe score.
Its timing cannot be directly compared with the earlier PCIe study. The
same-GPU screening evidence for the speed gain remains the 5.92% reduction
reported below. The official forty-seed PCIe evaluation is still separate.

Only the epoch default changed in the submission. `dev/configs/best.json` now
matches the confirmed recipe. Batched Muon and fused preparation remain isolated
experiments. The comparison runner explicitly retains seven epochs for its
baseline, optimizer and preprocessing variants.

Confirmation artifacts: `results/airbench_muon/20261003T223840Z-034593f6/`.
To reproduce the 40-seed confirmation:

```text
modal run --detach modal_runner.py::sweep --file dev/sweeps/compression_6_5_confirm.json
```

## Results: 3 October 2026

All 50 trials completed on one **NVIDIA A100 80GB PCIe**, using PyTorch
2.4.0+cu124 and four PyTorch CPU threads. The base recipe was `ea41efd`.
The production submission was unchanged during this screening study.

| Variant | Mean prepare + train | Mean accuracy | Time change | Mean prepare |
| --- | ---: | ---: | ---: | ---: |
| Baseline, 7 epochs | 8.6010 s | 75.691% | reference | 74.40 ms |
| 6.75 epochs | 8.3590 s | 75.510% | 2.81% faster | 75.77 ms |
| **6.5 epochs** | **8.0920 s** | **75.374%** | **5.92% faster** | 73.80 ms |
| Batched Muon | 8.6115 s | 75.716% | 0.12% slower | 75.85 ms |
| Fused preparation | 8.6420 s | 75.565% | 0.48% slower | 72.26 ms |

**6.5 epochs** saved 0.5091 s with a mean accuracy decrease of 0.317 percentage
points. This ten-seed screen was followed by the forty-seed confirmation above.
Batched Muon and fused preparation did not demonstrate a useful total-time
improvement in this study.

The baseline profile found **53.73 ms in weight reset**, versus 3.03 ms in
normalization and 1.69 ms in whitening. `Conv.reset_parameters()` calls
`torch.nn.init.dirac_`, whose pinned implementation writes each diagonal entry
in a Python loop. A vectorized equivalent is the next preparation hypothesis;
it has not been implemented or benchmarked here.

Synthetic validation found exact full-shape fused-normalization pixels, strides,
and RNG preservation. All reset checks passed. Batched Muon relative update
differences were 1.319% and 0.543% for its two matrix groups. No recompilation
messages appeared after the first completed trial in any benchmark block.

Raw results and source snapshots:
`results/compression/airbench_muon/20261003T220623Z/`.
The detailed report there discusses measurement limitations, including the
five-trial baseline blocks versus ten-trial variant blocks.

## Reproduce

Run the approved comparison with:

```text
modal run --detach modal_runner.py::compression
```

This runs on one Modal A100-80GB allocation. It does not modify the production
submission or the organizer harness. The study generates self-contained copies
of the current submission under its result directory and measures those copies
through `benchmark.run`.

| Variant | Change from the seven-epoch reference | Trials |
| --- | --- | ---: |
| baseline | None | 10 |
| epochs_6_75 | 169 rather than 175 training steps | 10 |
| epochs_6_5 | 163 rather than 175 training steps | 10 |
| batched_muon | Batch equal-shaped Newton-Schulz updates, retaining three iterations | 10 |
| fused_prepare | Compile uint8 conversion, normalization, FP16 conversion and layout conversion | 10 |

Each variant uses seeds 0 through 9. Baseline seeds 0-4 run before the other
variants and 5-9 run afterwards. These are exactly 50 uninstrumented harness
trials, with no discarded seeds. There are also three baseline profiling trials
and synthetic correctness checks; their timings are excluded from the comparison.

The entire comparison uses one GPU, with a shared per-study Inductor cache to
reuse model autotuning choices where keys match. Each harness invocation has a
fresh process and performs the original synthetic warmup. Compilation and all
real-data preparation use the same phase boundaries and limits as the harness.
The fused preprocessing variant additionally warms its actual 50,000-image shape
using synthetic data during build. No real-data work is moved outside timing.

The checks exercise resets after intervening training, immutable input tensors,
fractional step counts, full-shape normalization for all 256 uint8 values, strides,
RNG preservation, and batched Muon numerical differences. BF16 batched Muon is
not assumed bitwise equivalent to serial Muon.

Profiling uses the seven-epoch reference, inserts asynchronous CUDA events around
the original statements and
verifies that removing these calls restores the original function AST. It does
not synchronize within the loop. Reported component intervals include stream
idle time caused by host launch gaps; they are not sums of busy-kernel time.
The two preparation categories `transfer_cast` and `normalize` are the work
targeted by the fused variant, with transfer itself remaining unchanged.

Results are saved under `results/compression/airbench_muon/<timestamp>/`, both
locally and in the Modal results volume. They include source snapshots, hashes,
validation, profile intervals, complete harness records, compiler diagnostics,
and `comparison.json`. Paired confidence intervals describe seed variation;
they do not account for systematic timing drift across sequential variants.

These are development results. The actual GPU identity is retained in each
harness configuration. A ten-seed mean above 75% is a screening result, not an
official qualification or a substitute for a 40-seed confirmation.
