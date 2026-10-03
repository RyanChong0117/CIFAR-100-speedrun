# CIFAR-100 development experiments

The user-reported baseline is 74.8% accuracy and 81.4 seconds preparation plus
training. This is a reference observation, not a three-seed result from this sweep.
The architecture sweep is paused. The current study holds standard ResNet11 and
30 epochs fixed and tests recipes intended to converge within that budget.

## Rapid convergence study (current)

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
modal run --detach modal_rapid_sweep.py
```

This launches twelve one-seed screens in parallel, subject to the provider's GPU
quota. Each benchmark owns one A100-80GB, has four CPUs, and uses the unchanged
harness with `--n 1 --seed 0 --no-accuracy-target --params`. PCIe and SXM hosts
remain accepted for development. Actual device names accompany every result.
Architecture is fixed to ResNet11, widths `[64,128,256,512]`, residual depths
`[1,1,1]`, global max pooling and max-pool downsampling. Epochs=30, batch=512,
momentum=0.9, Nesterov, BF16, and all data handling remain fixed.

The one-epoch anchor uses LR=0.2, smoothing=0.05, cutout=8, decay=0.0005 and the
existing linear-warmup/cosine schedule. This isolates shorter warmup from other
recipe changes. The submission's default model is now standard ResNet11; default
training settings are retained until measured validation justifies changing them.

| ID | Difference from the one-epoch anchor | Information sought |
| --- | --- | --- |
| 001 | Warmup 5 | Reproduce the existing 30-epoch ResNet11 control |
| 002 | None | Establish the short-warmup anchor |
| 003 | Warmup 0 | Does starting at peak LR destabilize this model? |
| 004 | Warmup 2 | Does a second ramp epoch help convergence? |
| 005 | LR 0.12 | Are smaller updates more useful over 30 epochs? |
| 006 | LR 0.28 | Can stronger early updates converge faster? |
| 007 | Smoothing 0 | Does reduced smoothing help short training? |
| 008 | Cutout 0 | Does occlusion slow convergence at this budget? |
| 009 | Decay 0.001 | Test the stronger decay that helped the earlier recipe sweep |
| 010 | One-cycle, peak 0.2 | Compare complete LR schedule with 002 |
| 011 | One-cycle, peak 0.28 | Schedule pair with 006; LR pair with 010 |
| 012 | One-cycle, warmup 2, smoothing 0, cutout 4, decay 0.001 | Test a combined short-training hypothesis |

The one-cycle option is a two-phase cosine rise/fall matching the pinned
[PyTorch 2.4 OneCycleLR](https://github.com/pytorch/pytorch/blob/v2.4.0/torch/optim/lr_scheduler.py).
`warmup_epochs` sets the rising duration rather than using the usual 30% fraction.
`one_cycle_div_factor` defaults to 25 (initial LR = peak/25), and
`one_cycle_final_div_factor` defaults to 10000 (final LR = initial/10000).
Momentum is constant; changing the LR schedule does not cycle SGD momentum.
Zero warmup starts at peak LR. A single rising update also uses the peak to avoid
the reference scheduler's division by zero. Cosine retains exactly the previous
LR values, including its 10%-of-peak linear ramp and 0.1%-of-peak final LR.
LR lists are generated anew in timed `prepare`; the training loop still performs
one list lookup per update and adds no scheduler call overhead.

Completed screens at **>=74.7%** automatically receive three fresh seeds
`100,101,102`. The width comparison requires a completed three-seed mean
**>=75.5%** and minimum **>=75%**. This is provisional robustness evidence,
not official qualification. If multiple recipes pass, the lowest observed timed
runtime chooses the comparison candidate; hardware remains recorded and mixed
PCIe/SXM timings do not establish an official winner.

Only then does the runner compare the exact chosen recipe on standard ResNet11
and final-width-384 ResNet11. Both use seeds `200,201,202` and run sequentially
on the **same GPU allocation**. The only parameter difference is
`stage_widths=[64,128,256,384]`. No further architecture changes are searched.
If validation does not pass the gate, this comparison is explicitly skipped.

All harness measurements, logs, commands and frozen submission sources persist
in `cifar100-experiments/rapid-<timestamp>-<id>/`. Consolidated CSV/JSON rows
contain parameters, seeds, accuracy mean/range/SD, separate prepare/train/total
times, hardware and completion/failure status. `promotion.json`, `selection.json`,
`width_comparison.json` (when executed), and `completion.json` record every gate.
The CPU coordinator continues after independent job failures and preserves results
if the local terminal disconnects. Finished summaries are downloaded automatically
to `results/sweeps/<run-id>/`; raw artifacts can be retrieved as described below.
Custom recipe lists use `--params-file`; architecture, batch, momentum and epoch
changes are rejected by this runner.

Synthetic CPU checks precede all GPU jobs: native scheduler equivalence, all twelve
recipe paths, finite/read-only inference, immutable inputs and full fresh-trial
reset. These checks generate no CIFAR accuracy or GPU performance measurements.

## Parallel architecture study (previous; frozen)

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
modal run --detach modal_architecture_sweep.py
```

This launches a remote CPU coordinator and up to eight independent A100-80GB
containers. Each container runs one benchmark at a time on its own GPU with four
CPUs. Both PCIe and SXM4 hosts are accepted and their actual device names are
recorded. Compare runtimes on the same GPU variant; SXM results cannot establish
official PCIe speed. Provider quotas may reduce actual parallelism.

The architecture study used **30 epochs**. Its SGD/Nesterov optimizer,
batch size 512, LR 0.2, five-epoch warmup, cosine schedule, smoothing 0.05,
cutout 8, decay 0.0005, momentum 0.9, BF16, layout, and augmentation are unchanged.
The earlier recipe sweep still explicitly requests 60 epochs.

Model parameters accepted through `context.parameters`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `architecture` | `resnet11` | `resnet9`, `resnet11`, or `resnet15` selects residual depth |
| `width` | `64` | Derives widths `[width, 2*width, 4*width, 8*width]` |
| `stage_widths` | `[64,128,256,512]` | Independent stem and three stage widths; overrides derived widths |
| `residual_blocks` | `[1,1,1]` | Two-convolution skip blocks in stages 1-3; overrides preset depth |
| `global_pool` | `max` | Global `max` or `avg` pooling before the linear head |
| `downsampling` | `maxpool` | `maxpool`, `avgpool`, or a stride-2 stage-transition convolution |

`resnet11` uses depths `[1,1,1]`; `resnet15` uses `[2,1,2]`. The names count
convolutions plus the classifier. Residual blocks retain the baseline's
Conv-BN-ReLU ordering and unscaled identity addition. Added depth increases
capacity while preserving the existing CIFAR stem and three spatial reductions.
Narrower final stages probe the cost of late residual convolutions. Pooling pairs
isolate aggregation, and the two downsampling probes isolate spatial reduction.

| ID | Architecture | Widths | Global pool | Downsampling | Comparison |
| --- | --- | --- | --- | --- | --- |
| 001 | ResNet9 | 64/128/256/512 | max | maxpool | Unchanged model control |
| 002 | ResNet11 | 64/128/256/512 | max | maxpool | Add a middle-stage residual block |
| 003 | ResNet15 | 64/128/256/512 | max | maxpool | Add more early/late depth versus 002 |
| 004 | ResNet15 | 64/128/256/512 | avg | maxpool | Global-pooling pair with 003 |
| 005 | ResNet11 | 64/128/256/384 | max | maxpool | Narrow the final stage versus 002 |
| 006 | ResNet11 | 64/128/256/384 | avg | maxpool | Global-pooling pair with 005 |
| 007 | ResNet11 | 64/128/256/512 | max | stride | Strided convolution versus 002 |
| 008 | ResNet11 | 64/128/256/512 | max | avgpool | Average downsampling versus 002 |

Synthetic CPU checks run first in the pinned image: numerical control equivalence,
training/remainder batches, finite read-only inference, immutable inputs, and full
same-seed resets for all eight models. These checks do not measure CIFAR accuracy.

Screening uses seed `0`, `--n 1 --no-accuracy-target`. Every complete, successful
architecture scoring **at least 74.5%** is automatically run with `--n 3` on fresh
seeds `100,101,102`; the screening seed is excluded from validation statistics.
Failures are recorded and do not prevent other jobs or promotions. No recipe
setting changes during promotion, and three seeds do not establish a final winner.
If none passes 74.5%, the runner records that outcome without launching validation.

Full per-model harness results, source snapshots, commands and logs persist in
`cifar100-experiments/architectures-<timestamp>-<id>/screen/arch-NNN/` and
`validation/arch-NNN/`. Consolidated `summary.json` and `summary.csv` have one row
per architecture per stage, including every seed's accuracy, separate prepare/train
times, their total, completion, failures, hardware, and validation mean/range/SD.
`promotion.json` records the promoted models; `completion.json` records workflow
and benchmark completion separately. Worker directories are disjoint; the remote
coordinator commits consolidated summaries after each collected result.
Local summaries appear in `results/sweeps/<run-id>/` when the client completes.

The table ranks completed means >=75.5% by timed runtime first; 75-75.5% is
provisional and below 75% is non-qualifying. Promotion at 74.5% is a screening
decision, not qualification. Screen and validation tables are printed separately.
For custom architectures, pass `--params-file architectures.json`, a JSON list
of parameter dictionaries. This runner rejects training-recipe changes and keeps
30 epochs; the general recipe runner below remains available for future tuning.
Seed overrides are `--screen-seed` and `--validation-seed` (must be disjoint).

## First screening sweep

Run from the repository root using your authenticated Modal CLI:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
modal run modal_sweep.py --dry-run
modal run --detach modal_sweep.py
```

The UTF-8 settings avoid Windows console errors from Modal's Unicode status output.

Before any benchmark starts, the sweep checks that the provider supplies exactly
one A100 80GB GPU, accepting both PCIe and SXM4 variants for development screening.
Other GPU types, 40GB A100s, and multiple GPUs are rejected. The actual GPU name is
recorded in the summaries; the runner does not retry or choose hosts based on speed.
SXM timings are development-only and cannot establish PCIe performance. Competition
hardware checks remain unchanged and still require the PCIe variant for judging.

The default sweep uses one A100-80GB container, four CPUs, sequential benchmark
processes, three shared seeds (`0, 1, 2`), and `--no-accuracy-target`. It makes no
changes to benchmark settings, the harness, evaluation, or the submission defaults.
Only SGD momentum was added to the recipe's exposed parameters; it defaults to 0.9.
Momentum 0 disables Nesterov because PyTorch requires positive momentum for it.

All experiments hold `epochs=60` and `width=64`. The control uses batch size 512,
warmup 5, cutout 8, smoothing 0.05, LR 0.2, decay 0.0005, and momentum 0.9.

| ID | Change from control | Information sought |
| --- | --- | --- |
| 001 | None | Reproduce baseline across shared seeds |
| 002 | LR 0.1 | Is current LR too high? |
| 003 | LR 0.3 | Would stronger updates help convergence? |
| 004 | Warmup 0 | Does warmup help? |
| 005 | Warmup 2 | Does a shorter ramp improve useful learning? |
| 006 | Smoothing 0 | Does smoothing suppress useful confidence? |
| 007 | Smoothing 0.1 | Does stronger regularization help? |
| 008 | Cutout 0 | Does occlusion make short training harder? |
| 009 | Cutout 12 | Does stronger occlusion improve generalization? |
| 010 | Decay 0.00025 | Is weight regularization too strong? |
| 011 | Decay 0.001 | Would more weight regularization help? |
| 012 | Momentum 0.85 | Do less persistent updates help? |
| 013 | Momentum 0.95 | Do more persistent updates help? |
| 014 | Batch 256, explicit LR 0.2 | More updates: convergence versus runtime |
| 015 | Batch 1024, explicit LR 0.2 | Fewer updates: speed versus convergence |

The batch probes keep LR explicit to isolate batch size from automatic LR scaling.
The smaller batch is a diagnostic and may be too slow at 60 epochs. These probes
identify main effects; do not assume combining improvements preserves their benefit.
If needed, test a small follow-up combination on the same seeds before reducing epochs.

## Results and ranking

Full artifacts persist in Modal Volume `cifar100-experiments/<run-id>/`. The run ID
is printed when the GPU function starts. Every configuration retains its parameters,
command, stdout/stderr log, harness source snapshot, config, trial records, summary,
and any failure trace. The sweep updates consolidated `summary.json` and `summary.csv`
and explicitly commits the volume after each configuration, including failures.
If the local client stays connected, consolidated summaries are also saved under
`results/sweeps/<run-id>/` after completion. Retrieve all artifacts with:

```powershell
modal volume get cifar100-experiments /RUN_ID results/sweeps/RUN_ID-full
```

On Windows, create the local destination directory before recursively downloading
to avoid a directory-creation race in Modal CLI 1.6.0:

```powershell
New-Item -ItemType Directory -Force -Path results/sweeps/RUN_ID-full | Out-Null
modal volume get cifar100-experiments /RUN_ID results/sweeps/RUN_ID-full
```

Both summaries record per-seed accuracies/statuses, mean accuracy, minimum, maximum,
sample standard deviation, separate mean preparation/training/total times, and
completion. Accuracies are fractions and times are seconds. The table shows percent
accuracy and percentage-point standard deviation. Partial means are explicitly
diagnostic and never qualify an incomplete configuration.

Ranking tiers: complete mean >=75.5% (`safe-screen`, lowest total time first),
75–75.5% (`marginal`), below 75% (`non-qualifying`), then incomplete runs. These labels
are provisional development screening, not official qualification or a final winner.
The table reports runtime change relative to the user-reported 81.4 seconds.

## Later phases

Pass a UTF-8 JSON list of parameter dictionaries with `--params-file`. Missing keys
use current recipe defaults; omitted LR follows the recipe's batch scaling. For
controlled comparisons, copy the full selected parameter dictionary into each entry
and explicitly hold LR fixed. The runner does not automatically change the recipe.

```powershell
modal run --detach modal_sweep.py --params-file experiments.json --n 3 --seed 0
modal run --detach modal_sweep.py --params-file finalists.json --n 10 --seed 100
```

Once a 60-epoch recipe clears the safety target without materially increasing time,
hold its other settings fixed and test 55, then 50, then 45 epochs, reviewing each
step before going shorter. Then compare batch sizes with explicit LR and the same
training length. Validate the best two or three recipes on ten fresh shared seeds
(e.g. `100..109`), reviewing mean, range, variability, and runtime. A recipe that
qualified only on the first three seeds should not be retained. Consider 20–40
seeds only after ten-seed validation; official judging still uses organizer seeds.

Profile augmentation, indexing, layout conversions, optimizer overhead, or structural
changes only after recipe tuning. Any change requires a measured comparison.

Development reporting reads only harness-produced results, never dataset labels.
All training statistics, data transfers, augmentation, and fitting remain in timed
`prepare`/`train`. No training state, dataset-derived setup, or trial seed is moved
into `build`, and inference remains single-view and read-only.
