# RC3 optimization workflow

RC3 is protected by tag `rc3-control-20261004`, commit
`edca55e7f0fb19eaca0c3521ac622cf9e37490d8`. Its complete configuration and source
hash are in `dev/controls/rc3.json`. Production and benchmark files stay at this
control until a candidate passes the full validation gate.

The autonomous session started 2026-10-03 23:01:58 UTC and is bounded by
2026-10-04 04:01:58 UTC. The user authorized ordinary Modal experiments without a
separate spending cap, with at most four GPU containers and a stop on Codex usage
exhaustion. Requests carry the absolute deadline; the worker refuses new work
near it and bounds running subprocesses. Launch only one wave at a time.

`dev/sweeps/overnight_wave1.json` records the first requested experiments. Run:

```text
modal run overnight_modal.py::wave --file dev/sweeps/overnight_wave1.json
```

Each batch runs configurations sequentially on one GPU. Independent batches use
separate containers. Reusing a compilation cache does not reuse model state:
each harness worker builds independently and resets weights, BN state and
optimizers for every seed. Real-data-dependent whitening remains in prepare().

Synthetic CUDA checks and the train-only component profile precede implementation
screens. The profiler uses frozen, unmodified RC3 source; instrumented timings
are diagnostic and cannot establish a submission speedup.

Import actual harness output and create immutable reports after each wave:

```text
python -m dev.experiment_ledger import --results-root results --ledger results/overnight/ledger.jsonl --report-dir results/overnight/reports
```

Annotate promotion/rejection decisions through `Ledger.annotate`; never edit
recorded metrics. Record every attempted seed, failures and missing trials.
Keep all logs, source copies, requests and metadata under the results session.

Use 2–3 seeds for screening, then 8–10, 20 and 40 fresh seeds. Each stage has its
own experiment ID and parent. Controls may share a seed list with candidates for
paired comparison. Validation seeds must be fresh relative to candidate ancestors.
Three-seed means are screening evidence only; inspect variance and worst seeds.

Final promotion requires all 40 fresh trials to complete, mean accuracy above
75% with a credible margin, and clearly better timed runtime on official-like
A100 80GB PCIe hardware. The worker's `require_pcie` gate rejects incompatible
hardware before finalist trials. Record mismatches; do not hunt for fast hosts.
Historical RC3's 7.066516 s confirmation ran on A100 SXM4; its 10-seed PCIe screen
measured 8.091955 s. Small timing claims need a matching-hardware control.

No benchmark code, evaluation logic or timing boundaries are changed. The
trusted harness alone evaluates test data. Candidate training sees only its
current training split. Keep profiler and experiment patches out of a final
submitted model.
