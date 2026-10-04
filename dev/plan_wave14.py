"""Final structural screen: retain first-stage residual while removing one convolution."""
import json
import subprocess
from pathlib import Path

from dev.experiment_ledger import Ledger


def main():
    ledger = Ledger(Path('results/overnight/ledger.jsonl'))
    for r in ledger.records():
        if r['experiment_id'].startswith('w13_') and 'control' not in r['experiment_id']:
            ledger.annotate(r['experiment_id'], promotion_status='rejected',
                            conclusion='Fresh accuracy evidence or local retune insufficient.')
    template = json.loads(Path('dev/sweeps/overnight_wave13.json').read_text())[0]
    template = {k: v for k, v in template.items() if k not in ['batch_id', 'experiments']}
    template['source_commit'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip()
    requests = []
    short = dict(stage_depths=[2, 3, 3], fast_reset=True, head_lr=3.25, muon_lr=.25)
    for label, epochs, retained in [('61', 6.1, True), ('62', 6.2, True),
                                    ('63', 6.3, True), ('control', 6.2, False)]:
        params = short | dict(epochs=epochs, stage_residuals=[retained, True, True])
        experiment = dict(id=f'w14_first_residual_{label}', params=params, n=3, seed=2400,
                          parent_experiment='w13_first62_10', stage=1,
                          fresh_seed_validation=False, hypothesis=(
                              'First-depth2 formerly disabled its skip. Retain original '
                              'stage skip with only tensor-add cost; screen accuracy recovery '
                              'under shorter budgets. Reference separately allocated; '
                              'no small cross-host timing claim.'))
        requests.append(template | dict(batch_id=f'w14-first-residual-{label}',
                                        experiments=[experiment]))
    Path('dev/sweeps/overnight_wave14.json').write_text(json.dumps(requests, indent=2)+'\n')
    ledger.write_reports(Path('results/overnight/reports'))


if __name__ == '__main__':
    main()
