"""Read-only replacement gate; never writes a submission or changes measurements."""

import argparse
import json
import math
from pathlib import Path

from dev.experiment_ledger import T95, Ledger, finite, summarize_trials


def evaluate(record, records=()):
    stats = summarize_trials(record.get('trials', []), record.get('requested_trials', 0))
    total = stats['mean_prepare_train_time']
    n = stats['prepare_train_time_count']
    sd = stats['prepare_train_time_std']
    upper = (total + T95[min(n-1, 30)] * sd / math.sqrt(n)
             if n > 1 and finite(total) and finite(sd) else None)
    parent = next((r for r in records
                   if r['experiment_id'] == record.get('parent_experiment')), {})
    checks = dict(
        forty_fresh_complete_trials=(stats['complete'] and stats['trial_count'] == 40
                                     and record.get('fresh_seed_validation') is True
                                     and not record.get('seed_overlap_with_ancestors')),
        supported_fresh20_parent=(parent.get('complete') is True
                                  and parent.get('trial_count', 0) >= 20
                                  and parent.get('fresh_seed_validation') is True
                                  and finite(parent.get('accuracy_lower_95_one_sided'))
                                  and parent['accuracy_lower_95_one_sided'] > .75
                                  and parent.get('parameters') == record.get('parameters')),
        official_gpu=record.get('gpu_models') == ['NVIDIA A100 80GB PCIe'],
        accuracy_mean_above_threshold=(finite(stats['mean_accuracy'])
                                      and stats['mean_accuracy'] > .75),
        accuracy_bound_has_margin=(finite(stats['accuracy_lower_95_one_sided'])
                                   and stats['accuracy_lower_95_one_sided'] > .7505),
        runtime_improves_clearly=(finite(total) and total < 7.0665159039-.05
                                 and finite(upper) and upper < 7.0665159039),
        all_timed_metrics_present=(n == 40 and stats['prepare_time_count'] == 40
                                   and stats['train_time_count'] == 40),
        frozen_defaults_tested=record.get('raw_metadata', {}).get('materialize_defaults') is True,
        ordinary_benchmark=not record.get('instrumented')
                           and record.get('record_kind') != 'hardware_gate',
    )
    return dict(experiment_id=record['experiment_id'], promotable=all(checks.values()),
                checks=checks, recomputed_statistics=stats, runtime_upper95=upper,
                policy='Fresh40 PCIe, exploratory accuracy lower95 above75.05%, all timing '
                       'fields, frozen defaults, at least50ms mean runtime margin and '
                       'exploratory runtime upper95 below7.066516s. No automatic writes. '
                       'Time bounds assume independent trials; review thermal trends separately.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_id')
    args = parser.parse_args()
    records = Ledger(Path('results/overnight/ledger.jsonl')).records()
    record = next(r for r in records if r['experiment_id'] == args.experiment_id)
    print(json.dumps(evaluate(record, records), indent=2))
