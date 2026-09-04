import gzip
import json
from collections import Counter

import scripts.mppd_r1c_distributed_posterior_shard_20260905 as base_worker

_orig_init_stats = base_worker.init_stats
_orig_add_mstep_stats = base_worker.add_mstep_stats
_orig_write_kernel_stats = base_worker.write_kernel_stats


def init_stats_v3():
    stats = _orig_init_stats()
    stats['transfer_movement_raw_weight'] = Counter()
    return stats


def add_mstep_stats_v3(stats, factor, weight):
    w = float(weight)
    if w > 0 and factor.get('type') == 'INTERVAL' and factor.get('kind') == 'TRANSFER':
        movement = factor.get('movement') or 'UNKNOWN'
        stats['transfer_movement_raw_weight'][movement] += w
    _orig_add_mstep_stats(stats, factor, weight)


def write_kernel_stats_v3(path, stats):
    _orig_write_kernel_stats(path, stats)
    with gzip.open(path, 'at', encoding='utf-8') as f:
        for movement, weight in sorted(stats['transfer_movement_raw_weight'].items()):
            f.write(json.dumps({
                'scope': 'TRANSFER_MOVEMENT_RAW_WEIGHT',
                'group': movement,
                'kind': 'WEIGHT',
                'weight': float(weight),
            }, ensure_ascii=False) + '\n')


def main():
    base_worker.init_stats = init_stats_v3
    base_worker.add_mstep_stats = add_mstep_stats_v3
    base_worker.write_kernel_stats = write_kernel_stats_v3
    base_worker.main()


if __name__ == '__main__':
    main()
