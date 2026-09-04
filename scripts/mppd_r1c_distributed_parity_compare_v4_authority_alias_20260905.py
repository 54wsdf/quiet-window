import json
import sys
import tempfile
from pathlib import Path

import scripts.mppd_r1c_distributed_parity_compare_20260905 as base_compare


def main():
    argv = list(sys.argv)
    try:
        i = argv.index('--monolithic')
    except ValueError as exc:
        raise RuntimeError('--monolithic is required') from exc
    src = Path(argv[i + 1])
    payload = json.loads(src.read_text(encoding='utf-8'))
    move = payload['iteration']['posterior_redistribution']
    if 'mean_route_total_variation' not in move:
        if 'mean_route_total_variation_among_finite_both' not in move:
            raise RuntimeError('monolithic authority contains neither route-TV field')
        move['mean_route_total_variation'] = move['mean_route_total_variation_among_finite_both']
    tmp = Path(tempfile.mkdtemp(prefix='mppd_parity_authority_')) / 'monolithic_normalized.json'
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    argv[i + 1] = str(tmp)
    sys.argv = argv
    base_compare.main()


if __name__ == '__main__':
    main()
