import argparse
import gzip
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_cohorts(path):
    out = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r['cohort_id']] = r
    return out


def component(kernel):
    return kernel['components'][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--monolithic', required=True)
    ap.add_argument('--kernels', required=True)
    ap.add_argument('--offsets', required=True)
    ap.add_argument('--e0-summary', required=True)
    ap.add_argument('--e1-summary', required=True)
    ap.add_argument('--e0-cohorts', required=True)
    ap.add_argument('--e1-cohorts', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    mono = load(args.monolithic)
    kernels = load(args.kernels).get('kernels', load(args.kernels))
    offsets = load(args.offsets)
    e0 = load(args.e0_summary)
    e1 = load(args.e1_summary)
    m0 = mono['iteration']['E0_passenger_posterior']
    m1 = mono['iteration']['E1_passenger_posterior']
    mk = mono['iteration']['M_kernel_update']['kernels_after']
    mo = mono['iteration']['M_service_timing_update']

    global_diff = {}
    for key in ('access', 'egress', 'transfer_global'):
        a = component(kernels[key]); b = component(mk[key])
        global_diff[key] = {
            'median_sec_distributed': a['median_sec'],
            'median_sec_monolithic': b['median_sec'],
            'median_abs_diff': abs(a['median_sec'] - b['median_sec']),
            'sigma_distributed': a['sigma'],
            'sigma_monolithic': b['sigma'],
            'sigma_abs_diff': abs(a['sigma'] - b['sigma']),
        }

    dist_mov = kernels.get('transfer_by_movement', {})
    mono_mov = mk.get('transfer_by_movement', {})
    movement_keys_match = set(dist_mov) == set(mono_mov)
    movement_diffs = []
    for key in sorted(set(dist_mov) | set(mono_mov)):
        if key not in dist_mov or key not in mono_mov:
            movement_diffs.append({'movement': key, 'missing_side': 'distributed' if key not in dist_mov else 'monolithic'})
            continue
        a = component(dist_mov[key]); b = component(mono_mov[key])
        movement_diffs.append({
            'movement': key,
            'median_abs_diff': abs(a['median_sec'] - b['median_sec']),
            'sigma_abs_diff': abs(a['sigma'] - b['sigma']),
            'median_sec_distributed': a['median_sec'],
            'median_sec_monolithic': b['median_sec'],
            'sigma_distributed': a['sigma'],
            'sigma_monolithic': b['sigma'],
        })
    comparable = [x for x in movement_diffs if 'median_abs_diff' in x]
    movement_max = {
        'median_abs_diff': max((x['median_abs_diff'] for x in comparable), default=0.0),
        'sigma_abs_diff': max((x['sigma_abs_diff'] for x in comparable), default=0.0),
        'worst_median': max(comparable, key=lambda x: x['median_abs_diff']) if comparable else None,
        'worst_sigma': max(comparable, key=lambda x: x['sigma_abs_diff']) if comparable else None,
    }

    mono_offsets = {k: int(v) for k, v in mo.get('nonzero_offsets', {}).items()}
    dist_offsets = {k: int(v) for k, v in offsets.get('nonzero_offsets', {}).items()}
    offset_disagreement = {
        k: {'monolithic': mono_offsets.get(k, 0), 'distributed': dist_offsets.get(k, 0)}
        for k in sorted(set(mono_offsets) | set(dist_offsets))
        if mono_offsets.get(k, 0) != dist_offsets.get(k, 0)
    }

    a = load_cohorts(args.e0_cohorts)
    b = load_cohorts(args.e1_cohorts)
    if set(a) != set(b):
        raise RuntimeError(f'E0/E1 cohort key mismatch: {len(a)} vs {len(b)}')
    finite_both = route_changed = first_changed = tv_num = became = lost = 0.0
    for cid, x in a.items():
        y = b[cid]
        mass = float(x['mass'])
        xf = x.get('status') == 'FINITE_POSTERIOR'
        yf = y.get('status') == 'FINITE_POSTERIOR'
        if not xf and yf: became += mass
        if xf and not yf: lost += mass
        if not (xf and yf): continue
        finite_both += mass
        route_changed += mass * (x.get('top_route') != y.get('top_route'))
        first_changed += mass * (x.get('top_first_root') != y.get('top_first_root'))
        keys = set(x.get('route_probs', {})) | set(y.get('route_probs', {}))
        tv = 0.5 * sum(abs(float(x.get('route_probs', {}).get(k, 0.0)) - float(y.get('route_probs', {}).get(k, 0.0))) for k in keys)
        tv_num += mass * tv
    dist_move = {
        'finite_in_both_mass': finite_both,
        'became_finite_mass': became,
        'lost_finite_mass': lost,
        'top_route_changed_mass': route_changed,
        'top_first_service_changed_mass': first_changed,
        'mean_route_total_variation': tv_num / finite_both if finite_both else None,
    }
    mono_move = mono['iteration']['posterior_redistribution']

    posterior_diff = {
        'E0_finite_mass_abs': abs(float(e0['posterior']['finite_posterior_mass']) - float(m0['finite_posterior_mass'])),
        'E1_finite_mass_abs': abs(float(e1['posterior']['finite_posterior_mass']) - float(m1['finite_posterior_mass'])),
        'E1_route_entropy_abs': abs(float(e1['posterior']['weighted_mean_route_entropy_nats']) - float(m1['weighted_mean_route_entropy_nats'])),
        'E1_first_boarding_contraction_abs': abs(float(e1['posterior']['weighted_mean_first_boarding_route_entropy_contraction_nats']) - float(m1['weighted_mean_first_boarding_route_entropy_contraction_nats'])),
        'route_changed_mass_abs': abs(dist_move['top_route_changed_mass'] - float(mono_move['top_route_changed_mass'])),
        'first_service_changed_mass_abs': abs(dist_move['top_first_service_changed_mass'] - float(mono_move['top_first_service_changed_mass'])),
        'route_tv_abs': abs(dist_move['mean_route_total_variation'] - float(mono_move['mean_route_total_variation'])),
    }

    gates = {
        'E0_exact_finite_and_failure': (
            e0['posterior']['finite_posterior_mass'] == m0['finite_posterior_mass']
            and e0['posterior']['failure_mass'] == m0['failure_mass']
        ),
        'E1_exact_finite_and_failure': (
            e1['posterior']['finite_posterior_mass'] == m1['finite_posterior_mass']
            and e1['posterior']['failure_mass'] == m1['failure_mass']
        ),
        'movement_key_set_exact': movement_keys_match,
        'global_kernel_median_max_lt_1ms': max(x['median_abs_diff'] for x in global_diff.values()) < 0.001,
        'global_kernel_sigma_max_lt_1e-5': max(x['sigma_abs_diff'] for x in global_diff.values()) < 1e-5,
        'movement_kernel_median_max_lt_5ms': movement_max['median_abs_diff'] < 0.005,
        'movement_kernel_sigma_max_lt_5e-5': movement_max['sigma_abs_diff'] < 5e-5,
        'offset_decision_disagreement_le_2': len(offset_disagreement) <= 2,
        'E1_entropy_diff_lt_5e-5': posterior_diff['E1_route_entropy_abs'] < 5e-5,
        'route_changed_mass_diff_le_2': posterior_diff['route_changed_mass_abs'] <= 2,
        'first_service_changed_mass_diff_le_5': posterior_diff['first_service_changed_mass_abs'] <= 5,
        'route_tv_diff_lt_5e-5': posterior_diff['route_tv_abs'] < 5e-5,
    }
    status = 'R1C_DISTRIBUTED_PARITY_HIGH_PRECISION_PASS' if all(gates.values()) else 'R1C_DISTRIBUTED_PARITY_HIGH_PRECISION_FAIL'
    result = {
        'schema': 'mppd.r1c-distributed-parity-high-precision.v3',
        'date': '2026-09-05',
        'status': status,
        'scope': {'cohort_count': len(a), 'passenger_mass': sum(float(x['mass']) for x in a.values())},
        'global_kernel_diff': global_diff,
        'movement_kernel_count': {'distributed': len(dist_mov), 'monolithic': len(mono_mov)},
        'movement_kernel_max_diff': movement_max,
        'top_movement_kernel_diffs': sorted(comparable, key=lambda x: max(x['median_abs_diff'], 100*x['sigma_abs_diff']), reverse=True)[:20],
        'offsets': {
            'monolithic_nonzero_count': len(mono_offsets),
            'distributed_nonzero_count': len(dist_offsets),
            'disagreement_count': len(offset_disagreement),
            'disagreement': offset_disagreement,
        },
        'distributed_posterior_movement': dist_move,
        'monolithic_posterior_movement': mono_move,
        'posterior_diff': posterior_diff,
        'gates': gates,
        'scientific_boundary': [
            'This parity test uses the exact deterministic modulo-50 cohort used by the monolithic context-aware authority.',
            'The comparator reports parameter differences before judging them; no gate is relaxed based on a failed run.',
            'High-precision sufficient statistics preserve microsecond-level relative-time geometry while still aggregating identical posterior likelihood signatures.',
        ],
        'no_email_notification_logic': True,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'parity_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if status.endswith('_FAIL'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
