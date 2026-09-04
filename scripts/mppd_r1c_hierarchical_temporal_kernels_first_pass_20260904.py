import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base
import scripts.mppd_r1b_first_bidirectional_joint_update_v2_20260904 as v2
import scripts.mppd_r1b_first_bidirectional_joint_update_v4_residual_records_20260904 as v4

_ORIG_V2_ROUTE = v2.route_beam_joint_v2
_ORIG_V2_UPDATE = v2.update_kernels_v2
CONTEXT_MIN_WEIGHT = 30.0
CONTEXT_SHRINK_TAU = 75.0
HOUR_MIN_WEIGHT = 40.0
HOUR_SHRINK_TAU = 100.0


def direction_key(meta, origin, destination):
    so = meta.get(origin, {}).get('seq')
    sd = meta.get(destination, {}).get('seq')
    if so is None or sd is None or float(so) == float(sd):
        return 'UNK'
    return 'INC' if float(sd) > float(so) else 'DEC'


def route_contexts(cand, meta):
    legs = [x for x in base.r0.path_legs(cand.get('path') or [], meta) if x[1] != x[2]]
    if not legs:
        return None, None, None, None
    fl, fo, fd = legs[0]
    ll, lo, ld = legs[-1]
    ad = direction_key(meta, fo, fd)
    ed = direction_key(meta, lo, ld)
    return f'{fl}:{fo}:{ad}', f'{ll}:{ld}:{ed}', fl, ll


def choose_context_kernel(kernels, kind, context, hour):
    if kind == 'ACCESS':
        if context is not None:
            hk = kernels.get('access_by_context_hour', {}).get(f'{context}|h{hour:02d}')
            if hk is not None:
                return hk
            ck = kernels.get('access_by_context', {}).get(context)
            if ck is not None:
                return ck
        return kernels['access']
    if context is not None:
        hk = kernels.get('egress_by_context_hour', {}).get(f'{context}|h{hour:02d}')
        if hk is not None:
            return hk
        ck = kernels.get('egress_by_context', {}).get(context)
        if ck is not None:
            return ck
    return kernels['egress']


def annotate_chain_context(ch, access_context, egress_context, tin, tout):
    interval_idx = 0
    for f in ch.get('factors', []):
        if f.get('type') == 'INTERVAL':
            if f.get('kind') == 'ACCESS':
                f['station_context'] = access_context
                f['time_bin_hour'] = int(tin.hour)
            else:
                f['time_bin_hour'] = int(f.get('ready').hour if f.get('ready') is not None else tin.hour)
            interval_idx += 1
        elif f.get('type') == 'EGRESS':
            f['station_context'] = egress_context
            f['time_bin_hour'] = int(tout.hour)


def route_beam_joint_r1c(cand, meta, rides_fn, tin, tout, beam, kernels, max_skip):
    access_context, egress_context, first_line, last_line = route_contexts(cand, meta)
    local = dict(kernels)
    local['access'] = choose_context_kernel(kernels, 'ACCESS', access_context, tin.hour)
    local['egress'] = choose_context_kernel(kernels, 'EGRESS', egress_context, tout.hour)
    chains, failure = _ORIG_V2_ROUTE(cand, meta, rides_fn, tin, tout, beam, local, max_skip)
    for ch in chains:
        annotate_chain_context(ch, access_context, egress_context, tin, tout)
    return chains, failure


def adaptive_blend(parent, raw, effective_weight, tau, evidence_class):
    alpha = max(0.0, min(1.0, float(effective_weight) / (float(effective_weight) + float(tau))))
    return v2.blend_kernel(parent, raw, alpha, evidence_class), alpha


def fit_interval_groups(factors, parent, key_fn, min_weight, tau, evidence_class):
    groups = defaultdict(list)
    for f in factors:
        k = key_fn(f)
        if k is not None:
            groups[k].append(f)
    kernels = {}
    diags = {}
    for key, rows in groups.items():
        raw, diag = base.weighted_interval_fit(rows, parent, min_weight=min_weight)
        ew = float(diag.get('effective_weight', 0.0))
        if not diag.get('fitted') or ew < min_weight:
            continue
        k, alpha = adaptive_blend(parent, raw, ew, tau, evidence_class)
        kernels[str(key)] = k
        diags[str(key)] = {**diag, 'shrinkage_alpha': alpha, 'parent': 'HIERARCHICAL_PARENT'}
    return kernels, diags


def fit_egress_groups(factors, parent, key_fn, min_weight, tau, evidence_class):
    groups = defaultdict(list)
    for f in factors:
        if f.get('station_only_proxy') or f.get('fit_eligible') is False:
            continue
        k = key_fn(f)
        if k is not None:
            groups[k].append(f)
    kernels = {}
    diags = {}
    for key, rows in groups.items():
        raw, diag = base.weighted_egress_fit(rows, parent, min_weight=min_weight)
        ew = float(diag.get('effective_weight', 0.0))
        if not diag.get('fitted') or ew < min_weight:
            continue
        k, alpha = adaptive_blend(parent, raw, ew, tau, evidence_class)
        kernels[str(key)] = k
        diags[str(key)] = {**diag, 'shrinkage_alpha': alpha, 'parent': 'HIERARCHICAL_PARENT'}
    return kernels, diags


def update_kernels_r1c(factors, kernels):
    global_updated, diag = _ORIG_V2_UPDATE(factors, kernels)
    access_factors = [f for f in factors if f.get('type') == 'INTERVAL' and f.get('kind') == 'ACCESS']
    egress_factors = [f for f in factors if f.get('type') == 'EGRESS']

    access_ctx, access_ctx_diag = fit_interval_groups(
        access_factors,
        global_updated['access'],
        lambda f: f.get('station_context'),
        CONTEXT_MIN_WEIGHT,
        CONTEXT_SHRINK_TAU,
        'R1C_HIERARCHICAL_ACCESS_CONTEXT_K1',
    )
    egress_ctx, egress_ctx_diag = fit_egress_groups(
        egress_factors,
        global_updated['egress'],
        lambda f: f.get('station_context'),
        CONTEXT_MIN_WEIGHT,
        CONTEXT_SHRINK_TAU,
        'R1C_HIERARCHICAL_EGRESS_CONTEXT_K1',
    )

    access_hour = {}
    access_hour_diag = {}
    for ctx, parent in access_ctx.items():
        subset = [f for f in access_factors if f.get('station_context') == ctx]
        k, d = fit_interval_groups(
            subset,
            parent,
            lambda f: f"{ctx}|h{int(f.get('time_bin_hour', -1)):02d}" if f.get('time_bin_hour') is not None else None,
            HOUR_MIN_WEIGHT,
            HOUR_SHRINK_TAU,
            'R1C_HIERARCHICAL_ACCESS_CONTEXT_HOUR_K1',
        )
        access_hour.update(k); access_hour_diag.update(d)

    egress_hour = {}
    egress_hour_diag = {}
    for ctx, parent in egress_ctx.items():
        subset = [f for f in egress_factors if f.get('station_context') == ctx]
        k, d = fit_egress_groups(
            subset,
            parent,
            lambda f: f"{ctx}|h{int(f.get('time_bin_hour', -1)):02d}" if f.get('time_bin_hour') is not None else None,
            HOUR_MIN_WEIGHT,
            HOUR_SHRINK_TAU,
            'R1C_HIERARCHICAL_EGRESS_CONTEXT_HOUR_K1',
        )
        egress_hour.update(k); egress_hour_diag.update(d)

    updated = dict(global_updated)
    updated['schema'] = 'mppd.r1c-hierarchical-temporal-kernels.v1-k1-first-pass-mixture-capable'
    updated['access_by_context'] = access_ctx
    updated['access_by_context_hour'] = access_hour
    updated['egress_by_context'] = egress_ctx
    updated['egress_by_context_hour'] = egress_hour
    updated['hierarchy'] = {
        'access': 'GLOBAL_TO_LINE_STATION_DIRECTION_TO_HOUR',
        'egress': 'GLOBAL_TO_LINE_STATION_DIRECTION_TO_HOUR',
        'transfer': 'GLOBAL_TO_MOVEMENT',
        'day_specific': 'NOT_IDENTIFIABLE_IN_SINGLE_SEOUL_DAY_FIRST_PASS',
    }
    updated['mixture_policy'] = {
        'access': 'K1_FIRST_PASS_SCHEMA_MIXTURE_CAPABLE',
        'egress': 'K1_FIRST_PASS_SCHEMA_MIXTURE_CAPABLE',
        'transfer': 'MOVEMENT_CONDITIONED_K1_FIRST_PASS_SCHEMA_MIXTURE_CAPABLE_DATA_SELECTED_K_LATER',
    }

    diag['r1c_hierarchy'] = {
        'access_context_count': len(access_ctx),
        'access_context_hour_count': len(access_hour),
        'egress_context_count': len(egress_ctx),
        'egress_context_hour_count': len(egress_hour),
        'transfer_movement_count': len(updated.get('transfer_by_movement', {})),
        'context_min_effective_weight': CONTEXT_MIN_WEIGHT,
        'context_shrink_tau': CONTEXT_SHRINK_TAU,
        'hour_min_effective_weight': HOUR_MIN_WEIGHT,
        'hour_shrink_tau': HOUR_SHRINK_TAU,
        'access_context_diagnostics': access_ctx_diag,
        'access_context_hour_diagnostics': access_hour_diag,
        'egress_context_diagnostics': egress_ctx_diag,
        'egress_context_hour_diagnostics': egress_hour_diag,
        'day_specific_status': 'DEFERRED_TO_MULTI_DAY_R2',
        'transfer_regime_mixture_status': 'K1_FIRST_PASS_ONLY_MIXTURE_CAPABLE_NOT_MIXTURE_FORCED',
    }
    diag['update_role'] = 'R1C_FIRST_FORMAL_HIERARCHICAL_K1_PASS_NOT_FINAL_MULTI_DAY_KERNEL'
    return updated, diag


def find_out_dir(argv):
    for i, arg in enumerate(argv):
        if arg == '--out' and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith('--out='):
            return Path(arg.split('=', 1)[1])
    return None


def main():
    v2.route_beam_joint_v2 = route_beam_joint_r1c
    v2.update_kernels_v2 = update_kernels_r1c
    v4.main()

    out = find_out_dir(sys.argv[1:])
    if out is None:
        return
    summary_path = out / 'r1b_first_bidirectional_joint_update_smoke_summary.json'
    payload = json.loads(summary_path.read_text(encoding='utf-8'))
    payload['schema'] = 'mppd.r1c-hierarchical-temporal-kernels-first-pass.v1'
    payload['status'] = 'R1C_FIRST_FORMAL_HIERARCHICAL_TEMPORAL_KERNEL_PASS_COMPLETED'
    payload['r1c_design'] = {
        'access': 'GLOBAL_TO_LINE_STATION_DIRECTION_TO_HOUR_K1_FIRST_PASS',
        'egress': 'GLOBAL_TO_LINE_STATION_DIRECTION_TO_HOUR_K1_FIRST_PASS',
        'transfer': 'GLOBAL_TO_MOVEMENT_K1_FIRST_PASS',
        'mixture_capable': True,
        'mixture_forced': False,
        'day_specific': 'DEFERRED_SINGLE_DAY_NOT_IDENTIFIABLE',
        'service_support_mutation': False,
        'unresolved_missing_service_residual_retained': True,
    }
    payload.setdefault('scientific_boundary', []).extend([
        'This R1C first pass does not mutate service support; G1F remains the service substrate and unresolved missing-service residuals remain explicit.',
        'Access and egress use K=1 in this first pass but the serialized kernel interface remains mixture-capable; K=1 is a computational simplification, not a permanent scientific claim.',
        'Transfer kernels remain movement-conditioned and mixture-capable; transfer-regime component-count selection is deferred until held-out likelihood and posterior-predictive qualification.',
        'Station/direction and hour effects are empirical-Bayes shrinkage layers over shared global kernels, not independent unconstrained per-station fits.',
        'Day-specific deviations require multi-day data and are not identifiable from the current Seoul single-day 07:00-10:00 slice.',
    ])
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
