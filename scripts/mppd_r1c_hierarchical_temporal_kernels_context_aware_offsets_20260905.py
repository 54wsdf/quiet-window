import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base
import scripts.mppd_r1c_hierarchical_temporal_kernels_first_pass_20260904 as r1c


def factor_kernel_r1c(factor, kernels):
    if factor.get('type') == 'EGRESS':
        ctx = factor.get('station_context')
        hour = factor.get('time_bin_hour')
        if hour is None and factor.get('exit_time') is not None:
            hour = int(factor['exit_time'].hour)
        if hour is None:
            return kernels['egress']
        return r1c.choose_context_kernel(kernels, 'EGRESS', ctx, int(hour))
    kind = factor.get('kind')
    if kind == 'ACCESS':
        ctx = factor.get('station_context')
        hour = factor.get('time_bin_hour')
        if hour is None and factor.get('ready') is not None:
            hour = int(factor['ready'].hour)
        if hour is None:
            return kernels['access']
        return r1c.choose_context_kernel(kernels, 'ACCESS', ctx, int(hour))
    return base.kernel_for(kind, factor.get('movement'), kernels)


def factor_loglik_with_root_shift_r1c(factor, target_root, delta_sec, kernels):
    if factor['type'] == 'EGRESS':
        arr = base.shift_dt(factor['arr'], delta_sec if factor.get('arr_root') == target_root else 0.0)
        kern = factor_kernel_r1c(factor, kernels)
        return base.egress_logdensity_kernel(factor['exit_time'], arr, factor['arr_sd'], kern)

    ready = base.shift_dt(factor['ready'], delta_sec if factor.get('ready_root') == target_root else 0.0)
    lower = base.shift_dt(factor['lower'], delta_sec if factor.get('lower_root') == target_root else 0.0)
    upper = base.shift_dt(factor['upper'], delta_sec if factor.get('upper_root') == target_root else 0.0)
    kern = factor_kernel_r1c(factor, kernels)
    return base.interval_logprob_kernel(
        lower,
        factor['lower_sd'],
        upper,
        factor['upper_sd'],
        ready,
        factor['ready_sd'],
        kern,
    )


def update_service_offsets_r1c(roots, factors, kernels, root_usage, grid, min_usage):
    rmeta = base.root_metadata(roots)
    by_root_factors = defaultdict(list)
    for f in factors:
        touched = set()
        if f['type'] == 'EGRESS':
            if f.get('arr_root'):
                touched.add(f['arr_root'])
        else:
            for key in ('ready_root', 'lower_root', 'upper_root'):
                if f.get(key):
                    touched.add(f[key])
        for root in touched:
            by_root_factors[root].append(f)

    offsets = {}
    diagnostics = {}
    for root, usage in root_usage.items():
        md = rmeta.get(root)
        if not md or usage < min_usage:
            continue
        if md['evidence_class'] == 'PARTIAL_DIRECT_SERVICE_ANCHOR':
            offsets[root] = 0
            diagnostics[root] = {
                'usage_mass': usage,
                'evidence_class': md['evidence_class'],
                'selected_offset_sec': 0,
                'frozen': True,
                'likelihood_kernel_policy': 'R1C_CONTEXT_AWARE_AE_MOVEMENT_AWARE_K',
            }
            continue
        rows = by_root_factors.get(root, [])
        if not rows:
            continue
        total_w = sum(float(f.get('weight', 0.0)) for f in rows)
        if total_w <= 0:
            continue
        prior_sd = max(30.0, md['median_event_sd'], 90.0 if 'WEAK' in md['evidence_class'] else 60.0)
        scored = []
        for delta in grid:
            ll = 0.0
            used_w = 0.0
            for f in rows:
                w = float(f.get('weight', 0.0))
                lp = factor_loglik_with_root_shift_r1c(f, root, delta, kernels)
                if math.isfinite(lp) and w > 0:
                    ll += w * lp
                    used_w += w
            if used_w <= 0:
                continue
            mean_ll = ll / used_w
            prior = -0.5 * (float(delta) / prior_sd) ** 2
            scored.append((mean_ll + prior, delta, mean_ll, prior, used_w))
        if not scored:
            continue
        scored.sort(reverse=True)
        best = scored[0]
        zero = next((x for x in scored if x[1] == 0), None)
        gain = best[0] - zero[0] if zero else None
        selected = int(best[1])
        if gain is not None and gain < 1e-4:
            selected = 0
        offsets[root] = selected
        diagnostics[root] = {
            'usage_mass': usage,
            'factor_weight': best[4],
            'evidence_class': md['evidence_class'],
            'prior_sd_sec': prior_sd,
            'selected_offset_sec': selected,
            'score_gain_vs_zero': gain,
            'best_mean_loglik': best[2],
            'best_prior_log': best[3],
            'frozen': False,
            'likelihood_kernel_policy': 'R1C_CONTEXT_AWARE_AE_MOVEMENT_AWARE_K',
        }
    return offsets, diagnostics


def main():
    base.update_service_offsets = update_service_offsets_r1c
    r1c.main()
    out = r1c.find_out_dir(sys.argv[1:])
    if out is None:
        return
    path = Path(out) / 'r1b_first_bidirectional_joint_update_smoke_summary.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['schema'] = 'mppd.r1c-hierarchical-temporal-kernels-first-pass.v2-context-aware-service-offsets'
    payload['status'] = 'R1C_CONTEXT_AWARE_SERVICE_OFFSET_CONSISTENCY_PASS_COMPLETED'
    payload['implementation_patch'] = {
        'wrapper': 'scripts/mppd_r1c_hierarchical_temporal_kernels_context_aware_offsets_20260905.py',
        'service_offset_mstep_access_kernel': 'STATION_DIRECTION_HOUR_CONTEXT_IF_AVAILABLE_ELSE_GLOBAL',
        'service_offset_mstep_egress_kernel': 'STATION_DIRECTION_HOUR_CONTEXT_IF_AVAILABLE_ELSE_GLOBAL',
        'service_offset_mstep_transfer_kernel': 'MOVEMENT_SPECIFIC_IF_AVAILABLE_ELSE_GLOBAL',
        'service_support_mutation': False,
        'parameter_grid_unchanged': True,
        'hierarchy_hyperparameters_unchanged': True,
    }
    payload.setdefault('scientific_boundary', []).extend([
        'This patch changes no kernel hyperparameter, service-support state, route candidate set, beam width, skip support, or sampling rule; it only makes the service-offset M-step evaluate the same hierarchical A/E and movement-specific K likelihood used by the passenger E-step.',
        'The exact paired 1-of-50 rerun is a one-time consistency qualification before full-denominator scaling and must not become a new tuning loop.',
    ])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
