import json
import sys
from pathlib import Path

import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base
import scripts.mppd_r1b_first_bidirectional_joint_update_v2_20260904 as v2
import scripts.mppd_r1b_first_bidirectional_joint_update_v4_residual_records_20260904 as v4
import scripts.mppd_r1c_hierarchical_temporal_kernels_first_pass_20260904 as r1c
import scripts.mppd_r1c_hierarchical_temporal_kernels_context_aware_offsets_20260905 as ctx


def update_kernels_proxy_filtered(factors, kernels):
    excluded = [
        f for f in factors
        if f.get('type') == 'EGRESS'
        and (f.get('station_only_proxy') or f.get('fit_eligible') is False)
    ]
    fit_factors = [
        f for f in factors
        if not (
            f.get('type') == 'EGRESS'
            and (f.get('station_only_proxy') or f.get('fit_eligible') is False)
        )
    ]
    updated, diag = r1c.update_kernels_r1c(fit_factors, kernels)
    diag['station_only_egress_proxy_filter'] = {
        'excluded_factor_count': len(excluded),
        'excluded_posterior_weight': sum(float(f.get('weight', 0.0)) for f in excluded),
        'policy': 'EXCLUDE_STATION_ONLY_OR_FIT_INELIGIBLE_EGRESS_FROM_ALL_THETA_E_MSTEPS',
        'passenger_likelihood_unchanged': True,
    }
    diag['update_role'] = 'R1C_PROXY_FILTERED_CONTEXT_AWARE_K1_QUALIFICATION'
    return updated, diag


def find_out_dir(argv):
    return r1c.find_out_dir(argv)


def main():
    # Keep station-only routes in the passenger E-step likelihood. Filter them only
    # from the kernel M-step, exactly as the intended scientific boundary states.
    v2.route_beam_joint_v2 = r1c.route_beam_joint_r1c
    v2.update_kernels_v2 = update_kernels_proxy_filtered
    base.update_service_offsets = ctx.update_service_offsets_r1c
    v4.main()

    out = find_out_dir(sys.argv[1:])
    if out is None:
        return
    summary_path = Path(out) / 'r1b_first_bidirectional_joint_update_smoke_summary.json'
    payload = json.loads(summary_path.read_text(encoding='utf-8'))
    payload['schema'] = 'mppd.r1c-hierarchical-temporal-kernels.v3-proxy-filtered-context-aware-offsets'
    payload['status'] = 'R1C_PROXY_FILTERED_CONTEXT_AWARE_CONSISTENCY_QUALIFICATION_COMPLETED'
    payload['r1c_design'] = {
        'access': 'GLOBAL_TO_LINE_STATION_DIRECTION_TO_HOUR_K1_FIRST_PASS',
        'egress': 'GLOBAL_TO_LINE_STATION_DIRECTION_TO_HOUR_K1_FIRST_PASS_PROXY_FILTERED',
        'transfer': 'GLOBAL_TO_MOVEMENT_K1_FIRST_PASS',
        'mixture_capable': True,
        'mixture_forced': False,
        'day_specific': 'DEFERRED_SINGLE_DAY_NOT_IDENTIFIABLE',
        'service_support_mutation': False,
        'unresolved_missing_service_residual_retained': True,
        'service_offset_mstep': 'CONTEXT_AWARE_AE_MOVEMENT_AWARE_K',
        'station_only_proxy_in_passenger_likelihood': True,
        'station_only_proxy_in_egress_kernel_fit': False,
    }
    payload['implementation_patch'] = {
        'wrapper': 'scripts/mppd_r1c_proxy_filtered_context_aware_consistency_20260905.py',
        'fix': 'EXCLUDE_STATION_ONLY_PROXY_AND_FIT_INELIGIBLE_EGRESS_FACTORS_FROM_GLOBAL_AND_HIERARCHICAL_THETA_E_MSTEPS',
        'passenger_likelihood': 'UNCHANGED',
        'service_support': 'UNCHANGED_G1F',
        'kernel_hyperparameters': 'UNCHANGED',
        'offset_grid': 'UNCHANGED',
        'sample': 'UNCHANGED_EXACT_DETERMINISTIC_1_OF_50',
    }
    payload.setdefault('scientific_boundary', []).extend([
        'Station-only proxy chains remain valid passenger-likelihood objects but cannot provide observed train-arrival-to-exit propagation time and therefore are excluded from all Theta_E fitting.',
        'The predecessor global Theta_E update accidentally included fit-ineligible station-only proxy factors despite the documented intent; this result supersedes that kernel-M-step implementation while permanently retaining the predecessor validation records.',
        'This is a defect-correction qualification, not a new 1-of-50 tuning round.',
    ])
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
