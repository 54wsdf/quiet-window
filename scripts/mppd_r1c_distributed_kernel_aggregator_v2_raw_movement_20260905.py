import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import scripts.mppd_g2v2_uncertain_service_full_network_posterior_20260904 as g2
import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base
import scripts.mppd_r1b_first_bidirectional_joint_update_v2_20260904 as v2

GLOBAL_DAMPING = 0.35
CONTEXT_MIN_WEIGHT = 30.0
CONTEXT_SHRINK_TAU = 75.0
HOUR_MIN_WEIGHT = 40.0
HOUR_SHRINK_TAU = 100.0
TRANSFER_MOVEMENT_RAW_MIN_WEIGHT = 50.0


def default_kernels():
    return {
        'access': base.kernel_from_median_sigma(180.0, 0.90, 'G2V2_BROAD_INITIAL_PRIOR'),
        'transfer_global': base.kernel_from_median_sigma(180.0, 0.85, 'G2V2_BROAD_INITIAL_PRIOR'),
        'egress': base.kernel_from_median_sigma(120.0, 0.80, 'G2V2_BROAD_INITIAL_PRIOR'),
    }


def load_stats(root):
    hists = defaultdict(Counter)
    raw_movement_weight = Counter()
    files = sorted(Path(root).glob('**/e0_kernel_sufficient_stats.jsonl.gz'))
    if not files:
        raise RuntimeError('no E0 kernel sufficient-stat files found')
    for path in files:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get('scope') == 'TRANSFER_MOVEMENT_RAW_WEIGHT':
                    raw_movement_weight[str(r.get('group') or 'UNKNOWN')] += float(r['weight'])
                    continue
                key = (r['scope'], r.get('group'))
                if r['kind'] == 'INTERVAL':
                    sig = (float(r['lower_sec']), float(r['upper_sec']))
                elif r['kind'] == 'POINT':
                    sig = (float(r['x_sec']),)
                else:
                    raise RuntimeError(f"unexpected kernel sufficient-stat kind {r.get('kind')}")
                hists[key][sig] += float(r['weight'])
    return hists, raw_movement_weight, files


def interval_fit(hist, initial_kernel, min_weight, evidence_class):
    rows = [(lo, hi, w) for (lo, hi), w in hist.items() if hi > max(0.0, lo) + 1e-6 and w > 0]
    total = sum(w for _, _, w in rows)
    if total < min_weight:
        return initial_kernel, {'fitted': False, 'effective_weight': total, 'unique_signature_count': len(rows)}
    c0 = initial_kernel['components'][0]
    x0 = np.array([math.log(max(1.0, float(c0['median_sec']))), math.log(max(0.15, float(c0['sigma'])))])
    bounds = [(math.log(10.0), math.log(1800.0)), (math.log(0.12), math.log(2.5))]
    def obj(x):
        med = math.exp(float(x[0])); sig = math.exp(float(x[1])); val = 0.0
        for lo, hi, w in rows:
            fu = g2.lognorm_cdf(hi, med, sig); fl = g2.lognorm_cdf(lo, med, sig) if lo > 0 else 0.0
            val -= w * math.log(max(base.EPS, fu - fl))
        return val / total
    fit = minimize(obj, x0, method='L-BFGS-B', bounds=bounds)
    med = math.exp(float(fit.x[0])); sig = math.exp(float(fit.x[1]))
    return base.kernel_from_median_sigma(med, sig, evidence_class), {
        'fitted': bool(fit.success), 'effective_weight': total, 'unique_signature_count': len(rows),
        'median_sec': med, 'sigma': sig, 'objective_per_mass': float(fit.fun),
    }


def point_fit(hist, initial_kernel, min_weight, evidence_class):
    rows = [(x[0], w) for x, w in hist.items() if 0 < x[0] <= 3600 and w > 0]
    total = sum(w for _, w in rows)
    if total < min_weight:
        return initial_kernel, {'fitted': False, 'effective_weight': total, 'unique_signature_count': len(rows)}
    logs = np.array([math.log(x) for x, _ in rows], dtype=float); weights = np.array([w for _, w in rows], dtype=float)
    mu = float(np.average(logs, weights=weights)); var = float(np.average((logs-mu)**2, weights=weights))
    sig = min(2.5, max(0.12, math.sqrt(max(1e-8, var)))); med = math.exp(mu)
    return base.kernel_from_median_sigma(med, sig, evidence_class), {
        'fitted': True, 'effective_weight': total, 'unique_signature_count': len(rows), 'median_sec': med, 'sigma': sig,
    }


def adaptive_blend(parent, raw, effective_weight, tau, evidence_class):
    alpha = max(0.0, min(1.0, effective_weight/(effective_weight+tau)))
    return v2.blend_kernel(parent, raw, alpha, evidence_class), alpha


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stats-root', required=True); ap.add_argument('--out', required=True); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    hists, raw_movement_weight, files = load_stats(args.stats_root); prior = default_kernels()

    raw_a, diag_a = interval_fit(hists[('ACCESS_GLOBAL',None)], prior['access'], 20.0, 'R1C_FULL_RAW_ACCESS_INTERVAL_FIT')
    raw_k, diag_k = interval_fit(hists[('TRANSFER_GLOBAL',None)], prior['transfer_global'], 20.0, 'R1C_FULL_RAW_TRANSFER_INTERVAL_FIT')
    raw_e, diag_e = point_fit(hists[('EGRESS_GLOBAL',None)], prior['egress'], 20.0, 'R1C_FULL_RAW_EGRESS_POINT_FIT')
    access = v2.blend_kernel(prior['access'], raw_a, GLOBAL_DAMPING, 'R1C_FULL_DAMPED_GLOBAL_ACCESS')
    transfer_global = v2.blend_kernel(prior['transfer_global'], raw_k, GLOBAL_DAMPING, 'R1C_FULL_DAMPED_GLOBAL_TRANSFER')
    egress = v2.blend_kernel(prior['egress'], raw_e, GLOBAL_DAMPING, 'R1C_FULL_DAMPED_GLOBAL_EGRESS')

    transfer_by_movement = {}; transfer_diag = {}
    for movement, raw_weight in raw_movement_weight.items():
        if raw_weight < TRANSFER_MOVEMENT_RAW_MIN_WEIGHT:
            continue
        hist = hists.get(('TRANSFER_MOVEMENT', movement), Counter())
        raw, d = interval_fit(hist, raw_k, 50.0, 'R1C_FULL_RAW_TRANSFER_MOVEMENT_K1')
        # Exact monolithic semantics: membership is decided by raw movement mass.
        # If the cleaned interval mass falls below 50, weighted_interval_fit returns
        # its initial kernel (raw global K); v2 still creates a movement-specific
        # damped kernel by blending global damped K toward that raw global K.
        transfer_by_movement[movement] = v2.blend_kernel(
            transfer_global, raw, GLOBAL_DAMPING, 'R1C_FULL_DAMPED_TRANSFER_MOVEMENT_K1'
        )
        transfer_diag[movement] = {**d, 'raw_movement_weight': raw_weight, 'membership_gate': 'RAW_MOVEMENT_WEIGHT_GE_50'}

    access_by_context = {}; access_context_diag = {}
    for (scope, group), hist in hists.items():
        if scope != 'ACCESS_CONTEXT' or group is None: continue
        raw, d = interval_fit(hist, access, CONTEXT_MIN_WEIGHT, 'R1C_FULL_RAW_ACCESS_CONTEXT_K1')
        if not d.get('fitted') or d.get('effective_weight',0.0) < CONTEXT_MIN_WEIGHT: continue
        kernel, alpha = adaptive_blend(access, raw, d['effective_weight'], CONTEXT_SHRINK_TAU, 'R1C_FULL_HIERARCHICAL_ACCESS_CONTEXT_K1')
        access_by_context[group] = kernel; access_context_diag[group] = {**d, 'shrinkage_alpha':alpha}

    egress_by_context = {}; egress_context_diag = {}
    for (scope, group), hist in hists.items():
        if scope != 'EGRESS_CONTEXT' or group is None: continue
        raw, d = point_fit(hist, egress, CONTEXT_MIN_WEIGHT, 'R1C_FULL_RAW_EGRESS_CONTEXT_K1')
        if not d.get('fitted') or d.get('effective_weight',0.0) < CONTEXT_MIN_WEIGHT: continue
        kernel, alpha = adaptive_blend(egress, raw, d['effective_weight'], CONTEXT_SHRINK_TAU, 'R1C_FULL_HIERARCHICAL_EGRESS_CONTEXT_K1')
        egress_by_context[group] = kernel; egress_context_diag[group] = {**d, 'shrinkage_alpha':alpha}

    access_by_context_hour = {}; access_hour_diag = {}
    for (scope, group), hist in hists.items():
        if scope != 'ACCESS_CONTEXT_HOUR' or group is None: continue
        ctx = group.rsplit('|h',1)[0]; parent = access_by_context.get(ctx)
        if parent is None: continue
        raw, d = interval_fit(hist, parent, HOUR_MIN_WEIGHT, 'R1C_FULL_RAW_ACCESS_CONTEXT_HOUR_K1')
        if not d.get('fitted') or d.get('effective_weight',0.0) < HOUR_MIN_WEIGHT: continue
        kernel, alpha = adaptive_blend(parent, raw, d['effective_weight'], HOUR_SHRINK_TAU, 'R1C_FULL_HIERARCHICAL_ACCESS_CONTEXT_HOUR_K1')
        access_by_context_hour[group] = kernel; access_hour_diag[group] = {**d, 'shrinkage_alpha':alpha}

    egress_by_context_hour = {}; egress_hour_diag = {}
    for (scope, group), hist in hists.items():
        if scope != 'EGRESS_CONTEXT_HOUR' or group is None: continue
        ctx = group.rsplit('|h',1)[0]; parent = egress_by_context.get(ctx)
        if parent is None: continue
        raw, d = point_fit(hist, parent, HOUR_MIN_WEIGHT, 'R1C_FULL_RAW_EGRESS_CONTEXT_HOUR_K1')
        if not d.get('fitted') or d.get('effective_weight',0.0) < HOUR_MIN_WEIGHT: continue
        kernel, alpha = adaptive_blend(parent, raw, d['effective_weight'], HOUR_SHRINK_TAU, 'R1C_FULL_HIERARCHICAL_EGRESS_CONTEXT_HOUR_K1')
        egress_by_context_hour[group] = kernel; egress_hour_diag[group] = {**d, 'shrinkage_alpha':alpha}

    kernels = {
        'schema':'mppd.r1c-full-denominator-hierarchical-temporal-kernels.v2-raw-movement-membership-k1-mixture-capable',
        'access':access,'egress':egress,'transfer_global':transfer_global,'transfer_by_movement':transfer_by_movement,
        'access_by_context':access_by_context,'access_by_context_hour':access_by_context_hour,
        'egress_by_context':egress_by_context,'egress_by_context_hour':egress_by_context_hour,
        'hierarchy':{'access':'GLOBAL_TO_LINE_STATION_DIRECTION_TO_HOUR','egress':'GLOBAL_TO_LINE_STATION_DIRECTION_TO_HOUR','transfer':'GLOBAL_TO_MOVEMENT','day_specific':'DEFERRED_TO_R2_MULTI_DAY'},
        'mixture_policy':{'access':'K1_FIRST_PASS_SCHEMA_MIXTURE_CAPABLE','egress':'K1_FIRST_PASS_SCHEMA_MIXTURE_CAPABLE','transfer':'MOVEMENT_CONDITIONED_K1_FIRST_PASS_SCHEMA_MIXTURE_CAPABLE_DATA_SELECTED_K_LATER'},
    }
    diagnostics = {
        'schema':'mppd.r1c-full-denominator-kernel-aggregation-diagnostics.v2-raw-movement-membership','date':'2026-09-05','source_shard_file_count':len(files),
        'global':{'access':diag_a,'transfer':diag_k,'egress':diag_e,'damping':GLOBAL_DAMPING},
        'counts':{
            'access_context_count':len(access_by_context),'access_context_hour_count':len(access_by_context_hour),
            'egress_context_count':len(egress_by_context),'egress_context_hour_count':len(egress_by_context_hour),
            'transfer_movement_count':len(transfer_by_movement),
            'transfer_raw_movement_count':len(raw_movement_weight),
            'transfer_raw_movement_ge50_count':sum(1 for v in raw_movement_weight.values() if v >= TRANSFER_MOVEMENT_RAW_MIN_WEIGHT),
        },
        'hyperparameters':{
            'context_min_weight':CONTEXT_MIN_WEIGHT,'context_shrink_tau':CONTEXT_SHRINK_TAU,
            'hour_min_weight':HOUR_MIN_WEIGHT,'hour_shrink_tau':HOUR_SHRINK_TAU,
            'transfer_movement_membership_raw_min_weight':TRANSFER_MOVEMENT_RAW_MIN_WEIGHT,
            'transfer_movement_clean_fit_min_weight':50.0,
        },
        'access_context':access_context_diag,'access_context_hour':access_hour_diag,
        'egress_context':egress_context_diag,'egress_context_hour':egress_hour_diag,'transfer_movement':transfer_diag,
        'scientific_boundary':[
            'All input records are shard-local posterior sufficient statistics and are merged by additive posterior weight before fitting.',
            'Transfer movement kernel membership reproduces the canonical R1B/R1C rule: raw posterior movement mass gates membership before invalid interval geometry is removed by the interval fitter.',
            'If cleaned movement mass is below 50 after membership passes, the raw global transfer kernel is returned and the canonical second damping step is still applied.',
            'Global damping and hierarchical shrinkage retain the declared R1C first-pass hyperparameters; no full-denominator retuning is performed.',
            'K=1 remains a first-pass representation and all serialized kernels remain mixture-capable.'
        ],
        'no_email_notification_logic':True,
    }
    (out/'global_kernels.json').write_text(json.dumps({'kernels':kernels},ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'kernel_aggregation_diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'counts':diagnostics['counts'],'global':diagnostics['global']},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
