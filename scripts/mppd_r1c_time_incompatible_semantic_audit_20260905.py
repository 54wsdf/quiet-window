import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r0_cached_full_network_support_rescan_20260904 as support
import scripts.mppd_g2v2_uncertain_service_full_network_posterior_20260904 as g2
import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch


def load_residuals(path):
    out = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def candidate_hard_failure(cand, meta, rides_fn, tout):
    legs = [x for x in base.r0.path_legs(cand.get('path') or [], meta) if x[1] != x[2]]
    if not legs:
        return {'reason': 'STATION_ONLY_OR_EMPTY_ROUTE'}
    for leg_index, (line, origin, destination) in enumerate(legs):
        rides = rides_fn(line, origin, destination)
        if not rides:
            return {
                'reason': 'MISSING_SERVICE_SEGMENT',
                'leg_index': leg_index,
                'line': line,
                'origin': origin,
                'destination': destination,
            }
        admissible = []
        late = []
        for rd in rides:
            late_sec = (rd['arr'] - tout).total_seconds()
            gate_sec = 3.0 * max(1.0, float(rd['arr_sd']))
            if late_sec > gate_sec:
                late.append((late_sec - gate_sec, rd))
            else:
                admissible.append(rd)
        if not admissible:
            min_excess = min(x[0] for x in late) if late else None
            best = min(late, key=lambda x: x[0])[1] if late else None
            return {
                'reason': 'SERVICE_EXIT_HORIZON_INCOMPATIBLE',
                'leg_index': leg_index,
                'line': line,
                'origin': origin,
                'destination': destination,
                'ride_count': len(rides),
                'all_rides_later_than_exit_plus_3sigma': True,
                'minimum_excess_beyond_exit_3sigma_sec': min_excess,
                'closest_root_key': best.get('root_key') if best else None,
                'closest_arrival': best.get('arr').isoformat() if best else None,
                'closest_arrival_sd_sec': float(best.get('arr_sd')) if best else None,
            }
    return {'reason': 'HARD_CHAIN_EXISTS_KERNEL_SCORE_ONLY'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p1c', required=True)
    ap.add_argument('--routes', required=True)
    ap.add_argument('--service-init', required=True)
    ap.add_argument('--r1c-summary', required=True)
    ap.add_argument('--residual-cohorts', required=True)
    ap.add_argument('--topology-patch', required=True)
    ap.add_argument('--gtxa-overlay', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    apply_topology_patch(G, meta, load_patch(args.topology_patch))
    apply_gtxa_overlay(G, meta, code_to_nodes, load_overlay(args.gtxa_overlay))
    routes = support.load_routes(args.routes)
    roots, service_manifest, service_payload = g2.load_uncertain_service(args.service_init)
    summary = json.loads(Path(args.r1c_summary).read_text(encoding='utf-8'))
    residuals = load_residuals(args.residual_cohorts)

    offsets = summary.get('iteration', {}).get('M_service_timing_update', {}).get('nonzero_offsets', {})
    base.apply_root_offsets(roots, offsets)
    rides_fn, _ = base.build_joint_rides(roots)

    total_residual_mass = 0.0
    stored_reason_mass = Counter()
    replay_reason_mass = Counter()
    time_subtype_mass = Counter()
    time_by_line = Counter()
    time_by_leg = Counter()
    excess_samples = []
    unexplained_rows = []
    rows_out = []

    for rr in residuals:
        oc = str(rr['origin_code']); dc = str(rr['destination_code'])
        tin = datetime.fromisoformat(rr['entry_time']); tout = datetime.fromisoformat(rr['exit_time'])
        mass = float(rr['passenger_mass'])
        total_residual_mass += mass
        stored = rr.get('failure_reason_probs') or {}
        for reason, p in stored.items():
            stored_reason_mass[reason] += mass * float(p)

        cands = routes.get((oc, dc), [])
        weights = base.structural_pressure_weights(cands) if cands else []
        candidate_rows = []
        replay_w = Counter()
        for ci, cand in enumerate(cands):
            w = float(weights[ci]) if ci < len(weights) else 0.0
            fail = candidate_hard_failure(cand, meta, rides_fn, tout)
            reason = fail['reason']
            if reason == 'SERVICE_EXIT_HORIZON_INCOMPATIBLE':
                replay_reason = 'TIME_INCOMPATIBLE_CHAIN'
            elif reason == 'HARD_CHAIN_EXISTS_KERNEL_SCORE_ONLY':
                replay_reason = 'FINITE_HARD_CHAIN_EXISTS'
            elif reason == 'STATION_ONLY_OR_EMPTY_ROUTE':
                replay_reason = 'STATION_ONLY_OR_EMPTY_ROUTE'
            else:
                replay_reason = reason
            replay_w[replay_reason] += w
            if reason == 'SERVICE_EXIT_HORIZON_INCOMPATIBLE':
                time_subtype_mass[reason] += mass * w
                time_by_line[str(fail.get('line'))] += mass * w
                time_by_leg[str(fail.get('leg_index'))] += mass * w
                if fail.get('minimum_excess_beyond_exit_3sigma_sec') is not None:
                    excess_samples.append((float(fail['minimum_excess_beyond_exit_3sigma_sec']), mass * w))
            candidate_rows.append({
                'candidate_index': ci,
                'structural_weight': w,
                'line_sequence': cand.get('line_sequence'),
                'transfer_count': cand.get('transfer_count'),
                **fail,
            })

        z = sum(replay_w.values())
        replay_probs = {k: v / z for k, v in replay_w.items()} if z > 0 else {}
        for reason, p in replay_probs.items():
            replay_reason_mass[reason] += mass * p

        stored_time = float(stored.get('TIME_INCOMPATIBLE_CHAIN', 0.0))
        explained_time = float(replay_probs.get('TIME_INCOMPATIBLE_CHAIN', 0.0))
        if stored_time > 1e-12 and explained_time + 1e-9 < stored_time:
            unexplained_rows.append({
                'cohort_id': rr.get('cohort_id'),
                'passenger_mass': mass,
                'stored_time_probability': stored_time,
                'replayed_time_probability': explained_time,
                'candidate_rows': candidate_rows,
            })
        rows_out.append({
            'cohort_id': rr.get('cohort_id'),
            'origin_code': oc,
            'destination_code': dc,
            'entry_time': tin.isoformat(),
            'exit_time': tout.isoformat(),
            'passenger_mass': mass,
            'stored_failure_reason_probs': stored,
            'replay_hard_reason_probs': replay_probs,
            'candidate_rows': candidate_rows,
        })

    def weighted_quantile(pairs, q):
        if not pairs:
            return None
        pairs = sorted(pairs)
        total = sum(w for _, w in pairs)
        target = q * total
        acc = 0.0
        for x, w in pairs:
            acc += w
            if acc >= target:
                return x
        return pairs[-1][0]

    result = {
        'schema': 'mppd.r1c-time-incompatible-semantic-audit.v1',
        'date': '2026-09-05',
        'status': 'R1C_TIME_INCOMPATIBLE_SEMANTIC_AUDIT_COMPLETED',
        'source': {
            'r1c_schema': summary.get('schema'),
            'service_schema': service_payload.get('schema'),
            'residual_record_count': len(residuals),
            'residual_passenger_mass': total_residual_mass,
            'service_root_offset_count': len(offsets),
        },
        'stored_failure_mass': dict(stored_reason_mass),
        'replayed_hard_reason_mass': dict(replay_reason_mass),
        'time_incompatible_subtype_mass': dict(time_subtype_mass),
        'time_incompatible_by_line': dict(time_by_line.most_common()),
        'time_incompatible_by_leg_index': dict(time_by_leg.most_common()),
        'service_exit_horizon_excess_sec_weighted_quantiles': {
            'q05': weighted_quantile(excess_samples, 0.05),
            'q50': weighted_quantile(excess_samples, 0.50),
            'q95': weighted_quantile(excess_samples, 0.95),
        },
        'unexplained_time_incompatible_cohort_count': len(unexplained_rows),
        'semantic_judgement': (
            'TIME_INCOMPATIBLE_CHAIN_IS_HARD_SERVICE_EXIT_HORIZON_INCOMPATIBILITY_UNDER_CURRENT_ENGINE'
            if not unexplained_rows else
            'TIME_INCOMPATIBLE_CHAIN_HAS_UNEXPLAINED_ENGINE_PATHS_REQUIRING_IMPLEMENTATION_AUDIT'
        ),
        'scientific_boundary': [
            'The current R1B/R1C kernel densities score hard-feasible service chains; they do not create a service chain when every candidate train arrival lies later than the observed AFC exit time beyond the configured 3-sigma service-timing envelope.',
            'Therefore a density-only hierarchical Theta_A/Theta_K/Theta_E update is not expected to reduce SERVICE_EXIT_HORIZON_INCOMPATIBLE mass by construction.',
            'This audit changes semantic attribution only and performs no service mutation, route mutation, kernel tuning, or passenger-behavior relabeling.',
            'SERVICE_EXIT_HORIZON_INCOMPATIBLE is still model-based residual evidence, not observed anomalous passenger behavior.',
        ],
        'next_gate': (
            'If unexplained mass is zero, correct the R1C qualification criterion and advance full-denominator distributed R1C; otherwise repair the engine before scaling.'
        ),
        'no_email_notification_logic': True,
    }
    (out / 'r1c_time_incompatible_semantic_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with gzip.open(out / 'r1c_time_incompatible_semantic_audit_cohorts.jsonl.gz', 'wt', encoding='utf-8') as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    if unexplained_rows:
        (out / 'unexplained_time_incompatible_rows.json').write_text(json.dumps(unexplained_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
