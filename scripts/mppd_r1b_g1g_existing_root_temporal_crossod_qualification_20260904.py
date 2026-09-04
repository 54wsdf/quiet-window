import argparse
import gzip
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_g2v2_uncertain_service_full_network_posterior_20260904 as g2
import scripts.mppd_r1b_missing_service_support_proposal_audit_20260904 as audit
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch


def dt(x):
    return datetime.fromisoformat(x) if x else None


def event_time(ev):
    return ev.get('departure') or ev.get('arrival')


def predict_at_node(vp, node, meta, per_edge_sec):
    if node in vp['events']:
        t = event_time(vp['events'][node])
        return {'time': t, 'mode': 'EXACT_VARIANT_EVENT', 'distance_seq': 0.0}
    target_seq = meta.get(node, {}).get('seq')
    if target_seq is None:
        return None
    pts = []
    for n, ev in vp['events'].items():
        seq = meta.get(n, {}).get('seq')
        t = event_time(ev)
        if seq is None or t is None:
            continue
        pts.append((float(seq), t, n))
    if not pts:
        return None
    pts.sort(key=lambda x: x[0])
    left = None
    right = None
    for p in pts:
        if p[0] <= float(target_seq):
            left = p
        if p[0] >= float(target_seq) and right is None:
            right = p
    if left and right and left[0] != right[0]:
        frac = (float(target_seq) - left[0]) / (right[0] - left[0])
        pred = left[1] + (right[1] - left[1]) * frac
        return {
            'time': pred,
            'mode': 'INTERPOLATED_BETWEEN_VARIANT_EVENTS',
            'distance_seq': min(abs(float(target_seq)-left[0]), abs(right[0]-float(target_seq))),
        }
    nearest = min(pts, key=lambda p: abs(p[0] - float(target_seq)))
    sign = vp.get('direction_sign')
    if sign is None:
        return None
    delta_seq = float(target_seq) - nearest[0]
    pred = nearest[1] + timedelta(seconds=(delta_seq / float(sign)) * float(per_edge_sec))
    return {
        'time': pred,
        'mode': 'EXTRAPOLATED_FROM_VARIANT_EVENT',
        'distance_seq': abs(delta_seq),
    }


def root_prediction_candidates(profile, origin, destination, meta, per_edge_sec):
    out = []
    for vp in profile['variants']:
        po = predict_at_node(vp, origin, meta, per_edge_sec)
        pd = predict_at_node(vp, destination, meta, per_edge_sec)
        if not po or not pd:
            continue
        if po['time'] >= pd['time']:
            continue
        out.append({
            'variant_id': vp.get('variant_id'),
            'origin_time': po['time'],
            'destination_time': pd['time'],
            'origin_prediction_mode': po['mode'],
            'destination_prediction_mode': pd['mode'],
            'max_sequence_extrapolation': max(po['distance_seq'], pd['distance_seq']),
            'evidence_class': vp.get('evidence_class'),
        })
    return out


def load_residual_segment_records(path):
    seg = defaultdict(list)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            seen = set()
            ready_by_seg = {}
            for c in row.get('candidate_records', []):
                s = c.get('missing_segment')
                env = c.get('ready_envelope')
                if s and env and s not in ready_by_seg:
                    ready_by_seg[s] = env
            for m in row.get('missing_segments', []):
                s = m.get('segment')
                if not s or s in seen:
                    continue
                seen.add(s)
                env = ready_by_seg.get(s, {})
                seg[s].append({
                    'cohort_id': row.get('cohort_id'),
                    'origin_code': row.get('origin_code'),
                    'destination_code': row.get('destination_code'),
                    'passenger_mass': float(row.get('passenger_mass', 0.0)),
                    'pressure_mass': float(m.get('pressure_mass', 0.0)),
                    'structural_weight': float(m.get('structural_weight', 0.0)),
                    'ready_q05': dt(env.get('ready_q05_min')),
                    'ready_q50': dt(env.get('ready_q50_weighted_mean')),
                    'ready_q95': dt(env.get('ready_q95_max')),
                    'observed_exit': dt(env.get('observed_exit_time') or row.get('exit_time')),
                })
    return seg


def evaluate_prediction(pred, records):
    result = {'q05_support_mass': 0.0, 'q50_support_mass': 0.0, 'q95_support_mass': 0.0, 'exit_necessary_support_mass': 0.0,
              'q05_supported_cohorts': 0, 'q50_supported_cohorts': 0, 'q95_supported_cohorts': 0}
    dep = pred['origin_time']; arr = pred['destination_time']
    for r in records:
        mass = float(r['pressure_mass'])
        exit_ok = r['observed_exit'] is None or arr <= r['observed_exit']
        if exit_ok:
            result['exit_necessary_support_mass'] += mass
        if exit_ok and r['ready_q05'] is not None and dep >= r['ready_q05']:
            result['q05_support_mass'] += mass; result['q05_supported_cohorts'] += 1
        if exit_ok and r['ready_q50'] is not None and dep >= r['ready_q50']:
            result['q50_support_mass'] += mass; result['q50_supported_cohorts'] += 1
        if exit_ok and r['ready_q95'] is not None and dep >= r['ready_q95']:
            result['q95_support_mass'] += mass; result['q95_supported_cohorts'] += 1
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p1c', required=True)
    ap.add_argument('--service-init', required=True)
    ap.add_argument('--evidence-join', required=True)
    ap.add_argument('--residual-cohorts', required=True)
    ap.add_argument('--topology-patch', required=True)
    ap.add_argument('--gtxa-overlay', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    apply_topology_patch(G, meta, load_patch(args.topology_patch))
    apply_gtxa_overlay(G, meta, code_to_nodes, load_overlay(args.gtxa_overlay))
    roots, service_manifest, service_payload = g2.load_uncertain_service(args.service_init)
    profiles_by_line = audit.root_profiles(roots, meta)
    profile_lookup = {p['root_key']: p for rows in profiles_by_line.values() for p in rows}
    join = json.loads(Path(args.evidence_join).read_text(encoding='utf-8'))
    residual = load_residual_segment_records(args.residual_cohorts)
    line_runtime = service_manifest.get('line_median_per_edge_runtime_sec', {})
    global_runtime = float(service_manifest.get('global_median_per_edge_runtime_sec', 121.0))

    proposals = []
    skipped = defaultdict(float)
    for segrow in join.get('segments', []):
        if segrow.get('proposal_tier') != 'EXISTING_ROOT_SUPPORT_PROPOSAL_AUDIT_ELIGIBLE':
            continue
        parsed = audit.parse_pressure_segment(segrow.get('segment'))
        if not parsed:
            continue
        line, origin, destination = parsed
        records = residual.get(segrow['segment'], [])
        if not records:
            skipped['NO_COHORT_READY_RECORDS'] += float(segrow.get('pressure_mass', 0.0))
            continue
        roots_for_segment = []
        for r in segrow.get('cross_variant_roots', []):
            if r not in roots_for_segment: roots_for_segment.append(r)
        for r in segrow.get('interior_roots', []):
            if r not in roots_for_segment: roots_for_segment.append(r)
        for x in segrow.get('nearby_roots', []):
            r = x.get('root')
            if r and r not in roots_for_segment: roots_for_segment.append(r)
        per = float(line_runtime.get(line, global_runtime))
        root_rows = []
        for root in roots_for_segment:
            prof = profile_lookup.get(root)
            if not prof:
                continue
            preds = root_prediction_candidates(prof, origin, destination, meta, per)
            best = None
            for pred in preds:
                ev = evaluate_prediction(pred, records)
                row = {
                    'root': root,
                    'variant_id': pred['variant_id'],
                    'root_evidence_classes': prof.get('evidence_classes', []),
                    'origin_time': pred['origin_time'].isoformat(),
                    'destination_time': pred['destination_time'].isoformat(),
                    'origin_prediction_mode': pred['origin_prediction_mode'],
                    'destination_prediction_mode': pred['destination_prediction_mode'],
                    'max_sequence_extrapolation': pred['max_sequence_extrapolation'],
                    **ev,
                }
                key = (row['q50_support_mass'], row['q95_support_mass'], row['q05_support_mass'], -row['max_sequence_extrapolation'])
                if best is None or key > best[0]: best = (key, row)
            if best is not None:
                root_rows.append(best[1])
        root_rows.sort(key=lambda r: (r['q50_support_mass'], r['q95_support_mass'], r['q05_support_mass']), reverse=True)
        pressure = float(segrow.get('pressure_mass', 0.0))
        best = root_rows[0] if root_rows else None
        proposals.append({
            'segment': segrow['segment'],
            'line': line,
            'classification': segrow.get('direction_aware_classification'),
            'pressure_mass': pressure,
            'cohort_count': segrow.get('cohort_count'),
            'distinct_od_count': segrow.get('distinct_od_count'),
            'cross_od_support': segrow.get('cross_od_support'),
            'candidate_root_count': len(root_rows),
            'best_root': best,
            'best_q05_support_fraction': (best['q05_support_mass']/pressure if best and pressure > 0 else 0.0),
            'best_q50_support_fraction': (best['q50_support_mass']/pressure if best and pressure > 0 else 0.0),
            'best_q95_support_fraction': (best['q95_support_mass']/pressure if best and pressure > 0 else 0.0),
            'root_candidates': root_rows[:20],
        })
    proposals.sort(key=lambda r: (r['best_q50_support_fraction'], r['pressure_mass']), reverse=True)

    total_pressure = sum(float(r['pressure_mass']) for r in proposals)
    q50_any = sum(float(r['pressure_mass']) for r in proposals if r['best_q50_support_fraction'] > 0)
    q50_half = sum(float(r['pressure_mass']) for r in proposals if r['best_q50_support_fraction'] >= 0.5)
    q95_half = sum(float(r['pressure_mass']) for r in proposals if r['best_q95_support_fraction'] >= 0.5)
    multi_od_q50_half = sum(float(r['pressure_mass']) for r in proposals if r['best_q50_support_fraction'] >= 0.5 and int(r.get('distinct_od_count') or 0) >= 2)

    result = {
        'schema': 'mppd.r1b-g1g-existing-root-temporal-crossod-qualification.v1',
        'date': '2026-09-04',
        'status': 'G1G_EXISTING_ROOT_TEMPORAL_CROSSOD_QUALIFICATION_COMPLETED_NO_SERVICE_MUTATION',
        'source_service_schema': service_payload.get('schema'),
        'source_service_status': service_payload.get('status'),
        'source_evidence_join_schema': join.get('schema'),
        'scope': {
            'proposal_segment_count': len(proposals),
            'eligible_pressure_mass_represented': total_pressure,
            'q50_any_temporal_support_pressure': q50_any,
            'q50_majority_temporal_support_pressure': q50_half,
            'q95_majority_temporal_support_pressure': q95_half,
            'q50_majority_and_multi_od_pressure': multi_od_q50_half,
            'skipped_pressure': dict(skipped),
        },
        'proposals': proposals,
        'interpretation_rules': {
            'q05': 'WEAK_NECESSARY_READY_SUPPORT_APPROXIMATION',
            'q50': 'MEDIAN_READY_SUPPORT_APPROXIMATION',
            'q95': 'STRONG_READY_SUPPORT_APPROXIMATION',
            'exit_check': 'DESTINATION_EVENT_MUST_NOT_OCCUR_AFTER_OBSERVED_FINAL_EXIT',
            'prediction': 'EXISTING_VARIANT_EVENT_OR_SEQ_INTERPOLATION_EXTRAPOLATION_FOR_AUDIT_ONLY',
        },
        'scientific_boundary': [
            'G1G performs no service mutation and authorizes no new latent service root.',
            'Temporal support fractions are approximate necessary-support diagnostics based on residual ready envelopes and predicted existing-root event times; they are not observed boarding truth.',
            'A high support fraction only qualifies a bounded existing-root repair trial. Acceptance requires preserving every source-root original event exactly and improving the exact paired R1B posterior without degrading held-out service evidence.',
            'Sequence-based interpolation/extrapolation is used only to rank bounded repair trials and must not be promoted to observed ATS.',
            'Cross-OD support is reported separately so a single-cohort or single-OD coincidence cannot be mistaken for network-level service evidence.',
        ],
        'next_gate': 'Select bounded existing-root repair trials from high temporal-support, multi-OD candidates; apply observed-event-preserving completion only within existing roots; rerun exact paired R1B broad before any promotion.',
        'no_email_notification_logic': True,
    }
    (out/'r1b_g1g_existing_root_temporal_crossod_qualification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'scope': result['scope'], 'top_proposals': proposals[:20]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
