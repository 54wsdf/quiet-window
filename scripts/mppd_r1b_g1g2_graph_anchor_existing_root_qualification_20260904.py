import argparse
import json
from datetime import datetime
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r1b_g1g_existing_root_temporal_crossod_qualification_20260904 as g1g
import scripts.mppd_r1b_missing_service_support_proposal_audit_20260904 as audit
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch

OBSERVED_CLASSES = {'PARTIAL_DIRECT_SERVICE_ANCHOR'}
STRONG_GRADES = {'ORIGINAL_EXACT', 'GRAPH_BRACKETED_BETWEEN_ORIGINAL_EVENTS'}


def parse_time(x):
    return datetime.fromisoformat(x) if x else None


def event_time(e):
    return parse_time(e.get('departure') or e.get('arrival') or e.get('time'))


def event_grade(state, e):
    if e.get('is_original_event') is True:
        return 'ORIGINAL_EXACT'
    if state.get('evidence_class') in OBSERVED_CLASSES and e.get('is_original_event') is not False:
        return 'ORIGINAL_EXACT'
    role = e.get('interpolation_role')
    if role == 'BETWEEN_OBSERVED_ANCHORS':
        return 'LATENT_EXACT_BETWEEN_OBSERVED'
    if role in ('EXTRAPOLATED_AFTER_LAST_OBSERVED', 'EXTRAPOLATED_BEFORE_FIRST_OBSERVED'):
        return 'LATENT_EXACT_EXTRAPOLATED'
    if state.get('evidence_class') == 'SERVICE_TRAJECTORY_COMPLETION_HYPOTHESIS':
        return 'LATENT_EXACT_COMPLETION_UNCLASSIFIED'
    return 'INFERRED_EXACT_OTHER'


def unique_shortest_path(H, a, b):
    if a == b:
        return [a], True
    if a not in H or b not in H:
        return None, False
    try:
        gen = nx.all_shortest_paths(H, a, b)
        first = list(next(gen))
        try:
            next(gen)
            return first, False
        except StopIteration:
            return first, True
    except (nx.NetworkXNoPath, nx.NodeNotFound, StopIteration):
        return None, False


def build_root_variants(payload):
    roots = {}
    for state in payload.get('states', []):
        root = state.get('root_service_id') or state.get('service_id')
        line = state.get('line')
        events = {}
        for e in state.get('station_events', []):
            t = event_time(e)
            if t is None:
                continue
            events[e['node']] = {'time': t, 'raw': e, 'grade': event_grade(state, e)}
        roots.setdefault(f'{line}||{root}', []).append({
            'variant_id': state.get('service_id'),
            'root': root,
            'line': line,
            'direction': state.get('direction'),
            'evidence_class': state.get('evidence_class'),
            'root_evidence_class': state.get('root_evidence_class'),
            'events': events,
        })
    return roots


def predict_target_strict(variant, target, H):
    if target in variant['events']:
        x = variant['events'][target]
        return {
            'time': x['time'],
            'mode': 'EXACT_EVENT',
            'anchor_grade': x['grade'],
            'bracket_nodes': [],
            'bracket_span_edges': 0,
            'target_offset_edges': 0,
            'unique_graph_path': True,
        }

    originals = [(n, x['time']) for n, x in variant['events'].items() if x['grade'] == 'ORIGINAL_EXACT']
    best = None
    for i, (a, ta) in enumerate(originals):
        for b, tb in originals[i + 1:]:
            if ta == tb:
                continue
            if tb < ta:
                a2, ta2, b2, tb2 = b, tb, a, ta
            else:
                a2, ta2, b2, tb2 = a, ta, b, tb
            path, unique = unique_shortest_path(H, a2, b2)
            if not path or not unique or target not in path:
                continue
            idx = path.index(target)
            span = len(path) - 1
            if span <= 0:
                continue
            frac = idx / span
            pred = ta2 + (tb2 - ta2) * frac
            row = {
                'time': pred,
                'mode': 'GRAPH_BRACKETED_INTERPOLATION',
                'anchor_grade': 'GRAPH_BRACKETED_BETWEEN_ORIGINAL_EVENTS',
                'bracket_nodes': [a2, b2],
                'bracket_span_edges': span,
                'target_offset_edges': min(idx, span - idx),
                'unique_graph_path': True,
            }
            key = (span, row['target_offset_edges'])
            if best is None or key < best[0]:
                best = (key, row)
    return best[1] if best else None


def evaluate_prediction(origin_pred, destination_pred, records):
    out = {
        'q05_support_mass': 0.0,
        'q50_support_mass': 0.0,
        'q95_support_mass': 0.0,
        'exit_necessary_support_mass': 0.0,
        'q05_supported_cohorts': 0,
        'q50_supported_cohorts': 0,
        'q95_supported_cohorts': 0,
    }
    dep = origin_pred['time']
    arr = destination_pred['time']
    for r in records:
        mass = float(r['pressure_mass'])
        exit_ok = r['observed_exit'] is None or arr <= r['observed_exit']
        if exit_ok:
            out['exit_necessary_support_mass'] += mass
        if exit_ok and r['ready_q05'] is not None and dep >= r['ready_q05']:
            out['q05_support_mass'] += mass
            out['q05_supported_cohorts'] += 1
        if exit_ok and r['ready_q50'] is not None and dep >= r['ready_q50']:
            out['q50_support_mass'] += mass
            out['q50_supported_cohorts'] += 1
        if exit_ok and r['ready_q95'] is not None and dep >= r['ready_q95']:
            out['q95_support_mass'] += mass
            out['q95_supported_cohorts'] += 1
    return out


def candidate_roots(segrow):
    out = []
    for r in segrow.get('cross_variant_roots', []):
        if r not in out:
            out.append(r)
    for r in segrow.get('interior_roots', []):
        if r not in out:
            out.append(r)
    for x in segrow.get('nearby_roots', []):
        r = x.get('root')
        if r and r not in out:
            out.append(r)
    return out


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
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    apply_topology_patch(G, meta, load_patch(args.topology_patch))
    apply_gtxa_overlay(G, meta, code_to_nodes, load_overlay(args.gtxa_overlay))
    line_graph = audit.line_graphs(G, meta)

    payload = json.loads(Path(args.service_init).read_text(encoding='utf-8'))
    roots = build_root_variants(payload)
    join = json.loads(Path(args.evidence_join).read_text(encoding='utf-8'))
    residual = g1g.load_residual_segment_records(args.residual_cohorts)

    rows = []
    for segrow in join.get('segments', []):
        if segrow.get('proposal_tier') != 'EXISTING_ROOT_SUPPORT_PROPOSAL_AUDIT_ELIGIBLE':
            continue
        parsed = audit.parse_pressure_segment(segrow.get('segment'))
        if not parsed:
            continue
        line, origin, destination = parsed
        H = line_graph.get(line, nx.Graph())
        seg_path, seg_unique = unique_shortest_path(H, origin, destination)
        records = residual.get(segrow['segment'], [])
        pressure = float(segrow.get('pressure_mass', 0.0))
        variants = []
        for root in candidate_roots(segrow):
            for variant in roots.get(root, []):
                op = predict_target_strict(variant, origin, H)
                dp = predict_target_strict(variant, destination, H)
                if not op or not dp or op['time'] >= dp['time']:
                    continue
                ev = evaluate_prediction(op, dp, records)
                strong_anchor = op['anchor_grade'] in STRONG_GRADES and dp['anchor_grade'] in STRONG_GRADES
                exact_original_endpoint_count = int(op['anchor_grade'] == 'ORIGINAL_EXACT') + int(dp['anchor_grade'] == 'ORIGINAL_EXACT')
                row = {
                    'root': root,
                    'variant_id': variant['variant_id'],
                    'variant_evidence_class': variant['evidence_class'],
                    'origin_prediction': {**op, 'time': op['time'].isoformat()},
                    'destination_prediction': {**dp, 'time': dp['time'].isoformat()},
                    'strong_original_anchor_path': strong_anchor,
                    'exact_original_endpoint_count': exact_original_endpoint_count,
                    **ev,
                }
                key = (
                    int(strong_anchor),
                    ev['q50_support_mass'],
                    ev['q95_support_mass'],
                    exact_original_endpoint_count,
                    -max(op['bracket_span_edges'], dp['bracket_span_edges']),
                )
                variants.append((key, row))
        variants.sort(key=lambda x: x[0], reverse=True)
        best = variants[0][1] if variants else None
        best_strong = next((x[1] for x in variants if x[1]['strong_original_anchor_path']), None)
        strong_q50_frac = best_strong['q50_support_mass'] / pressure if best_strong and pressure > 0 else 0.0
        strong_q95_frac = best_strong['q95_support_mass'] / pressure if best_strong and pressure > 0 else 0.0
        distinct_od = int(segrow.get('distinct_od_count') or 0)
        strong_trial = bool(seg_unique and best_strong and strong_q50_frac >= 0.5 and distinct_od >= 2)
        rows.append({
            'segment': segrow['segment'],
            'line': line,
            'classification': segrow.get('direction_aware_classification'),
            'pressure_mass': pressure,
            'cohort_count': segrow.get('cohort_count'),
            'distinct_od_count': distinct_od,
            'cross_od_support': segrow.get('cross_od_support'),
            'segment_unique_shortest_path': seg_unique,
            'segment_graph_path_edges': (len(seg_path) - 1 if seg_path else None),
            'candidate_variant_count_with_two_endpoint_predictions': len(variants),
            'best_any_variant': best,
            'best_strong_variant': best_strong,
            'best_strong_q50_support_fraction': strong_q50_frac,
            'best_strong_q95_support_fraction': strong_q95_frac,
            'strong_bounded_repair_trial_eligible': strong_trial,
            'top_variants': [x[1] for x in variants[:20]],
        })

    rows.sort(key=lambda r: (int(r['strong_bounded_repair_trial_eligible']), r['best_strong_q50_support_fraction'], r['pressure_mass']), reverse=True)
    total = sum(r['pressure_mass'] for r in rows)
    strong_any = sum(r['pressure_mass'] for r in rows if r['best_strong_variant'] is not None)
    strong_q50_majority = sum(r['pressure_mass'] for r in rows if r['best_strong_q50_support_fraction'] >= 0.5)
    strong_multi_od = sum(r['pressure_mass'] for r in rows if r['strong_bounded_repair_trial_eligible'])
    eligible_segments = [r for r in rows if r['strong_bounded_repair_trial_eligible']]

    result = {
        'schema': 'mppd.r1b-g1g2-graph-anchor-existing-root-qualification.v1',
        'date': '2026-09-04',
        'status': 'G1G2_GRAPH_ANCHOR_EXISTING_ROOT_QUALIFICATION_COMPLETED_NO_SERVICE_MUTATION',
        'source_service_schema': payload.get('schema'),
        'source_evidence_join_schema': join.get('schema'),
        'scope': {
            'segment_count': len(rows),
            'eligible_pressure_mass_represented': total,
            'pressure_with_any_strong_original_anchor_variant': strong_any,
            'pressure_with_strong_q50_majority_support': strong_q50_majority,
            'strong_bounded_repair_trial_pressure_multi_od': strong_multi_od,
            'strong_bounded_repair_trial_segment_count': len(eligible_segments),
        },
        'strong_bounded_repair_trials': eligible_segments,
        'segments': rows,
        'qualification_rules': {
            'graph_path': 'TARGET_SEGMENT_MUST_HAVE_UNIQUE_SAME_LINE_SHORTEST_PATH',
            'endpoint_prediction': 'EXACT_ORIGINAL_EVENT_OR_UNIQUE_GRAPH_BRACKET_BETWEEN_TWO_ORIGINAL_EVENTS_ONLY',
            'forbidden_strong_anchor': [
                'SEQUENCE_ONLY_EXTRAPOLATION',
                'G1F_TERMINAL_EXTRAPOLATED_EVENT',
                'LATENT_COMPLETION_EVENT_WITHOUT_ORIGINAL_BRACKET',
                'NON_UNIQUE_GRAPH_PATH',
            ],
            'temporal_support': 'BEST_STRONG_VARIANT_Q50_SUPPORT_FRACTION_AT_LEAST_0_5',
            'network_support': 'DISTINCT_OD_COUNT_AT_LEAST_2',
        },
        'scientific_boundary': [
            'G1G2 performs no service mutation and creates no latent root.',
            'Only source-root original events or unique graph interpolation between two source-root original events can provide strong endpoint timing support.',
            'G1F terminal extrapolation and sequence-only extrapolation are explicitly excluded from strong repair qualification.',
            'A strong bounded-repair trial is still not observed service truth; it only authorizes a controlled observed-event-preserving repair trial followed by exact paired posterior rerun.',
            'If no candidate survives, the correct result is to retain the service-support residual and advance temporal/kernel inference rather than fabricate service support.',
        ],
        'next_gate': 'If strong bounded-repair trials exist, apply only those controlled existing-root repairs and exact-paired rerun; otherwise freeze service-support scope and advance R1C temporal/kernel inference while retaining strict-new-root residuals.',
        'no_email_notification_logic': True,
    }
    (outdir/'r1b_g1g2_graph_anchor_existing_root_qualification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'scope': result['scope'], 'strong_trials': eligible_segments[:10], 'top_segments': rows[:10]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
