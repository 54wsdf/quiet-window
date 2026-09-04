import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_g2v2_uncertain_service_full_network_posterior_20260904 as g2
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch


def parse_pressure_segment(value):
    parts = str(value).split('|')
    if len(parts) < 5:
        return None
    line = parts[0]
    origin = '|'.join(parts[1:3])
    destination = '|'.join(parts[3:5])
    return line, origin, destination


def line_graphs(G, meta):
    out = {}
    for line in sorted({m.get('line') for m in meta.values() if m.get('line')}):
        H = nx.Graph()
        H.add_nodes_from([n for n, m in meta.items() if m.get('line') == line])
        for u, v, d in G.edges(data=True):
            if u not in meta or v not in meta:
                continue
            if meta[u].get('line') != line or meta[v].get('line') != line:
                continue
            if str(d.get('kind', '')).startswith('transfer'):
                continue
            H.add_edge(u, v, **d)
        out[line] = H
    return out


def root_profiles(roots, meta):
    profiles = defaultdict(list)
    for (line, root), variants in roots.items():
        union_nodes = set()
        variant_nodes = []
        evidence = set()
        for v in variants:
            nodes = set(v.get('events', {}))
            union_nodes.update(nodes)
            variant_nodes.append(nodes)
            evidence.add(v.get('evidence_class', 'UNKNOWN'))
        seqs = [meta[n].get('seq') for n in union_nodes if n in meta and meta[n].get('seq') is not None]
        profiles[line].append({
            'root': root,
            'root_key': f'{line}||{root}',
            'union_nodes': union_nodes,
            'variant_nodes': variant_nodes,
            'evidence_classes': sorted(evidence),
            'seq_min': min(seqs) if seqs else None,
            'seq_max': max(seqs) if seqs else None,
        })
    return profiles


def safe_distance(H, a, b):
    if a not in H or b not in H:
        return None
    try:
        return nx.shortest_path_length(H, a, b)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def classify_segment(line, origin, destination, pressure, H, profiles, meta, nearby_edges):
    target_path_len = safe_distance(H, origin, destination)
    oseq = meta.get(origin, {}).get('seq')
    dseq = meta.get(destination, {}).get('seq')
    lo = min(oseq, dseq) if oseq is not None and dseq is not None else None
    hi = max(oseq, dseq) if oseq is not None and dseq is not None else None

    exact = []
    cross_variant = []
    interior = []
    nearby = []
    nearest = None
    nearest_root = None
    nearest_evidence = None

    for p in profiles:
        if any(origin in nodes and destination in nodes for nodes in p['variant_nodes']):
            exact.append(p['root_key'])
            continue
        if origin in p['union_nodes'] and destination in p['union_nodes']:
            cross_variant.append(p['root_key'])

        if lo is not None and p['seq_min'] is not None and p['seq_max'] is not None:
            if p['seq_min'] <= lo and hi <= p['seq_max']:
                interior.append(p['root_key'])

        distances = []
        for n in p['union_nodes']:
            do = safe_distance(H, n, origin)
            dd = safe_distance(H, n, destination)
            if do is not None:
                distances.append(do)
            if dd is not None:
                distances.append(dd)
        if distances:
            md = min(distances)
            if nearest is None or md < nearest:
                nearest = md
                nearest_root = p['root_key']
                nearest_evidence = p['evidence_classes']
            if md <= nearby_edges:
                nearby.append({'root': p['root_key'], 'distance_edges': md, 'evidence_classes': p['evidence_classes']})

    if exact:
        cls = 'SUPPORT_ALREADY_PRESENT_CONTRADICTION_AUDIT'
        action = 'AUDIT_RIDE_CACHE_OR_VARIANT_SELECTION_NO_SERVICE_CREATION'
    elif cross_variant:
        cls = 'ROOT_VARIANT_FRAGMENTATION_CANDIDATE'
        action = 'RECOMPLETE_OR_REWEIGHT_COMPETING_VARIANTS_NO_NEW_ROOT'
    elif interior:
        cls = 'EXISTING_ROOT_INTERIOR_HOLE_CANDIDATE'
        action = 'EVIDENCE_QUALIFY_TRAJECTORY_COMPLETION_WITHIN_EXISTING_ROOT'
    elif nearby:
        cls = 'NEAR_EXISTING_ROOT_EXTENSION_CANDIDATE'
        action = 'EVIDENCE_QUALIFY_BOUNDED_ROOT_EXTENSION_WITH_INCREASED_TIMING_UNCERTAINTY'
    else:
        cls = 'NO_NEARBY_EXISTING_ROOT_SUPPORT'
        action = 'REQUIRES_COHORT_LEVEL_TEMPORAL_AND_CROSS_OD_EVIDENCE_BEFORE_ANY_NEW_LATENT_ROOT_PROPOSAL'

    return {
        'line': line,
        'origin': origin,
        'destination': destination,
        'pressure_mass': float(pressure),
        'target_path_length_edges': target_path_len,
        'classification': cls,
        'recommended_action': action,
        'exact_support_roots': exact[:20],
        'cross_variant_roots': cross_variant[:20],
        'interior_roots': interior[:20],
        'nearby_roots': sorted(nearby, key=lambda x: x['distance_edges'])[:20],
        'nearest_root_distance_edges': nearest,
        'nearest_root': nearest_root,
        'nearest_root_evidence_classes': nearest_evidence,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p1c', required=True)
    ap.add_argument('--service-init', required=True)
    ap.add_argument('--r1b-summary', required=True)
    ap.add_argument('--topology-patch')
    ap.add_argument('--gtxa-overlay')
    ap.add_argument('--nearby-edges', type=int, default=3)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    if args.topology_patch:
        apply_topology_patch(G, meta, load_patch(args.topology_patch))
    if args.gtxa_overlay:
        apply_gtxa_overlay(G, meta, code_to_nodes, load_overlay(args.gtxa_overlay))

    roots, service_manifest, service_payload = g2.load_uncertain_service(args.service_init)
    lgraphs = line_graphs(G, meta)
    profiles = root_profiles(roots, meta)
    summary = json.loads(Path(args.r1b_summary).read_text(encoding='utf-8'))
    pressure_rows = (
        summary.get('iteration', {})
        .get('E1_passenger_posterior', {})
        .get('top_missing_service_pressure', [])
    )

    rows = []
    invalid = []
    for item in pressure_rows:
        parsed = parse_pressure_segment(item.get('segment'))
        if not parsed:
            invalid.append(item)
            continue
        line, origin, destination = parsed
        H = lgraphs.get(line, nx.Graph())
        row = classify_segment(
            line,
            origin,
            destination,
            item.get('pressure_mass', 0.0),
            H,
            profiles.get(line, []),
            meta,
            args.nearby_edges,
        )
        rows.append(row)

    rows.sort(key=lambda x: x['pressure_mass'], reverse=True)
    class_pressure = Counter()
    line_pressure = Counter()
    for r in rows:
        class_pressure[r['classification']] += r['pressure_mass']
        line_pressure[r['line']] += r['pressure_mass']

    result = {
        'schema': 'mppd.r1b-missing-service-support-proposal-audit.v1',
        'date': '2026-09-04',
        'status': 'R1B_MISSING_SERVICE_SUPPORT_PROPOSAL_AUDIT_COMPLETED',
        'scope': {
            'source_r1b_schema': summary.get('schema'),
            'source_scientific_unit': summary.get('scientific_unit'),
            'source_sample': summary.get('scope'),
            'pressure_row_count': len(pressure_rows),
            'classified_row_count': len(rows),
            'invalid_pressure_row_count': len(invalid),
            'nearby_root_threshold_edges': args.nearby_edges,
        },
        'service_input': {
            'schema': service_payload.get('schema'),
            'status': service_payload.get('status'),
            'manifest': service_manifest,
        },
        'pressure_by_classification': [
            {'classification': k, 'pressure_mass': v}
            for k, v in class_pressure.most_common()
        ],
        'pressure_by_line': [
            {'line': k, 'pressure_mass': v}
            for k, v in line_pressure.most_common()
        ],
        'top_segments': rows[:100],
        'proposal_gate': {
            'SUPPORT_ALREADY_PRESENT_CONTRADICTION_AUDIT': 'NO_CREATION_AUDIT_CACHE_OR_VARIANT_LOGIC',
            'ROOT_VARIANT_FRAGMENTATION_CANDIDATE': 'NO_NEW_ROOT_RECOMPLETE_OR_REWEIGHT_VARIANTS',
            'EXISTING_ROOT_INTERIOR_HOLE_CANDIDATE': 'CAN_ADVANCE_TO_EXISTING_ROOT_COMPLETION_QUALIFICATION',
            'NEAR_EXISTING_ROOT_EXTENSION_CANDIDATE': 'CAN_ADVANCE_TO_BOUNDED_EXTENSION_QUALIFICATION_WITH_UNCERTAINTY',
            'NO_NEARBY_EXISTING_ROOT_SUPPORT': 'CANNOT_CREATE_SERVICE_FROM_SPATIAL_PRESSURE_ALONE_NEEDS_TEMPORAL_CROSS_OD_EVIDENCE',
        },
        'scientific_boundary': [
            'Pressure mass is posterior-weighted diagnostic support from R1B and is not observed service truth.',
            'This audit classifies whether a missing segment can plausibly be repaired inside an existing latent service root; it does not activate or create any service event.',
            'A new latent service root is forbidden from spatial pressure alone. It requires cohort-level temporal windows, cross-OD consistency, trajectory continuity, and no contradiction with direct service anchors.',
            'Existing-root completion or extension remains an inferred hypothesis with increased timing uncertainty and must be re-evaluated by a new passenger posterior.',
            'Direct service anchors remain immutable observations; only latent support around them may be proposed where evidence permits.',
        ],
        'next_gate': 'Use cohort-level residual timing evidence to qualify only the highest-pressure existing-root completion/extension candidates; separately audit no-nearby-root segments before proposing any new latent service support.',
        'no_email_notification_logic': True,
    }
    (out / 'r1b_missing_service_support_proposal_audit_summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
