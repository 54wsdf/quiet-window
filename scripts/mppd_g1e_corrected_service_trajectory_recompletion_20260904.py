import argparse
import json
import time
from collections import Counter
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_g1c_full_network_service_trajectory_completion_20260904 as g1c
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch


STALE_COMPLETION_CLASS = 'SERVICE_TRAJECTORY_COMPLETION_HYPOTHESIS'
WEAK_LATTICE_CLASS = 'AFC_INFERRED_SERVICE_FIELD_WEAK_LATTICE_INITIALIZATION'


def corrected_line_graphs(G, meta):
    out = {}
    lines = sorted({m.get('line') for m in meta.values() if m.get('line')})
    for line in lines:
        H = nx.Graph()
        nodes = [n for n, m in meta.items() if m.get('line') == line]
        H.add_nodes_from(nodes)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p1c', required=True)
    ap.add_argument('--service-init', required=True)
    ap.add_argument('--topology-patch', required=True)
    ap.add_argument('--gtxa-overlay', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-variants-per-state', type=int, default=3)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    base_graph = {'nodes': G.number_of_nodes(), 'edges': G.number_of_edges()}
    topology_result = apply_topology_patch(G, meta, load_patch(args.topology_patch))
    after_patch = {'nodes': G.number_of_nodes(), 'edges': G.number_of_edges()}
    overlay_result = apply_gtxa_overlay(G, meta, code_to_nodes, load_overlay(args.gtxa_overlay))
    final_graph = {'nodes': G.number_of_nodes(), 'edges': G.number_of_edges()}

    payload = json.loads(Path(args.service_init).read_text(encoding='utf-8'))
    input_states = payload.get('states', [])
    retained_states = [s for s in input_states if s.get('evidence_class') != STALE_COMPLETION_CLASS]
    removed_stale = [s for s in input_states if s.get('evidence_class') == STALE_COMPLETION_CLASS]

    lgraphs = corrected_line_graphs(G, meta)
    corridors_by_line = {line: g1c.component_corridors(H, meta) for line, H in lgraphs.items()}
    line_runtime, global_runtime = g1c.line_runtime_stats(retained_states, meta)

    completions = []
    completion_by_root = Counter()
    completion_by_line = Counter()
    skipped = Counter()

    for st in retained_states:
        if st.get('evidence_class') == WEAK_LATTICE_CLASS:
            skipped['WEAK_LATTICE_ALREADY_SPATIAL_SUPPORT'] += 1
            continue
        obs = g1c.event_map(st)
        if len(obs) < 2:
            skipped['LT2_EVENTS'] += 1
            continue
        corridors = corridors_by_line.get(st.get('line'), [])
        scored = []
        for c in corridors:
            overlap, coverage = g1c.corridor_compatibility(c, set(obs))
            if overlap < 2:
                continue
            scored.append((coverage, overlap, c['length_nodes'], c))
        if not scored:
            skipped['NO_COMPATIBLE_CORRIDOR'] += 1
            continue
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best_cov = scored[0][0]
        selected = [x for x in scored if x[0] >= max(0.5, best_cov - 0.15)][:args.max_variants_per_state]
        per = float(line_runtime.get(st.get('line'), global_runtime))
        for vi, (_, _, _, corridor) in enumerate(selected):
            comp = g1c.complete_state(st, corridor, per, vi)
            if comp is None:
                continue
            comp['service_id'] = f"{st['service_id']}::G1E_CORR::{vi}"
            comp['root_service_id'] = st.get('root_service_id') or st['service_id']
            comp['evidence_class'] = STALE_COMPLETION_CLASS
            comp['recompletion_evidence_class'] = 'SERVICE_TRAJECTORY_RECOMPLETED_ON_CORRECTED_G0E_G0H_GRAPH'
            comp['source_graph_authority'] = 'G0E_PLUS_G0H_CORRECTED_FULL_NETWORK_GRAPH'
            for e in comp.get('station_events', []):
                if not e.get('is_original_event'):
                    e['event_evidence_class'] = 'SERVICE_TRAJECTORY_INTERPOLATED_ON_CORRECTED_GRAPH'
            completions.append(comp)
            completion_by_root[comp['root_service_id']] += 1
            completion_by_line[st.get('line')] += 1

    final_states = retained_states + completions
    manifest = {
        'input_schema': payload.get('schema'),
        'input_status': payload.get('status'),
        'input_state_count': len(input_states),
        'removed_stale_completion_hypothesis_count': len(removed_stale),
        'retained_noncompletion_state_count': len(retained_states),
        'new_corrected_completion_hypothesis_count': len(completions),
        'final_state_count': len(final_states),
        'roots_with_corrected_completion': len(completion_by_root),
        'completion_by_line': dict(sorted(completion_by_line.items())),
        'skipped': dict(skipped),
        'global_median_per_edge_runtime_sec': global_runtime,
        'line_median_per_edge_runtime_sec': dict(sorted(line_runtime.items())),
        'corridor_count_by_line': {k: len(v) for k, v in sorted(corridors_by_line.items())},
        'base_graph': base_graph,
        'after_topology_patch_graph': after_patch,
        'final_graph': final_graph,
        'topology_patch_inserted_edge_count': topology_result.get('inserted_edge_count') if isinstance(topology_result, dict) else None,
        'gtxa_overlay_schema': overlay_result.get('schema') if isinstance(overlay_result, dict) else None,
    }

    result_payload = {
        'schema': 'mppd.city-day-service-state-initialization.v4-corrected-graph-recompletion',
        'date': '2026-09-04',
        'status': 'FULL_NETWORK_SERVICE_STATE_INITIALIZATION_RECOMPLETED_ON_CORRECTED_G0E_G0H_GRAPH',
        'authority': '00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md',
        'city': payload.get('city', 'Seoul'),
        'business_date': payload.get('business_date', '2026-08-29'),
        'time_window': payload.get('time_window', '07:00-10:00'),
        'states': final_states,
        'manifest': manifest,
        'hard_boundaries': [
            'All non-completion input states are retained unchanged, including direct/inferred roots and G1D date-aware GTX-A weak component states.',
            'Old G1C completion hypotheses are removed because they were generated on the pre-G0E/G0H graph and are not active authority after topology/crosswalk correction.',
            'New completion hypotheses are regenerated on the corrected G0E+G0H full-network graph and remain low-confidence latent alternatives, never observed ATS.',
            'Original station events inside each retained root remain unchanged; only missing station events in completion variants are interpolated.',
            'Corrected same-line continuity edges and GTX-A date-aware topology may support completion geometry but never become observed service events.',
            'Weak lattice states are preserved rather than recompleted because they are already AFC-inferred wide spatial support and must remain explicitly weak.',
            'No new independent service root is created by this recompletion step.',
            'No passenger residual is used as a direct instruction to create or time a train in G1E.',
        ],
        'no_email_notification_logic': True,
    }

    service_out = outdir / 'seoul_20260829_0700_1000_service_state_initialization_v4_corrected_recompletion.json'
    service_out.write_text(json.dumps(result_payload, ensure_ascii=False), encoding='utf-8')

    summary = {
        'schema': 'mppd.g1e-corrected-service-trajectory-recompletion-summary.v1',
        'date': '2026-09-04',
        'status': 'G1E_CORRECTED_SERVICE_TRAJECTORY_RECOMPLETION_COMPLETED',
        'manifest': manifest,
        'performance': {'wall_sec': time.perf_counter() - wall0},
        'next_gate': 'Audit R1B posterior-weighted missing-service segments against this corrected recompletion before any new latent service root proposal; then rerun R1B E-step on the corrected service substrate.',
        'scientific_boundary': [
            'G1E is a representation-consistency repair between the corrected route graph and latent service completion graph.',
            'A support improvement after G1E demonstrates removal of stale graph-induced service fragmentation; it is not evidence that interpolated service events are observed truth.',
            'Any residual requiring genuinely new service support remains subject to cohort-level temporal and cross-OD evidence qualification.',
        ],
        'no_email_notification_logic': True,
    }
    (outdir / 'g1e_corrected_service_trajectory_recompletion_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
