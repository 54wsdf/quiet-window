import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timedelta
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


def event_time(ev):
    return ev['departure'] if ev.get('departure') is not None else ev['arrival']


def observed_order(state):
    obs = g1c.event_map(state)
    rows = [(event_time(v), n) for n, v in obs.items()]
    rows.sort(key=lambda x: (x[0], x[1]))
    return [n for _, n in rows], obs


def orient_full_corridor(nodes, ordered_obs):
    pos = {n: i for i, n in enumerate(nodes)}
    if not set(ordered_obs).issubset(pos):
        return None
    idx = [pos[n] for n in ordered_obs]
    if all(a < b for a, b in zip(idx, idx[1:])):
        return list(nodes)
    if all(a > b for a, b in zip(idx, idx[1:])):
        return list(reversed(nodes))
    return None


def capped_all_shortest_paths(H, a, b, cap=3):
    if a == b:
        return [[a]]
    try:
        gen = nx.all_shortest_paths(H, a, b)
        out = []
        for p in gen:
            out.append(list(p))
            if len(out) >= cap:
                break
        return out
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def all_observed_path_variants(H, ordered_obs, max_variants=3, pair_cap=3):
    beams = [[]]
    for a, b in zip(ordered_obs, ordered_obs[1:]):
        pair_paths = capped_all_shortest_paths(H, a, b, cap=pair_cap)
        if not pair_paths:
            return []
        nxt = []
        for prefix in beams:
            for p in pair_paths:
                cand = list(p) if not prefix else prefix + p[1:]
                if len(cand) != len(set(cand)):
                    continue
                nxt.append(cand)
        if not nxt:
            return []
        seen = set()
        dedup = []
        for p in sorted(nxt, key=lambda x: (len(x), tuple(x))):
            sig = tuple(p)
            if sig in seen:
                continue
            seen.add(sig)
            dedup.append(p)
            if len(dedup) >= max_variants:
                break
        beams = dedup
    if len(ordered_obs) == 1:
        return [[ordered_obs[0]]]
    return beams[:max_variants]


def edge_distance(H, a, b):
    d = H.get_edge_data(a, b) or {}
    try:
        w = float(d.get('weight', 1.0))
        return max(1.0, w)
    except Exception:
        return 1.0


def cumulative_distances(H, nodes):
    out = [0.0]
    for a, b in zip(nodes, nodes[1:]):
        out.append(out[-1] + edge_distance(H, a, b))
    return out


def complete_path_preserving(state, nodes, H, per_edge_sec, variant_index, mode):
    ordered_obs, obs = observed_order(state)
    if len(obs) < 2 or not set(obs).issubset(set(nodes)):
        return None
    pos = {n: i for i, n in enumerate(nodes)}
    obs_idx = [pos[n] for n in ordered_obs]
    if not all(a < b for a, b in zip(obs_idx, obs_idx[1:])):
        return None

    cum = cumulative_distances(H, nodes)
    obs_points = []
    for n in ordered_obs:
        i = pos[n]
        obs_points.append((i, cum[i], event_time(obs[n]), n))

    base_unc = float(state.get('timing_uncertainty_sec') or 15.0)
    station_events = []
    interpolated = 0
    for i, n in enumerate(nodes):
        if n in obs:
            raw = obs[n]['raw']
            arr = raw.get('arrival') or raw.get('time')
            dep = raw.get('departure') or raw.get('time')
            station_events.append({
                'node': n,
                'arrival': arr,
                'departure': dep,
                'event_evidence_class': state.get('evidence_class'),
                'timing_uncertainty_sec': base_unc,
                'is_original_event': True,
            })
            continue

        left = None
        right = None
        for p in obs_points:
            if p[0] < i:
                left = p
            if p[0] > i:
                right = p
                break

        if left is not None and right is not None:
            denom = right[1] - left[1]
            frac = 0.5 if denom <= 1e-9 else (cum[i] - left[1]) / denom
            dtsec = (right[2] - left[2]).total_seconds()
            pred = left[2] + timedelta(seconds=frac * dtsec)
            dist = min(cum[i] - left[1], right[1] - cum[i])
            interpolation_role = 'BETWEEN_OBSERVED_ANCHORS'
        elif left is not None:
            dist = cum[i] - left[1]
            pred = left[2] + timedelta(seconds=dist * per_edge_sec)
            interpolation_role = 'EXTRAPOLATED_AFTER_LAST_OBSERVED'
        elif right is not None:
            dist = right[1] - cum[i]
            pred = right[2] - timedelta(seconds=dist * per_edge_sec)
            interpolation_role = 'EXTRAPOLATED_BEFORE_FIRST_OBSERVED'
        else:
            return None

        unc = min(600.0, max(base_unc, 30.0 + 25.0 * float(dist)))
        station_events.append({
            'node': n,
            'arrival': pred.isoformat(),
            'departure': pred.isoformat(),
            'event_evidence_class': 'SERVICE_TRAJECTORY_INTERPOLATED_ON_CORRECTED_GRAPH_G1F',
            'timing_uncertainty_sec': unc,
            'is_original_event': False,
            'interpolation_role': interpolation_role,
        })
        interpolated += 1

    original_nodes_out = {e['node'] for e in station_events if e.get('is_original_event')}
    if original_nodes_out != set(obs):
        return None

    # Strong value-preservation check: every original event must retain the exact
    # arrival/departure strings from the retained root.
    by_node = {e['node']: e for e in station_events if e.get('is_original_event')}
    for n, x in obs.items():
        raw = x['raw']
        expected_arr = raw.get('arrival') or raw.get('time')
        expected_dep = raw.get('departure') or raw.get('time')
        if by_node[n]['arrival'] != expected_arr or by_node[n]['departure'] != expected_dep:
            return None

    return {
        'service_id': f"{state['service_id']}::G1F::{variant_index}",
        'root_service_id': state.get('root_service_id') or state['service_id'],
        'line': state['line'],
        'direction': state.get('direction'),
        'evidence_class': STALE_COMPLETION_CLASS,
        'root_evidence_class': state.get('evidence_class'),
        'recompletion_evidence_class': 'OBSERVED_EVENT_PRESERVING_SERVICE_RECOMPLETION_ON_CORRECTED_G0E_G0H_GRAPH',
        'source_graph_authority': 'G0E_PLUS_G0H_CORRECTED_FULL_NETWORK_GRAPH',
        'completion_mode': mode,
        'timing_uncertainty_sec': max(
            base_unc,
            max((float(e['timing_uncertainty_sec']) for e in station_events if not e.get('is_original_event')), default=0.0),
        ),
        'path': {
            'length_nodes': len(nodes),
            'terminal_u': nodes[0],
            'terminal_v': nodes[-1],
            'observed_event_count': len(obs),
            'observed_event_preservation_pass': True,
        },
        'per_edge_runtime_sec': per_edge_sec,
        'interpolated_event_count': interpolated,
        'station_events': station_events,
    }


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
    completion_mode = Counter()
    skipped = Counter()
    preservation_failures = []

    for st in retained_states:
        if st.get('evidence_class') == WEAK_LATTICE_CLASS:
            skipped['WEAK_LATTICE_ALREADY_SPATIAL_SUPPORT'] += 1
            continue
        ordered_obs, obs = observed_order(st)
        if len(obs) < 2:
            skipped['LT2_EVENTS'] += 1
            continue
        line = st.get('line')
        H = lgraphs.get(line)
        if H is None:
            skipped['NO_LINE_GRAPH'] += 1
            continue
        per = float(line_runtime.get(line, global_runtime))

        path_variants = []
        for c in corridors_by_line.get(line, []):
            oriented = orient_full_corridor(c['nodes'], ordered_obs)
            if oriented is None:
                continue
            path_variants.append(('FULL_CORRIDOR_ALL_OBSERVED_PRESERVED', oriented))
            if len(path_variants) >= args.max_variants_per_state:
                break

        if not path_variants:
            fallback = all_observed_path_variants(
                H, ordered_obs, max_variants=args.max_variants_per_state, pair_cap=args.max_variants_per_state
            )
            for p in fallback:
                path_variants.append(('OBSERVED_ORDER_CORE_PATH_NO_TERMINAL_EXTRAPOLATION', p))

        if not path_variants:
            skipped['NO_ALL_OBSERVED_CONNECTED_ORDERED_PATH'] += 1
            continue

        produced = 0
        for vi, (mode, nodes) in enumerate(path_variants[:args.max_variants_per_state]):
            comp = complete_path_preserving(st, nodes, H, per, vi, mode)
            if comp is None:
                preservation_failures.append({
                    'service_id': st.get('service_id'),
                    'root_service_id': st.get('root_service_id') or st.get('service_id'),
                    'line': line,
                    'mode': mode,
                    'observed_nodes': sorted(obs),
                    'path_nodes': nodes,
                })
                continue
            completions.append(comp)
            produced += 1
            completion_by_root[comp['root_service_id']] += 1
            completion_by_line[line] += 1
            completion_mode[mode] += 1
        if produced == 0:
            skipped['ALL_CANDIDATE_COMPLETIONS_FAILED_PRESERVATION_CHECK'] += 1

    # Global invariant check over every completion that will be emitted.
    retained_by_root = {}
    for st in retained_states:
        retained_by_root[st.get('root_service_id') or st.get('service_id')] = st
    invariant_violations = []
    for comp in completions:
        root = comp['root_service_id']
        src = retained_by_root.get(root)
        if src is None:
            invariant_violations.append({'root': root, 'reason': 'MISSING_SOURCE_ROOT'})
            continue
        src_obs = g1c.event_map(src)
        out_original = {e['node']: e for e in comp.get('station_events', []) if e.get('is_original_event')}
        if set(out_original) != set(src_obs):
            invariant_violations.append({
                'root': root,
                'reason': 'ORIGINAL_EVENT_NODE_SET_CHANGED',
                'source_nodes': sorted(src_obs),
                'output_original_nodes': sorted(out_original),
            })
            continue
        for n, x in src_obs.items():
            raw = x['raw']
            ea = raw.get('arrival') or raw.get('time')
            ed = raw.get('departure') or raw.get('time')
            if out_original[n].get('arrival') != ea or out_original[n].get('departure') != ed:
                invariant_violations.append({'root': root, 'reason': 'ORIGINAL_EVENT_TIME_CHANGED', 'node': n})
                break

    if invariant_violations:
        raise RuntimeError(f'observed-event preservation invariant failed for {len(invariant_violations)} completions')

    final_states = retained_states + completions
    manifest = {
        'input_schema': payload.get('schema'),
        'input_status': payload.get('status'),
        'input_state_count': len(input_states),
        'removed_stale_completion_hypothesis_count': len(removed_stale),
        'retained_noncompletion_state_count': len(retained_states),
        'new_observed_event_preserving_completion_count': len(completions),
        'final_state_count': len(final_states),
        'roots_with_completion': len(completion_by_root),
        'completion_by_line': dict(sorted(completion_by_line.items())),
        'completion_mode': dict(completion_mode),
        'skipped': dict(skipped),
        'candidate_preservation_failure_count': len(preservation_failures),
        'emitted_invariant_violation_count': len(invariant_violations),
        'observed_event_preservation_global_pass': len(invariant_violations) == 0,
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
        'schema': 'mppd.city-day-service-state-initialization.v5-observed-event-preserving-corrected-recompletion',
        'date': '2026-09-04',
        'status': 'FULL_NETWORK_SERVICE_STATE_RECOMPLETED_WITH_OBSERVED_EVENT_PRESERVATION_ON_CORRECTED_GRAPH',
        'authority': '00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md',
        'city': payload.get('city', 'Seoul'),
        'business_date': payload.get('business_date', '2026-08-29'),
        'time_window': payload.get('time_window', '07:00-10:00'),
        'states': final_states,
        'manifest': manifest,
        'hard_boundaries': [
            'Every emitted completion hypothesis contains every original station event from its source root with the exact original arrival/departure values.',
            'Completion hypotheses may add latent station events but may never delete, replace, or retime original source-root events.',
            'A full corrected-graph corridor is used only when it contains all original events in their observed temporal order.',
            'If no full corridor preserves all observed events, G1F uses only a corrected-graph shortest path connecting all original events in observed time order; that fallback does not extrapolate to unobserved terminals.',
            'Fallback shortest-path ambiguity is retained up to max-variants-per-state and remains low-confidence latent geometry, not observed train path truth.',
            'Weak lattice states are retained unchanged and are not recompleted.',
            'No independent service root is created by G1F.',
            'No passenger residual directly creates or times a train in G1F.',
            'Direct and inferred source-root evidence classes remain provenance authority for original events.',
        ],
        'no_email_notification_logic': True,
    }

    service_out = outdir / 'seoul_20260829_0700_1000_service_state_initialization_v5_observed_event_preserving_recompletion.json'
    service_out.write_text(json.dumps(result_payload, ensure_ascii=False), encoding='utf-8')

    summary = {
        'schema': 'mppd.g1f-observed-event-preserving-service-recompletion-summary.v1',
        'date': '2026-09-04',
        'status': 'G1F_OBSERVED_EVENT_PRESERVING_SERVICE_RECOMPLETION_COMPLETED',
        'manifest': manifest,
        'candidate_preservation_failures_preview': preservation_failures[:50],
        'performance': {'wall_sec': time.perf_counter() - wall0},
        'next_gate': 'Paired R1B weighted-residual micro/broad rerun on G1F v5 substrate; only after that classify remaining missing-service support.',
        'scientific_boundary': [
            'G1F fixes an implementation/representation invariant: completion cannot discard source-root events.',
            'A posterior improvement after G1F is evidence that prior service completion geometry was internally inconsistent, not evidence that newly interpolated events are observed ATS.',
            'Core-path fallback support is intentionally narrower than speculative full-terminal extension when a full all-observed corridor cannot be established.',
            'Any remaining genuine service-support gap still requires passenger temporal, cross-OD, trajectory, and direct-anchor consistency evidence.',
        ],
        'no_email_notification_logic': True,
    }
    (outdir / 'g1f_observed_event_preserving_service_recompletion_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
