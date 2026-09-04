import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0


def dt(s):
    return datetime.fromisoformat(s)


def line_graphs(G, meta):
    out = {}
    lines = sorted({m['line'] for m in meta.values()})
    for line in lines:
        H = nx.Graph()
        nodes = [n for n, m in meta.items() if m.get('line') == line]
        H.add_nodes_from(nodes)
        for u, v, d in G.edges(data=True):
            if u not in meta or v not in meta:
                continue
            if meta[u].get('line') != line or meta[v].get('line') != line:
                continue
            if d.get('kind') not in {'inline', 'inline_uncertain'}:
                continue
            H.add_edge(u, v, kind=d.get('kind'))
        out[line] = H
    return out


def component_corridors(H, meta):
    corridors = []
    seen = set()
    for ci, comp_nodes in enumerate(nx.connected_components(H)):
        C = H.subgraph(comp_nodes).copy()
        if len(C) < 2:
            continue
        terminals = sorted([n for n in C if C.degree(n) == 1])
        candidate_paths = []
        if len(terminals) >= 2:
            for a, b in combinations(terminals, 2):
                try:
                    p = nx.shortest_path(C, a, b)
                except nx.NetworkXNoPath:
                    continue
                candidate_paths.append(p)
        else:
            # Circular or ambiguity-closed component. Use sequence order as an
            # explicit low-confidence corridor hypothesis rather than pretending
            # there is a unique observed train path.
            seq_nodes = sorted(
                C.nodes,
                key=lambda n: (
                    meta[n].get('seq') if meta[n].get('seq') is not None else 10**9,
                    n,
                ),
            )
            if len(seq_nodes) >= 2:
                candidate_paths.append(seq_nodes)

        # Keep unique maximal-ish terminal paths. No station is removed from the
        # line graph; corridor hypotheses are alternatives, not spatial filters.
        candidate_paths.sort(key=len, reverse=True)
        for p in candidate_paths:
            sig = tuple(p)
            rsig = tuple(reversed(p))
            if sig in seen or rsig in seen:
                continue
            seen.add(sig)
            corridors.append({
                'component': ci,
                'nodes': p,
                'length_nodes': len(p),
                'terminal_u': p[0],
                'terminal_v': p[-1],
            })
    return corridors


def event_map(state):
    out = {}
    for e in state.get('station_events', []):
        arr = e.get('arrival') or e.get('time')
        dep = e.get('departure') or e.get('time')
        if not arr or not dep:
            continue
        out[e['node']] = {'arrival': dt(arr), 'departure': dt(dep), 'raw': e}
    return out


def line_runtime_stats(states, meta):
    vals = defaultdict(list)
    global_vals = []
    for st in states:
        if st.get('evidence_class') != 'PARTIAL_DIRECT_SERVICE_ANCHOR':
            continue
        line = st['line']
        ev = event_map(st)
        pts = []
        for n, x in ev.items():
            if n not in meta or meta[n].get('seq') is None:
                continue
            pts.append((int(meta[n]['seq']), x['departure']))
        pts.sort()
        for (s0, t0), (s1, t1) in zip(pts, pts[1:]):
            ds = abs(s1 - s0)
            if ds <= 0:
                continue
            per = abs((t1 - t0).total_seconds()) / ds
            if 30 <= per <= 300:
                vals[line].append(per)
                global_vals.append(per)
    gmed = statistics.median(global_vals) if global_vals else 121.0
    return {line: statistics.median(v) for line, v in vals.items() if v}, gmed


def corridor_positions(nodes):
    return {n: i for i, n in enumerate(nodes)}


def direction_sign(state, obs, pos):
    d = state.get('direction')
    if d == 'INC':
        return 1
    if d == 'DEC':
        return -1
    pairs = [(pos[n], x['departure'].timestamp()) for n, x in obs.items() if n in pos]
    if len(pairs) < 2:
        return 1
    pairs.sort()
    return 1 if pairs[-1][1] >= pairs[0][1] else -1


def corridor_compatibility(corridor, obs_nodes):
    cset = set(corridor['nodes'])
    overlap = len(cset & obs_nodes)
    coverage = overlap / len(obs_nodes) if obs_nodes else 0.0
    return overlap, coverage


def complete_state(state, corridor, per_edge_sec, variant_index):
    obs = event_map(state)
    nodes = corridor['nodes']
    pos = corridor_positions(nodes)
    obs_on = [(pos[n], x['departure'], n) for n, x in obs.items() if n in pos]
    if len(obs_on) < 2:
        return None
    sign = direction_sign(state, obs, pos)
    refpos = statistics.median([p for p, _, _ in obs_on])
    intercepts = [
        t - timedelta(seconds=sign * (p - refpos) * per_edge_sec)
        for p, t, _ in obs_on
    ]
    ref_time = datetime.fromtimestamp(statistics.median([x.timestamp() for x in intercepts]))

    observed_positions = [p for p, _, _ in obs_on]
    station_events = []
    interpolated = 0
    for n in nodes:
        p = pos[n]
        if n in obs:
            raw = obs[n]['raw']
            arr = raw.get('arrival') or raw.get('time')
            dep = raw.get('departure') or raw.get('time')
            station_events.append({
                'node': n,
                'arrival': arr,
                'departure': dep,
                'event_evidence_class': state.get('evidence_class'),
                'timing_uncertainty_sec': float(state.get('timing_uncertainty_sec') or 15.0),
                'is_original_event': True,
            })
            continue
        pred = ref_time + timedelta(seconds=sign * (p - refpos) * per_edge_sec)
        dist = min(abs(p - q) for q in observed_positions)
        unc = min(600.0, 30.0 + 25.0 * dist)
        station_events.append({
            'node': n,
            'arrival': pred.isoformat(),
            'departure': pred.isoformat(),
            'event_evidence_class': 'SERVICE_TRAJECTORY_INTERPOLATED_FROM_PARTIAL_STATE',
            'timing_uncertainty_sec': unc,
            'is_original_event': False,
        })
        interpolated += 1

    overlap, coverage = corridor_compatibility(corridor, set(obs))
    return {
        'service_id': f"{state['service_id']}::COMP::{variant_index}",
        'root_service_id': state['service_id'],
        'line': state['line'],
        'direction': state.get('direction'),
        'evidence_class': 'SERVICE_TRAJECTORY_COMPLETION_HYPOTHESIS',
        'root_evidence_class': state.get('evidence_class'),
        'timing_uncertainty_sec': max(
            float(state.get('timing_uncertainty_sec') or 15.0),
            max((e['timing_uncertainty_sec'] for e in station_events if not e['is_original_event']), default=0.0),
        ),
        'corridor': {
            'component': corridor['component'],
            'terminal_u': corridor['terminal_u'],
            'terminal_v': corridor['terminal_v'],
            'length_nodes': corridor['length_nodes'],
            'observed_overlap': overlap,
            'observed_coverage': coverage,
        },
        'per_edge_runtime_sec': per_edge_sec,
        'interpolated_event_count': interpolated,
        'station_events': station_events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p1c', required=True)
    ap.add_argument('--service-init', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-variants-per-state', type=int, default=3)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()
    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    base = json.loads(Path(args.service_init).read_text(encoding='utf-8'))
    states = base.get('states', [])
    lgraphs = line_graphs(G, meta)
    corridors_by_line = {line: component_corridors(H, meta) for line, H in lgraphs.items()}
    line_runtime, global_runtime = line_runtime_stats(states, meta)

    completions = []
    completion_by_root = Counter()
    completion_by_line = Counter()
    skipped = Counter()
    for st in states:
        if st.get('evidence_class') == 'AFC_INFERRED_SERVICE_FIELD_WEAK_LATTICE_INITIALIZATION':
            # These states already exist solely as wide spatial support.
            continue
        obs = event_map(st)
        if len(obs) < 2:
            skipped['LT2_EVENTS'] += 1
            continue
        corridors = corridors_by_line.get(st['line'], [])
        scored = []
        for c in corridors:
            overlap, coverage = corridor_compatibility(c, set(obs))
            if overlap < 2:
                continue
            scored.append((coverage, overlap, c['length_nodes'], c))
        if not scored:
            skipped['NO_COMPATIBLE_CORRIDOR'] += 1
            continue
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best_cov = scored[0][0]
        # Prefer corridors covering all observed nodes; otherwise retain a few
        # high-overlap alternatives rather than forcing one branch hypothesis.
        selected = [x for x in scored if x[0] >= max(0.5, best_cov - 0.15)][:args.max_variants_per_state]
        per = float(line_runtime.get(st['line'], global_runtime))
        for vi, (_, _, _, corridor) in enumerate(selected):
            comp = complete_state(st, corridor, per, vi)
            if comp is None:
                continue
            completions.append(comp)
            completion_by_root[st['service_id']] += 1
            completion_by_line[st['line']] += 1

    merged_states = list(states) + completions
    manifest = {
        'base_state_count': len(states),
        'completion_hypothesis_count': len(completions),
        'merged_state_count': len(merged_states),
        'roots_with_completion': len(completion_by_root),
        'completion_by_line': dict(sorted(completion_by_line.items())),
        'skipped': dict(skipped),
        'global_median_per_edge_runtime_sec': global_runtime,
        'line_median_per_edge_runtime_sec': dict(sorted(line_runtime.items())),
        'corridor_count_by_line': {k: len(v) for k, v in sorted(corridors_by_line.items())},
    }
    payload = {
        'schema': 'mppd.city-day-service-state-initialization.v2-trajectory-completion',
        'date': '2026-09-04',
        'status': 'FULL_NETWORK_SERVICE_STATE_INITIALIZATION_WITH_COMPLETION_HYPOTHESES',
        'authority': '00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md',
        'city': 'Seoul',
        'business_date': '2026-08-29',
        'time_window': '07:00-10:00',
        'states': merged_states,
        'manifest': manifest,
        'hard_boundaries': [
            'Original PARTIAL_DIRECT_SERVICE_ANCHOR and AFC_INFERRED_SERVICE_FIELD states are preserved unchanged.',
            'Completion hypotheses are separate low-confidence latent service alternatives; interpolated station events are never promoted to observed operational truth.',
            'Observed station events inside a completion hypothesis retain their original timestamps and evidence class.',
            'Missing station times are inferred along line-graph corridor hypotheses; branch ambiguity may create multiple variants sharing one root_service_id.',
            'G3 must treat completion variants as mutually competing/movable latent hypotheses and may remove them.',
            'No line, OD, segment, or transfer-count class is removed.'
        ],
        'no_email_notification_logic': True,
    }
    (outdir/'seoul_20260829_0700_1000_service_state_initialization_v2.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    summary = {
        'schema':'mppd.g1c-service-trajectory-completion-summary.v1',
        'status':'G1C_SERVICE_TRAJECTORY_COMPLETION_HYPOTHESES_COMPLETED',
        'manifest':manifest,
        'performance':{'wall_sec':time.perf_counter()-wall0},
        'next_gate':'Rerun cached full-network chain support using v2 service initialization. Qualify only support restoration; do not claim completed trajectories as realized service truth.',
        'no_email_notification_logic':True,
    }
    (outdir/'g1c_service_trajectory_completion_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
