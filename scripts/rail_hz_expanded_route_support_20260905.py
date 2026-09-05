from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

TRANSFER_RANKING_PENALTY_S = 300.0
DEFAULT_K = 32


def line_membership(line_paths: dict[str, dict[str, Any]]) -> tuple[dict[str, set[int]], dict[int, set[str]]]:
    by_line: dict[str, set[int]] = defaultdict(set)
    by_station: dict[int, set[str]] = defaultdict(set)
    for meta in line_paths.values():
        line = str(meta["afc_line"])
        for station in meta["nodes"]:
            s = int(station)
            by_line[line].add(s)
            by_station[s].add(line)
    return by_line, by_station


def build_graph(prior: dict[str, Any]) -> dict[tuple[int, str], list[tuple[tuple[int, str], float, dict[str, Any]]]]:
    graph: dict[tuple[int, str], list[tuple[tuple[int, str], float, dict[str, Any]]]] = defaultdict(list)
    _by_line, by_station = line_membership(prior["line_paths"])

    # Use the best available directed structural runtime for each station-line edge.
    edge_best: dict[tuple[int, str, int, str], tuple[float, dict[str, Any]]] = {}
    for e in prior["edges"]:
        lag = e.get("arrival_to_arrival_lag_sec_median")
        if lag is None:
            continue
        line = str(e["afc_line"])
        u, v = int(e["from_station"]), int(e["to_station"])
        key = (u, line, v, line)
        meta = {
            "kind": "RIDE",
            "line": line,
            "path_id": e["path_id"],
            "direction": e["direction"],
            "from_station": u,
            "to_station": v,
            "structural_runtime_s": float(lag),
        }
        old = edge_best.get(key)
        if old is None or float(lag) < old[0]:
            edge_best[key] = (float(lag), meta)

    for (u, line_u, v, line_v), (cost, meta) in edge_best.items():
        graph[(u, line_u)].append(((v, line_v), cost, meta))

    for station, lines in by_station.items():
        lines = sorted(lines)
        for a in lines:
            for b in lines:
                if a == b:
                    continue
                graph[(station, a)].append(((station, b), TRANSFER_RANKING_PENALTY_S, {
                    "kind": "TRANSFER",
                    "station": station,
                    "from_line": a,
                    "to_line": b,
                    "ranking_penalty_s": TRANSFER_RANKING_PENALTY_S,
                    "scientific_semantics": "candidate-ranking penalty only; transfer-time likelihood is inferred later by Theta_K",
                }))

    # Deterministic order makes support generation replayable.
    for state in graph:
        graph[state].sort(key=lambda x: (x[1], x[0][0], x[0][1], x[2]["kind"]))
    return graph


def shortest_path(
    graph: dict[tuple[int, str], list[tuple[tuple[int, str], float, dict[str, Any]]]],
    start: tuple[int, str],
    goal: tuple[int, str],
    banned_nodes: set[tuple[int, str]] | None = None,
    banned_edges: set[tuple[tuple[int, str], tuple[int, str]]] | None = None,
) -> tuple[float, list[tuple[int, str]], list[dict[str, Any]]] | None:
    banned_nodes = banned_nodes or set()
    banned_edges = banned_edges or set()
    if start in banned_nodes or goal in banned_nodes:
        return None
    heap: list[tuple[float, tuple[tuple[int, str], ...], tuple[int, str], list[dict[str, Any]]]] = [(0.0, (start,), start, [])]
    best: dict[tuple[int, str], float] = {start: 0.0}
    while heap:
        cost, path_tuple, u, edge_meta = heapq.heappop(heap)
        if cost > best.get(u, math.inf) + 1e-12:
            continue
        if u == goal:
            return cost, list(path_tuple), edge_meta
        for v, w, meta in graph.get(u, []):
            if v in banned_nodes or (u, v) in banned_edges:
                continue
            nc = cost + float(w)
            if nc + 1e-12 < best.get(v, math.inf):
                best[v] = nc
                heapq.heappush(heap, (nc, path_tuple + (v,), v, edge_meta + [meta]))
    return None


def path_cost(graph, states: list[tuple[int, str]]) -> tuple[float, list[dict[str, Any]]]:
    total = 0.0
    metas: list[dict[str, Any]] = []
    for u, v in zip(states, states[1:]):
        opts = [(w, m) for vv, w, m in graph.get(u, []) if vv == v]
        if not opts:
            raise KeyError((u, v))
        w, m = min(opts, key=lambda x: x[0])
        total += float(w)
        metas.append(m)
    return total, metas


def yen_k_shortest_simple_paths(graph, start, goal, k: int) -> list[tuple[float, list[tuple[int, str]], list[dict[str, Any]]]]:
    first = shortest_path(graph, start, goal)
    if first is None:
        return []
    accepted = [first]
    candidates: list[tuple[float, tuple[tuple[int, str], ...], list[dict[str, Any]]]] = []
    candidate_keys: set[tuple[tuple[int, str], ...]] = set()

    for _ in range(1, k):
        prev_cost, prev_states, _prev_meta = accepted[-1]
        del prev_cost
        for i in range(len(prev_states) - 1):
            spur = prev_states[i]
            root = prev_states[: i + 1]
            banned_edges: set[tuple[tuple[int, str], tuple[int, str]]] = set()
            for _c, states, _m in accepted:
                if len(states) > i and states[: i + 1] == root:
                    banned_edges.add((states[i], states[i + 1]))
            banned_nodes = set(root[:-1])
            spur_result = shortest_path(graph, spur, goal, banned_nodes=banned_nodes, banned_edges=banned_edges)
            if spur_result is None:
                continue
            _spur_cost, spur_states, _spur_meta = spur_result
            total_states = root[:-1] + spur_states
            key = tuple(total_states)
            if key in candidate_keys or any(tuple(x[1]) == key for x in accepted):
                continue
            total_cost, total_meta = path_cost(graph, total_states)
            heapq.heappush(candidates, (total_cost, key, total_meta))
            candidate_keys.add(key)
        if not candidates:
            break
        cost, key, meta = heapq.heappop(candidates)
        candidate_keys.discard(key)
        accepted.append((float(cost), list(key), meta))
    return accepted


def compress_candidate(cost: float, states: list[tuple[int, str]], edges: list[dict[str, Any]], rank: int) -> dict[str, Any]:
    transfers = [e for e in edges if e["kind"] == "TRANSFER"]
    rides = [e for e in edges if e["kind"] == "RIDE"]
    line_sequence: list[str] = []
    for _station, line in states:
        if not line_sequence or line_sequence[-1] != line:
            line_sequence.append(line)
    physical_path: list[int] = []
    for station, _line in states:
        if not physical_path or physical_path[-1] != station:
            physical_path.append(station)
    return {
        "rank": rank,
        "base_ranking_cost_s": float(cost),
        "state_path": [{"station": s, "line": l} for s, l in states],
        "physical_station_path": physical_path,
        "line_sequence": line_sequence,
        "transfer_count": len(transfers),
        "transfer_movements": [f"{e['from_line']}:{e['station']}->{e['to_line']}:{e['station']}" for e in transfers],
        "ride_segments": rides,
        "simple_state_path": len(states) == len(set(states)),
    }


def build_support(prior_path: Path, output: Path, k: int) -> dict[str, Any]:
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    graph = build_graph(prior)
    by_line, by_station = line_membership(prior["line_paths"])
    states = sorted(graph)
    support: dict[str, list[dict[str, Any]]] = {}
    unreachable: list[str] = []
    candidate_count = 0
    max_transfer = 0
    pairs_with_k = 0

    # Generate only AFC-observable endpoint surfaces: line must actually serve the station.
    endpoints = sorted((station, line) for station, lines in by_station.items() for line in lines)
    for start in endpoints:
        for goal in endpoints:
            if start == goal:
                continue
            key = f"{start[1]}:{start[0]}->{goal[1]}:{goal[0]}"
            paths = yen_k_shortest_simple_paths(graph, start, goal, k)
            if not paths:
                unreachable.append(key)
                continue
            payload = [compress_candidate(c, s, e, i) for i, (c, s, e) in enumerate(paths, start=1)]
            support[key] = payload
            candidate_count += len(payload)
            max_transfer = max(max_transfer, max(p["transfer_count"] for p in payload))
            pairs_with_k += int(len(payload) == k)

    gates = {
        "all_endpoint_surfaces_reachable": len(unreachable) == 0,
        "no_transfer_count_filter": True,
        "line_aware_expanded_state_graph": True,
        "absolute_auxiliary_timestamp_not_used": prior["absolute_timestamp_policy"] == "FORBIDDEN_AS_2019_REALIZED_TRUTH",
        "at_least_one_multitransfer_candidate_exists": max_transfer >= 2,
    }
    result = {
        "schema": "rail.hz-expanded-route-support.v1",
        "dataset_id": "CN_HZ_Tianchi_2019",
        "status": "QUALIFIED_LINE_AWARE_ROUTE_SUPPORT" if all(gates.values()) else "ROUTE_SUPPORT_GATE_FAILED",
        "endpoint_surface_count": len(endpoints),
        "expanded_state_count": len(states),
        "route_pair_count": len(support),
        "candidate_count": candidate_count,
        "k_shortest_support_beam": k,
        "pairs_reaching_k_beam": pairs_with_k,
        "max_transfer_count_present": max_transfer,
        "unreachable_endpoint_pairs": unreachable,
        "integrity_gates": gates,
        "scientific_boundary": {
            "transfer_count_cap": None,
            "k_path_beam_is_computational_support_approximation": True,
            "routes_outside_k_are_not_claimed_empirically_zero": True,
            "non_simple_behavioral_support": "DEFERRED_TO_CONTROLLED_SIDECAR_BEFORE_BEHAVIORAL_INVARIANT_CLAIMS",
            "transfer_penalty_role": "RANKING_ONLY_NOT_THETA_K",
        },
        "line_membership": {line: sorted(nodes) for line, nodes in sorted(by_line.items())},
        "route_support": support,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "endpoint_surface_count": len(endpoints),
        "expanded_state_count": len(states),
        "route_pair_count": len(support),
        "candidate_count": candidate_count,
        "max_transfer_count_present": max_transfer,
        "pairs_reaching_k_beam": pairs_with_k,
        "integrity_gates": gates,
    }, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)
    return result


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    args=p.parse_args()
    build_support(args.prior,args.output,args.k)


if __name__=="__main__":
    main()
