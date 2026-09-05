from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

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

    # A shared physical edge may belong to more than one service family. Keep every
    # compatible family instead of selecting one path_id prematurely.
    edge_options: dict[tuple[int, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for e in prior["edges"]:
        lag = e.get("arrival_to_arrival_lag_sec_median")
        if lag is None:
            continue
        line = str(e["afc_line"])
        u, v = int(e["from_station"]), int(e["to_station"])
        edge_options[(u, line, v, line)].append({
            "path_id": str(e["path_id"]),
            "direction": str(e["direction"]),
            "structural_runtime_s": float(lag),
        })

    for (u, line_u, v, line_v), options in edge_options.items():
        options = sorted(options, key=lambda x: (x["structural_runtime_s"], x["path_id"], x["direction"]))
        ranking_cost = min(x["structural_runtime_s"] for x in options)
        graph[(u, line_u)].append(((v, line_v), ranking_cost, {
            "kind": "RIDE",
            "line": line_u,
            "from_station": u,
            "to_station": v,
            "ranking_runtime_s": ranking_cost,
            "service_options": options,
        }))

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

    for state in graph:
        graph[state].sort(key=lambda x: (x[1], x[0][0], x[0][1], x[2]["kind"]))
    return graph


def shortest_path(graph, start, goal, banned_nodes=None, banned_edges=None):
    banned_nodes = banned_nodes or set()
    banned_edges = banned_edges or set()
    if start in banned_nodes or goal in banned_nodes:
        return None
    heap = [(0.0, (start,), start, [])]
    best = {start: 0.0}
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


def path_cost(graph, states):
    total = 0.0
    metas = []
    for u, v in zip(states, states[1:]):
        opts = [(w, m) for vv, w, m in graph.get(u, []) if vv == v]
        if not opts:
            raise KeyError((u, v))
        w, m = min(opts, key=lambda x: x[0])
        total += float(w)
        metas.append(m)
    return total, metas


def yen_k_shortest_simple_paths(graph, start, goal, k):
    first = shortest_path(graph, start, goal)
    if first is None:
        return []
    accepted = [first]
    candidates = []
    candidate_keys = set()
    for _ in range(1, k):
        _prev_cost, prev_states, _prev_meta = accepted[-1]
        for i in range(len(prev_states) - 1):
            spur = prev_states[i]
            root = prev_states[: i + 1]
            banned_edges = set()
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


def ride_legs(edges: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    legs = []
    current = []
    for e in edges + [{"kind": "END"}]:
        if e["kind"] == "RIDE":
            current.append(e)
            continue
        if current:
            compatible = None
            runtime_by_option: dict[tuple[str, str], float] = defaultdict(float)
            for edge in current:
                opts = {(x["path_id"], x["direction"]) for x in edge["service_options"]}
                compatible = opts if compatible is None else compatible & opts
            if not compatible:
                return None
            for option in compatible:
                for edge in current:
                    matches = [x for x in edge["service_options"] if (x["path_id"], x["direction"]) == option]
                    if not matches:
                        return None
                    runtime_by_option[option] += min(float(x["structural_runtime_s"]) for x in matches)
            legs.append({
                "line": current[0]["line"],
                "from_station": current[0]["from_station"],
                "to_station": current[-1]["to_station"],
                "edge_count": len(current),
                "compatible_service_options": [
                    {"path_id": p, "direction": d, "structural_runtime_s": runtime_by_option[(p, d)]}
                    for p, d in sorted(compatible)
                ],
            })
            current = []
    return legs


def compress_candidate(cost, states, edges, rank):
    transfers = [e for e in edges if e["kind"] == "TRANSFER"]
    rides = [e for e in edges if e["kind"] == "RIDE"]
    legs = ride_legs(edges)
    if legs is None:
        return None
    line_sequence = []
    for _station, line in states:
        if not line_sequence or line_sequence[-1] != line:
            line_sequence.append(line)
    physical_path = []
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
        "ride_legs": legs,
        "simple_state_path": len(states) == len(set(states)),
        "service_family_not_prematurely_resolved": True,
    }


def build_support(prior_path: Path, output: Path, k: int) -> dict[str, Any]:
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    graph = build_graph(prior)
    by_line, by_station = line_membership(prior["line_paths"])
    states = sorted(graph)
    support = {}
    unreachable = []
    candidate_count = 0
    incoherent_filtered = 0
    max_transfer = 0
    pairs_reaching_k = 0
    endpoints = sorted((station, line) for station, lines in by_station.items() for line in lines)
    for start in endpoints:
        for goal in endpoints:
            if start == goal:
                continue
            key = f"{start[1]}:{start[0]}->{goal[1]}:{goal[0]}"
            raw_paths = yen_k_shortest_simple_paths(graph, start, goal, k)
            payload = []
            for raw_rank, (c, s, e) in enumerate(raw_paths, start=1):
                item = compress_candidate(c, s, e, raw_rank)
                if item is None:
                    incoherent_filtered += 1
                    continue
                item["rank"] = len(payload) + 1
                payload.append(item)
            if not payload:
                unreachable.append(key)
                continue
            support[key] = payload
            candidate_count += len(payload)
            max_transfer = max(max_transfer, max(p["transfer_count"] for p in payload))
            pairs_reaching_k += int(len(payload) == k)

    gates = {
        "all_endpoint_surfaces_reachable": len(unreachable) == 0,
        "no_transfer_count_filter": True,
        "line_aware_expanded_state_graph": True,
        "shared_edges_retain_all_service_families": True,
        "all_retained_ride_legs_have_coherent_service_options": all(
            all(leg["compatible_service_options"] for cand in cands for leg in cand["ride_legs"])
            for cands in support.values()
        ),
        "absolute_auxiliary_timestamp_not_used": prior["absolute_timestamp_policy"] == "FORBIDDEN_AS_2019_REALIZED_TRUTH",
        "at_least_one_multitransfer_candidate_exists": max_transfer >= 2,
    }
    result = {
        "schema": "rail.hz-expanded-route-support.v2-service-family-aware",
        "dataset_id": "CN_HZ_Tianchi_2019",
        "status": "QUALIFIED_LINE_AWARE_ROUTE_SUPPORT" if all(gates.values()) else "ROUTE_SUPPORT_GATE_FAILED",
        "endpoint_surface_count": len(endpoints),
        "expanded_state_count": len(states),
        "route_pair_count": len(support),
        "candidate_count": candidate_count,
        "incoherent_raw_paths_filtered": incoherent_filtered,
        "k_shortest_support_beam": k,
        "pairs_reaching_k_beam": pairs_reaching_k,
        "max_transfer_count_present": max_transfer,
        "unreachable_endpoint_pairs": unreachable,
        "integrity_gates": gates,
        "scientific_boundary": {
            "transfer_count_cap": None,
            "k_path_beam_is_computational_support_approximation": True,
            "routes_outside_k_are_not_claimed_empirically_zero": True,
            "non_simple_behavioral_support": "DEFERRED_TO_CONTROLLED_SIDECAR_BEFORE_BEHAVIORAL_INVARIANT_CLAIMS",
            "transfer_penalty_role": "RANKING_ONLY_NOT_THETA_K",
            "shared_service_family_ambiguity": "RETAINED_UNTIL_PASSENGER_SERVICE_POSTERIOR_UPDATE",
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
        "incoherent_raw_paths_filtered": incoherent_filtered,
        "max_transfer_count_present": max_transfer,
        "pairs_reaching_k_beam": pairs_reaching_k,
        "integrity_gates": gates,
    }, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    a = p.parse_args()
    build_support(a.prior, a.output, a.k)


if __name__ == "__main__":
    main()
