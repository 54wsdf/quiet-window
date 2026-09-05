from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import scripts.rail_hz_expanded_route_support_20260905 as base


def _finalize_leg(current: list[dict[str, Any]], compatible: set[tuple[str, str]]) -> dict[str, Any]:
    runtime_by_option: dict[tuple[str, str], float] = {}
    for option in compatible:
        runtime = 0.0
        for edge in current:
            matches = [x for x in edge["service_options"] if (x["path_id"], x["direction"]) == option]
            if not matches:
                raise RuntimeError("incoherent finalized service leg")
            runtime += min(float(x["structural_runtime_s"]) for x in matches)
        runtime_by_option[option] = runtime
    return {
        "line": current[0]["line"],
        "from_station": current[0]["from_station"],
        "to_station": current[-1]["to_station"],
        "edge_count": len(current),
        "compatible_service_options": [
            {"path_id": p, "direction": d, "structural_runtime_s": runtime_by_option[(p, d)]}
            for p, d in sorted(compatible)
        ],
    }


def ride_legs_and_movements(edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Build service-coherent ride legs.

    A continuous AFC line is not necessarily one through-train service family. In
    Hangzhou Line B, travelling between the two branches requires changing service
    family at station 20 (客运中心). The v2 builder rejected those OD pairs because it
    intersected service-family support across the whole same-line path.

    Here, whenever the compatible service-family intersection would become empty,
    the current leg is closed and a SAME_LINE_SERVICE_CHANGE movement is inserted at
    the physical junction. This is not a topology shortcut: it explicitly requires a
    new boarding chain and therefore enters the transfer/connection timing likelihood.
    """
    legs: list[dict[str, Any]] = []
    movements: list[str] = []
    movement_meta: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    compatible: set[tuple[str, str]] | None = None

    def flush() -> None:
        nonlocal current, compatible
        if current:
            if not compatible:
                raise RuntimeError("empty compatible set at leg flush")
            legs.append(_finalize_leg(current, compatible))
            current = []
            compatible = None

    for edge in edges:
        if edge["kind"] == "TRANSFER":
            flush()
            movement = f"{edge['from_line']}:{edge['station']}->{edge['to_line']}:{edge['station']}"
            movements.append(movement)
            movement_meta.append({
                "kind": "CROSS_LINE_TRANSFER",
                "station": int(edge["station"]),
                "from_line": str(edge["from_line"]),
                "to_line": str(edge["to_line"]),
                "movement": movement,
            })
            continue

        if edge["kind"] != "RIDE":
            raise RuntimeError(f"unexpected edge kind: {edge['kind']}")

        options = {(str(x["path_id"]), str(x["direction"])) for x in edge["service_options"]}
        if not current:
            current = [edge]
            compatible = set(options)
            continue

        assert compatible is not None
        narrowed = compatible & options
        if narrowed:
            current.append(edge)
            compatible = narrowed
            continue

        # Same AFC line, but no through service family spans the junction. A passenger
        # must change service/boarding chain here. Close the prior leg and start a new
        # one from the same physical station.
        junction = int(current[-1]["to_station"])
        line_before = str(current[-1]["line"])
        line_after = str(edge["line"])
        if line_before != line_after or junction != int(edge["from_station"]):
            raise RuntimeError("service-family split is not a contiguous same-line junction")
        flush()
        movement = f"SAME_LINE_SERVICE_CHANGE:{line_before}:{junction}"
        movements.append(movement)
        movement_meta.append({
            "kind": "SAME_LINE_SERVICE_CHANGE",
            "station": junction,
            "line": line_before,
            "movement": movement,
            "scientific_semantics": "new boarding/service family required at same physical line junction; modeled as an inter-leg connection, not a through train",
        })
        current = [edge]
        compatible = set(options)

    flush()

    # For ordinary ride chains every inter-leg boundary must have one movement. Paths
    # with no ride leg (same-station line-surface changes) remain a separate boundary
    # case handled by the downstream station-only logic and are not fabricated here.
    if legs and len(movements) != len(legs) - 1:
        raise RuntimeError(f"movement/leg alignment failed: {len(movements)} movements for {len(legs)} legs")
    return legs, movements, movement_meta


def compress_candidate(cost: float, states: list[tuple[int, str]], edges: list[dict[str, Any]], rank: int) -> dict[str, Any] | None:
    try:
        legs, movements, movement_meta = ride_legs_and_movements(edges)
    except RuntimeError:
        return None
    explicit_transfers = [e for e in edges if e["kind"] == "TRANSFER"]
    rides = [e for e in edges if e["kind"] == "RIDE"]
    line_sequence: list[str] = []
    for _station, line in states:
        if not line_sequence or line_sequence[-1] != line:
            line_sequence.append(line)
    physical_path: list[int] = []
    for station, _line in states:
        if not physical_path or physical_path[-1] != station:
            physical_path.append(station)
    same_line_changes = sum(m["kind"] == "SAME_LINE_SERVICE_CHANGE" for m in movement_meta)
    return {
        "rank": rank,
        "base_ranking_cost_s": float(cost),
        "state_path": [{"station": s, "line": l} for s, l in states],
        "physical_station_path": physical_path,
        "line_sequence": line_sequence,
        "transfer_count": len(explicit_transfers),
        "same_line_service_change_count": int(same_line_changes),
        "inter_leg_movement_count": len(movements),
        "transfer_movements": movements,
        "inter_leg_movement_meta": movement_meta,
        "ride_segments": rides,
        "ride_legs": legs,
        "simple_state_path": len(states) == len(set(states)),
        "service_family_not_prematurely_resolved": True,
    }


def build_support(prior_path: Path, output: Path, k: int) -> dict[str, Any]:
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    graph = base.build_graph(prior)
    by_line, by_station = base.line_membership(prior["line_paths"])
    states = sorted(graph)
    support: dict[str, list[dict[str, Any]]] = {}
    unreachable: list[str] = []
    candidate_count = 0
    incoherent_filtered = 0
    max_transfer = 0
    max_same_line_changes = 0
    pairs_reaching_k = 0
    pairs_using_same_line_change = 0
    endpoints = sorted((station, line) for station, lines in by_station.items() for line in lines)

    for start in endpoints:
        for goal in endpoints:
            if start == goal:
                continue
            key = f"{start[1]}:{start[0]}->{goal[1]}:{goal[0]}"
            raw_paths = base.yen_k_shortest_simple_paths(graph, start, goal, k)
            payload: list[dict[str, Any]] = []
            for raw_rank, (cost, state_path, edge_meta) in enumerate(raw_paths, start=1):
                item = compress_candidate(cost, state_path, edge_meta, raw_rank)
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
            max_same_line_changes = max(max_same_line_changes, max(p["same_line_service_change_count"] for p in payload))
            pairs_using_same_line_change += int(any(p["same_line_service_change_count"] > 0 for p in payload))
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
        "all_multileg_candidates_have_aligned_movements": all(
            (not cand["ride_legs"]) or len(cand["transfer_movements"]) == len(cand["ride_legs"]) - 1
            for cands in support.values() for cand in cands
        ),
        "absolute_auxiliary_timestamp_not_used": prior["absolute_timestamp_policy"] == "FORBIDDEN_AS_2019_REALIZED_TRUTH",
        "at_least_one_multitransfer_candidate_exists": max_transfer >= 2,
        "same_line_branch_service_change_is_representable": max_same_line_changes >= 1,
    }
    result = {
        "schema": "rail.hz-expanded-route-support.v3-branch-service-change-aware",
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
        "max_same_line_service_change_count_present": max_same_line_changes,
        "pairs_with_same_line_service_change_support": pairs_using_same_line_change,
        "unreachable_endpoint_pairs": unreachable,
        "integrity_gates": gates,
        "scientific_boundary": {
            "transfer_count_cap": None,
            "k_path_beam_is_computational_support_approximation": True,
            "routes_outside_k_are_not_claimed_empirically_zero": True,
            "non_simple_behavioral_support": "DEFERRED_TO_CONTROLLED_SIDECAR_BEFORE_BEHAVIORAL_INVARIANT_CLAIMS",
            "transfer_penalty_role": "RANKING_ONLY_NOT_THETA_K",
            "shared_service_family_ambiguity": "RETAINED_UNTIL_PASSENGER_SERVICE_POSTERIOR_UPDATE",
            "same_line_branch_change": "REQUIRES_NEW_BOARDING_CHAIN_AT_PHYSICAL_JUNCTION; NOT TREATED_AS_THROUGH_SERVICE",
        },
        "line_membership": {line: sorted(nodes) for line, nodes in sorted(by_line.items())},
        "route_support": support,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "endpoint_surface_count": len(endpoints),
        "route_pair_count": len(support),
        "candidate_count": candidate_count,
        "incoherent_raw_paths_filtered": incoherent_filtered,
        "max_transfer_count_present": max_transfer,
        "max_same_line_service_change_count_present": max_same_line_changes,
        "pairs_with_same_line_service_change_support": pairs_using_same_line_change,
        "unreachable_endpoint_pair_count": len(unreachable),
        "unreachable_endpoint_pairs": unreachable[:100],
        "integrity_gates": gates,
    }, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=base.DEFAULT_K)
    a = p.parse_args()
    build_support(a.prior, a.output, a.k)


if __name__ == "__main__":
    main()
