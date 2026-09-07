from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.mppd_r1_hz_operating_audit import AUDIT_RESOURCES, _events_by_station, _trajectory_rows
from scripts.mppd_r1_hz_operating_authority import (
    ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S,
    build_operating_authority,
    normal_headway_floor_for,
    validate_operating_authority,
)

SCHEMA = "mppd.r1-hz-headway-conflict-graph.v1"
EXACT_COMPONENT_NODE_LIMIT = 28
EXACT_COMPONENT_EDGE_LIMIT = 70
EXACT_STATE_LIMIT = 250_000


class ExactStateLimitExceeded(RuntimeError):
    pass


def _node_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": str(row["trajectory_id"]),
        "line_id": str(row.get("afc_line", row.get("line_id", ""))),
        "path_id": str(row.get("path_id", "")),
        "direction": str(row.get("direction", row.get("direction_id", ""))),
        "path_ambiguous": bool(row.get("path_ambiguous", False)),
        "direction_ambiguous": bool(row.get("direction_ambiguous", False)),
        "support_station_count": int(row.get("support_station_count", 0) or 0),
        "support_event_count": int(row.get("support_event_count", 0) or 0),
        "support_weight": float(row.get("support_weight", 0.0) or 0.0),
        "evidence_score": float(row.get("evidence_score", 0.0) or 0.0),
        "objective_gain": float(row.get("objective_gain", 0.0) or 0.0),
    }


def _resource_index() -> dict[str, Mapping[str, Any]]:
    return {str(row["resource_id"]): row for row in AUDIT_RESOURCES}


def build_conflict_graph(
    discovery: Mapping[str, Any],
    authority: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    authority = authority or build_operating_authority()
    errors = validate_operating_authority(authority)
    if errors:
        raise ValueError("invalid operating authority: " + "; ".join(errors))
    rows = _trajectory_rows(discovery)
    nodes = {str(row["trajectory_id"]): _node_record(row) for row in rows}
    events = {str(row["trajectory_id"]): _events_by_station(row) for row in rows}
    resources = _resource_index()

    # An edge is a pair of service hypotheses that cannot coexist in the same
    # reconstructed world.  A pair may be adjacent at many stations/resources;
    # retain the observation with the largest positive violation margin while
    # accumulating all resource/station evidence for auditability.
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    for resource_id, resource in resources.items():
        allowed_paths = set(str(x) for x in resource["path_ids"])
        for direction in ("Down", "Up"):
            for station in tuple(int(x) for x in resource["station_ids"]):
                seq: list[tuple[float, str]] = []
                for tid, node in nodes.items():
                    if node["path_id"] not in allowed_paths or node["direction"] != direction:
                        continue
                    if station in events[tid]:
                        seq.append((float(events[tid][station]), tid))
                seq.sort(key=lambda item: (item[0], item[1]))
                for (left_t, left_id), (right_t, right_id) in zip(seq, seq[1:]):
                    gap = float(right_t - left_t)
                    if gap < 0:
                        raise AssertionError("sorted resource events yielded negative headway")
                    midpoint = (left_t + right_t) / 2.0
                    left_path = str(nodes[left_id]["path_id"])
                    right_path = str(nodes[right_id]["path_id"])
                    left_floor, left_band = normal_headway_floor_for(
                        authority,
                        resource_id=resource_id,
                        path_id=left_path,
                        direction=direction,
                        event_time_s=midpoint,
                    )
                    right_floor, right_band = normal_headway_floor_for(
                        authority,
                        resource_id=resource_id,
                        path_id=right_path,
                        direction=direction,
                        event_time_s=midpoint,
                    )
                    required = max(float(left_floor), float(right_floor))
                    if gap >= required:
                        continue
                    key = tuple(sorted((left_id, right_id)))
                    physical = gap < ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S
                    observation = {
                        "resource_id": resource_id,
                        "direction": direction,
                        "station_id": station,
                        "gap_s": gap,
                        "required_floor_s": required,
                        "violation_margin_s": required - gap,
                        "physical_90s_violation": physical,
                        "authority_band_ids": sorted(
                            {x for x in (left_band, right_band) if x is not None}
                        ),
                    }
                    current = edge_map.get(key)
                    if current is None:
                        edge_map[key] = {
                            "left_trajectory_id": key[0],
                            "right_trajectory_id": key[1],
                            "physical_90s_violation": physical,
                            "operating_envelope_violation": required
                            > ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S,
                            "observation_count": 1,
                            "resources": [resource_id],
                            "authority_band_ids": list(observation["authority_band_ids"]),
                            "worst_observation": observation,
                        }
                    else:
                        current["physical_90s_violation"] = bool(
                            current["physical_90s_violation"] or physical
                        )
                        current["operating_envelope_violation"] = bool(
                            current["operating_envelope_violation"]
                            or required > ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S
                        )
                        current["observation_count"] = int(current["observation_count"]) + 1
                        current["resources"] = sorted(
                            set(current["resources"]) | {resource_id}
                        )
                        current["authority_band_ids"] = sorted(
                            set(current["authority_band_ids"])
                            | set(observation["authority_band_ids"])
                        )
                        if observation["violation_margin_s"] > current["worst_observation"][
                            "violation_margin_s"
                        ]:
                            current["worst_observation"] = observation

    edges = sorted(
        edge_map.values(),
        key=lambda row: (
            -float(row["worst_observation"]["violation_margin_s"]),
            row["left_trajectory_id"],
            row["right_trajectory_id"],
        ),
    )
    return nodes, edges


def _edge_tuples(edges: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    return {
        tuple(sorted((str(row["left_trajectory_id"]), str(row["right_trajectory_id"]))))
        for row in edges
    }


def connected_components(edge_set: set[tuple[str, str]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for u, v in edge_set:
        adjacency[u].add(v)
        adjacency[v].add(u)
    components: list[set[str]] = []
    unseen = set(adjacency)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            u = stack.pop()
            if u in component:
                continue
            component.add(u)
            unseen.discard(u)
            stack.extend(adjacency[u] - component)
        components.append(component)
    components.sort(key=lambda comp: (-len(comp), min(comp)))
    return components


def greedy_matching_lower_bound(edge_set: set[tuple[str, str]]) -> tuple[int, list[tuple[str, str]]]:
    if not edge_set:
        return 0, []
    degree: Counter[str] = Counter()
    for u, v in edge_set:
        degree[u] += 1
        degree[v] += 1
    # Try several deterministic orders; every result is a valid matching lower bound.
    orders = (
        sorted(edge_set),
        sorted(edge_set, key=lambda e: (degree[e[0]] + degree[e[1]], e)),
        sorted(edge_set, key=lambda e: (-(degree[e[0]] + degree[e[1]]), e)),
    )
    best: list[tuple[str, str]] = []
    for order in orders:
        used: set[str] = set()
        matching: list[tuple[str, str]] = []
        for u, v in order:
            if u in used or v in used:
                continue
            used.update((u, v))
            matching.append((u, v))
        if len(matching) > len(best):
            best = matching
    return len(best), best


def _evidence_loss_key(cover: frozenset[str], nodes: Mapping[str, Mapping[str, Any]]) -> tuple[Any, ...]:
    return (
        len(cover),
        sum(float(nodes[x]["evidence_score"]) for x in cover),
        sum(int(nodes[x]["support_station_count"]) for x in cover),
        sum(float(nodes[x]["support_weight"]) for x in cover),
        tuple(sorted(cover)),
    )


def greedy_vertex_cover_upper_bound(
    edge_set: set[tuple[str, str]],
    nodes: Mapping[str, Mapping[str, Any]],
) -> frozenset[str]:
    remaining = set(edge_set)
    cover: set[str] = set()
    while remaining:
        degree: Counter[str] = Counter()
        for u, v in remaining:
            degree[u] += 1
            degree[v] += 1
        max_degree = max(degree.values())
        candidates = [u for u, d in degree.items() if d == max_degree]
        # Cardinality is primary.  When degree ties, sacrifice the weaker AFC
        # hypothesis first.  Passenger support is intentionally not used here.
        chosen = min(
            candidates,
            key=lambda u: (
                float(nodes[u]["evidence_score"]),
                int(nodes[u]["support_station_count"]),
                float(nodes[u]["support_weight"]),
                u,
            ),
        )
        cover.add(chosen)
        remaining = {e for e in remaining if chosen not in e}
    return frozenset(cover)


def exact_minimum_vertex_cover(
    edge_set: set[tuple[str, str]],
    nodes: Mapping[str, Mapping[str, Any]],
    *,
    state_limit: int = EXACT_STATE_LIMIT,
) -> tuple[frozenset[str], int]:
    normalized = frozenset(tuple(sorted(e)) for e in edge_set)
    calls = 0

    @lru_cache(maxsize=None)
    def solve(state: frozenset[tuple[str, str]]) -> frozenset[str]:
        nonlocal calls
        calls += 1
        if calls > state_limit:
            raise ExactStateLimitExceeded(f"exact state limit exceeded: {state_limit}")
        if not state:
            return frozenset()
        degree: Counter[str] = Counter()
        for u, v in state:
            degree[u] += 1
            degree[v] += 1
        # Branch on a high-degree conflict edge to collapse the state quickly.
        u, v = max(
            state,
            key=lambda e: (degree[e[0]] + degree[e[1]], degree[e[0]], degree[e[1]], e),
        )
        candidates: list[frozenset[str]] = []
        for chosen in (u, v):
            reduced = frozenset(edge for edge in state if chosen not in edge)
            candidates.append(solve(reduced) | {chosen})
        return min(candidates, key=lambda cover: _evidence_loss_key(cover, nodes))

    return solve(normalized), calls


def analyze_conflicts(discovery: Mapping[str, Any]) -> dict[str, Any]:
    authority = build_operating_authority()
    nodes, edges = build_conflict_graph(discovery, authority)
    edge_set = _edge_tuples(edges)
    components = connected_components(edge_set)
    incident = {x for edge in edge_set for x in edge}
    physical_edges = [x for x in edges if x["physical_90s_violation"]]
    operating_only_edges = [
        x
        for x in edges
        if not x["physical_90s_violation"] and x["operating_envelope_violation"]
    ]

    component_rows: list[dict[str, Any]] = []
    total_lb = 0
    total_ub = 0
    total_exact = 0
    all_exact = True
    exact_removed: set[str] = set()
    greedy_removed: set[str] = set()
    for index, component in enumerate(components):
        comp_edges = {e for e in edge_set if e[0] in component and e[1] in component}
        lb, matching = greedy_matching_lower_bound(comp_edges)
        greedy_cover = greedy_vertex_cover_upper_bound(comp_edges, nodes)
        total_lb += lb
        total_ub += len(greedy_cover)
        greedy_removed.update(greedy_cover)
        exact_cover: frozenset[str] | None = None
        exact_states: int | None = None
        exact_status = "NOT_ATTEMPTED_COMPONENT_TOO_LARGE"
        if len(component) <= EXACT_COMPONENT_NODE_LIMIT and len(comp_edges) <= EXACT_COMPONENT_EDGE_LIMIT:
            try:
                exact_cover, exact_states = exact_minimum_vertex_cover(comp_edges, nodes)
                exact_status = "EXACT"
            except ExactStateLimitExceeded:
                exact_status = "STATE_LIMIT_EXCEEDED"
        if exact_cover is None:
            all_exact = False
        else:
            total_exact += len(exact_cover)
            exact_removed.update(exact_cover)
            if not lb <= len(exact_cover) <= len(greedy_cover):
                raise AssertionError("exact cover falls outside valid component bounds")
        component_rows.append(
            {
                "component_index": index,
                "node_count": len(component),
                "edge_count": len(comp_edges),
                "matching_lower_bound_removals": lb,
                "greedy_cover_upper_bound_removals": len(greedy_cover),
                "exact_status": exact_status,
                "exact_minimum_removals": len(exact_cover) if exact_cover is not None else None,
                "exact_states_explored": exact_states,
                "nodes": sorted(component),
                "matching_edges": [list(x) for x in matching],
                "greedy_removed_nodes": sorted(greedy_cover),
                "exact_removed_nodes": sorted(exact_cover) if exact_cover is not None else None,
            }
        )

    removal_set = exact_removed if all_exact else greedy_removed
    retained_count = len(nodes) - len(removal_set)
    removed_path_direction: Counter[str] = Counter()
    removed_ambiguity = 0
    removed_evidence = 0.0
    for tid in removal_set:
        node = nodes[tid]
        removed_path_direction[f"{node['path_id']}:{node['direction']}"] += 1
        removed_ambiguity += int(bool(node["path_ambiguous"] or node["direction_ambiguous"]))
        removed_evidence += float(node["evidence_score"])

    degree: Counter[str] = Counter()
    for u, v in edge_set:
        degree[u] += 1
        degree[v] += 1
    by_path_direction: Counter[str] = Counter()
    ambiguous_incident = 0
    for tid in incident:
        node = nodes[tid]
        by_path_direction[f"{node['path_id']}:{node['direction']}"] += 1
        ambiguous_incident += int(bool(node["path_ambiguous"] or node["direction_ambiguous"]))

    return {
        "schema": SCHEMA,
        "status": (
            "EXACT_MINIMUM_HEADWAY_REPAIR_IDENTIFIED_AFC_TIEBREAK_ONLY"
            if all_exact
            else "HEADWAY_CONFLICT_BOUNDS_IDENTIFIED_GREEDY_REPAIR_ONLY"
        ),
        "service_count_before": len(nodes),
        "conflict_graph": {
            "union_conflict_edge_count": len(edges),
            "physical_90s_conflict_edge_count": len(physical_edges),
            "operating_only_conflict_edge_count": len(operating_only_edges),
            "incident_service_count": len(incident),
            "nonincident_service_count": len(nodes) - len(incident),
            "component_count": len(components),
            "largest_component_node_count": max((len(x) for x in components), default=0),
            "largest_component_edge_count": max(
                (
                    sum(1 for e in edge_set if e[0] in comp and e[1] in comp)
                    for comp in components
                ),
                default=0,
            ),
            "maximum_node_degree": max(degree.values(), default=0),
            "incident_service_count_by_path_direction": dict(sorted(by_path_direction.items())),
            "incident_ambiguous_service_count": ambiguous_incident,
        },
        "repair_bounds": {
            "minimum_removals_lower_bound": total_lb,
            "minimum_removals_upper_bound": total_ub,
            "maximum_physically_admissible_retained_count_lower_bound": len(nodes) - total_ub,
            "maximum_physically_admissible_retained_count_upper_bound": len(nodes) - total_lb,
            "all_components_solved_exactly": all_exact,
            "exact_minimum_removals": total_exact if all_exact else None,
            "exact_maximum_retained_count": len(nodes) - total_exact if all_exact else None,
        },
        "diagnostic_repair": {
            "method": "EXACT_CARDINALITY_AFC_EVIDENCE_TIEBREAK" if all_exact else "GREEDY_MAX_DEGREE_AFC_TIEBREAK",
            "removed_service_count": len(removal_set),
            "retained_service_count": retained_count,
            "removed_ambiguous_service_count": removed_ambiguity,
            "removed_evidence_score_sum": removed_evidence,
            "removed_count_by_path_direction": dict(sorted(removed_path_direction.items())),
            "removed_trajectory_ids": sorted(removal_set),
            "passenger_posterior_used_for_tiebreak": False,
            "circulation_used_for_tiebreak": False,
        },
        "components": component_rows,
        "conflict_edges": edges,
        "semantics": {
            "service_world_mutated": False,
            "diagnostic_repair_is_formally_applied": False,
            "planned_absolute_timetable_used": False,
            "planned_trip_count_used": False,
            "planned_trip_ids_used": False,
            "headway_resource_competition_used": True,
            "shared_trunk_competition_used": True,
            "depot_boundary_resolution_applied": False,
            "physical_vehicle_identity_claimed": False,
        },
        "scientific_boundary": (
            "The exact/greedy repair resolves only headway incompatibility in the current 1841 "
            "hypothesis world. AFC evidence is used only to choose among equal-cardinality covers. "
            "Passenger posterior, movement refit and circulation balance must be rerun before any "
            "removed trajectory is accepted as a formal death/merge."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--services", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    discovery = json.loads(args.services.read_text(encoding="utf-8"))
    result = analyze_conflicts(discovery)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "service_count_before": result["service_count_before"],
                "conflict_graph": result["conflict_graph"],
                "repair_bounds": result["repair_bounds"],
                "diagnostic_repair": {
                    key: value
                    for key, value in result["diagnostic_repair"].items()
                    if key != "removed_trajectory_ids"
                },
                "largest_components": [
                    {
                        key: row[key]
                        for key in (
                            "component_index",
                            "node_count",
                            "edge_count",
                            "matching_lower_bound_removals",
                            "greedy_cover_upper_bound_removals",
                            "exact_status",
                            "exact_minimum_removals",
                            "exact_states_explored",
                        )
                    }
                    for row in result["components"][:20]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
