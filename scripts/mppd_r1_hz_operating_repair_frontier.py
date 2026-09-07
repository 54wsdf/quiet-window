from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from scripts.mppd_r1_hz_operating_audit import audit_service_world
from scripts.mppd_r1_hz_operating_authority import build_operating_authority

SCHEMA = "mppd.r1-hz-operating-repair-frontier.v1"
METHOD = "ITERATED_EXACT_WEIGHTED_VERTEX_COVER_SCIPY_MILP"
DEFAULT_PASSENGER_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0)
AFC_WEIGHT = 0.25


def _rows(discovery: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = discovery.get("trajectories")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("service discovery requires trajectories sequence")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("trajectory rows must be mappings")
        row = copy.deepcopy(dict(item))
        tid = str(row.get("trajectory_id", ""))
        if not tid or tid in seen:
            raise ValueError(f"invalid/duplicate trajectory_id {tid!r}")
        seen.add(tid)
        rows.append(row)
    expected = discovery.get("inferred_service_trajectory_count")
    if expected is not None and int(expected) != len(rows):
        raise ValueError("service count disagrees with trajectories")
    return rows


def _direction(row: Mapping[str, Any]) -> str:
    return str(row.get("direction", row.get("direction_id", "")))


def _path(row: Mapping[str, Any]) -> str:
    return str(row.get("path_id", ""))


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _rank01(values: Mapping[str, float]) -> dict[str, float]:
    """Deterministic empirical rank in [0,1], ties share the same mid-rank."""
    if not values:
        return {}
    groups: dict[float, list[str]] = {}
    for key, value in values.items():
        groups.setdefault(float(value), []).append(key)
    ordered = sorted(groups)
    n = len(values)
    if n == 1:
        return {next(iter(values)): 0.0}
    out: dict[str, float] = {}
    offset = 0
    for value in ordered:
        keys = sorted(groups[value])
        lo = offset
        hi = offset + len(keys) - 1
        rank = ((lo + hi) / 2.0) / (n - 1)
        for key in keys:
            out[key] = rank
        offset += len(keys)
    return out


def _assignment_mass(passenger_feedback: Mapping[str, Any]) -> dict[str, float]:
    raw = passenger_feedback.get("trajectory_assignment_mass", {})
    if not isinstance(raw, Mapping):
        raise ValueError("passenger feedback trajectory_assignment_mass must be mapping")
    return {str(k): max(0.0, _finite_number(v)) for k, v in raw.items()}


def _unique_edges(conflicts: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for item in conflicts:
        left = str(item.get("left_trajectory_id", ""))
        right = str(item.get("right_trajectory_id", ""))
        if not left or not right or left == right:
            raise ValueError("invalid operating conflict pair")
        edges.add(tuple(sorted((left, right))))
    return sorted(edges)


def _path_direction_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    return Counter((_path(row), _direction(row)) for row in rows)


def _surplus_flag(row: Mapping[str, Any], counts: Counter[tuple[str, str]]) -> bool:
    path = _path(row)
    direction = _direction(row)
    down = counts[(path, "Down")]
    up = counts[(path, "Up")]
    if down == up:
        return False
    return (direction == "Down" and down > up) or (direction == "Up" and up > down)


def _deletion_costs(
    rows: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, float],
    passenger_weight: float,
) -> dict[str, float]:
    if passenger_weight < 0 or not math.isfinite(passenger_weight):
        raise ValueError("passenger_weight must be finite and non-negative")
    ids = [str(row["trajectory_id"]) for row in rows]
    passenger_rank = _rank01({tid: math.log1p(float(assignment.get(tid, 0.0))) for tid in ids})
    afc_rank = _rank01({tid: _finite_number(row.get("evidence_score")) for tid, row in zip(ids, rows)})
    counts = _path_direction_counts(rows)
    costs: dict[str, float] = {}
    for tid, row in zip(ids, rows):
        # Unit count cost is primary. Passenger/AFC terms make a heavily used,
        # well-supported trajectory more expensive to delete. Direction surplus
        # and ambiguity are mild discounts, never independent deletion reasons.
        cost = 1.0
        cost += passenger_weight * passenger_rank[tid]
        cost += AFC_WEIGHT * afc_rank[tid]
        if _surplus_flag(row, counts):
            cost -= 0.08
        if bool(row.get("path_ambiguous", False)):
            cost -= 0.08
        if bool(row.get("direction_ambiguous", False)):
            cost -= 0.08
        costs[tid] = max(0.05, cost)
    return costs


def _solve_weighted_vertex_cover(
    rows: Sequence[Mapping[str, Any]],
    edges: Sequence[tuple[str, str]],
    assignment: Mapping[str, float],
    passenger_weight: float,
) -> tuple[set[str], dict[str, Any]]:
    if not edges:
        return set(), {"objective": 0.0, "status": "NO_CONFLICTS"}
    ids = sorted({tid for edge in edges for tid in edge})
    by_id = {str(row["trajectory_id"]): row for row in rows}
    missing = [tid for tid in ids if tid not in by_id]
    if missing:
        raise ValueError(f"conflict graph references missing services: {missing[:5]}")
    conflict_rows = [by_id[tid] for tid in ids]
    costs_map = _deletion_costs(conflict_rows, assignment, passenger_weight)
    index = {tid: i for i, tid in enumerate(ids)}
    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    for r, (left, right) in enumerate(edges):
        row_idx.extend((r, r))
        col_idx.extend((index[left], index[right]))
        data.extend((1.0, 1.0))
    A = coo_matrix((data, (row_idx, col_idx)), shape=(len(edges), len(ids))).tocsr()
    result = milp(
        c=np.array([costs_map[tid] for tid in ids], dtype=float),
        integrality=np.ones(len(ids), dtype=int),
        bounds=Bounds(np.zeros(len(ids)), np.ones(len(ids))),
        constraints=LinearConstraint(A, np.ones(len(edges)), np.full(len(edges), np.inf)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"weighted vertex cover failed: {result.message}")
    selected = {tid for tid, value in zip(ids, result.x) if value >= 0.5}
    if any(left not in selected and right not in selected for left, right in edges):
        raise AssertionError("MILP solution does not cover all conflict edges")
    return selected, {
        "status": "OPTIMAL" if int(result.status) == 0 else f"SCIPY_STATUS_{result.status}",
        "objective": float(result.fun),
        "selected_count": len(selected),
        "conflict_edge_count": len(edges),
        "passenger_weight": passenger_weight,
        "afc_weight": AFC_WEIGHT,
    }


def build_candidate(
    discovery: Mapping[str, Any],
    passenger_feedback: Mapping[str, Any],
    *,
    passenger_weight: float,
    max_rounds: int = 20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = build_operating_authority()
    assignment = _assignment_mass(passenger_feedback)
    original_rows = _rows(discovery)
    current_rows = copy.deepcopy(original_rows)
    removed: set[str] = set()
    rounds: list[dict[str, Any]] = []

    for round_index in range(1, max_rounds + 1):
        world = copy.deepcopy(dict(discovery))
        world["trajectories"] = current_rows
        world["inferred_service_trajectory_count"] = len(current_rows)
        audit = audit_service_world(world, authority)
        conflicts = list(audit["headway"].get("violation_pairs", ()))
        edges = _unique_edges(conflicts)
        if not edges:
            break
        selected, solver = _solve_weighted_vertex_cover(
            current_rows, edges, assignment, passenger_weight
        )
        if not selected:
            raise RuntimeError("frontier repair made no progress")
        removed.update(selected)
        rounds.append(
            {
                "round_index": round_index,
                "service_count_before": len(current_rows),
                "conflict_pair_count_before": len(conflicts),
                "unique_conflict_edge_count_before": len(edges),
                "selected_death_count": len(selected),
                "selected_assignment_mass_sum": sum(float(assignment.get(tid, 0.0)) for tid in selected),
                "solver": solver,
            }
        )
        current_rows = [row for row in current_rows if str(row["trajectory_id"]) not in selected]
    else:
        raise RuntimeError("frontier repair did not converge")

    candidate = copy.deepcopy(dict(discovery))
    candidate["trajectories"] = current_rows
    candidate["inferred_service_trajectory_count"] = len(current_rows)
    candidate["semantics"] = dict(candidate.get("semantics", {}))
    candidate["semantics"].update(
        {
            "service_count_is_fixed": False,
            "operating_constraint_repair_applied": True,
            "passenger_assignment_used_only_for_conflict_resolution": True,
            "planned_timetable_used": False,
            "planned_trip_count_used": False,
            "planned_trip_ids_used": False,
        }
    )
    final_audit = audit_service_world(candidate, authority)
    if final_audit["headway"]["total_normal_floor_violation_pair_count"] != 0:
        raise RuntimeError("frontier candidate retains enforced headway conflicts")
    removed_assignment = sum(float(assignment.get(tid, 0.0)) for tid in removed)
    total_assignment = sum(float(v) for v in assignment.values())
    report = {
        "passenger_weight": passenger_weight,
        "service_count_before": len(original_rows),
        "service_count_after": len(current_rows),
        "removed_service_count": len(removed),
        "removed_trajectory_ids": sorted(removed),
        "removed_baseline_assignment_mass_sum": removed_assignment,
        "baseline_assignment_mass_sum": total_assignment,
        "removed_assignment_mass_share": removed_assignment / total_assignment if total_assignment else None,
        "rounds": rounds,
        "direction_balance_after": final_audit["direction_balance"]["by_path"],
        "remaining_headway_conflicts": 0,
    }
    candidate["operating_repair_frontier"] = {
        "schema": SCHEMA,
        "method": METHOD,
        "passenger_weight": passenger_weight,
        "candidate_world_is_final_r1": False,
    }
    return candidate, report


def build_frontier(
    discovery: Mapping[str, Any],
    passenger_feedback: Mapping[str, Any],
    *,
    passenger_weights: Sequence[float] = DEFAULT_PASSENGER_WEIGHTS,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for weight in passenger_weights:
        label = f"pw_{str(float(weight)).replace('.', 'p')}"
        candidate, report = build_candidate(
            discovery, passenger_feedback, passenger_weight=float(weight)
        )
        candidates[label] = candidate
        rows.append({"label": label, **report})
    return candidates, {
        "schema": SCHEMA,
        "status": "PASSENGER_AWARE_OPERATING_REPAIR_FRONTIER_COMPLETED_NO_CANDIDATE_SELECTED",
        "method": METHOD,
        "passenger_weights": [float(x) for x in passenger_weights],
        "candidates": rows,
        "semantics": {
            "planned_timetable_used": False,
            "passenger_assignment_is_baseline_diagnostic_not_fixed_truth": True,
            "all_candidates_satisfy_enforced_headway_constraints": True,
            "candidate_world_is_final_r1": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--services", type=Path, required=True)
    parser.add_argument("--passenger-feedback", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=list(DEFAULT_PASSENGER_WEIGHTS))
    args = parser.parse_args()
    discovery = json.loads(args.services.read_text(encoding="utf-8"))
    passenger = json.loads(args.passenger_feedback.read_text(encoding="utf-8"))
    candidates, frontier = build_frontier(
        discovery, passenger, passenger_weights=tuple(args.weights)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, candidate in candidates.items():
        (args.output_dir / f"services_{label}.json").write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (args.output_dir / "frontier.json").write_text(
        json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": frontier["status"],
        "candidates": [
            {
                "label": row["label"],
                "N": row["service_count_after"],
                "removed": row["removed_service_count"],
                "removed_assignment_share": row["removed_assignment_mass_share"],
            }
            for row in frontier["candidates"]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
