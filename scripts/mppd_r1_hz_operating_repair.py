from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.mppd_r1_hz_operating_audit import audit_service_world
from scripts.mppd_r1_hz_operating_authority import build_operating_authority

REPAIR_SCHEMA = "mppd.r1-hz-operating-structure-repair.v1"
REPAIR_METHOD = "ITERATED_CONFLICT_GRAPH_GREEDY_VERTEX_COVER"


def _numeric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _support_size(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, (str, bytes)) or value is None:
        return 0.0
    if isinstance(value, Sequence):
        return float(len(value))
    return 0.0


def _direction(row: Mapping[str, Any]) -> str:
    return str(row.get("direction", row.get("direction_id", "")))


def _path_id(row: Mapping[str, Any]) -> str:
    return str(row.get("path_id", ""))


def _line_id(row: Mapping[str, Any]) -> str:
    return str(row.get("afc_line", row.get("line_id", "")))


def _trajectory_rows(discovery: Mapping[str, Any]) -> list[dict[str, Any]]:
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
        if not tid:
            raise ValueError("trajectory requires trajectory_id")
        if tid in seen:
            raise ValueError(f"duplicate trajectory_id {tid}")
        seen.add(tid)
        if _direction(row) not in {"Up", "Down"} or not _path_id(row) or not _line_id(row):
            raise ValueError(f"{tid}: incomplete line/path/direction identity")
        rows.append(row)
    expected = discovery.get("inferred_service_trajectory_count")
    if expected is not None and int(expected) != len(rows):
        raise ValueError("inferred_service_trajectory_count disagrees with trajectories")
    return rows


def _path_direction_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    return Counter((_path_id(row), _direction(row)) for row in rows)


def _is_surplus_direction(
    row: Mapping[str, Any],
    counts: Counter[tuple[str, str]],
) -> bool:
    path = _path_id(row)
    direction = _direction(row)
    down = int(counts[(path, "Down")])
    up = int(counts[(path, "Up")])
    if down == up:
        return False
    return (direction == "Down" and down > up) or (direction == "Up" and up > down)


def _evidence_view(row: Mapping[str, Any]) -> dict[str, Any]:
    support_event_ids = row.get("support_event_ids", ())
    if isinstance(support_event_ids, (str, bytes)) or not isinstance(support_event_ids, Sequence):
        support_event_ids = ()
    return {
        "evidence_score": _numeric(row.get("evidence_score")),
        "station_support": _support_size(row.get("station_support")),
        "support_weight": _numeric(row.get("support_weight")),
        "support_event_count": _support_size(row.get("support_event_count", support_event_ids)),
        "trajectory_residual_p90_s": _numeric(row.get("trajectory_residual_p90_s")),
        "path_ambiguous": bool(row.get("path_ambiguous", False)),
        "direction_ambiguous": bool(row.get("direction_ambiguous", False)),
    }


def _event_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    def ids(row: Mapping[str, Any]) -> set[str]:
        raw = row.get("support_event_ids", ())
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            return set()
        return {str(item) for item in raw if str(item)}

    a, b = ids(left), ids(right)
    shared = a & b
    denom = min(len(a), len(b))
    return {
        "shared_support_event_count": len(shared),
        "minimum_side_overlap_share": (len(shared) / denom) if denom else None,
    }


def _removal_priority(
    tid: str,
    *,
    degree: int,
    row: Mapping[str, Any],
    counts: Counter[tuple[str, str]],
) -> tuple[Any, ...]:
    """Higher tuple means easier to remove.

    Conflict coverage dominates so the first candidate world seeks a small death
    set. Within equal coverage, direction surplus and ambiguity make removal easier;
    stronger multi-station/AFC evidence makes removal harder. This is deliberately a
    deterministic diagnostic rule, not the final joint likelihood.
    """

    evidence = _evidence_view(row)
    return (
        int(degree),
        int(_is_surplus_direction(row, counts)),
        int(evidence["path_ambiguous"]),
        int(evidence["direction_ambiguous"]),
        -float(evidence["evidence_score"]),
        -float(evidence["station_support"]),
        -float(evidence["support_weight"]),
        -float(evidence["support_event_count"]),
        float(evidence["trajectory_residual_p90_s"]),
        tid,
    )


def _unique_conflict_edges(
    conflicts: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    edges: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for conflict in conflicts:
        left = str(conflict.get("left_trajectory_id", ""))
        right = str(conflict.get("right_trajectory_id", ""))
        if not left or not right or left == right:
            raise ValueError("operating conflict contains invalid trajectory pair")
        key = tuple(sorted((left, right)))
        edges[key].append(conflict)
    return dict(edges)


def _select_conflict_cover(
    rows: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Greedy vertex cover with deterministic evidence-aware tie breaking."""

    by_id = {str(row["trajectory_id"]): row for row in rows}
    counts = _path_direction_counts(rows)
    remaining = _unique_conflict_edges(conflicts)
    selected: set[str] = set()
    selection_meta: dict[str, dict[str, Any]] = {}

    while remaining:
        incident: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for edge in remaining:
            incident[edge[0]].append(edge)
            incident[edge[1]].append(edge)
        missing = [tid for tid in incident if tid not in by_id]
        if missing:
            raise ValueError(f"audit conflict references missing trajectories: {missing[:5]}")

        best_tid = max(
            incident,
            key=lambda tid: _removal_priority(
                tid,
                degree=len(incident[tid]),
                row=by_id[tid],
                counts=counts,
            ),
        )
        covered = list(incident[best_tid])
        selected.add(best_tid)
        selection_meta[best_tid] = {
            "conflict_degree_at_selection": len(covered),
            "path_direction_surplus_at_selection": _is_surplus_direction(by_id[best_tid], counts),
            "evidence": _evidence_view(by_id[best_tid]),
            "covered_pair_keys": [list(edge) for edge in sorted(covered)],
        }
        for edge in covered:
            remaining.pop(edge, None)

    return selected, selection_meta


def _operation_for_removed(
    tid: str,
    *,
    row: Mapping[str, Any],
    selection: Mapping[str, Any],
    conflicts: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    round_index: int,
) -> dict[str, Any]:
    incidents = [
        conflict
        for conflict in conflicts
        if tid in {
            str(conflict.get("left_trajectory_id", "")),
            str(conflict.get("right_trajectory_id", "")),
        }
    ]
    incidents.sort(
        key=lambda item: (
            float(item.get("gap_s", 0.0)) - float(item.get("required_floor_s", 0.0)),
            str(item.get("resource_id", "")),
        )
    )
    representative = incidents[0] if incidents else {}
    left = str(representative.get("left_trajectory_id", ""))
    right = str(representative.get("right_trajectory_id", ""))
    counterpart = right if left == tid else left
    overlap = (
        _event_overlap(row, rows_by_id[counterpart])
        if counterpart in rows_by_id
        else {"shared_support_event_count": 0, "minimum_side_overlap_share": None}
    )
    return {
        "operation_type": "death_candidate",
        "stage": "OPERATING_CONSTRAINT_REPAIR_DIAGNOSTIC",
        "round_index": round_index,
        "trajectory_id": tid,
        "line_id": _line_id(row),
        "path_id": _path_id(row),
        "direction": _direction(row),
        "reason_class": (
            "ABSOLUTE_PHYSICAL_HEADWAY_CONFLICT"
            if bool(representative.get("absolute_physical_violation"))
            else "VERIFIED_OPERATING_ENVELOPE_CONFLICT"
        ),
        "representative_conflict": dict(representative),
        "representative_counterpart_trajectory_id": counterpart or None,
        "support_event_overlap_with_counterpart": overlap,
        "selection": dict(selection),
        "scientific_boundary": (
            "Death is a constrained candidate operation only. Evidence is not inherited "
            "by the survivor, so this operation is not yet a semantic merge."
        ),
    }


def repair_service_world(
    discovery: Mapping[str, Any],
    authority: Mapping[str, Any] | None = None,
    *,
    max_rounds: int = 20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")
    authority = authority or build_operating_authority()
    original_rows = _trajectory_rows(discovery)
    current_rows = copy.deepcopy(original_rows)
    before = audit_service_world(discovery, authority)
    operations: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []

    for round_index in range(1, max_rounds + 1):
        current_discovery = copy.deepcopy(dict(discovery))
        current_discovery["trajectories"] = current_rows
        current_discovery["inferred_service_trajectory_count"] = len(current_rows)
        audit = audit_service_world(current_discovery, authority)
        conflicts = list(audit["headway"].get("violation_pairs", ()))
        if not conflicts:
            break

        selected, selection_meta = _select_conflict_cover(current_rows, conflicts)
        if not selected:
            raise RuntimeError("conflict repair made no progress")
        rows_by_id = {str(row["trajectory_id"]): row for row in current_rows}
        for tid in sorted(selected):
            operations.append(
                _operation_for_removed(
                    tid,
                    row=rows_by_id[tid],
                    selection=selection_meta[tid],
                    conflicts=conflicts,
                    rows_by_id=rows_by_id,
                    round_index=round_index,
                )
            )
        round_summaries.append(
            {
                "round_index": round_index,
                "service_count_before": len(current_rows),
                "conflict_pair_count_before": len(conflicts),
                "selected_death_count": len(selected),
                "selected_trajectory_ids": sorted(selected),
            }
        )
        current_rows = [
            row for row in current_rows if str(row["trajectory_id"]) not in selected
        ]
    else:
        raise RuntimeError(f"operating repair did not converge within {max_rounds} rounds")

    repaired = copy.deepcopy(dict(discovery))
    repaired["trajectories"] = current_rows
    repaired["inferred_service_trajectory_count"] = len(current_rows)
    semantics = dict(repaired.get("semantics", {}))
    semantics.update(
        {
            "service_count_is_fixed": False,
            "operating_constraint_repair_applied": True,
            "planned_timetable_used": False,
            "planned_trip_count_used": False,
            "planned_trip_ids_used": False,
        }
    )
    repaired["semantics"] = semantics

    after = audit_service_world(repaired, authority)
    remaining = list(after["headway"].get("violation_pairs", ()))
    if remaining:
        raise RuntimeError(f"operating repair left {len(remaining)} enforced headway conflicts")

    initial_ids = {str(row["trajectory_id"]) for row in original_rows}
    final_ids = {str(row["trajectory_id"]) for row in current_rows}
    if not final_ids.issubset(initial_ids):
        raise AssertionError("death-only repair invented a trajectory")
    removed_ids = sorted(initial_ids - final_ids)
    if len(removed_ids) != len(operations):
        raise AssertionError("removed service count disagrees with operation log")

    report = {
        "schema": REPAIR_SCHEMA,
        "status": "OPERATING_CONSTRAINT_DEATH_CANDIDATE_WORLD_COMPLETED",
        "method": REPAIR_METHOD,
        "service_count_before": len(original_rows),
        "service_count_after": len(current_rows),
        "service_count_change": len(current_rows) - len(original_rows),
        "removed_service_count": len(removed_ids),
        "removed_trajectory_ids": removed_ids,
        "round_count": len(round_summaries),
        "rounds": round_summaries,
        "before": {
            "headway": before["headway"],
            "direction_balance": before["direction_balance"],
        },
        "after": {
            "headway": after["headway"],
            "direction_balance": after["direction_balance"],
        },
        "operations": operations,
        "semantics": {
            "birth_operations_applied": 0,
            "death_candidate_operations_applied": len(operations),
            "merge_operations_applied": 0,
            "event_times_changed": False,
            "direction_balance_forced_by_nonconflict_deletion": False,
            "planned_absolute_timetable_used": False,
            "planned_trip_count_used": False,
            "planned_trip_ids_used": False,
            "depot_boundary_resolution_applied": False,
            "physical_vehicle_identity_claimed": False,
            "candidate_world_is_final_r1": False,
        },
    }
    repaired["operating_constraint_repair"] = {
        "schema": REPAIR_SCHEMA,
        "method": REPAIR_METHOD,
        "removed_service_count": len(removed_ids),
        "report_status": report["status"],
        "candidate_world_is_final_r1": False,
    }
    return repaired, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--services", type=Path, required=True)
    parser.add_argument("--output-services", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=20)
    args = parser.parse_args()

    discovery = json.loads(args.services.read_text(encoding="utf-8"))
    repaired, report = repair_service_world(discovery, max_rounds=args.max_rounds)
    args.output_services.write_text(
        json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "service_count_before": report["service_count_before"],
                "service_count_after": report["service_count_after"],
                "removed_service_count": report["removed_service_count"],
                "round_count": report["round_count"],
                "remaining_headway_conflicts": report["after"]["headway"][
                    "total_normal_floor_violation_pair_count"
                ],
                "direction_balance_after": report["after"]["direction_balance"]["by_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
