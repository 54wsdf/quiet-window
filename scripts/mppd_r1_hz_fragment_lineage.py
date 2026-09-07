from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

SCHEMA = "mppd.r1-hz-operating-fragment-lineage.v1"


def _rows(world: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = world.get("trajectories")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("service world requires trajectories sequence")
    out: dict[str, Mapping[str, Any]] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("trajectory must be mapping")
        tid = str(row.get("trajectory_id", ""))
        if not tid or tid in out:
            raise ValueError(f"invalid/duplicate trajectory_id {tid!r}")
        out[tid] = row
    return out


def _events(row: Mapping[str, Any]) -> dict[str, float]:
    raw = row.get("events")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("trajectory events must be sequence")
    out: dict[str, float] = {}
    for event in raw:
        if not isinstance(event, Mapping):
            raise ValueError("event must be mapping")
        station = str(event.get("station", event.get("station_id", "")))
        value = event.get("time_s", event.get("anchor_time_s"))
        if not station or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("trajectory event requires station and finite time")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("trajectory event time must be finite")
        out[station] = value
    return out


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    line = str(row.get("afc_line", row.get("line_id", "")))
    path = str(row.get("path_id", ""))
    direction = str(row.get("direction", row.get("direction_id", "")))
    return line, path, direction


def _geometry(removed: Mapping[str, Any], survivor: Mapping[str, Any]) -> dict[str, Any] | None:
    r_line, r_path, r_dir = _identity(removed)
    s_line, s_path, s_dir = _identity(survivor)
    if r_line != s_line or r_dir != s_dir:
        return None
    cross_path = r_path != s_path
    # The only admitted cross-path lineage is the known B main/branch shared trunk.
    if cross_path and not (r_line == "B" and {r_path, s_path} == {"B_main", "B_branch"}):
        return None
    re = _events(removed)
    se = _events(survivor)
    shared = sorted(set(re) & set(se), key=lambda x: int(x) if x.isdigit() else x)
    if len(shared) < 4:
        return None
    diffs = [re[station] - se[station] for station in shared]
    center = float(median(diffs))
    residual = [abs(value - center) for value in diffs]
    residual_sorted = sorted(residual)
    p90 = residual_sorted[min(len(residual_sorted) - 1, int(math.ceil(0.90 * len(residual_sorted))) - 1)]
    return {
        "shared_station_count": len(shared),
        "shared_station_ids": shared,
        "median_offset_s_removed_minus_survivor": center,
        "absolute_median_offset_s": abs(center),
        "p90_parallel_residual_s": float(p90),
        "max_parallel_residual_s": max(residual),
        "cross_path": cross_path,
    }


def _candidate_score(geometry: Mapping[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(geometry["p90_parallel_residual_s"]),
        float(geometry["absolute_median_offset_s"]),
        1.0 if geometry["cross_path"] else 0.0,
        -int(geometry["shared_station_count"]),
    )


def _assignment_mass(feedback: Mapping[str, Any]) -> dict[str, float]:
    raw = feedback.get("trajectory_assignment_mass", {})
    if not isinstance(raw, Mapping):
        raise ValueError("trajectory_assignment_mass must be mapping")
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        out[str(key)] = max(0.0, float(value))
    return out


def audit_fragment_lineage(
    original: Mapping[str, Any],
    candidate: Mapping[str, Any],
    passenger_feedback: Mapping[str, Any],
    *,
    parallel_residual_gate_s: float = 20.0,
    max_offset_s: float = 450.0,
) -> dict[str, Any]:
    if parallel_residual_gate_s < 0 or max_offset_s <= 0:
        raise ValueError("lineage gates must be positive")
    original_rows = _rows(original)
    candidate_rows = _rows(candidate)
    if not set(candidate_rows).issubset(original_rows):
        raise ValueError("candidate is not a death-only subset of original services")
    removed_ids = sorted(set(original_rows) - set(candidate_rows))
    assignment = _assignment_mass(passenger_feedback)

    lineage_rows: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    classes: Counter[str] = Counter()
    for removed_id in removed_ids:
        removed = original_rows[removed_id]
        candidates: list[tuple[tuple[float, float, float, int], str, dict[str, Any]]] = []
        for survivor_id, survivor in candidate_rows.items():
            geometry = _geometry(removed, survivor)
            if geometry is None:
                continue
            if float(geometry["absolute_median_offset_s"]) > max_offset_s:
                continue
            candidates.append((_candidate_score(geometry), survivor_id, geometry))
        if not candidates:
            row = {
                "removed_trajectory_id": removed_id,
                "lineage_class": "UNRESOLVED_NO_COMPATIBLE_SURVIVOR",
                "baseline_assignment_mass": assignment.get(removed_id, 0.0),
            }
            lineage_rows.append(row)
            classes[row["lineage_class"]] += 1
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        _score, survivor_id, geometry = candidates[0]
        if float(geometry["p90_parallel_residual_s"]) <= parallel_residual_gate_s:
            lineage_class = (
                "PARALLEL_SHARED_TRUNK_ALTERNATE_PATH"
                if geometry["cross_path"]
                else "PARALLEL_SAME_PATH_FRAGMENT"
            )
        else:
            lineage_class = "UNRESOLVED_NONPARALLEL_FRAGMENT"
        row = {
            "removed_trajectory_id": removed_id,
            "survivor_trajectory_id": survivor_id,
            "lineage_class": lineage_class,
            "baseline_assignment_mass": assignment.get(removed_id, 0.0),
            "survivor_baseline_assignment_mass": assignment.get(survivor_id, 0.0),
            **geometry,
        }
        lineage_rows.append(row)
        classes[lineage_class] += 1
        if lineage_class.startswith("PARALLEL_"):
            groups[survivor_id].append(row)

    group_rows: list[dict[str, Any]] = []
    for survivor_id, members in sorted(groups.items()):
        survivor_mass = assignment.get(survivor_id, 0.0)
        fragment_mass = sum(float(row["baseline_assignment_mass"]) for row in members)
        offsets = [float(row["median_offset_s_removed_minus_survivor"]) for row in members]
        # Evidence-preserving retiming is deliberately not applied here.  This
        # suggested window only records the latent timing support that the next
        # joint AFC+passenger update should score.
        group_rows.append({
            "survivor_trajectory_id": survivor_id,
            "parallel_fragment_count": len(members),
            "parallel_fragment_ids": [row["removed_trajectory_id"] for row in members],
            "survivor_baseline_assignment_mass": survivor_mass,
            "fragment_baseline_assignment_mass": fragment_mass,
            "total_lineage_assignment_mass": survivor_mass + fragment_mass,
            "fragment_offset_min_s": min(offsets),
            "fragment_offset_max_s": max(offsets),
            "fragment_offset_median_s": float(median(offsets)),
            "event_retime_applied": False,
        })

    parallel_rows = [row for row in lineage_rows if str(row["lineage_class"]).startswith("PARALLEL_")]
    parallel_mass = sum(float(row.get("baseline_assignment_mass", 0.0)) for row in parallel_rows)
    removed_mass = sum(float(assignment.get(tid, 0.0)) for tid in removed_ids)
    offsets = [float(row["absolute_median_offset_s"]) for row in parallel_rows]
    residuals = [float(row["p90_parallel_residual_s"]) for row in parallel_rows]
    return {
        "schema": SCHEMA,
        "status": "OPERATING_FRAGMENT_LINEAGE_AUDITED_NO_EVENT_RETIME_APPLIED",
        "service_count_original": len(original_rows),
        "service_count_candidate": len(candidate_rows),
        "removed_service_count": len(removed_ids),
        "lineage_class_counts": dict(sorted(classes.items())),
        "parallel_lineage_count": len(parallel_rows),
        "parallel_lineage_share_of_removed": len(parallel_rows) / len(removed_ids) if removed_ids else 0.0,
        "removed_baseline_assignment_mass": removed_mass,
        "parallel_lineage_baseline_assignment_mass": parallel_mass,
        "parallel_lineage_assignment_mass_share_of_removed": parallel_mass / removed_mass if removed_mass else None,
        "parallel_absolute_offset_median_s": float(median(offsets)) if offsets else None,
        "parallel_p90_residual_median_s": float(median(residuals)) if residuals else None,
        "survivor_group_count": len(group_rows),
        "lineages": lineage_rows,
        "survivor_groups": group_rows,
        "semantics": {
            "candidate_service_count_changed": False,
            "event_times_changed": False,
            "deleted_fragment_evidence_discarded": False,
            "physical_vehicle_identity_claimed": False,
            "planned_timetable_used": False,
            "lineage_is_evidence_association_not_proven_same_vehicle": True,
            "candidate_world_is_final_r1": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-services", type=Path, required=True)
    parser.add_argument("--candidate-services", type=Path, required=True)
    parser.add_argument("--passenger-feedback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel-residual-gate-s", type=float, default=20.0)
    parser.add_argument("--max-offset-s", type=float, default=450.0)
    args = parser.parse_args()
    result = audit_fragment_lineage(
        json.loads(args.original_services.read_text(encoding="utf-8")),
        json.loads(args.candidate_services.read_text(encoding="utf-8")),
        json.loads(args.passenger_feedback.read_text(encoding="utf-8")),
        parallel_residual_gate_s=args.parallel_residual_gate_s,
        max_offset_s=args.max_offset_s,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "N_original": result["service_count_original"],
        "N_candidate": result["service_count_candidate"],
        "removed": result["removed_service_count"],
        "lineage_classes": result["lineage_class_counts"],
        "parallel_lineage_share": result["parallel_lineage_share_of_removed"],
        "parallel_assignment_share": result["parallel_lineage_assignment_mass_share_of_removed"],
        "median_abs_offset_s": result["parallel_absolute_offset_median_s"],
        "survivor_groups": result["survivor_group_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
