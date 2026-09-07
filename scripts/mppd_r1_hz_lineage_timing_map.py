from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from scripts.mppd_r1_hz_operating_audit import AUDIT_RESOURCES, audit_service_world
from scripts.mppd_r1_hz_operating_authority import build_operating_authority, normal_headway_floor_for

SCHEMA = "mppd.r1-hz-lineage-timing-map.v1"
DEFAULT_PASSENGER_WEIGHTS = (0.0, 1.0, 2.0, 4.0)
AFC_WEIGHT = 0.5


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
            raise ValueError(f"invalid/duplicate trajectory {tid!r}")
        out[tid] = row
    return out


def _events(row: Mapping[str, Any]) -> dict[int, float]:
    raw = row.get("events")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("trajectory events must be sequence")
    out: dict[int, float] = {}
    for event in raw:
        if not isinstance(event, Mapping):
            raise ValueError("event must be mapping")
        station = int(event.get("station", event.get("station_id")))
        value = event.get("time_s", event.get("anchor_time_s"))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("event time must be finite")
        out[station] = float(value)
    return out


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("afc_line", row.get("line_id", ""))),
        str(row.get("path_id", "")),
        str(row.get("direction", row.get("direction_id", ""))),
    )


def _assignment(feedback: Mapping[str, Any]) -> dict[str, float]:
    raw = feedback.get("trajectory_assignment_mass", {})
    if not isinstance(raw, Mapping):
        raise ValueError("trajectory_assignment_mass must be mapping")
    return {
        str(k): max(0.0, float(v))
        for k, v in raw.items()
        if not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(float(v))
    }


def _build_groups(
    original: Mapping[str, Any],
    candidate: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Mapping[str, Any]]]:
    original_rows = _rows(original)
    candidate_rows = _rows(candidate)
    if not set(candidate_rows).issubset(original_rows):
        raise ValueError("candidate is not subset of original")
    groups = {survivor: [survivor] for survivor in candidate_rows}
    assigned_removed: set[str] = set()
    raw = lineage.get("lineages", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("lineage rows must be sequence")
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("lineage row must be mapping")
        cls = str(item.get("lineage_class", ""))
        if not cls.startswith("PARALLEL_"):
            continue
        removed = str(item.get("removed_trajectory_id", ""))
        survivor = str(item.get("survivor_trajectory_id", ""))
        if removed not in original_rows or survivor not in groups:
            raise ValueError("lineage references missing trajectory")
        if removed in assigned_removed:
            raise ValueError("removed trajectory assigned to multiple latent services")
        groups[survivor].append(removed)
        assigned_removed.add(removed)
    unresolved = (set(original_rows) - set(candidate_rows)) - assigned_removed
    if unresolved:
        raise ValueError(f"timing MAP requires resolved lineage for all removed fragments: {len(unresolved)} remain")
    return groups, original_rows


def _hypothesis_conflicts(
    hypotheses: Mapping[str, Mapping[str, Any]],
    group_of: Mapping[str, str],
) -> set[tuple[str, str]]:
    authority = build_operating_authority()
    event_maps = {hid: _events(row) for hid, row in hypotheses.items()}
    identity = {hid: _identity(row) for hid, row in hypotheses.items()}
    conflicts: set[tuple[str, str]] = set()
    for resource in AUDIT_RESOURCES:
        rid = str(resource["resource_id"])
        allowed_paths = set(str(x) for x in resource["path_ids"])
        for direction in ("Down", "Up"):
            for station in resource["station_ids"]:
                events: list[tuple[float, str, str]] = []
                for hid, row in hypotheses.items():
                    _line, path, d = identity[hid]
                    if path not in allowed_paths or d != direction or int(station) not in event_maps[hid]:
                        continue
                    events.append((event_maps[hid][int(station)], hid, path))
                events.sort(key=lambda item: (item[0], item[1]))
                for i, (left_t, left_id, left_path) in enumerate(events):
                    for right_t, right_id, right_path in events[i + 1 :]:
                        gap = right_t - left_t
                        # 450 s exceeds every currently verified lower-bound envelope.
                        if gap >= 450.0:
                            break
                        if group_of[left_id] == group_of[right_id]:
                            continue
                        midpoint = (left_t + right_t) / 2.0
                        left_floor, _ = normal_headway_floor_for(
                            authority, resource_id=rid, path_id=left_path,
                            direction=direction, event_time_s=midpoint,
                        )
                        right_floor, _ = normal_headway_floor_for(
                            authority, resource_id=rid, path_id=right_path,
                            direction=direction, event_time_s=midpoint,
                        )
                        if gap < max(left_floor, right_floor):
                            conflicts.add(tuple(sorted((left_id, right_id))))
    return conflicts


def _within_group_scores(
    groups: Mapping[str, Sequence[str]],
    original_rows: Mapping[str, Mapping[str, Any]],
    assignment: Mapping[str, float],
    passenger_weight: float,
) -> dict[str, float]:
    if passenger_weight < 0 or not math.isfinite(passenger_weight):
        raise ValueError("passenger_weight must be finite and nonnegative")
    scores: dict[str, float] = {}
    for _group, hids in groups.items():
        pvals = [math.log1p(assignment.get(hid, 0.0)) for hid in hids]
        avals = [float(original_rows[hid].get("evidence_score", 0.0) or 0.0) for hid in hids]
        pmin, pmax = min(pvals), max(pvals)
        amin, amax = min(avals), max(avals)
        for hid, pv, av in zip(hids, pvals, avals):
            pn = (pv - pmin) / (pmax - pmin) if pmax > pmin else 0.0
            an = (av - amin) / (amax - amin) if amax > amin else 0.0
            # Tiny preference for the survivor keeps exact ties stable without
            # overriding AFC/passenger evidence.
            survivor_bonus = 1e-6 if hid == _group else 0.0
            scores[hid] = passenger_weight * pn + AFC_WEIGHT * an + survivor_bonus
    return scores


def solve_timing_map(
    original: Mapping[str, Any],
    candidate: Mapping[str, Any],
    lineage: Mapping[str, Any],
    passenger_feedback: Mapping[str, Any],
    *,
    passenger_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups, original_rows = _build_groups(original, candidate, lineage)
    assignment = _assignment(passenger_feedback)
    hypothesis_ids = sorted(hid for hids in groups.values() for hid in hids)
    group_of = {hid: group for group, hids in groups.items() for hid in hids}
    hypotheses = {hid: original_rows[hid] for hid in hypothesis_ids}
    conflicts = _hypothesis_conflicts(hypotheses, group_of)
    scores = _within_group_scores(groups, original_rows, assignment, passenger_weight)
    index = {hid: i for i, hid in enumerate(hypothesis_ids)}

    rows: list[tuple[list[int], list[float], float, float]] = []
    for group, hids in sorted(groups.items()):
        rows.append(([index[hid] for hid in hids], [1.0] * len(hids), 1.0, 1.0))
    for left, right in sorted(conflicts):
        rows.append(([index[left], index[right]], [1.0, 1.0], -np.inf, 1.0))
    rr: list[int] = []
    cc: list[int] = []
    dd: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for r, (cols, vals, lo, hi) in enumerate(rows):
        rr.extend([r] * len(cols)); cc.extend(cols); dd.extend(vals); lower.append(lo); upper.append(hi)
    A = coo_matrix((dd, (rr, cc)), shape=(len(rows), len(hypothesis_ids))).tocsr()
    result = milp(
        c=-np.array([scores[hid] for hid in hypothesis_ids], dtype=float),
        integrality=np.ones(len(hypothesis_ids), dtype=int),
        bounds=Bounds(np.zeros(len(hypothesis_ids)), np.ones(len(hypothesis_ids))),
        constraints=LinearConstraint(A, np.array(lower), np.array(upper)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"lineage timing MAP failed: {result.message}")
    chosen_by_group: dict[str, str] = {}
    for group, hids in groups.items():
        chosen = [hid for hid in hids if result.x[index[hid]] >= 0.5]
        if len(chosen) != 1:
            raise AssertionError(f"latent service {group} selected {len(chosen)} timing hypotheses")
        chosen_by_group[group] = chosen[0]

    out = copy.deepcopy(dict(candidate))
    out_rows: list[dict[str, Any]] = []
    shifts: list[float] = []
    path_reassignments = 0
    non_survivor = 0
    candidate_rows = _rows(candidate)
    for group in sorted(groups):
        chosen_id = chosen_by_group[group]
        row = copy.deepcopy(dict(original_rows[chosen_id]))
        base = candidate_rows[group]
        base_events = _events(base); chosen_events = _events(row)
        shared = set(base_events) & set(chosen_events)
        shift = float(median([chosen_events[s] - base_events[s] for s in shared])) if shared else 0.0
        shifts.append(abs(shift))
        if chosen_id != group:
            non_survivor += 1
        if _identity(row)[1] != _identity(base)[1]:
            path_reassignments += 1
        row["trajectory_id"] = group
        row["lineage_timing_map"] = {
            "source_hypothesis_trajectory_id": chosen_id,
            "base_survivor_trajectory_id": group,
            "whole_trajectory_shift_from_survivor_s": shift,
            "path_reassigned": _identity(row)[1] != _identity(base)[1],
            "passenger_weight": passenger_weight,
        }
        out_rows.append(row)
    out["trajectories"] = out_rows
    out["inferred_service_trajectory_count"] = len(out_rows)
    out["semantics"] = dict(out.get("semantics", {}))
    out["semantics"].update({
        "service_count_is_fixed": False,
        "lineage_timing_map_applied": True,
        "planned_timetable_used": False,
        "planned_trip_count_used": False,
        "candidate_world_is_final_r1": False,
    })
    audit = audit_service_world(out)
    remaining = audit["headway"]["total_normal_floor_violation_pair_count"]
    if remaining != 0:
        raise AssertionError(f"MAP world retains {remaining} headway conflicts")
    report = {
        "schema": SCHEMA,
        "status": "COHERENT_LINEAGE_TIMING_MAP_COMPLETED_REQUIRES_PASSENGER_REEVALUATION",
        "passenger_weight": passenger_weight,
        "service_count": len(out_rows),
        "latent_service_group_count": len(groups),
        "timing_hypothesis_count": len(hypothesis_ids),
        "hypothesis_conflict_pair_count": len(conflicts),
        "non_survivor_hypothesis_selected_count": non_survivor,
        "path_reassignment_count": path_reassignments,
        "absolute_shift_median_s": float(median(shifts)) if shifts else 0.0,
        "absolute_shift_max_s": max(shifts) if shifts else 0.0,
        "remaining_headway_conflicts": 0,
        "solver_objective": float(-result.fun),
        "chosen_hypothesis_by_latent_service": chosen_by_group,
        "semantics": {
            "service_count_changed": False,
            "whole_trajectory_retiming_only": True,
            "arrival_departure_not_yet_separated": True,
            "planned_timetable_used": False,
            "candidate_world_is_final_r1": False,
        },
    }
    return out, report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--original-services", type=Path, required=True)
    p.add_argument("--candidate-services", type=Path, required=True)
    p.add_argument("--lineage", type=Path, required=True)
    p.add_argument("--passenger-feedback", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weights", type=float, nargs="*", default=list(DEFAULT_PASSENGER_WEIGHTS))
    a = p.parse_args()
    original = json.loads(a.original_services.read_text(encoding="utf-8"))
    candidate = json.loads(a.candidate_services.read_text(encoding="utf-8"))
    lineage = json.loads(a.lineage.read_text(encoding="utf-8"))
    passenger = json.loads(a.passenger_feedback.read_text(encoding="utf-8"))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for weight in a.weights:
        world, report = solve_timing_map(original, candidate, lineage, passenger, passenger_weight=float(weight))
        label = f"pw_{str(float(weight)).replace('.', 'p')}"
        (a.output_dir / f"services_{label}.json").write_text(json.dumps(world, ensure_ascii=False, indent=2), encoding="utf-8")
        (a.output_dir / f"report_{label}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports.append({"label": label, **{k: report[k] for k in (
            "passenger_weight", "service_count", "timing_hypothesis_count", "hypothesis_conflict_pair_count",
            "non_survivor_hypothesis_selected_count", "path_reassignment_count", "absolute_shift_median_s", "absolute_shift_max_s"
        )}})
    summary = {"schema": SCHEMA, "status": "LINEAGE_TIMING_MAP_FRONTIER_COMPLETED_NO_FINAL_TIMING_SELECTED", "candidates": reports}
    (a.output_dir / "frontier.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
