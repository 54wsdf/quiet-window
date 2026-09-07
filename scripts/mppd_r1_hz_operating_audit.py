from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from scripts.mppd_r1_hz_operating_authority import (
    ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S,
    build_operating_authority,
    normal_headway_floor_for,
    validate_operating_authority,
)

AUDIT_SCHEMA = "mppd.r1-hz-operating-service-audit.v1"

# Physical-resource scopes are deliberately explicit. B_main and B_branch compete
# together on their shared trunk; their exclusive branches are audited separately.
AUDIT_RESOURCES: tuple[dict[str, Any], ...] = (
    {
        "resource_id": "B_SHARED_TRUNK",
        "line_id": "B",
        "path_ids": ("B_main", "B_branch"),
        "station_ids": tuple(range(0, 21)),
    },
    {
        "resource_id": "B_MAIN_ONLY",
        "line_id": "B",
        "path_ids": ("B_main",),
        "station_ids": tuple(range(21, 28)),
    },
    {
        "resource_id": "B_BRANCH_ONLY",
        "line_id": "B",
        "path_ids": ("B_branch",),
        "station_ids": tuple(range(28, 34)),
    },
    {
        "resource_id": "C_MAIN",
        "line_id": "C",
        "path_ids": ("C_main",),
        "station_ids": tuple(range(34, 67)),
    },
    {
        "resource_id": "A_MAIN",
        "line_id": "A",
        "path_ids": ("A_main",),
        "station_ids": (67, 68, 69, 70, 71, 72, 73, 74, 5, 75, 76, 77, 46, 78, 79, 80, 15, 16),
    },
)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _trajectory_rows(discovery: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = discovery.get("trajectories")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("service discovery requires trajectories sequence")
    rows = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("trajectory rows must be mappings")
        trajectory_id = str(item.get("trajectory_id", ""))
        if not trajectory_id:
            raise ValueError("trajectory requires trajectory_id")
        if trajectory_id in seen:
            raise ValueError(f"duplicate trajectory_id {trajectory_id}")
        seen.add(trajectory_id)
        rows.append(item)
    expected = discovery.get("inferred_service_trajectory_count")
    if expected is not None and int(expected) != len(rows):
        raise ValueError("inferred_service_trajectory_count disagrees with trajectories")
    return rows


def _events_by_station(trajectory: Mapping[str, Any]) -> dict[int, float]:
    raw = trajectory.get("events")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{trajectory.get('trajectory_id')}: events must be a sequence")
    out: dict[int, float] = {}
    for event in raw:
        if not isinstance(event, Mapping):
            raise ValueError("event rows must be mappings")
        station_raw = event.get("station", event.get("station_id"))
        time_raw = event.get("time_s", event.get("anchor_time_s"))
        try:
            station = int(station_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("event station must be integer-like") from exc
        if not _finite(time_raw):
            raise ValueError(f"event time at station {station} must be finite")
        if station in out:
            raise ValueError(f"trajectory repeats station {station}")
        out[station] = float(time_raw)
    if len(out) < 2:
        raise ValueError("trajectory must contain at least two station events")
    return out


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(float(x) for x in values)
    if len(vals) == 1:
        return vals[0]
    position = (len(vals) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return vals[lo]
    alpha = position - lo
    return vals[lo] * (1.0 - alpha) + vals[hi] * alpha


def _resource_index() -> dict[str, Mapping[str, Any]]:
    return {str(row["resource_id"]): row for row in AUDIT_RESOURCES}


def audit_service_world(
    discovery: Mapping[str, Any],
    authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = authority or build_operating_authority()
    authority_errors = validate_operating_authority(authority)
    if authority_errors:
        raise ValueError("invalid operating authority: " + "; ".join(authority_errors))
    rows = _trajectory_rows(discovery)
    resources = _resource_index()

    service_meta: dict[str, dict[str, Any]] = {}
    path_direction_counts: Counter[tuple[str, str]] = Counter()
    line_direction_counts: Counter[tuple[str, str]] = Counter()
    path_ambiguous_count = 0
    direction_ambiguous_count = 0
    for row in rows:
        tid = str(row["trajectory_id"])
        path_id = str(row.get("path_id", ""))
        direction = str(row.get("direction", row.get("direction_id", "")))
        line_id = str(row.get("afc_line", row.get("line_id", "")))
        if not path_id or direction not in {"Up", "Down"} or not line_id:
            raise ValueError(f"{tid}: incomplete line/path/direction identity")
        service_meta[tid] = {
            "trajectory_id": tid,
            "path_id": path_id,
            "direction": direction,
            "line_id": line_id,
            "events": _events_by_station(row),
            "path_ambiguous": bool(row.get("path_ambiguous", False)),
            "direction_ambiguous": bool(row.get("direction_ambiguous", False)),
        }
        path_direction_counts[(path_id, direction)] += 1
        line_direction_counts[(line_id, direction)] += 1
        path_ambiguous_count += int(bool(row.get("path_ambiguous", False)))
        direction_ambiguous_count += int(bool(row.get("direction_ambiguous", False)))

    # At every physical operational point, sort all services sharing the resource.
    # A physical pair can be adjacent at several stations; retain its minimum gap so
    # one close pair is counted once rather than once per station.
    pair_min: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for resource_id, resource in resources.items():
        allowed_paths = set(str(x) for x in resource["path_ids"])
        station_ids = tuple(int(x) for x in resource["station_ids"])
        for direction in ("Down", "Up"):
            for station in station_ids:
                events: list[tuple[float, str]] = []
                for tid, meta in service_meta.items():
                    if meta["path_id"] not in allowed_paths or meta["direction"] != direction:
                        continue
                    if station in meta["events"]:
                        events.append((float(meta["events"][station]), tid))
                events.sort(key=lambda item: (item[0], item[1]))
                for (left_t, left_id), (right_t, right_id) in zip(events, events[1:]):
                    gap = float(right_t - left_t)
                    if gap < 0:
                        raise AssertionError("sorted physical event sequence yielded negative gap")
                    pair_ids = tuple(sorted((left_id, right_id)))
                    key = (resource_id, direction, pair_ids[0], pair_ids[1])
                    current = pair_min.get(key)
                    if current is None or gap < float(current["gap_s"]):
                        midpoint = (left_t + right_t) / 2.0
                        left_path = str(service_meta[left_id]["path_id"])
                        right_path = str(service_meta[right_id]["path_id"])
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
                        floor = max(left_floor, right_floor)
                        bands = sorted({x for x in (left_band, right_band) if x})
                        pair_min[key] = {
                            "resource_id": resource_id,
                            "direction": direction,
                            "left_trajectory_id": left_id,
                            "right_trajectory_id": right_id,
                            "left_path_id": left_path,
                            "right_path_id": right_path,
                            "station_id": station,
                            "left_time_s": left_t,
                            "right_time_s": right_t,
                            "midpoint_time_s": midpoint,
                            "gap_s": gap,
                            "required_floor_s": float(floor),
                            "authority_band_ids": bands,
                            "absolute_physical_violation": gap < ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S,
                            "normal_operating_violation": gap < floor,
                        }

    all_pairs = list(pair_min.values())
    physical_violations = [x for x in all_pairs if x["absolute_physical_violation"]]
    operating_violations = [
        x
        for x in all_pairs
        if x["normal_operating_violation"] and x["required_floor_s"] > ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S
    ]
    any_violations = [x for x in all_pairs if x["normal_operating_violation"]]

    by_resource: dict[str, Any] = {}
    for resource_id in resources:
        vals = [x for x in all_pairs if x["resource_id"] == resource_id]
        gaps = [float(x["gap_s"]) for x in vals]
        by_resource[resource_id] = {
            "adjacent_pair_count": len(vals),
            "minimum_gap_s": min(gaps) if gaps else None,
            "p05_gap_s": _quantile(gaps, 0.05),
            "median_gap_s": median(gaps) if gaps else None,
            "absolute_physical_violation_pair_count": sum(
                bool(x["absolute_physical_violation"]) for x in vals
            ),
            "operating_envelope_violation_pair_count": sum(
                bool(x["normal_operating_violation"])
                and float(x["required_floor_s"]) > ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S
                for x in vals
            ),
        }

    path_balance: dict[str, Any] = {}
    for path_id in sorted({path for path, _direction in path_direction_counts}):
        down = int(path_direction_counts[(path_id, "Down")])
        up = int(path_direction_counts[(path_id, "Up")])
        path_balance[path_id] = {
            "down": down,
            "up": up,
            "signed_down_minus_up": down - up,
            "absolute_imbalance": abs(down - up),
            "balance_is_not_depot_resolved": True,
        }

    line_balance: dict[str, Any] = {}
    for line_id in sorted({line for line, _direction in line_direction_counts}):
        down = int(line_direction_counts[(line_id, "Down")])
        up = int(line_direction_counts[(line_id, "Up")])
        line_balance[line_id] = {
            "down": down,
            "up": up,
            "signed_down_minus_up": down - up,
            "absolute_imbalance": abs(down - up),
        }

    violations_sorted = sorted(
        any_violations,
        key=lambda row: (
            float(row["gap_s"]) - float(row["required_floor_s"]),
            str(row["resource_id"]),
            str(row["left_trajectory_id"]),
        ),
    )

    return {
        "schema": AUDIT_SCHEMA,
        "status": "OPERATING_CONSTRAINT_AUDIT_COMPLETED_NO_STRUCTURE_MUTATION",
        "inferred_service_count": len(rows),
        "path_ambiguous_service_count": path_ambiguous_count,
        "direction_ambiguous_service_count": direction_ambiguous_count,
        "authority_semantics": dict(authority["semantics"]),
        "headway": {
            "absolute_physical_floor_s": ABSOLUTE_PHYSICAL_HEADWAY_FLOOR_S,
            "unique_adjacent_service_pair_count": len(all_pairs),
            "absolute_physical_violation_pair_count": len(physical_violations),
            "verified_operating_envelope_violation_pair_count": len(operating_violations),
            "total_normal_floor_violation_pair_count": len(any_violations),
            # Full frontier is required by structural repair. The compact top-200
            # view remains for human diagnostics and backwards compatibility.
            "violation_pairs": violations_sorted,
            "worst_violation_pairs": violations_sorted[:200],
            "by_resource": by_resource,
        },
        "direction_balance": {
            "by_path": path_balance,
            "by_line": line_balance,
            "depot_boundary_resolution_applied": False,
            "interpretation_boundary": (
                "Raw imbalance is reported before depot/service-day boundary resolution; "
                "it is not yet a physical fleet-balance claim."
            ),
        },
        "semantics": {
            "service_count_changed": False,
            "event_times_changed": False,
            "planned_absolute_timetable_used": False,
            "planned_trip_count_used": False,
            "planned_trip_ids_used": False,
            "shared_trunk_competition_audited": True,
            "depot_network_station_mapping_resolved": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--services", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authority-output", type=Path)
    args = parser.parse_args()

    discovery = json.loads(args.services.read_text(encoding="utf-8"))
    authority = build_operating_authority()
    result = audit_service_world(discovery, authority)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.authority_output is not None:
        args.authority_output.write_text(
            json.dumps(authority, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "inferred_service_count": result["inferred_service_count"],
                "headway": {
                    "absolute_physical_violation_pair_count": result["headway"][
                        "absolute_physical_violation_pair_count"
                    ],
                    "verified_operating_envelope_violation_pair_count": result["headway"][
                        "verified_operating_envelope_violation_pair_count"
                    ],
                    "by_resource": result["headway"]["by_resource"],
                },
                "direction_balance": result["direction_balance"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
