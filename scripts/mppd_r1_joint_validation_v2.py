from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.mppd_r1_joint_model import (
    ACCESS_MIN_S,
    EGRESS_MIN_S,
    FINE_STAGE,
    JOINT_STATE_SCHEMA,
    TRANSFER_MIN_S,
)

NETWORK_AUTHORITY_SCHEMA = "mppd.r1-network-authority.v2"
REPORT_SCHEMA = "mppd.r1-joint-validation.v2"


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def movement_key(x: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return tuple(
        str(x[k])
        for k in ("station_id", "from_line_id", "from_direction_id", "to_line_id", "to_direction_id")
    )


def validate(state: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    violations: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    def bad(code: str, message: str) -> None:
        violations[code] += 1
        if len(examples[code]) < 20:
            examples[code].append(message)

    if state.get("schema") != JOINT_STATE_SCHEMA:
        bad("state_schema", f"expected {JOINT_STATE_SCHEMA}")
    if authority.get("schema") != NETWORK_AUTHORITY_SCHEMA:
        bad("authority_schema", f"expected {NETWORK_AUTHORITY_SCHEMA}")

    # A network authority may define the physical domain, but it may not leak the
    # answer to the count-free service reconstruction problem.
    forbidden_authority_keys = {
        "service_event_keys",
        "expected_service_count",
        "train_count",
        "realized_service_count",
        "planned_service_count",
    }
    for key in sorted(forbidden_authority_keys & set(authority)):
        bad("authority_service_leakage", f"network authority may not contain {key}")

    cfg = state.get("config", {})
    if cfg.get("train_count_is_input") is not False:
        bad("count_free_semantics", "train_count_is_input must be false")
    if cfg.get("planned_timetable_used_in_primary_inference") is not False:
        bad("planned_timetable_semantics", "planned timetable cannot enter primary inference")
    if cfg.get("legacy_candidate_roots_used_as_input") is not False:
        bad("legacy_semantics", "legacy roots cannot define the service world")
    if float(cfg.get("access_min_s", 0)) < ACCESS_MIN_S:
        bad("physical_lower_bound", "access_min_s < 15")
    if float(cfg.get("egress_min_s", 0)) < EGRESS_MIN_S:
        bad("physical_lower_bound", "egress_min_s < 15")
    if float(cfg.get("transfer_min_s", 0)) < TRANSFER_MIN_S:
        bad("physical_lower_bound", "transfer_min_s < 5")
    if float(cfg.get("event_trust_region_s", 999)) > 5.0:
        bad("trust_region", "event_trust_region_s > 5")

    stations = {str(x) for x in authority.get("station_ids", [])}
    if not stations:
        bad("authority_domain", "station_ids is empty")

    path_defs: dict[tuple[str, str, str], list[str]] = {}
    for row in authority.get("line_paths", []):
        if not isinstance(row, dict) or not {"path_id", "line_id", "direction_id", "station_ids"} <= row.keys():
            bad("authority_path", f"invalid path definition {row!r}")
            continue
        key = (str(row["path_id"]), str(row["line_id"]), str(row["direction_id"]))
        path_defs[key] = [str(x) for x in row["station_ids"]]

    services = state.get("services", [])
    if not isinstance(services, list) or not services:
        bad("service_world", "no inferred services")
        services = []
    service_ids: set[str] = set()
    service_event_count = 0
    fine_event_count = 0
    for tr in services:
        if not isinstance(tr, dict):
            bad("service_world", "non-dict service")
            continue
        sid = str(tr.get("trajectory_id", ""))
        if not sid:
            bad("service_id", "empty trajectory id")
        elif sid in service_ids:
            bad("service_id", f"duplicate {sid}")
        service_ids.add(sid)
        key = (str(tr.get("path_id")), str(tr.get("line_id")), str(tr.get("direction_id")))
        events = sorted(tr.get("events", []), key=lambda e: int(e.get("sequence_index", -1)))
        service_event_count += len(events)
        if len(events) < 2:
            bad("service_continuity", f"{sid}: fewer than two events")
            continue
        if path_defs and key not in path_defs:
            bad("service_path", f"{sid}: undeclared path {key}")
        declared = path_defs.get(key)
        actual_stations = [str(e.get("station_id")) for e in events]
        if declared is not None and actual_stations != declared:
            bad("service_path", f"{sid}: event station sequence differs from authority path")
        if stations:
            for s in actual_stations:
                if s not in stations:
                    bad("service_station", f"{sid}: station {s} outside network authority")
        for a, b in zip(events, events[1:]):
            if not finite(a.get("anchor_time_s")) or not finite(b.get("anchor_time_s")):
                bad("service_anchor", f"{sid}: non-finite anchor")
                continue
            if float(b["anchor_time_s"]) <= float(a["anchor_time_s"]):
                bad("service_continuity", f"{sid}: non-increasing anchor sequence")
        if state.get("stage") == FINE_STAGE:
            for e in events:
                arr, dep = e.get("arrival_time_s"), e.get("departure_time_s")
                if not finite(arr) or not finite(dep):
                    bad("fine_event", f"{sid}/{e.get('station_id')}: arrival/departure missing")
                    continue
                fine_event_count += 1
                if float(dep) < float(arr):
                    bad("fine_event", f"{sid}/{e.get('station_id')}: departure < arrival")
            for a, b in zip(events, events[1:]):
                if finite(a.get("departure_time_s")) and finite(b.get("arrival_time_s")):
                    if float(b["arrival_time_s"]) <= float(a["departure_time_s"]):
                        bad("service_running_time", f"{sid}: non-positive running time")

    # Service count is an inferred property of the state, not an authority target.
    metadata = state.get("metadata", {})
    if metadata.get("service_count_is_fixed") is True:
        bad("count_free_semantics", "state metadata fixes service count")
    if "expected_service_count" in metadata:
        bad("count_free_semantics", "state metadata contains expected_service_count")

    station_rows = {str(x.get("station_id")): x for x in state.get("station_movements", []) if isinstance(x, dict)}
    for s in stations - set(station_rows):
        bad("station_movement_coverage", f"missing station movement {s}")
    for s, row in station_rows.items():
        identification = row.get("identification")
        if identification not in {"DIRECTLY_IDENTIFIED", "HIERARCHICALLY_INFERRED", "UNRESOLVED"}:
            bad("station_identification", f"{s}: invalid identification label")
        for prefix, lower in (("access", ACCESS_MIN_S), ("egress", EGRESS_MIN_S)):
            q = [row.get(f"{prefix}_q05_s"), row.get(f"{prefix}_q50_s"), row.get(f"{prefix}_q95_s")]
            if identification == "UNRESOLVED" and all(v is None for v in q):
                continue
            if not all(finite(v) for v in q):
                bad("station_distribution", f"{s}: incomplete {prefix} distribution")
                continue
            q05, q50, q95 = map(float, q)
            if q05 < lower:
                bad("physical_lower_bound", f"{s}: {prefix} q05={q05}")
            if not q05 <= q50 <= q95:
                bad("station_distribution", f"{s}: non-monotone {prefix} quantiles")

    expected_moves = {
        movement_key(x)
        for x in authority.get("transfer_movements", [])
        if isinstance(x, dict)
        and {"station_id", "from_line_id", "from_direction_id", "to_line_id", "to_direction_id"} <= x.keys()
    }
    actual_moves: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    transfer_path_count = 0
    for row in state.get("transfer_movements", []):
        if not isinstance(row, dict):
            bad("transfer_movement", "non-dict transfer movement")
            continue
        try:
            key = movement_key(row)
        except KeyError:
            bad("transfer_movement", f"missing fields {row!r}")
            continue
        actual_moves[key] = row
        paths = row.get("paths", [])
        if not paths:
            bad("transfer_paths", f"{key}: no inferred transfer path")
            continue
        transfer_path_count += len(paths)
        weights = []
        for p in paths:
            vals = [p.get("q05_s"), p.get("q50_s"), p.get("q95_s"), p.get("weight")]
            if not all(finite(v) for v in vals):
                bad("transfer_paths", f"{key}: non-finite path parameters")
                continue
            q05, q50, q95, weight = map(float, vals)
            weights.append(weight)
            if q05 < TRANSFER_MIN_S:
                bad("physical_lower_bound", f"{key}: transfer q05={q05}")
            if not q05 <= q50 <= q95:
                bad("transfer_paths", f"{key}: non-monotone transfer quantiles")
            if weight <= 0:
                bad("transfer_paths", f"{key}: non-positive weight")
        if weights and abs(sum(weights) - 1.0) > 1e-6:
            bad("transfer_paths", f"{key}: weights sum to {sum(weights)}")
    for key in expected_moves - set(actual_moves):
        bad("transfer_coverage", f"missing transfer movement {key}")
    for key in set(actual_moves) - expected_moves:
        bad("transfer_coverage", f"undeclared transfer movement {key}")

    passenger = state.get("passenger_summary")
    resolved_share = None
    if not isinstance(passenger, dict):
        bad("passenger_posterior", "passenger_summary missing")
    else:
        pm, rm = passenger.get("passenger_mass"), passenger.get("resolved_mass")
        if not finite(pm) or not finite(rm) or float(pm) <= 0 or not 0 <= float(rm) <= float(pm):
            bad("passenger_posterior", "invalid passenger mass accounting")
        else:
            resolved_share = float(rm) / float(pm)
        err = passenger.get("max_time_closure_error_s")
        if err is not None and (not finite(err) or abs(float(err)) > 1e-6):
            bad("passenger_time_closure", f"max closure error={err}")

    final_ready = state.get("stage") == FINE_STAGE
    if not final_ready:
        bad("final_resolution", "final qualification requires FINE_1S state")

    passed = not violations
    return {
        "schema": REPORT_SCHEMA,
        "status": "QUALIFIED_R1_JOINT_RECONSTRUCTION_V2" if passed else "R1_JOINT_RECONSTRUCTION_V2_NOT_QUALIFIED",
        "count_free": {
            "service_count_is_authority_input": False,
            "inferred_service_count": len(services),
        },
        "coverage": {
            "authority_station_count": len(stations),
            "station_movement_count": len(station_rows),
            "authority_transfer_movement_count": len(expected_moves),
            "transfer_movement_count": len(actual_moves),
            "transfer_path_count": transfer_path_count,
            "service_event_count": service_event_count,
            "fine_arrival_departure_event_count": fine_event_count,
            "passenger_resolved_share": resolved_share,
        },
        "violation_counts": dict(sorted(violations.items())),
        "violation_examples": dict(sorted(examples.items())),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--authority", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    state = json.loads(a.state.read_text(encoding="utf-8"))
    authority = json.loads(a.authority.read_text(encoding="utf-8"))
    report = validate(state, authority)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "QUALIFIED_R1_JOINT_RECONSTRUCTION_V2":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
