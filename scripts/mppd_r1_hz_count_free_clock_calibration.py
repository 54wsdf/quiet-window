from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import scripts.mppd_r1_hz_count_free_passenger_coverage as cov

SCHEMA_SHARD = "mppd.r1-hz-count-free-clock-calibration-shard.v2-small-step"
SCHEMA_MERGED = "mppd.r1-hz-count-free-clock-calibration.v2-small-step"
SCHEMA_SERVICE = "mppd.r1-hz-count-free-service-clock-calibrated.v2-small-step"
MAX_SINGLE_STEP_S = 5.0


def shift_values(start: int, stop: int, step: int) -> list[float]:
    if step <= 0 or stop < start:
        raise ValueError("invalid shift grid")
    values = [float(x) for x in range(start, stop + 1, step)]
    if not values or 0.0 not in values:
        raise ValueError("small-step shift grid must include zero")
    if any(abs(x) > MAX_SINGLE_STEP_S for x in values):
        raise ValueError(f"single clock update cannot exceed +/-{MAX_SINGLE_STEP_S:g} s")
    return values


def evaluate_row(row: dict[str, Any], routes: list[dict[str, Any]], index: cov.LegIndex, shift_s: float) -> str:
    # If actual service events are current passenger-facing ridge times minus c,
    # then dep-c >= entry+15 and arr-c <= exit-15 are exactly equivalent to
    # evaluating the unshifted field at entry+c and exit+c. Transfer inequalities
    # are invariant to a common clock shift.
    proxy = dict(row)
    proxy["entry_sec"] = float(row["entry_sec"]) + shift_s
    proxy["exit_sec"] = float(row["exit_sec"]) + shift_s
    failures = []
    for route in routes:
        result = cov.evaluate_route(proxy, route, index)
        if result["ok"]:
            return "FEASIBLE_COMPLETE_CHAIN"
        failures.append(result)
    return str(cov.choose_failure(failures)["reason"])


def run_shard(
    cohorts: Path,
    routes_path: Path,
    services_path: Path,
    output: Path,
    shard_index: int,
    shard_count: int,
    shifts: list[float],
) -> dict[str, Any]:
    if any(abs(c) > MAX_SINGLE_STEP_S for c in shifts):
        raise ValueError("clock calibration received a shift outside the five-second trust region")
    routes = cov.load_routes(routes_path)
    services, service_raw = cov.load_services(services_path)
    index = cov.LegIndex(services)
    stats = {
        c: {
            "passenger_mass": 0.0,
            "resolved_mass": 0.0,
            "failure_mass": Counter(),
        }
        for c in shifts
    }
    cohort_count = 0
    for _ordinal, row in cov.iter_shard_rows(cohorts, shard_index, shard_count):
        cohort_count += 1
        mass = float(row["passenger_mass"])
        rr = routes.get(cov.route_key(row))
        if not rr:
            for c in shifts:
                stats[c]["passenger_mass"] += mass
                stats[c]["failure_mass"]["NO_ROUTE_SUPPORT"] += mass
            continue
        for c in shifts:
            stats[c]["passenger_mass"] += mass
            reason = evaluate_row(row, rr, index, c)
            if reason == "FEASIBLE_COMPLETE_CHAIN":
                stats[c]["resolved_mass"] += mass
            else:
                stats[c]["failure_mass"][reason] += mass

    grid = []
    for c in shifts:
        x = stats[c]
        pm = float(x["passenger_mass"])
        rm = float(x["resolved_mass"])
        grid.append({
            "clock_shift_s": c,
            "passenger_mass": pm,
            "resolved_mass": rm,
            "resolved_share": rm / pm if pm else None,
            "failure_mass": dict(x["failure_mass"]),
        })
    result = {
        "schema": SCHEMA_SHARD,
        "service_date": service_raw.get("service_date", "2019-01-04"),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "cohort_count": cohort_count,
        "service_trajectory_count": int(service_raw["inferred_service_trajectory_count"]),
        "shift_grid": grid,
        "semantics": {
            "planned_timetable_used": False,
            "train_count_is_input": False,
            "legacy_roots_used": False,
            "common_shift_applied_to_all_service_events": True,
            "single_update_trust_region_s": MAX_SINGLE_STEP_S,
            "large_common_clock_jump_forbidden": True,
            "reestimate_passenger_world_after_each_small_step": True,
            "access_min_s": cov.ACCESS_MIN_S,
            "egress_min_s": cov.EGRESS_MIN_S,
            "transfer_min_s": cov.TRANSFER_MIN_S,
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    best = max(grid, key=lambda z: (z["resolved_mass"], -abs(z["clock_shift_s"]), -z["clock_shift_s"]))
    print(json.dumps({
        "shard_index": shard_index,
        "best_small_step_shift_s": best["clock_shift_s"],
        "best_resolved_share": best["resolved_share"],
        "zero_shift_share": next(x["resolved_share"] for x in grid if x["clock_shift_s"] == 0.0),
    }, ensure_ascii=False, indent=2))
    return result


def merge(input_dir: Path, output: Path) -> dict[str, Any]:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(input_dir.glob("shard_*.json"))]
    if not rows:
        raise SystemExit("no clock calibration shards")
    shard_count = int(rows[0]["shard_count"])
    if len(rows) != shard_count or {int(x["shard_index"]) for x in rows} != set(range(shard_count)):
        raise SystemExit("incomplete clock calibration shard set")
    shifts = [float(x["clock_shift_s"]) for x in rows[0]["shift_grid"]]
    if any(abs(c) > MAX_SINGLE_STEP_S for c in shifts):
        raise SystemExit("clock shard grid violates five-second trust region")
    totals = {c: {"passenger_mass": 0.0, "resolved_mass": 0.0, "failure_mass": Counter()} for c in shifts}
    for row in rows:
        row_by_shift = {float(x["clock_shift_s"]): x for x in row["shift_grid"]}
        if set(row_by_shift) != set(shifts):
            raise SystemExit("shift grid mismatch")
        for c in shifts:
            x = row_by_shift[c]
            totals[c]["passenger_mass"] += float(x["passenger_mass"])
            totals[c]["resolved_mass"] += float(x["resolved_mass"])
            totals[c]["failure_mass"].update({k: float(v) for k, v in x["failure_mass"].items()})
    grid = []
    for c in shifts:
        x = totals[c]
        pm = x["passenger_mass"]
        rm = x["resolved_mass"]
        grid.append({
            "clock_shift_s": c,
            "passenger_mass": pm,
            "resolved_mass": rm,
            "resolved_share": rm / pm if pm else None,
            "failure_mass": dict(x["failure_mass"]),
        })
    best = max(grid, key=lambda z: (z["resolved_mass"], -abs(z["clock_shift_s"]), -z["clock_shift_s"]))
    zero = next(x for x in grid if x["clock_shift_s"] == 0.0)
    result = {
        "schema": SCHEMA_MERGED,
        "service_date": rows[0]["service_date"],
        "status": "COUNT_FREE_SMALL_STEP_SERVICE_CLOCK_UPDATE_ESTIMATED_REQUIRES_REINFERENCE",
        "service_trajectory_count": int(rows[0]["service_trajectory_count"]),
        "best_clock_shift_s": best["clock_shift_s"],
        "best_resolved_mass": best["resolved_mass"],
        "best_resolved_share": best["resolved_share"],
        "zero_shift_resolved_mass": zero["resolved_mass"],
        "zero_shift_resolved_share": zero["resolved_share"],
        "resolved_mass_gain_vs_zero": best["resolved_mass"] - zero["resolved_mass"],
        "resolved_share_gain_vs_zero": best["resolved_share"] - zero["resolved_share"],
        "best_failure_mass": best["failure_mass"],
        "grid": grid,
        "semantics": {
            "clock_shift_definition": "actual_service_event_time = current_service_event_time - best_clock_shift_s",
            "planned_timetable_used": False,
            "service_count_fixed_during_this_update_only": True,
            "service_count_source": "count-free AFC inference",
            "single_update_trust_region_s": MAX_SINGLE_STEP_S,
            "large_common_clock_jump_forbidden": True,
            "previous_plus_60_second_solution_is_diagnostic_only_not_an_admissible_update": True,
            "each_small_step_requires_passenger_and_movement_reinference": True,
            "this_is_one_internal_update_of_the_same_joint_R1_objective": True,
            "station_specific_event_updates_are_preferred_over_repeated_global_drift": True,
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "best_clock_shift_s": result["best_clock_shift_s"],
        "best_resolved_mass": result["best_resolved_mass"],
        "best_resolved_share": result["best_resolved_share"],
        "zero_shift_resolved_share": result["zero_shift_resolved_share"],
        "resolved_mass_gain_vs_zero": result["resolved_mass_gain_vs_zero"],
        "best_failure_mass": result["best_failure_mass"],
    }, ensure_ascii=False, indent=2))
    return result


def apply_shift(services_path: Path, calibration_path: Path, output: Path) -> dict[str, Any]:
    services = json.loads(services_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    shift = float(calibration["best_clock_shift_s"])
    if abs(shift) > MAX_SINGLE_STEP_S + 1e-9:
        raise SystemExit(f"refusing inadmissible one-shot clock shift {shift:g}s; limit is +/-{MAX_SINGLE_STEP_S:g}s")
    out = json.loads(json.dumps(services))
    out["schema"] = SCHEMA_SERVICE
    out["status"] = "COUNT_FREE_SERVICE_CLOCK_SMALL_STEP_APPLIED_REQUIRES_FULL_JOINT_REINFERENCE"
    for tr in out["trajectories"]:
        tr["pre_small_step_reference_time_s"] = float(tr["reference_time_s"])
        tr["reference_time_s"] = float(tr["reference_time_s"]) - shift
        for event in tr["events"]:
            event["pre_small_step_time_s"] = float(event["time_s"])
            event["time_s"] = float(event["time_s"]) - shift
    phases = {str(k): float(v) for k, v in out.get("station_phase_nuisance_s", {}).items()}
    provisional = {s: shift + v for s, v in phases.items()}
    out["absolute_clock_calibration"] = {
        "common_shift_s": shift,
        "single_update_trust_region_s": MAX_SINGLE_STEP_S,
        "source": "one trust-region update from full-day AFC passenger chain feasibility under A>=15,E>=15,K>=5",
        "planned_timetable_used": False,
        "service_count": int(out["inferred_service_trajectory_count"]),
        "provisional_station_egress_center_s": provisional,
        "provisional_station_egress_center_min_s": min(provisional.values()) if provisional else None,
        "provisional_station_egress_center_max_s": max(provisional.values()) if provisional else None,
        "provisional_station_egress_semantics": "small common clock step plus previously inferred relative station phase; initialization only, not final egress distribution",
        "next_required_action": "rerun passenger posterior and station movement inference before any further service-time update",
    }
    out.setdefault("semantics", {})["absolute_service_clock_updated_from_passenger_feedback"] = True
    out["semantics"]["planned_timetable_used"] = False
    out["semantics"]["train_count_is_input"] = False
    out["semantics"]["large_common_clock_jump_forbidden"] = True
    out["semantics"]["single_update_trust_region_s"] = MAX_SINGLE_STEP_S
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "service_trajectory_count": out["inferred_service_trajectory_count"],
        "common_shift_s": shift,
        "single_update_trust_region_s": MAX_SINGLE_STEP_S,
        "provisional_station_egress_center_min_s": out["absolute_clock_calibration"]["provisional_station_egress_center_min_s"],
        "provisional_station_egress_center_max_s": out["absolute_clock_calibration"]["provisional_station_egress_center_max_s"],
    }, ensure_ascii=False, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--cohorts", type=Path, required=True)
    s.add_argument("--routes", type=Path, required=True)
    s.add_argument("--services", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--shard-index", type=int, required=True)
    s.add_argument("--shard-count", type=int, default=8)
    s.add_argument("--shift-start-s", type=int, default=-5)
    s.add_argument("--shift-stop-s", type=int, default=5)
    s.add_argument("--shift-step-s", type=int, default=5)
    s = sub.add_parser("merge")
    s.add_argument("--input-dir", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("apply")
    s.add_argument("--services", type=Path, required=True)
    s.add_argument("--calibration", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.command == "shard":
        run_shard(a.cohorts, a.routes, a.services, a.output, a.shard_index, a.shard_count, shift_values(a.shift_start_s, a.shift_stop_s, a.shift_step_s))
    elif a.command == "merge":
        merge(a.input_dir, a.output)
    else:
        apply_shift(a.services, a.calibration, a.output)


if __name__ == "__main__":
    main()
