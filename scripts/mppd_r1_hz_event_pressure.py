from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import scripts.mppd_r1_hz_count_free_passenger_coverage as cov

SCHEMA = "mppd.r1-hz-event-pressure.v1"
TRUST_REGION_S = 5.0


def event_key(trajectory_id: str, station: int) -> str:
    return f"{trajectory_id}|station={int(station)}"


def detailed_evaluate(row: dict[str, Any], route: dict[str, Any], index: cov.LegIndex) -> dict[str, Any]:
    legs = route.get("ride_legs", [])
    if not legs:
        return {"ok": False, "reason": "ROUTE_HAS_NO_RIDE_LEGS", "progress": -1, "chain": []}
    entry = float(row["entry_sec"])
    exit_t = float(row["exit_sec"])
    ready = entry + cov.ACCESS_MIN_S
    chain: list[dict[str, Any]] = []
    for j, leg in enumerate(legs):
        ride = index.earliest_arrival(leg, ready)
        if ride is None:
            reason = "FIRST_LEG_NO_SERVICE" if j == 0 else "TRANSFER_NO_DOWNSTREAM_SERVICE"
            return {
                "ok": False,
                "reason": reason,
                "progress": j - 1,
                "chain": chain,
                "failed_leg_index": j,
                "failed_ready_s": ready,
            }
        dep, arr, sid = ride
        chain.append(
            {
                "trajectory_id": sid,
                "from_station": int(leg["from_station"]),
                "to_station": int(leg["to_station"]),
                "departure_s": float(dep),
                "arrival_s": float(arr),
            }
        )
        if j < len(legs) - 1:
            ready = float(arr) + cov.TRANSFER_MIN_S
    final = float(chain[-1]["arrival_s"])
    deadline = exit_t - cov.EGRESS_MIN_S
    if final > deadline:
        return {
            "ok": False,
            "reason": "FINAL_EGRESS_HORIZON",
            "progress": len(legs),
            "chain": chain,
            "excess_s": final - deadline,
            "latest_allowed_arrival_s": deadline,
        }
    return {
        "ok": True,
        "reason": "FEASIBLE_COMPLETE_CHAIN",
        "progress": len(legs),
        "chain": chain,
    }


def choose_detailed_failure(failures: list[dict[str, Any]]) -> dict[str, Any]:
    rank = {
        "FINAL_EGRESS_HORIZON": 3,
        "TRANSFER_NO_DOWNSTREAM_SERVICE": 2,
        "FIRST_LEG_NO_SERVICE": 1,
        "ROUTE_HAS_NO_RIDE_LEGS": 0,
    }
    return max(
        failures,
        key=lambda x: (
            int(x.get("progress", -99)),
            rank.get(str(x.get("reason")), -1),
            -float(x.get("excess_s", 0.0)),
        ),
    )


def previous_service(index: cov.LegIndex, leg: dict[str, Any], ready_s: float) -> tuple[float, float, str] | None:
    data = index.get(leg)
    i = bisect.bisect_left(data["deps"], ready_s)
    if i <= 0:
        return None
    return data["rows"][i - 1]


def add(counter: dict[str, Counter[str]], key: str, field: str, mass: float) -> None:
    counter[key][field] += float(mass)


def run(cohorts_path: Path, routes_path: Path, services_path: Path, output: Path) -> dict[str, Any]:
    routes = cov.load_routes(routes_path)
    services, service_raw = cov.load_services(services_path)
    index = cov.LegIndex(services)

    event_pressure: dict[str, Counter[str]] = defaultdict(Counter)
    event_meta: dict[str, dict[str, Any]] = {}
    structural_pressure: Counter[str] = Counter()
    passenger_mass = 0.0
    resolved_mass = 0.0
    failure_mass: Counter[str] = Counter()
    final_excess_hist: Counter[int] = Counter()

    def remember(sid: str, station: int) -> str:
        key = event_key(sid, station)
        if key not in event_meta:
            tr = services[sid]
            event_meta[key] = {
                "trajectory_id": sid,
                "station_id": int(station),
                "path_id": tr["path_id"],
                "direction": tr["direction"],
                "path_ambiguous": bool(tr.get("path_ambiguous", False)),
                "anchor_time_s": float(tr["events"][str(int(station))]["time_s"]),
                "support_weight": float(tr.get("support_weight", 0.0)),
                "evidence_score": float(tr.get("evidence_score", 0.0)),
            }
        return key

    pf = pq.ParquetFile(cohorts_path)
    for batch in pf.iter_batches(batch_size=50000):
        for row in batch.to_pylist():
            mass = float(row["passenger_mass"])
            passenger_mass += mass
            rr = routes.get(cov.route_key(row))
            if not rr:
                failure_mass["NO_ROUTE_SUPPORT"] += mass
                continue

            success = None
            success_route = None
            failures: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for route in rr:
                result = detailed_evaluate(row, route, index)
                if result["ok"]:
                    success = result
                    success_route = route
                    break
                failures.append((result, route))

            if success is not None and success_route is not None:
                resolved_mass += mass
                chain = success["chain"]
                entry = float(row["entry_sec"])
                exit_t = float(row["exit_sec"])

                first = chain[0]
                access_slack = float(first["departure_s"]) - entry - cov.ACCESS_MIN_S
                if access_slack < TRUST_REGION_S:
                    key = remember(str(first["trajectory_id"]), int(first["from_station"]))
                    add(event_pressure, key, "protect_against_minus5_mass", mass)

                final = chain[-1]
                egress_slack = exit_t - float(final["arrival_s"]) - cov.EGRESS_MIN_S
                if egress_slack < TRUST_REGION_S:
                    key = remember(str(final["trajectory_id"]), int(final["to_station"]))
                    add(event_pressure, key, "protect_against_plus5_mass", mass)

                for left, right in zip(chain, chain[1:]):
                    transfer_slack = float(right["departure_s"]) - float(left["arrival_s"]) - cov.TRANSFER_MIN_S
                    if transfer_slack < TRUST_REGION_S:
                        up_key = remember(str(left["trajectory_id"]), int(left["to_station"]))
                        down_key = remember(str(right["trajectory_id"]), int(right["from_station"]))
                        add(event_pressure, up_key, "protect_against_plus5_mass", mass)
                        add(event_pressure, down_key, "protect_against_minus5_mass", mass)
                continue

            chosen = choose_detailed_failure([x[0] for x in failures])
            chosen_route = next(route for result, route in failures if result is chosen)
            reason = str(chosen["reason"])
            failure_mass[reason] += mass

            if reason == "FINAL_EGRESS_HORIZON" and chosen.get("chain"):
                final = chosen["chain"][-1]
                key = remember(str(final["trajectory_id"]), int(final["to_station"]))
                excess = float(chosen["excess_s"])
                final_excess_hist[int(min(3600, round(excess)))] += mass
                add(event_pressure, key, "minus5_residual_reduction_mass_seconds", mass * min(TRUST_REGION_S, excess))
                add(event_pressure, key, "minus5_pressure_mass", mass)
                if excess <= TRUST_REGION_S + 1e-9:
                    add(event_pressure, key, "resolvable_if_minus5_mass", mass)
                continue

            if reason in {"FIRST_LEG_NO_SERVICE", "TRANSFER_NO_DOWNSTREAM_SERVICE"}:
                j = int(chosen["failed_leg_index"])
                ready = float(chosen["failed_ready_s"])
                leg = chosen_route["ride_legs"][j]
                prev = previous_service(index, leg, ready)
                if prev is not None:
                    dep, _arr, sid = prev
                    missed = ready - float(dep)
                    if 0 < missed <= TRUST_REGION_S + 1e-9:
                        key = remember(str(sid), int(leg["from_station"]))
                        add(event_pressure, key, "resolvable_if_plus5_mass", mass)
                        add(event_pressure, key, "plus5_pressure_mass", mass)
                        continue
                opts = ",".join(sorted(f"{x['path_id']}:{x['direction']}" for x in leg["compatible_service_options"]))
                structural_pressure[f"{reason}|{int(leg['from_station'])}->{int(leg['to_station'])}|{opts}|bin={int(ready//300)}"] += mass

    rows = []
    for key, counts in event_pressure.items():
        meta = event_meta[key]
        minus_gain = float(counts["resolvable_if_minus5_mass"])
        plus_gain = float(counts["resolvable_if_plus5_mass"])
        minus_risk = float(counts["protect_against_minus5_mass"])
        plus_risk = float(counts["protect_against_plus5_mass"])
        rows.append(
            {
                **meta,
                "resolvable_if_minus5_mass": minus_gain,
                "resolvable_if_plus5_mass": plus_gain,
                "protect_against_minus5_mass": minus_risk,
                "protect_against_plus5_mass": plus_risk,
                "minus5_pressure_mass": float(counts["minus5_pressure_mass"]),
                "plus5_pressure_mass": float(counts["plus5_pressure_mass"]),
                "minus5_residual_reduction_mass_seconds": float(counts["minus5_residual_reduction_mass_seconds"]),
                "passenger_feasibility_net_mass_minus5": minus_gain - minus_risk,
                "passenger_feasibility_net_mass_plus5": plus_gain - plus_risk,
            }
        )
    rows.sort(
        key=lambda x: max(
            abs(float(x["passenger_feasibility_net_mass_minus5"])),
            abs(float(x["passenger_feasibility_net_mass_plus5"])),
            float(x["minus5_pressure_mass"]),
            float(x["plus5_pressure_mass"]),
        ),
        reverse=True,
    )

    result = {
        "schema": SCHEMA,
        "status": "EVENT_LEVEL_PASSENGER_PRESSURE_EXTRACTED_NOT_YET_APPLIED",
        "service_date": service_raw.get("service_date", "2019-01-04"),
        "service_trajectory_count": int(service_raw["inferred_service_trajectory_count"]),
        "passenger_mass": passenger_mass,
        "resolved_mass": resolved_mass,
        "resolved_share": resolved_mass / passenger_mass if passenger_mass else None,
        "failure_mass": dict(failure_mass),
        "event_pressure_count": len(rows),
        "event_pressure": rows,
        "structural_birth_or_reassignment_pressure": [
            {"key": k, "passenger_mass": float(v)} for k, v in structural_pressure.most_common()
        ],
        "final_egress_excess_histogram_s": {str(k): float(v) for k, v in sorted(final_excess_hist.items())},
        "semantics": {
            "single_candidate_event_move_s": TRUST_REGION_S,
            "pressure_is_passenger_feasibility_only": True,
            "pressure_is_not_a_complete_joint_objective": True,
            "afc_ridge_likelihood_and_service_smoothness_must_be_combined_before_apply": True,
            "no_event_time_was_modified": True,
            "planned_timetable_used": False,
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "service_trajectory_count": result["service_trajectory_count"],
        "passenger_mass": result["passenger_mass"],
        "resolved_mass": result["resolved_mass"],
        "resolved_share": result["resolved_share"],
        "failure_mass": result["failure_mass"],
        "event_pressure_count": result["event_pressure_count"],
        "top_event_pressure": rows[:10],
        "top_structural_pressure": result["structural_birth_or_reassignment_pressure"][:10],
    }, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohorts", type=Path, required=True)
    p.add_argument("--routes", type=Path, required=True)
    p.add_argument("--services", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.cohorts, a.routes, a.services, a.output)


if __name__ == "__main__":
    main()
