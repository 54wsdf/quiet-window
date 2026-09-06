from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from pathlib import Path
from typing import Any

import scripts.mppd_r1_hz_count_free_service as afc

SCHEMA = "mppd.r1-hz-afc-anchor-frontier.v1"
BIN_S = 5


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def nearest_event(events: list[afc.Event], t: float) -> tuple[afc.Event | None, float | None]:
    if not events:
        return None, None
    centers = [float(e.center_s) for e in events]
    i = bisect.bisect_left(centers, float(t))
    candidates = []
    if i < len(events):
        candidates.append(events[i])
    if i > 0:
        candidates.append(events[i - 1])
    best = min(candidates, key=lambda e: (abs(float(e.center_s) - float(t)), -float(e.weight)))
    return best, abs(float(best.center_s) - float(t))


def service_maps(services: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    trajectories: dict[str, dict[str, Any]] = {}
    events: dict[tuple[str, int], dict[str, Any]] = {}
    for tr in services.get("trajectories", []):
        sid = str(tr["trajectory_id"])
        trajectories[sid] = tr
        for e in tr.get("events", []):
            events[(sid, int(e["station"]))] = e
    return trajectories, events


def audit_updates(
    services: dict[str, Any],
    proposal: dict[str, Any],
    station_events: dict[int, list[afc.Event]],
) -> list[dict[str, Any]]:
    trajectories, event_map = service_maps(services)
    rows: list[dict[str, Any]] = []
    for u in proposal.get("updates", []):
        sid = str(u["trajectory_id"])
        station = int(u["station_id"])
        delta = float(u["delta_s"])
        tr = trajectories[sid]
        event = event_map[(sid, station)]
        phase = float(event.get("station_phase_nuisance_s", 0.0))
        anchor_before = float(event["time_s"])
        pulse_before = anchor_before + phase
        pulse_after = pulse_before + delta
        b_event, b_dist = nearest_event(station_events.get(station, []), pulse_before)
        a_event, a_dist = nearest_event(station_events.get(station, []), pulse_after)
        residual_p90 = tr.get("residual_p90_abs_s")
        supported_before = bool(
            b_dist is not None and finite(residual_p90) and float(b_dist) <= float(residual_p90) + 1e-9
        )
        distance_change = None if b_dist is None or a_dist is None else float(a_dist) - float(b_dist)
        nearest_weight = float(b_event.weight) if b_event is not None else 0.0
        rows.append(
            {
                "trajectory_id": sid,
                "station_id": station,
                "delta_s": delta,
                "direct_net_mass": float(u.get("direct_net_mass", 0.0)),
                "direct_rescue_mass": float(u.get("direct_rescue_mass", 0.0)),
                "direct_damage_mass": float(u.get("direct_damage_mass", 0.0)),
                "path_ambiguous": bool(u.get("path_ambiguous", tr.get("path_ambiguous", False))),
                "trajectory_residual_p90_abs_s": float(residual_p90) if finite(residual_p90) else None,
                "station_phase_nuisance_s": phase,
                "predicted_passenger_pulse_before_s": pulse_before,
                "predicted_passenger_pulse_after_s": pulse_after,
                "nearest_afc_pulse_before_id": b_event.event_id if b_event is not None else None,
                "nearest_afc_pulse_after_id": a_event.event_id if a_event is not None else None,
                "nearest_afc_pulse_before_distance_s": b_dist,
                "nearest_afc_pulse_after_distance_s": a_dist,
                "afc_nearest_distance_change_s": distance_change,
                "nearest_afc_pulse_weight_before": nearest_weight,
                "afc_weighted_distance_change": None if distance_change is None else nearest_weight * distance_change,
                "nearest_afc_pulse_identity_changed": bool(
                    b_event is not None and a_event is not None and b_event.event_id != a_event.event_id
                ),
                "afc_supported_before_under_trajectory_residual_p90": supported_before,
            }
        )
    return rows


def q(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    i = int(round((len(xs) - 1) * p))
    return float(xs[i])


def aggregate(rows: list[dict[str, Any]], lambda_event: float) -> dict[str, Any]:
    kept = [r for r in rows if float(r["direct_net_mass"]) > float(lambda_event)]
    changes = [float(r["afc_nearest_distance_change_s"]) for r in kept if r["afc_nearest_distance_change_s"] is not None]
    weighted = [float(r["afc_weighted_distance_change"]) for r in kept if r["afc_weighted_distance_change"] is not None]
    before = [float(r["nearest_afc_pulse_before_distance_s"]) for r in kept if r["nearest_afc_pulse_before_distance_s"] is not None]
    after = [float(r["nearest_afc_pulse_after_distance_s"]) for r in kept if r["nearest_afc_pulse_after_distance_s"] is not None]
    return {
        "lambda_event": float(lambda_event),
        "modified_event_count": len(kept),
        "afc_evaluable_event_count": len(changes),
        "afc_supported_before_count": sum(bool(r["afc_supported_before_under_trajectory_residual_p90"]) for r in kept),
        "afc_distance_improved_count": sum(x < -1e-9 for x in changes),
        "afc_distance_unchanged_count": sum(abs(x) <= 1e-9 for x in changes),
        "afc_distance_worsened_count": sum(x > 1e-9 for x in changes),
        "nearest_afc_pulse_identity_changed_count": sum(bool(r["nearest_afc_pulse_identity_changed"]) for r in kept),
        "baseline_nearest_distance_median_s": statistics.median(before) if before else None,
        "updated_nearest_distance_median_s": statistics.median(after) if after else None,
        "afc_distance_change_sum_s": sum(changes),
        "afc_distance_change_mean_s": statistics.mean(changes) if changes else None,
        "afc_distance_change_median_s": statistics.median(changes) if changes else None,
        "afc_distance_change_p90_s": q(changes, 0.90),
        "afc_weighted_distance_change_sum": sum(weighted),
        "path_ambiguous_modified_event_count": sum(bool(r["path_ambiguous"]) for r in kept),
        "direct_net_mass_sum_not_deduplicated": sum(float(r["direct_net_mass"]) for r in kept),
    }


def run(
    input_paths: list[Path],
    service_date: str,
    services_path: Path,
    proposal_path: Path,
    passenger_frontier_path: Path,
    output: Path,
) -> dict[str, Any]:
    services = json.loads(services_path.read_text(encoding="utf-8"))
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    passenger = json.loads(passenger_frontier_path.read_text(encoding="utf-8"))
    counts, input_meta = afc.load_exit_counts(input_paths, service_date, BIN_S)
    station_events = afc.detect_station_events(counts, BIN_S)
    audited = audit_updates(services, proposal, station_events)
    passenger_by_lambda = {float(x["lambda_event"]): x for x in passenger.get("points", [])}
    lambdas = sorted(passenger_by_lambda)
    points = []
    for lam in lambdas:
        a = aggregate(audited, lam)
        p = passenger_by_lambda[lam]
        if int(a["modified_event_count"]) != int(p["modified_event_count"]):
            raise ValueError(f"lambda={lam}: proposal/AFC/passenger modified-event counts disagree")
        points.append(
            {
                **a,
                "passenger_resolved_mass": float(p["resolved_mass"]),
                "passenger_resolved_mass_gain": float(p["resolved_mass_gain"]),
                "passenger_gain_per_modified_event": p.get("gain_per_modified_event"),
                "passenger_pareto_nondominated": bool(p.get("pareto_nondominated", False)),
            }
        )

    result = {
        "schema": SCHEMA,
        "status": "AFC_ANCHOR_DEGRADATION_FRONTIER_COMPLETED_NO_ITERATION1_SELECTED",
        "service_date": service_date,
        "service_trajectory_count": int(services["inferred_service_trajectory_count"]),
        "proposal_event_count": len(audited),
        "raw_afc": input_meta,
        "detected_station_event_count": sum(len(v) for v in station_events.values()),
        "stations_with_detected_events": sum(bool(v) for v in station_events.values()),
        "points": points,
        "event_audit": audited,
        "semantics": {
            "afc_anchor_definition": "service_event_time + station_phase_nuisance compared with nearest observed exit-AFC pulse detected by the same 5 s count-free event detector",
            "distance_change_sign": "negative improves AFC anchor agreement; positive worsens it",
            "this_is_not_a_normalized_log_likelihood": True,
            "no_planned_timetable_used": True,
            "no_final_lambda_selected": True,
            "final_iteration1_requires_joint_passenger_afc_and_structure_stability_decision": True,
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "detected_station_event_count": result["detected_station_event_count"],
        "points": points,
    }, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, action="append", required=True)
    p.add_argument("--service-date", required=True)
    p.add_argument("--services", type=Path, required=True)
    p.add_argument("--proposal", type=Path, required=True)
    p.add_argument("--passenger-frontier", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.input, a.service_date, a.services, a.proposal, a.passenger_frontier, a.output)


if __name__ == "__main__":
    main()
