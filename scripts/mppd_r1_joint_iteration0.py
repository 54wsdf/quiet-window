from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.mppd_r1_joint_model import (
    JointState,
    PassengerPosteriorSummary,
    R1Config,
    ServiceEventState,
    ServiceTrajectoryState,
    StationMovementState,
)


def load_warm_state(raw: dict[str, Any]) -> JointState:
    services = []
    for tr in raw.get("services", []):
        services.append(
            ServiceTrajectoryState(
                trajectory_id=str(tr["trajectory_id"]),
                line_id=str(tr["line_id"]),
                direction_id=str(tr["direction_id"]),
                path_id=str(tr["path_id"]),
                events=[ServiceEventState(**e) for e in tr.get("events", [])],
                path_ambiguous=bool(tr.get("path_ambiguous", False)),
                direction_ambiguous=bool(tr.get("direction_ambiguous", False)),
                support_weight=float(tr.get("support_weight", 0.0)),
                evidence_score=float(tr.get("evidence_score", 0.0)),
                warm_start_source=tr.get("warm_start_source"),
            )
        )
    return JointState(
        schema=str(raw["schema"]),
        stage=str(raw["stage"]),
        iteration=int(raw.get("iteration", 0)),
        config=R1Config(**raw.get("config", {})),
        services=services,
        metadata=dict(raw.get("metadata", {})),
    )


def assemble_iteration0(
    warm_raw: dict[str, Any],
    movement_raw: dict[str, Any],
    passenger_raw: dict[str, Any],
) -> JointState:
    state = load_warm_state(warm_raw)
    if abs(float(movement_raw.get("service_clock_shift_s", 0.0))) > 1e-9:
        raise ValueError("iteration-zero movement initialization must use unshifted count-free services")
    if int(movement_raw["service_trajectory_count"]) != state.inferred_service_count:
        raise ValueError("movement fit and joint warm start disagree on service count")
    if int(passenger_raw["count_free_service_trajectory_count"]) != state.inferred_service_count:
        raise ValueError("passenger feedback and joint warm start disagree on service count")

    access = movement_raw.get("access_intervals", {})
    egress = movement_raw.get("egress_intervals", {})
    station_ids = sorted(set(access) | set(egress), key=lambda x: int(x))
    station_movements: list[StationMovementState] = []
    for sid in station_ids:
        a = access.get(sid, {})
        e = egress.get(sid, {})
        a_direct = bool(a.get("identified_from_current_day", False))
        e_direct = bool(e.get("identified_from_current_day", False))
        identification = "DIRECTLY_IDENTIFIED" if a_direct and e_direct else "HIERARCHICALLY_INFERRED"
        station_movements.append(
            StationMovementState(
                station_id=str(sid),
                identification=identification,
                access_q05_s=float(a["q05_s"]),
                access_q50_s=float(a["median_s"]),
                access_q95_s=float(a["q95_s"]),
                egress_q05_s=float(e["q05_s"]),
                egress_q50_s=float(e["median_s"]),
                egress_q95_s=float(e["q95_s"]),
                evidence_mass=min(float(a.get("evidence_mass", 0.0)), float(e.get("evidence_mass", 0.0))),
            )
        )

    failure = passenger_raw.get("failure_mass", {})
    passenger = PassengerPosteriorSummary(
        passenger_mass=float(passenger_raw["passenger_mass"]),
        resolved_mass=float(passenger_raw["resolved_mass"]),
        no_route_support_mass=float(failure.get("NO_ROUTE_SUPPORT", 0.0)),
        unresolved_with_route_mass=float(passenger_raw.get("unresolved_route_exists_mass", 0.0)),
        map_chain_change_share=None,
        max_time_closure_error_s=None,
    )

    state.station_movements = station_movements
    # The current movement-fit transfer keys aggregate direction. They are kept as
    # an initialization sidecar only; the new R1 state will not pretend they are
    # direction-specific transfer posteriors.
    state.transfer_movements = []
    state.passenger_summary = passenger
    state.metadata = dict(state.metadata)
    state.metadata.update(
        {
            "iteration0_initialized": True,
            "movement_initialization_service_clock_shift_s": 0.0,
            "movement_initialization_access_hazard": movement_raw.get("access_hazard"),
            "movement_initialization_transfer_hazard": movement_raw.get("transfer_hazard"),
            "direction_aggregated_transfer_initialization": movement_raw.get("transfer_path_intervals", {}),
            "direction_specific_transfer_posterior_pending": True,
            "posterior_reweight_pending": True,
            "event_level_local_update_pending": True,
        }
    )
    errors = state.validate()
    if errors:
        raise ValueError("invalid iteration-zero joint state: " + "; ".join(errors[:20]))
    return state


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--warm-state", type=Path, required=True)
    p.add_argument("--movement-fit", type=Path, required=True)
    p.add_argument("--passenger-feedback", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()

    warm = json.loads(a.warm_state.read_text(encoding="utf-8"))
    movement = json.loads(a.movement_fit.read_text(encoding="utf-8"))
    passenger = json.loads(a.passenger_feedback.read_text(encoding="utf-8"))
    state = assemble_iteration0(warm, movement, passenger)
    state.write(a.output)

    direct = sum(x.identification == "DIRECTLY_IDENTIFIED" for x in state.station_movements)
    hierarchical = sum(x.identification == "HIERARCHICALLY_INFERRED" for x in state.station_movements)
    summary = {
        "schema": "mppd.r1-joint-iteration0-summary.v1",
        "status": "R1_JOINT_COARSE_ITERATION0_READY",
        "stage": state.stage,
        "inferred_service_count": state.inferred_service_count,
        "service_event_anchor_count": sum(len(s.events) for s in state.services),
        "station_movement_count": len(state.station_movements),
        "directly_identified_station_count": direct,
        "hierarchically_inferred_station_count": hierarchical,
        "passenger_mass": state.passenger_summary.passenger_mass if state.passenger_summary else None,
        "resolved_mass": state.passenger_summary.resolved_mass if state.passenger_summary else None,
        "resolved_share": state.passenger_summary.resolved_share if state.passenger_summary else None,
        "direction_specific_transfer_posterior_pending": True,
        "event_level_local_update_pending": True,
        "movement_initialization_service_clock_shift_s": 0.0,
    }
    a.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
