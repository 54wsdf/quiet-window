from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

SCHEMA_PROPOSAL = "mppd.r1-hz-event-update-proposal.v1"
SCHEMA_SERVICE = "mppd.r1-hz-count-free-service-local-update.v1"
DELTA_S = 5.0
MIN_EDGE_PROGRESS_S = 5.0


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def event_lookup(services: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    trajectories: dict[str, dict[str, Any]] = {}
    events: dict[tuple[str, int], dict[str, Any]] = {}
    for tr in services.get("trajectories", []):
        sid = str(tr["trajectory_id"])
        trajectories[sid] = tr
        for e in tr.get("events", []):
            events[(sid, int(e["station"]))] = e
    return trajectories, events


def trajectory_residual_envelope_s(tr: dict[str, Any]) -> float | None:
    x = tr.get("residual_p90_abs_s")
    return float(x) if finite(x) else None


def continuity_after_single_move(tr: dict[str, Any], station_id: int, delta_s: float) -> bool:
    rows = sorted(tr.get("events", []), key=lambda e: int(e["sequence_index"]))
    times = []
    found = False
    for e in rows:
        t = float(e["time_s"])
        if int(e["station"]) == int(station_id):
            t += float(delta_s)
            found = True
        times.append(t)
    if not found:
        return False
    return all(b - a >= MIN_EDGE_PROGRESS_S - 1e-9 for a, b in zip(times, times[1:]))


def candidate_for_direction(
    row: dict[str, Any],
    tr: dict[str, Any],
    delta_s: float,
) -> dict[str, Any] | None:
    if delta_s == -DELTA_S:
        benefit = float(row.get("resolvable_if_minus5_mass", 0.0))
        risk = float(row.get("protect_against_minus5_mass", 0.0))
        residual_pressure = float(row.get("minus5_pressure_mass", 0.0))
        residual_reduction = float(row.get("minus5_residual_reduction_mass_seconds", 0.0))
    elif delta_s == DELTA_S:
        benefit = float(row.get("resolvable_if_plus5_mass", 0.0))
        risk = float(row.get("protect_against_plus5_mass", 0.0))
        residual_pressure = float(row.get("plus5_pressure_mass", 0.0))
        residual_reduction = 0.0
    else:
        raise ValueError("coarse proposal only supports +/-5 s")

    # No weighted scalar objective is invented here. A move is eligible only if
    # it has directly demonstrable one-step passenger feasibility benefit and
    # that benefit strictly dominates directly demonstrable one-step damage.
    if benefit <= 0.0 or benefit <= risk:
        return None

    envelope = trajectory_residual_envelope_s(tr)
    if envelope is None or DELTA_S > envelope + 1e-9:
        return None

    station_id = int(row["station_id"])
    if not continuity_after_single_move(tr, station_id, delta_s):
        return None

    return {
        "trajectory_id": str(row["trajectory_id"]),
        "station_id": station_id,
        "delta_s": float(delta_s),
        "direct_rescue_mass": benefit,
        "direct_damage_mass": risk,
        "direct_net_mass": benefit - risk,
        "residual_pressure_mass": residual_pressure,
        "residual_reduction_mass_seconds": residual_reduction,
        "trajectory_residual_p90_abs_s": envelope,
        "path_ambiguous": bool(row.get("path_ambiguous", False)),
        "anchor_time_before_s": float(row["anchor_time_s"]),
        "selection_rule": "DIRECT_ONE_STEP_RESCUE_STRICTLY_EXCEEDS_DIRECT_ONE_STEP_DAMAGE_AND_5S_WITHIN_AFC_TRAJECTORY_RESIDUAL_P90",
    }


def build_proposal(services: dict[str, Any], pressure: dict[str, Any]) -> dict[str, Any]:
    if int(services.get("inferred_service_trajectory_count", -1)) != int(pressure.get("service_trajectory_count", -2)):
        raise ValueError("service/pressure trajectory count mismatch")
    sem = services.get("semantics", {})
    if sem.get("train_count_is_input") is not False or sem.get("planned_timetable_used") is not False:
        raise ValueError("proposal requires count-free service world without planned timetable")

    trajectories, event_map = event_lookup(services)
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = Counter()

    for row in pressure.get("event_pressure", []):
        sid = str(row["trajectory_id"])
        tr = trajectories.get(sid)
        if tr is None or (sid, int(row["station_id"])) not in event_map:
            rejected["missing_event"] += 1
            continue
        made = False
        for delta in (-DELTA_S, DELTA_S):
            c = candidate_for_direction(row, tr, delta)
            if c is not None:
                by_trajectory[sid].append(c)
                made = True
        if not made:
            rejected["no_pareto_safe_one_step_move"] += 1

    selected: list[dict[str, Any]] = []
    multiple_candidate_trajectories = 0
    for sid, rows in sorted(by_trajectory.items()):
        if len(rows) > 1:
            multiple_candidate_trajectories += 1
        # At most one event per trajectory per coarse iteration. Sort primarily
        # by exact one-step net rescue. Residual reduction only breaks ties.
        rows.sort(
            key=lambda x: (
                float(x["direct_net_mass"]),
                float(x["direct_rescue_mass"]),
                float(x["residual_reduction_mass_seconds"]),
                -float(x["direct_damage_mass"]),
            ),
            reverse=True,
        )
        best = rows[0]
        # Exact ties with opposite directions are not identifiable in this local
        # screen and are therefore rejected rather than arbitrarily resolved.
        ties = [
            x for x in rows
            if abs(float(x["direct_net_mass"]) - float(best["direct_net_mass"])) < 1e-9
            and abs(float(x["direct_rescue_mass"]) - float(best["direct_rescue_mass"])) < 1e-9
            and abs(float(x["residual_reduction_mass_seconds"]) - float(best["residual_reduction_mass_seconds"])) < 1e-9
        ]
        if len({float(x["delta_s"]) for x in ties}) > 1:
            rejected["direction_tie"] += 1
            continue
        selected.append(best)

    selected.sort(key=lambda x: (-float(x["direct_net_mass"]), x["trajectory_id"], int(x["station_id"])))
    proposal = {
        "schema": SCHEMA_PROPOSAL,
        "status": "SPARSE_FIVE_SECOND_EVENT_UPDATE_PROPOSED_REQUIRES_FULL_POSTERIOR_REEVALUATION",
        "service_trajectory_count": int(services["inferred_service_trajectory_count"]),
        "pressure_event_count": int(pressure.get("event_pressure_count", len(pressure.get("event_pressure", [])))),
        "proposed_event_update_count": len(selected),
        "proposed_trajectory_update_count": len({x["trajectory_id"] for x in selected}),
        "multiple_candidate_trajectory_count": multiple_candidate_trajectories,
        "sum_direct_rescue_mass_not_deduplicated": sum(float(x["direct_rescue_mass"]) for x in selected),
        "sum_direct_damage_mass_not_deduplicated": sum(float(x["direct_damage_mass"]) for x in selected),
        "updates": selected,
        "rejected_counts": dict(rejected),
        "semantics": {
            "coarse_resolution_s": 5.0,
            "single_event_trust_region_s": 5.0,
            "at_most_one_event_per_trajectory": True,
            "planned_timetable_used": False,
            "service_count_changed": False,
            "passenger_pressure_only_used_as_pareto_gate": True,
            "afc_anchor_gate": "5 s move must lie within trajectory AFC residual p90 envelope",
            "service_continuity_hard_gate": True,
            "full_posterior_reevaluation_required_before_acceptance": True,
            "proposal_is_not_qualified_update": True,
        },
    }
    return proposal


def apply_proposal(services: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    if proposal.get("schema") != SCHEMA_PROPOSAL:
        raise ValueError("wrong proposal schema")
    out = deepcopy(services)
    trajectories, event_map = event_lookup(out)
    applied = []
    for row in proposal.get("updates", []):
        sid = str(row["trajectory_id"])
        station = int(row["station_id"])
        delta = float(row["delta_s"])
        if delta not in (-DELTA_S, DELTA_S):
            raise ValueError("non-admissible coarse delta")
        tr = trajectories[sid]
        if not continuity_after_single_move(tr, station, delta):
            raise ValueError(f"continuity failure for {sid}/{station}")
        event = event_map[(sid, station)]
        before = float(event["time_s"])
        event["time_s"] = before + delta
        event["r1_local_update_iteration"] = 1
        event["r1_local_update_delta_s"] = delta
        applied.append({"trajectory_id": sid, "station_id": station, "delta_s": delta, "before_s": before, "after_s": before + delta})

    out["schema"] = SCHEMA_SERVICE
    out["status"] = "R1_COARSE_LOCAL_EVENT_UPDATE_APPLIED_REQUIRES_FULL_POSTERIOR_REEVALUATION"
    out.setdefault("semantics", {})["event_times_locally_updated_from_joint_passenger_pressure"] = True
    out["semantics"]["single_event_update_trust_region_s"] = 5.0
    out["semantics"]["coarse_time_resolution_s"] = 5.0
    out["semantics"]["planned_timetable_used"] = False
    out["semantics"]["train_count_is_input"] = False
    out["semantics"]["legacy_candidate_roots_used_as_input"] = False
    out["semantics"]["service_trajectory_count_is_inferred"] = True
    out["semantics"]["service_count_changed_in_this_update"] = False
    out["joint_local_update"] = {
        "iteration": 1,
        "proposal_schema": proposal["schema"],
        "applied_event_update_count": len(applied),
        "applied_trajectory_update_count": len({x["trajectory_id"] for x in applied}),
        "updates": applied,
        "reference_time_semantics": "reference_time_s remains the AFC-only trajectory construction reference; locally updated event times are authoritative for this working state",
        "acceptance_pending_full_posterior_reevaluation": True,
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("propose")
    s.add_argument("--services", type=Path, required=True)
    s.add_argument("--pressure", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("apply")
    s.add_argument("--services", type=Path, required=True)
    s.add_argument("--proposal", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    services = json.loads(a.services.read_text(encoding="utf-8"))
    if a.command == "propose":
        pressure = json.loads(a.pressure.read_text(encoding="utf-8"))
        result = build_proposal(services, pressure)
    else:
        proposal = json.loads(a.proposal.read_text(encoding="utf-8"))
        result = apply_proposal(services, proposal)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.command == "propose":
        print(json.dumps({
            "status": result["status"],
            "proposed_event_update_count": result["proposed_event_update_count"],
            "proposed_trajectory_update_count": result["proposed_trajectory_update_count"],
            "sum_direct_rescue_mass_not_deduplicated": result["sum_direct_rescue_mass_not_deduplicated"],
            "sum_direct_damage_mass_not_deduplicated": result["sum_direct_damage_mass_not_deduplicated"],
            "top_updates": result["updates"][:20],
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({
            "status": result["status"],
            "service_trajectory_count": result["inferred_service_trajectory_count"],
            "applied_event_update_count": result["joint_local_update"]["applied_event_update_count"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
