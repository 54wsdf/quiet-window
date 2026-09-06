from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "mppd.r1-joint-consensus-event-proposal.v1"


def key(row: dict[str, Any]) -> tuple[str, int, float]:
    return str(row["trajectory_id"]), int(row["station_id"]), float(row["delta_s"])


def build_consensus(
    proposal: dict[str, Any],
    afc_frontier: dict[str, Any],
    lambda_event: float = 0.0,
    require_direct_afc_support: bool = True,
) -> dict[str, Any]:
    if lambda_event < 0:
        raise ValueError("lambda_event must be nonnegative")
    audits = {key(x): x for x in afc_frontier.get("event_audit", [])}
    selected = []
    excluded = Counter()
    for u in proposal.get("updates", []):
        if float(u["direct_net_mass"]) <= float(lambda_event):
            excluded["below_event_complexity"] += 1
            continue
        a = audits.get(key(u))
        if a is None:
            excluded["missing_afc_audit"] += 1
            continue
        change = a.get("afc_nearest_distance_change_s")
        if change is None:
            excluded["no_detected_afc_anchor"] += 1
            continue
        if require_direct_afc_support and not bool(a.get("afc_supported_before_under_trajectory_residual_p90")):
            excluded["not_directly_afc_supported_under_trajectory_envelope"] += 1
            continue
        if float(change) > 1e-9:
            excluded["afc_anchor_worsens"] += 1
            continue
        row = json.loads(json.dumps(u))
        row["afc_nearest_distance_change_s"] = float(change)
        row["afc_weighted_distance_change"] = a.get("afc_weighted_distance_change")
        row["nearest_afc_pulse_before_distance_s"] = a.get("nearest_afc_pulse_before_distance_s")
        row["nearest_afc_pulse_after_distance_s"] = a.get("nearest_afc_pulse_after_distance_s")
        row["afc_supported_before_under_trajectory_residual_p90"] = bool(
            a.get("afc_supported_before_under_trajectory_residual_p90")
        )
        row["joint_evidence_class"] = "PASSENGER_DIRECT_GAIN_AND_NONWORSENING_DIRECT_AFC_ANCHOR"
        selected.append(row)

    selected.sort(
        key=lambda x: (
            -float(x["direct_net_mass"]),
            float(x["afc_nearest_distance_change_s"]),
            str(x["trajectory_id"]),
            int(x["station_id"]),
        )
    )
    return {
        "schema": SCHEMA,
        "status": "JOINT_PASSENGER_AFC_CONSENSUS_PROPOSAL_REQUIRES_HELDOUT_SELECTION",
        "lambda_event": float(lambda_event),
        "require_direct_afc_support": bool(require_direct_afc_support),
        "source_proposal_event_count": int(proposal.get("proposed_event_update_count", len(proposal.get("updates", [])))),
        "consensus_event_update_count": len(selected),
        "consensus_trajectory_update_count": len({str(x["trajectory_id"]) for x in selected}),
        "updates": selected,
        "excluded_counts": dict(excluded),
        "sum_direct_rescue_mass_not_deduplicated": sum(float(x["direct_rescue_mass"]) for x in selected),
        "sum_direct_damage_mass_not_deduplicated": sum(float(x["direct_damage_mass"]) for x in selected),
        "sum_direct_net_mass_not_deduplicated": sum(float(x["direct_net_mass"]) for x in selected),
        "afc_distance_change_sum_s": sum(float(x["afc_nearest_distance_change_s"]) for x in selected),
        "semantics": {
            "passenger_condition": "direct one-step rescue strictly exceeds direct one-step damage and direct_net_mass > lambda_event",
            "afc_condition": "predicted passenger-facing pulse after the 5 s event move does not increase distance to the nearest observed AFC exit pulse",
            "direct_afc_support_required": bool(require_direct_afc_support),
            "no_planned_timetable_used": True,
            "no_update_applied": True,
            "lambda_not_final_until_heldout_crossfit": True,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--proposal", type=Path, required=True)
    p.add_argument("--afc-frontier", type=Path, required=True)
    p.add_argument("--lambda-event", type=float, default=0.0)
    p.add_argument("--allow-propagated-without-direct-afc", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    proposal = json.loads(a.proposal.read_text(encoding="utf-8"))
    afc_frontier = json.loads(a.afc_frontier.read_text(encoding="utf-8"))
    result = build_consensus(
        proposal,
        afc_frontier,
        lambda_event=a.lambda_event,
        require_direct_afc_support=not a.allow_propagated_without_direct_afc,
    )
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "lambda_event": result["lambda_event"],
        "source_proposal_event_count": result["source_proposal_event_count"],
        "consensus_event_update_count": result["consensus_event_update_count"],
        "excluded_counts": result["excluded_counts"],
        "afc_distance_change_sum_s": result["afc_distance_change_sum_s"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
