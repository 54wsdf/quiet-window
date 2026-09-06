from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "mppd.r1-event-update-complexity-frontier.v1"


def build_frontier(proposal: dict[str, Any]) -> dict[str, Any]:
    updates = list(proposal.get("updates", []))
    if not updates:
        raise ValueError("proposal has no updates")

    positive_nets = sorted({float(x["direct_net_mass"]) for x in updates if float(x["direct_net_mass"]) > 0})
    # λ values are induced by the observed discrete net-rescue support rather than
    # an arbitrary event-count cap. λ=0 reproduces the unrestricted positive-net
    # stress test; λ just below each unique net value gives each nested frontier.
    lambdas = [0.0]
    for net in positive_nets:
        lambdas.append(max(0.0, net - 1e-9))
    lambdas = sorted(set(lambdas))

    rows = []
    for lam in lambdas:
        kept = [x for x in updates if float(x["direct_net_mass"]) > lam]
        by_direction = Counter("minus5" if float(x["delta_s"]) < 0 else "plus5" for x in kept)
        rows.append(
            {
                "lambda_event": lam,
                "update_count": len(kept),
                "trajectory_count": len({str(x["trajectory_id"]) for x in kept}),
                "sum_direct_rescue_mass_not_deduplicated": sum(float(x["direct_rescue_mass"]) for x in kept),
                "sum_direct_damage_mass_not_deduplicated": sum(float(x["direct_damage_mass"]) for x in kept),
                "sum_direct_net_mass_not_deduplicated": sum(float(x["direct_net_mass"]) for x in kept),
                "direction_counts": dict(by_direction),
                "minimum_kept_direct_net_mass": min((float(x["direct_net_mass"]) for x in kept), default=None),
            }
        )

    # Keep only distinct update-count operating points; when multiple lambdas
    # induce the same set, retain the largest lambda (strongest regularization).
    by_count: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_count[int(row["update_count"])] = row
    frontier = [by_count[k] for k in sorted(by_count, reverse=True)]

    return {
        "schema": SCHEMA,
        "status": "EVENT_UPDATE_COMPLEXITY_FRONTIER_READY_FOR_POSTERIOR_SWEEP",
        "source_proposal_schema": proposal.get("schema"),
        "source_update_count": len(updates),
        "frontier": frontier,
        "semantics": {
            "lambda_event_units": "passenger_mass_of_direct_one_step_net_feasibility_gain_per_modified_event",
            "lambda_zero_is_unregularized_stress_test": True,
            "frontier_does_not_select_final_lambda": True,
            "final_lambda_requires_full_posterior_and_AFC_stability_comparison": True,
            "no_arbitrary_max_modified_event_count": True,
        },
    }


def filter_proposal(proposal: dict[str, Any], lambda_event: float) -> dict[str, Any]:
    if lambda_event < 0:
        raise ValueError("lambda_event must be nonnegative")
    out = json.loads(json.dumps(proposal))
    kept = [x for x in out.get("updates", []) if float(x["direct_net_mass"]) > float(lambda_event)]
    out["status"] = "EVENT_UPDATE_PROPOSAL_FILTERED_BY_L0_COMPLEXITY_REQUIRES_POSTERIOR_REEVALUATION"
    out["updates"] = kept
    out["proposed_event_update_count"] = len(kept)
    out["proposed_trajectory_update_count"] = len({str(x["trajectory_id"]) for x in kept})
    out["sum_direct_rescue_mass_not_deduplicated"] = sum(float(x["direct_rescue_mass"]) for x in kept)
    out["sum_direct_damage_mass_not_deduplicated"] = sum(float(x["direct_damage_mass"]) for x in kept)
    out.setdefault("semantics", {})["lambda_event"] = float(lambda_event)
    out["semantics"]["event_l0_complexity_filter_applied"] = True
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("frontier")
    s.add_argument("--proposal", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("filter")
    s.add_argument("--proposal", type=Path, required=True)
    s.add_argument("--lambda-event", type=float, required=True)
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    proposal = json.loads(a.proposal.read_text(encoding="utf-8"))
    result = build_frontier(proposal) if a.command == "frontier" else filter_proposal(proposal, a.lambda_event)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.command == "frontier":
        print(json.dumps({
            "status": result["status"],
            "source_update_count": result["source_update_count"],
            "frontier": result["frontier"],
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({
            "status": result["status"],
            "lambda_event": result["semantics"]["lambda_event"],
            "proposed_event_update_count": result["proposed_event_update_count"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
