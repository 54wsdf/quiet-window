from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.mppd_r1_hz_network_authority import build_authority
from scripts.mppd_r1_joint_model import bootstrap_from_count_free_discovery
from scripts.mppd_r1_service_circulation import (
    infer_service_circulations,
    turnback_rules_from_network_authority,
)

SCHEMA = "mppd.r1-hz-circulation-sensitivity.v1"
DEFAULT_WINDOWS_S: tuple[tuple[float, float], ...] = (
    (30.0, 300.0),
    (60.0, 300.0),
    (90.0, 300.0),
    (30.0, 600.0),
    (60.0, 600.0),
    (90.0, 600.0),
    (60.0, 900.0),
    (90.0, 900.0),
    (120.0, 900.0),
    (30.0, 1200.0),
)


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(float(x) for x in values)
    if len(vals) == 1:
        return vals[0]
    p = (len(vals) - 1) * q
    lo = int(math.floor(p))
    hi = int(math.ceil(p))
    if lo == hi:
        return vals[lo]
    a = p - lo
    return vals[lo] * (1.0 - a) + vals[hi] * a


def _network_authority() -> dict[str, Any]:
    # build_authority consumes only the already-qualified line-aware route topology
    # here.  No timetable, service inventory or target count is supplied.
    return build_authority(
        {
            "status": "QUALIFIED_LINE_AWARE_ROUTE_SUPPORT",
            "route_support": {},
        }
    )


def _chain_summary(state: Any) -> dict[str, Any]:
    by_id = {service.trajectory_id: service for service in state.services}
    lengths: list[int] = []
    path_counts: Counter[str] = Counter()
    line_counts: Counter[str] = Counter()
    alternating_direction_failures = 0
    membership = 0
    for chain in state.service_circulations:
        ids = list(chain.ordered_trajectory_ids)
        lengths.append(len(ids))
        membership += len(ids)
        if ids:
            first = by_id[ids[0]]
            path_counts[first.path_id] += 1
            line_counts[first.line_id] += 1
        for left_id, right_id in zip(ids, ids[1:]):
            if by_id[left_id].direction_id == by_id[right_id].direction_id:
                alternating_direction_failures += 1
    return {
        "membership_count": membership,
        "chain_length_min": min(lengths) if lengths else None,
        "chain_length_p50": _quantile(lengths, 0.50),
        "chain_length_p95": _quantile(lengths, 0.95),
        "chain_length_max": max(lengths) if lengths else None,
        "chain_count_by_path": dict(sorted(path_counts.items())),
        "chain_count_by_line": dict(sorted(line_counts.items())),
        "alternating_direction_failure_count": alternating_direction_failures,
    }


def run_circulation_sensitivity(
    discovery: Mapping[str, Any],
    *,
    windows_s: Sequence[tuple[float, float]] = DEFAULT_WINDOWS_S,
) -> dict[str, Any]:
    state = bootstrap_from_count_free_discovery(dict(discovery))
    authority = _network_authority()
    if authority["semantics"].get("contains_planned_timetable") is not False:
        raise ValueError("circulation sensitivity requires plan-free network authority")
    if authority["semantics"].get("contains_expected_service_count") is not False:
        raise ValueError("circulation sensitivity may not use expected service count")

    rows: list[dict[str, Any]] = []
    for min_turnback_s, max_turnback_s in windows_s:
        if min_turnback_s < 0 or max_turnback_s < min_turnback_s:
            raise ValueError("invalid turnback sensitivity window")
        rules = turnback_rules_from_network_authority(
            authority,
            min_turnback_s=float(min_turnback_s),
            max_turnback_s=float(max_turnback_s),
        )
        inferred, audit = infer_service_circulations(state, rules)
        chain_summary = _chain_summary(inferred)
        if chain_summary["membership_count"] != state.inferred_service_count:
            raise AssertionError("circulation chains do not cover the service world exactly once")
        rows.append(
            {
                "min_turnback_s": float(min_turnback_s),
                "max_turnback_s": float(max_turnback_s),
                "audit": asdict(audit),
                "chain_summary": chain_summary,
                "link_share": (
                    audit.matched_link_count / state.inferred_service_count
                    if state.inferred_service_count
                    else 0.0
                ),
                "chain_count_share": (
                    audit.circulation_count / state.inferred_service_count
                    if state.inferred_service_count
                    else 0.0
                ),
            }
        )

    # No turnback window is selected here.  Report the maximal-link and minimal-chain
    # envelopes as diagnostics only; a calibrated terminal/depot model is required
    # before circulation becomes a hard structural acceptance gate.
    best_links = max((int(row["audit"]["matched_link_count"]) for row in rows), default=0)
    min_chains = min((int(row["audit"]["circulation_count"]) for row in rows), default=0)
    return {
        "schema": SCHEMA,
        "status": "COUNT_FREE_CIRCULATION_SENSITIVITY_COMPLETED_NO_WINDOW_SELECTED",
        "service_count": state.inferred_service_count,
        "service_stage": state.stage,
        "sensitivity_rows": rows,
        "envelope": {
            "maximum_matched_link_count": best_links,
            "minimum_circulation_chain_count": min_chains,
        },
        "semantics": {
            "service_count_changed": False,
            "event_times_changed": False,
            "planned_timetable_used": False,
            "planned_trip_count_used": False,
            "physical_vehicle_identity_claimed": False,
            "turnback_window_selected": False,
            "depot_boundary_resolution_applied": False,
        },
        "scientific_boundary": (
            "Each row is an order-preserving maximum-cardinality revenue-service "
            "successor reconstruction under an explicit experimental turnback window. "
            "The chain count is not yet a claimed physical fleet count."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--services", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    discovery = json.loads(args.services.read_text(encoding="utf-8"))
    result = run_circulation_sensitivity(discovery)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "service_count": result["service_count"],
                "envelope": result["envelope"],
                "sensitivity": [
                    {
                        "window_s": [row["min_turnback_s"], row["max_turnback_s"]],
                        "matched_links": row["audit"]["matched_link_count"],
                        "chains": row["audit"]["circulation_count"],
                        "singletons": row["audit"]["singleton_circulation_count"],
                        "p50_chain_length": row["chain_summary"]["chain_length_p50"],
                    }
                    for row in result["sensitivity_rows"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
