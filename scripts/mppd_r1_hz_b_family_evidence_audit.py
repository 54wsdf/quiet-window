from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import scripts.mppd_r1_hz_birth_bootstrap_stability as stability
import scripts.mppd_r1_hz_progressive_birth_refinement as birth

SCHEMA = "mppd.r1-hz-b-family-evidence-audit.v1"
B_SHARED_TRUNK = frozenset(range(0, 21))
FRONTIER_POINTS: tuple[tuple[str, float, float], ...] = (
    ("n60e100", 60.0, 1.00),
    ("n45e090", 45.0, 0.90),
    ("n45e080", 45.0, 0.80),
    ("n30e100", 30.0, 1.00),
    ("n30e090", 30.0, 0.90),
    ("n30e080", 30.0, 0.80),
)


def trunk_only(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    ids = []
    for token in row.get("support_event_ids", []):
        try:
            station = int(str(token).split("@", 1)[0])
        except (ValueError, TypeError):
            continue
        if station in B_SHARED_TRUNK:
            ids.append(str(token))
    out["support_event_ids"] = ids
    return out


def aligned_on_b_trunk(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    aa = trunk_only(a)
    bb = trunk_only(b)
    align = stability.event_alignment(aa, bb, tolerance_s=15.0)
    duplicate = (
        align["common_support_station_count"] >= 4
        and align["matched_support_station_count"] >= 4
        and align["matched_support_fraction"] >= 0.60
        and align["median_nearest_event_delta_s"] is not None
        and float(align["median_nearest_event_delta_s"]) <= 15.0
    )
    return {**align, "shared_trunk_duplicate_evidence": bool(duplicate)}


def audit_b_births(parent: dict[str, Any], births: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parent_b = [
        p for p in parent.get("trajectories", [])
        if str(p.get("afc_line")) == "B"
    ]
    b_births = [x for x in births if str(x.get("afc_line")) == "B"]
    non_b = [x for x in births if str(x.get("afc_line")) != "B"]

    ordered = sorted(
        b_births,
        key=lambda x: (-float(x.get("evidence_score", 0.0)), -int(x.get("support_station_count", 0))),
    )
    accepted: list[dict[str, Any]] = []
    audited: list[dict[str, Any]] = []

    for candidate in ordered:
        compatible_parent = [
            p for p in parent_b
            if str(p.get("direction")) == str(candidate.get("direction"))
        ]
        parent_rows = []
        for p in compatible_parent:
            a = aligned_on_b_trunk(candidate, p)
            if a["common_support_station_count"] > 0:
                parent_rows.append((p, a))
        parent_rows.sort(
            key=lambda z: (
                -float(z[1]["matched_support_fraction"]),
                -int(z[1]["matched_support_station_count"]),
                float(z[1]["median_nearest_event_delta_s"] or 1e9),
            )
        )
        best_parent = parent_rows[0] if parent_rows else None
        parent_duplicate = bool(best_parent and best_parent[1]["shared_trunk_duplicate_evidence"])

        accepted_rows = []
        for other in accepted:
            if str(other.get("direction")) != str(candidate.get("direction")):
                continue
            a = aligned_on_b_trunk(candidate, other)
            if a["common_support_station_count"] > 0:
                accepted_rows.append((other, a))
        accepted_rows.sort(
            key=lambda z: (
                -float(z[1]["matched_support_fraction"]),
                -int(z[1]["matched_support_station_count"]),
                float(z[1]["median_nearest_event_delta_s"] or 1e9),
            )
        )
        best_birth = accepted_rows[0] if accepted_rows else None
        birth_duplicate = bool(best_birth and best_birth[1]["shared_trunk_duplicate_evidence"])

        if parent_duplicate:
            cls = "REJECT_BIRTH_SHARED_TRUNK_DUPLICATE_OF_PARENT"
        elif birth_duplicate:
            cls = "REJECT_BIRTH_SHARED_TRUNK_DUPLICATE_OF_STRONGER_BIRTH"
        else:
            cls = "SURVIVES_B_SHARED_TRUNK_EVIDENCE_GATE"
            accepted.append(candidate)

        audited.append({
            "trajectory_id": str(candidate.get("trajectory_id")),
            "direction": str(candidate.get("direction")),
            "path_id": str(candidate.get("path_id")),
            "path_ambiguous": bool(candidate.get("path_ambiguous", False)),
            "anchor_time_s": birth.anchor_time(candidate),
            "support_station_count": int(candidate.get("support_station_count", 0)),
            "evidence_score": float(candidate.get("evidence_score", 0.0)),
            "classification": cls,
            "nearest_shared_trunk_parent": (
                {
                    "trajectory_id": str(best_parent[0].get("trajectory_id")),
                    "path_id": str(best_parent[0].get("path_id")),
                    **best_parent[1],
                }
                if best_parent else None
            ),
            "nearest_shared_trunk_accepted_birth": (
                {
                    "trajectory_id": str(best_birth[0].get("trajectory_id")),
                    "path_id": str(best_birth[0].get("path_id")),
                    **best_birth[1],
                }
                if best_birth else None
            ),
        })

    survivors = non_b + accepted
    return survivors, audited


def build_audit(parent: dict[str, Any], fine: dict[str, Any]) -> dict[str, Any]:
    points = {}
    for label, novelty, evidence in FRONTIER_POINTS:
        births, excluded = birth.select_residual_births(
            parent,
            fine,
            novelty_radius_s=novelty,
            minimum_evidence_score=evidence,
            minimum_support_stations=6,
        )
        survivors, audited_b = audit_b_births(parent, births)
        counts = Counter(x["classification"] for x in audited_b)
        points[label] = {
            "novelty_radius_s": novelty,
            "minimum_evidence_score": evidence,
            "raw_birth_count": len(births),
            "raw_b_birth_count": sum(str(x.get("afc_line")) == "B" for x in births),
            "b_family_class_counts": dict(counts),
            "surviving_birth_count_after_b_family_gate": len(survivors),
            "surviving_b_birth_count": sum(str(x.get("afc_line")) == "B" for x in survivors),
            "b_family_adjusted_service_count": len(parent.get("trajectories", [])) + len(survivors),
            "surviving_births_by_line_direction": dict(Counter(f"{x['afc_line']}:{x['direction']}" for x in survivors)),
            "excluded_counts_before_b_family_gate": excluded,
            "b_birth_audit": audited_b,
        }
    return {
        "schema": SCHEMA,
        "status": "B_MAIN_BRANCH_SHARED_TRUNK_EVIDENCE_AUDIT_COMPLETED_REQUIRES_BOOTSTRAP_INTERSECTION",
        "parent_service_count": len(parent.get("trajectories", [])),
        "shared_trunk_station_ids": sorted(B_SHARED_TRUNK),
        "event_time_tolerance_s": 15.0,
        "minimum_matched_trunk_stations": 4,
        "minimum_matched_trunk_fraction": 0.60,
        "frontier": points,
        "semantics": {
            "b_main_and_branch_share_stations_0_through_20": True,
            "shared_trunk_afc_pulse_identity_overrides_path_label_for_duplicate_detection": True,
            "path_ambiguity_alone_does_not_reject_a_birth": True,
            "planned_timetable_used": False,
            "this_audit_does_not_set_final_service_count": True,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parent-services", type=Path, required=True)
    p.add_argument("--fine-services", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    parent = json.loads(a.parent_services.read_text(encoding="utf-8"))
    fine = json.loads(a.fine_services.read_text(encoding="utf-8"))
    result = build_audit(parent, fine)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "frontier": {
            label: {
                "raw_births": x["raw_birth_count"],
                "raw_B_births": x["raw_b_birth_count"],
                "surviving_births": x["surviving_birth_count_after_b_family_gate"],
                "adjusted_N": x["b_family_adjusted_service_count"],
                "B_classes": x["b_family_class_counts"],
            }
            for label, x in result["frontier"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
