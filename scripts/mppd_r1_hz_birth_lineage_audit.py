from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import scripts.mppd_r1_hz_progressive_birth_refinement as birth

SCHEMA = "mppd.r1-hz-progressive-birth-lineage-audit.v1"


def event_map(row: dict[str, Any]) -> dict[int, float]:
    return {int(e["station"]): float(e["time_s"]) for e in row.get("events", [])}


def anchor(row: dict[str, Any]) -> float | None:
    return birth.anchor_time(row)


def nearest_parent(
    parent_rows: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> tuple[dict[str, Any] | None, float | None]:
    t = anchor(candidate)
    if t is None:
        return None, None
    compatible = [
        p for p in parent_rows
        if str(p.get("afc_line")) == str(candidate.get("afc_line"))
        and str(p.get("direction")) == str(candidate.get("direction"))
        and anchor(p) is not None
    ]
    if not compatible:
        return None, None
    p = min(compatible, key=lambda x: abs(float(anchor(x)) - t))
    return p, abs(float(anchor(p)) - t)


def q(values: list[float], frac: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * frac
    lo = math.floor(k); hi = math.ceil(k)
    if lo == hi:
        return float(xs[lo])
    a = k - lo
    return float(xs[lo] * (1-a) + xs[hi] * a)


def lineage_row(parent_rows: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
    p, d = nearest_parent(parent_rows, candidate)
    cm = event_map(candidate)
    pm = event_map(p) if p else {}
    common = sorted(set(cm) & set(pm))
    signed = [cm[s] - pm[s] for s in common]
    med = median(signed) if signed else None
    abs_dev = [abs(x - med) for x in signed] if med is not None else []
    same_path = bool(p is not None and str(p.get("path_id")) == str(candidate.get("path_id")))
    return {
        "birth_source_trajectory_id": str(candidate.get("trajectory_id")),
        "afc_line": str(candidate.get("afc_line")),
        "direction": str(candidate.get("direction")),
        "path_id": str(candidate.get("path_id")),
        "path_ambiguous": bool(candidate.get("path_ambiguous", False)),
        "support_station_count": int(candidate.get("support_station_count", 0)),
        "support_event_count": int(candidate.get("support_event_count", 0)),
        "evidence_score": float(candidate.get("evidence_score", 0.0)),
        "reference_time_s": float(candidate.get("reference_time_s", 0.0)),
        "anchor_time_s": anchor(candidate),
        "nearest_parent_trajectory_id": str(p.get("trajectory_id")) if p else None,
        "nearest_parent_anchor_distance_s": d,
        "nearest_parent_same_path": same_path,
        "common_station_count": len(common),
        "parallel_signed_offset_median_s": float(med) if med is not None else None,
        "parallel_offset_mad_s": float(median(abs_dev)) if abs_dev else None,
        "parallel_offset_p90_abs_dev_s": q(abs_dev, 0.90),
        "lineage_class": (
            "DISTINCT_PARALLEL_SIBLING_SAME_PATH"
            if p is not None and same_path and len(common) >= 4
            else "DISTINCT_SIBLING_ALTERNATE_PATH"
            if p is not None
            else "NO_PARENT_ON_LINE_DIRECTION"
        ),
    }


def headway_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = defaultdict(list)
    for r in rows:
        t = anchor(r)
        if t is not None:
            by[(str(r.get("afc_line")), str(r.get("direction")))].append(float(t))
    gaps = []
    by_key = {}
    for key, times in sorted(by.items()):
        times.sort()
        g = [b-a for a,b in zip(times, times[1:])]
        gaps.extend(g)
        by_key[f"{key[0]}:{key[1]}"] = {
            "service_count": len(times),
            "minimum_gap_s": min(g) if g else None,
            "q10_gap_s": q(g,0.10),
            "median_gap_s": q(g,0.50),
            "gap_lt_30_count": sum(x < 30 for x in g),
            "gap_lt_45_count": sum(x < 45 for x in g),
            "gap_lt_60_count": sum(x < 60 for x in g),
        }
    return {
        "pair_count": len(gaps),
        "minimum_gap_s": min(gaps) if gaps else None,
        "q10_gap_s": q(gaps,0.10),
        "median_gap_s": q(gaps,0.50),
        "gap_lt_30_count": sum(x < 30 for x in gaps),
        "gap_lt_45_count": sum(x < 45 for x in gaps),
        "gap_lt_60_count": sum(x < 60 for x in gaps),
        "by_line_direction": by_key,
    }


def build_audit(parent: dict[str, Any], births: list[dict[str, Any]], augmented: dict[str, Any]) -> dict[str, Any]:
    parent_rows = list(parent.get("trajectories", []))
    lineage = [lineage_row(parent_rows, c) for c in births]
    distances = [float(x["nearest_parent_anchor_distance_s"]) for x in lineage if x["nearest_parent_anchor_distance_s"] is not None]
    evidence = [float(x["evidence_score"]) for x in lineage]
    supports = [float(x["support_station_count"]) for x in lineage]
    mads = [float(x["parallel_offset_mad_s"]) for x in lineage if x["parallel_offset_mad_s"] is not None]
    return {
        "schema": SCHEMA,
        "status": "PROGRESSIVE_BIRTH_LINEAGE_AUDITED_REQUIRES_JOINT_MOVEMENT_QUALIFICATION",
        "parent_service_count": len(parent_rows),
        "birth_count": len(births),
        "augmented_service_count": len(augmented.get("trajectories", [])),
        "births_by_line_direction": dict(Counter(f"{x['afc_line']}:{x['direction']}" for x in lineage)),
        "lineage_class_counts": dict(Counter(x["lineage_class"] for x in lineage)),
        "path_ambiguous_birth_count": sum(bool(x["path_ambiguous"]) for x in lineage),
        "nearest_parent_anchor_distance_s": {
            "minimum": min(distances) if distances else None,
            "q10": q(distances,0.10),
            "median": q(distances,0.50),
            "q90": q(distances,0.90),
            "maximum": max(distances) if distances else None,
        },
        "birth_support_station_count": {
            "minimum": min(supports) if supports else None,
            "median": q(supports,0.50),
            "q90": q(supports,0.90),
        },
        "birth_evidence_score": {
            "minimum": min(evidence) if evidence else None,
            "median": q(evidence,0.50),
            "q90": q(evidence,0.90),
        },
        "parallel_offset_mad_s": {
            "median": q(mads,0.50),
            "q90": q(mads,0.90),
        },
        "parent_headway_audit": headway_audit(parent_rows),
        "augmented_headway_audit": headway_audit(list(augmented.get("trajectories", []))),
        "birth_lineage": lineage,
        "semantics": {
            "five_second_grid_is_not_minimum_headway": True,
            "lineage_uses_line_direction_anchor_and_common_station_event_times": True,
            "parallel_sibling_is_not_automatically_qualified_as_distinct_train": True,
            "planned_timetable_used": False,
        },
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--parent-services",type=Path,required=True)
    p.add_argument("--fine-services",type=Path,required=True)
    p.add_argument("--novelty-radius-s",type=float,required=True)
    p.add_argument("--minimum-evidence-score",type=float,required=True)
    p.add_argument("--minimum-support-stations",type=int,default=6)
    p.add_argument("--augmented-output",type=Path,required=True)
    p.add_argument("--audit-output",type=Path,required=True)
    a=p.parse_args()
    parent=json.loads(a.parent_services.read_text(encoding="utf-8"))
    fine=json.loads(a.fine_services.read_text(encoding="utf-8"))
    births,_=birth.select_residual_births(parent,fine,novelty_radius_s=a.novelty_radius_s,minimum_evidence_score=a.minimum_evidence_score,minimum_support_stations=a.minimum_support_stations)
    augmented=birth.build_augmented_world(parent,births,novelty_radius_s=a.novelty_radius_s,minimum_evidence_score=a.minimum_evidence_score,minimum_support_stations=a.minimum_support_stations)
    audit=build_audit(parent,births,augmented)
    a.augmented_output.write_text(json.dumps(augmented,ensure_ascii=False,indent=2),encoding="utf-8")
    a.audit_output.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ("status","parent_service_count","birth_count","augmented_service_count","births_by_line_direction","lineage_class_counts","nearest_parent_anchor_distance_s","birth_support_station_count","birth_evidence_score","augmented_headway_audit")},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
