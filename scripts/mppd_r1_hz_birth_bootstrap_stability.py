from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

SCHEMA = "mppd.r1-hz-birth-bootstrap-stability.v1"


def parse_support_events(row: dict[str, Any]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for token in row.get("support_event_ids", []):
        try:
            station_s, time_s = str(token).split("@", 1)
            station = int(station_s)
            t = float(time_s)
        except (ValueError, TypeError):
            continue
        out.setdefault(station, []).append(t)
    for values in out.values():
        values.sort()
    return out


def q(values: list[float], frac: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    k = (len(xs) - 1) * frac
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    a = k - lo
    return xs[lo] * (1.0 - a) + xs[hi] * a


def event_alignment(a: dict[str, Any], b: dict[str, Any], tolerance_s: float = 15.0) -> dict[str, Any]:
    am = parse_support_events(a)
    bm = parse_support_events(b)
    common = sorted(set(am) & set(bm))
    nearest = []
    for station in common:
        nearest.append(min(abs(x - y) for x in am[station] for y in bm[station]))
    matched = [x for x in nearest if x <= tolerance_s]
    denom = max(1, min(len(am), len(bm)))
    return {
        "support_station_count_a": len(am),
        "support_station_count_b": len(bm),
        "common_support_station_count": len(common),
        "matched_support_station_count": len(matched),
        "matched_support_fraction": len(matched) / denom,
        "median_nearest_event_delta_s": median(nearest) if nearest else None,
        "q90_nearest_event_delta_s": q(nearest, 0.90),
    }


def pair_score(
    full_birth: dict[str, Any],
    candidate: dict[str, Any],
    *,
    event_tolerance_s: float = 15.0,
    anchor_tolerance_s: float = 120.0,
) -> dict[str, Any] | None:
    if str(full_birth.get("afc_line")) != str(candidate.get("afc_line")):
        return None
    if str(full_birth.get("direction")) != str(candidate.get("direction")):
        return None
    fa = full_birth.get("anchor_time_s")
    ca = candidate.get("anchor_time_s")
    if fa is None or ca is None:
        return None
    anchor_delta = abs(float(fa) - float(ca))
    if anchor_delta > anchor_tolerance_s:
        return None
    align = event_alignment(full_birth, candidate, tolerance_s=event_tolerance_s)
    med = align["median_nearest_event_delta_s"]
    if align["common_support_station_count"] < 4:
        return None
    if align["matched_support_station_count"] < 4:
        return None
    if align["matched_support_fraction"] < 0.50:
        return None
    if med is None or float(med) > event_tolerance_s:
        return None
    return {
        **align,
        "anchor_delta_s": anchor_delta,
        "same_path": str(full_birth.get("path_id")) == str(candidate.get("path_id")),
        "candidate_path_ambiguous": bool(candidate.get("path_ambiguous", False)),
        "candidate_trajectory_id": str(candidate.get("trajectory_id")),
    }


def match_one_replicate(
    full_births: list[dict[str, Any]],
    replicate: dict[str, Any],
    label: str,
) -> list[dict[str, Any] | None]:
    pool: list[tuple[str, dict[str, Any]]] = []
    for row in replicate.get("coarse_support_trajectories", []):
        pool.append(("RECOVERED_AS_COARSE_SERVICE", row))
    for row in replicate.get("frontier", {}).get(label, {}).get("births", []):
        pool.append(("RECOVERED_AS_RESIDUAL_BIRTH", row))

    possible: list[tuple[tuple[float, float, float, float], int, int, str, dict[str, Any]]] = []
    for i, full in enumerate(full_births):
        for j, (source, candidate) in enumerate(pool):
            score = pair_score(full, candidate)
            if score is None:
                continue
            rank = (
                -float(score["matched_support_fraction"]),
                -float(score["matched_support_station_count"]),
                float(score["median_nearest_event_delta_s"]),
                float(score["anchor_delta_s"]),
            )
            possible.append((rank, i, j, source, score))
    possible.sort(key=lambda x: x[0])

    assigned_full: set[int] = set()
    assigned_pool: set[int] = set()
    out: list[dict[str, Any] | None] = [None] * len(full_births)
    for _rank, i, j, source, score in possible:
        if i in assigned_full or j in assigned_pool:
            continue
        assigned_full.add(i)
        assigned_pool.add(j)
        out[i] = {"recovery_source": source, **score}
    return out


def summarize(full: dict[str, Any], replicates: list[dict[str, Any]]) -> dict[str, Any]:
    nrep = len(replicates)
    if nrep < 3:
        raise ValueError("at least three bootstrap replicates are required")
    stable_threshold = math.ceil((2.0 / 3.0) * nrep)
    strong_threshold = math.ceil((5.0 / 6.0) * nrep)
    frontier: dict[str, Any] = {}

    for label, full_point in full.get("frontier", {}).items():
        full_births = list(full_point.get("births", []))
        matches_by_rep = [match_one_replicate(full_births, rep, label) for rep in replicates]
        candidates = []
        for i, row in enumerate(full_births):
            recovered = []
            for rep, matches in zip(replicates, matches_by_rep):
                m = matches[i]
                if m is None:
                    continue
                recovered.append({
                    "bootstrap_seed": rep.get("source_profile", {}).get("bootstrap_seed"),
                    **m,
                })
            count = len(recovered)
            source_counts = {
                source: sum(x["recovery_source"] == source for x in recovered)
                for source in ("RECOVERED_AS_COARSE_SERVICE", "RECOVERED_AS_RESIDUAL_BIRTH")
            }
            candidates.append({
                "trajectory_id": row.get("trajectory_id"),
                "afc_line": row.get("afc_line"),
                "direction": row.get("direction"),
                "path_id": row.get("path_id"),
                "path_ambiguous": row.get("path_ambiguous", False),
                "anchor_time_s": row.get("anchor_time_s"),
                "support_station_count": row.get("support_station_count"),
                "evidence_score": row.get("evidence_score"),
                "recovered_replicate_count": count,
                "recovered_fraction": count / nrep,
                "recovery_source_counts": source_counts,
                "stable_two_thirds": count >= stable_threshold,
                "strong_stable_five_sixths": count >= strong_threshold,
                "replicate_matches": recovered,
            })

        stable = [x for x in candidates if x["stable_two_thirds"]]
        strong = [x for x in candidates if x["strong_stable_five_sixths"]]
        replicate_birth_counts = [int(r.get("frontier", {}).get(label, {}).get("birth_count", 0)) for r in replicates]
        replicate_augmented_counts = [
            int(r.get("coarse_service_count", 0)) + int(r.get("frontier", {}).get(label, {}).get("birth_count", 0))
            for r in replicates
        ]
        frontier[label] = {
            "novelty_radius_s": full_point.get("novelty_radius_s"),
            "minimum_evidence_score": full_point.get("minimum_evidence_score"),
            "full_data_birth_count": len(full_births),
            "stable_birth_count_two_thirds": len(stable),
            "strong_stable_birth_count_five_sixths": len(strong),
            "full_data_stable_augmented_count_two_thirds": int(full["coarse_service_count"]) + len(stable),
            "full_data_strong_stable_augmented_count_five_sixths": int(full["coarse_service_count"]) + len(strong),
            "bootstrap_birth_count_min": min(replicate_birth_counts) if replicate_birth_counts else None,
            "bootstrap_birth_count_median": q([float(x) for x in replicate_birth_counts], 0.50),
            "bootstrap_birth_count_max": max(replicate_birth_counts) if replicate_birth_counts else None,
            "bootstrap_augmented_count_min": min(replicate_augmented_counts) if replicate_augmented_counts else None,
            "bootstrap_augmented_count_median": q([float(x) for x in replicate_augmented_counts], 0.50),
            "bootstrap_augmented_count_max": max(replicate_augmented_counts) if replicate_augmented_counts else None,
            "stable_births_by_line_direction": {
                key: sum(f"{x['afc_line']}:{x['direction']}" == key for x in stable)
                for key in sorted({f"{x['afc_line']}:{x['direction']}" for x in stable})
            },
            "candidates": candidates,
        }

    coarse_counts = [int(r.get("coarse_service_count", 0)) for r in replicates]
    fine_counts = [int(r.get("fine_five_second_service_count", 0)) for r in replicates]
    return {
        "schema": SCHEMA,
        "status": "BIRTH_BOOTSTRAP_STABILITY_COMPLETED_REQUIRES_B_FAMILY_AND_DEATH_SPLIT_AUDIT",
        "bootstrap_replicate_count": nrep,
        "stable_recovery_threshold": stable_threshold,
        "strong_stable_recovery_threshold": strong_threshold,
        "full_data_coarse_service_count": int(full.get("coarse_service_count", 0)),
        "full_data_fine_five_second_service_count": int(full.get("fine_five_second_service_count", 0)),
        "bootstrap_coarse_service_count": {
            "min": min(coarse_counts),
            "median": q([float(x) for x in coarse_counts], 0.50),
            "max": max(coarse_counts),
        },
        "bootstrap_fine_five_second_service_count": {
            "min": min(fine_counts),
            "median": q([float(x) for x in fine_counts], 0.50),
            "max": max(fine_counts),
        },
        "frontier": frontier,
        "semantics": {
            "recovery_can_occur_as_coarse_service_or_residual_birth": True,
            "matching_uses_same_line_direction_anchor_and_station_pulse_evidence": True,
            "support_event_time_tolerance_s": 15.0,
            "anchor_match_tolerance_s": 120.0,
            "minimum_matched_support_stations": 4,
            "planned_timetable_used": False,
            "bootstrap_stability_does_not_by_itself_select_final_service_count": True,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--replicate", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    full = json.loads(a.full.read_text(encoding="utf-8"))
    reps = [json.loads(path.read_text(encoding="utf-8")) for path in a.replicate]
    result = summarize(full, reps)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "bootstrap_coarse_service_count": result["bootstrap_coarse_service_count"],
        "frontier": {
            label: {
                "full_births": x["full_data_birth_count"],
                "stable_births": x["stable_birth_count_two_thirds"],
                "strong_stable_births": x["strong_stable_birth_count_five_sixths"],
                "stable_augmented_N": x["full_data_stable_augmented_count_two_thirds"],
            }
            for label, x in result["frontier"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
