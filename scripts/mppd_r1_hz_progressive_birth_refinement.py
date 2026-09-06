from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "mppd.r1-hz-progressive-5s-birth-refinement.v1"
LINE_ANCHOR_STATION = {"A": 5, "B": 5, "C": 46}


def anchor_time(row: dict[str, Any]) -> float | None:
    station = LINE_ANCHOR_STATION.get(str(row.get("afc_line")))
    if station is None:
        return None
    for event in row.get("events", []):
        if int(event["station"]) == station:
            return float(event["time_s"])
    return None


def group_anchor_times(trajectories: list[dict[str, Any]]) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in trajectories:
        t = anchor_time(row)
        if t is not None:
            out[(str(row["afc_line"]), str(row["direction"]))].append(t)
    for values in out.values():
        values.sort()
    return out


def nearest_distance(values: list[float], x: float) -> float | None:
    if not values:
        return None
    i = bisect.bisect_left(values, x)
    ds = []
    if i < len(values):
        ds.append(abs(values[i] - x))
    if i > 0:
        ds.append(abs(values[i - 1] - x))
    return min(ds) if ds else None


def candidate_sort_key(row: dict[str, Any]):
    return (
        -float(row.get("evidence_score", 0.0)),
        -int(row.get("support_station_count", 0)),
        -float(row.get("support_weight", 0.0)),
        str(row.get("trajectory_id", "")),
    )


def select_residual_births(
    parent: dict[str, Any],
    fine: dict[str, Any],
    *,
    novelty_radius_s: float,
    minimum_evidence_score: float,
    minimum_support_stations: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if novelty_radius_s <= 0:
        raise ValueError("novelty_radius_s must be positive")
    parent_tr = list(parent.get("trajectories", []))
    fine_tr = list(fine.get("trajectories", []))
    parent_index = group_anchor_times(parent_tr)

    eligible = []
    excluded = Counter()
    for row in fine_tr:
        if float(row.get("evidence_score", 0.0)) < minimum_evidence_score:
            excluded["below_evidence_threshold"] += 1
            continue
        if int(row.get("support_station_count", 0)) < minimum_support_stations:
            excluded["below_station_support"] += 1
            continue
        t = anchor_time(row)
        if t is None:
            excluded["missing_line_anchor"] += 1
            continue
        key = (str(row["afc_line"]), str(row["direction"]))
        d = nearest_distance(parent_index.get(key, []), t)
        if d is not None and d < novelty_radius_s:
            excluded["explained_by_parent_service"] += 1
            continue
        candidate = json.loads(json.dumps(row))
        candidate["nearest_parent_anchor_distance_s"] = d
        eligible.append(candidate)

    # Compete residual fine candidates with one another. The five-second grid is
    # the time-estimation resolution; novelty_radius_s is a distinct-train
    # competition radius and must not be silently equated with 5 s.
    selected: list[dict[str, Any]] = []
    selected_times: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in sorted(eligible, key=candidate_sort_key):
        t = anchor_time(row)
        assert t is not None
        key = (str(row["afc_line"]), str(row["direction"]))
        d = nearest_distance(selected_times[key], t)
        if d is not None and d < novelty_radius_s:
            excluded["competes_with_stronger_birth"] += 1
            continue
        row["structure_operation"] = "birth"
        row["birth_evidence_class"] = "FIVE_SECOND_AFC_RESIDUAL_NOT_EXPLAINED_BY_COARSE_SERVICE"
        selected.append(row)
        bisect.insort(selected_times[key], t)

    return selected, dict(excluded)


def build_augmented_world(
    parent: dict[str, Any],
    births: list[dict[str, Any]],
    *,
    novelty_radius_s: float,
    minimum_evidence_score: float,
    minimum_support_stations: int,
) -> dict[str, Any]:
    out = json.loads(json.dumps(parent))
    parent_count = len(out.get("trajectories", []))
    existing_ids = {str(x.get("trajectory_id")) for x in out.get("trajectories", [])}
    for i, row in enumerate(births):
        new = json.loads(json.dumps(row))
        source_id = str(new.get("trajectory_id", f"fine-{i}"))
        new_id = f"p5birth:{i}:{source_id}"
        if new_id in existing_ids:
            raise ValueError("duplicate progressive birth id")
        new["source_five_second_trajectory_id"] = source_id
        new["trajectory_id"] = new_id
        existing_ids.add(new_id)
        out["trajectories"].append(new)

    total = len(out["trajectories"])
    out["schema"] = SCHEMA
    out["status"] = "PROGRESSIVE_5S_RESIDUAL_BIRTH_WORLD_REQUIRES_PASSENGER_JOINT_SCORING"
    out["inferred_service_trajectory_count"] = total
    sem = dict(out.get("semantics", {}))
    sem.update(
        {
            "train_count_is_input": False,
            "service_trajectory_count_is_inferred": True,
            "planned_timetable_used": False,
            "legacy_candidate_roots_used_as_input": False,
            "coarse_parent_services_retained_this_birth_probe": True,
            "five_second_grid_is_time_resolution_not_minimum_train_headway": True,
            "births_come_only_from_five_second_afc_residual": True,
        }
    )
    out["semantics"] = sem
    out["progressive_birth_refinement"] = {
        "parent_service_count": parent_count,
        "birth_count": len(births),
        "augmented_service_count": total,
        "novelty_radius_s": float(novelty_radius_s),
        "minimum_evidence_score": float(minimum_evidence_score),
        "minimum_support_stations": int(minimum_support_stations),
        "scientific_boundary": "This probe only opens evidence-supported service birth while retaining the coarse service world. Death/split replacement is deferred until joint AFC-passenger scoring can show that removing/replacing a parent does not degrade the posterior.",
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parent-services", type=Path, required=True)
    p.add_argument("--fine-services", type=Path, required=True)
    p.add_argument("--novelty-radius-s", type=float, required=True)
    p.add_argument("--minimum-evidence-score", type=float, required=True)
    p.add_argument("--minimum-support-stations", type=int, default=6)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()

    parent = json.loads(a.parent_services.read_text(encoding="utf-8"))
    fine = json.loads(a.fine_services.read_text(encoding="utf-8"))
    births, excluded = select_residual_births(
        parent,
        fine,
        novelty_radius_s=a.novelty_radius_s,
        minimum_evidence_score=a.minimum_evidence_score,
        minimum_support_stations=a.minimum_support_stations,
    )
    world = build_augmented_world(
        parent,
        births,
        novelty_radius_s=a.novelty_radius_s,
        minimum_evidence_score=a.minimum_evidence_score,
        minimum_support_stations=a.minimum_support_stations,
    )
    a.output.write_text(json.dumps(world, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "schema": SCHEMA + ".summary",
        "status": world["status"],
        **world["progressive_birth_refinement"],
        "excluded_counts": excluded,
        "births_by_line_direction": dict(Counter(f"{x['afc_line']}:{x['direction']}" for x in births)),
        "birth_evidence_score_min": min((float(x["evidence_score"]) for x in births), default=None),
        "birth_evidence_score_median": sorted(float(x["evidence_score"]) for x in births)[len(births)//2] if births else None,
    }
    a.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
