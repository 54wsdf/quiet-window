from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scripts.mppd_r1_hz_count_free_service as base

SCHEMA_LEVEL = "mppd.r1-hz-multires-service-level.v1"
SCHEMA_LADDER = "mppd.r1-hz-multires-service-count-ladder.v1"
OBSERVATION_BIN_S = 5


@dataclass(frozen=True)
class ResolutionConfig:
    structural_resolution_s: int
    event_minimum_separation_s: int
    short_smoothing_s: int
    event_support_half_window_s: int
    cluster_radius_s: float
    split_std_s: float
    split_minimum_separation_s: float
    merge_time_s: float


# The 60 s level deliberately reproduces the structural scale of the existing
# 1841-trajectory solution. Finer levels then reopen the service structure on
# the same 5 s AFC observation grid; no previous service count is constrained.
RESOLUTION_LADDER: tuple[ResolutionConfig, ...] = (
    ResolutionConfig(60, 30, 15, 60, 55.0, 55.0, 25.0, 30.0),
    ResolutionConfig(30, 20, 15, 60, 30.0, 30.0, 15.0, 20.0),
    ResolutionConfig(15, 10, 10, 60, 15.0, 15.0, 10.0, 10.0),
    ResolutionConfig(5, 5, 5, 60, 5.0, 5.0, 5.0, 5.0),
)


def resolution_configs() -> tuple[ResolutionConfig, ...]:
    return RESOLUTION_LADDER


def detect_station_events(
    counts: np.ndarray,
    bin_s: int,
    cfg: ResolutionConfig,
    threshold_quantile: float = 0.95,
    minimum_score: float = 1.0,
) -> dict[int, list[base.Event]]:
    out: dict[int, list[base.Event]] = defaultdict(list)
    short_bins = max(1, round(cfg.short_smoothing_s / bin_s))
    background_bins = max(5, round(300 / bin_s))
    min_sep_bins = max(1, math.ceil(cfg.event_minimum_separation_s / bin_s))
    half = max(1, math.ceil(cfg.event_support_half_window_s / bin_s))
    for station in range(counts.shape[0]):
        x = counts[station].astype(float)
        if x.sum() <= 0:
            continue
        short = base.moving_average(x, short_bins)
        background = base.moving_average(x, background_bins)
        residual = short - background
        score = residual / np.sqrt(np.maximum(background, 0.25))
        threshold = max(float(np.quantile(score, threshold_quantile)), minimum_score)
        left = np.r_[score[0], score[:-1]]
        right = np.r_[score[1:], score[-1]]
        candidates = np.flatnonzero((score >= threshold) & (score >= left) & (score >= right) & (residual > 0))
        ranked = candidates[np.argsort(-score[candidates], kind="stable")]
        selected: list[int] = []
        for idx in ranked:
            i = int(idx)
            if all(abs(i - j) >= min_sep_bins for j in selected):
                selected.append(i)
        selected.sort()
        for i in selected:
            lo = i
            hi = i
            while lo > 0 and i - lo < half and residual[lo - 1] > 0:
                lo -= 1
            while hi + 1 < len(x) and hi - i < half and residual[hi + 1] > 0:
                hi += 1
            raw = float(np.sum(x[lo : hi + 1]))
            bg = float(np.sum(background[lo : hi + 1]))
            center_s = float((i + 0.5) * bin_s)
            out[station].append(base.Event(
                event_id=f"{station}@{center_s:.1f}",
                station=station,
                center_s=center_s,
                score=float(score[i]),
                excess_mass=max(0.0, raw - bg),
            ))
    return out


def two_means_split(
    items: list[tuple[float, base.Event]],
    minimum_separation_s: float,
) -> tuple[list[tuple[float, base.Event]], list[tuple[float, base.Event]]] | None:
    if len(items) < 6:
        return None
    xs = np.array([x for x, _e in items], dtype=float)
    c1 = float(np.quantile(xs, 0.25))
    c2 = float(np.quantile(xs, 0.75))
    if abs(c2 - c1) < minimum_separation_s:
        return None
    for _ in range(12):
        a: list[tuple[float, base.Event]] = []
        b: list[tuple[float, base.Event]] = []
        for item in items:
            (a if abs(item[0] - c1) <= abs(item[0] - c2) else b).append(item)
        if not a or not b:
            return None
        nc1 = float(np.average([x for x, _e in a], weights=[e.weight for _x, e in a]))
        nc2 = float(np.average([x for x, _e in b], weights=[e.weight for _x, e in b]))
        if abs(nc1 - c1) + abs(nc2 - c2) < 1e-3:
            c1, c2 = nc1, nc2
            break
        c1, c2 = nc1, nc2
    if abs(c2 - c1) < minimum_separation_s:
        return None
    return a, b


def discover_path_candidates(
    events: dict[int, list[base.Event]],
    path_id: str,
    direction: str,
    offsets: dict[int, float],
    station_phase: dict[int, float],
    cfg: ResolutionConfig,
    min_station_support: int,
    complexity_penalty: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    projected: list[tuple[float, base.Event]] = []
    for station, off in offsets.items():
        phase = float(station_phase.get(station, 0.0))
        for event in events.get(station, []):
            projected.append((event.center_s - off - phase, event))

    operations: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    queue = list(base.weighted_cluster(projected, cfg.cluster_radius_s))
    while queue:
        group = queue.pop(0)
        items = group["items"]
        center = float(group["center_s"])
        summary = base.support_summary(items, center)
        if summary["station_count"] < min_station_support:
            operations["death_low_support"] += 1
            continue
        evidence_score = base.candidate_evidence_score(
            summary, len(base.LINE_PATHS[path_id]["nodes"]), cfg.cluster_radius_s
        )
        if evidence_score <= complexity_penalty:
            operations["death_complexity_penalty"] += 1
            continue
        if summary["residual_std_s"] is not None and summary["residual_std_s"] > cfg.split_std_s:
            split = two_means_split(items, cfg.split_minimum_separation_s)
            if split:
                left, right = split
                lc = float(np.average([x for x, _e in left], weights=[e.weight for _x, e in left]))
                rc = float(np.average([x for x, _e in right], weights=[e.weight for _x, e in right]))
                ls = base.support_summary(left, lc)
                rs = base.support_summary(right, rc)
                if ls["station_count"] >= min_station_support and rs["station_count"] >= min_station_support:
                    queue.insert(0, {"center_s": rc, "items": right})
                    queue.insert(0, {"center_s": lc, "items": left})
                    operations["split"] += 1
                    continue
        out.append({
            "path_id": path_id,
            "afc_line": base.LINE_PATHS[path_id]["afc_line"],
            "direction": direction,
            "reference_time_s": center,
            "evidence_score": evidence_score,
            "objective_gain": evidence_score - complexity_penalty,
            **summary,
        })
        operations["birth_retained"] += 1
    out.sort(key=lambda row: row["reference_time_s"])
    return out, operations


def discover_level(
    counts: np.ndarray,
    cfg: ResolutionConfig,
    *,
    iterations: int = 3,
    min_station_support: int = 4,
    complexity_penalty: float = 0.60,
) -> dict[str, Any]:
    events = detect_station_events(counts, OBSERVATION_BIN_S, cfg)
    edge_lags = base.estimate_directed_edge_lags(counts, OBSERVATION_BIN_S)
    offsets_by_pd = {
        (path_id, direction): base.cumulative_offsets(path_id, direction, edge_lags)
        for path_id in base.LINE_PATHS
        for direction in ("Down", "Up")
    }
    station_phase: dict[int, float] = {}
    operation_total: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    iteration_summaries = []

    for iteration in range(iterations):
        all_candidates: list[dict[str, Any]] = []
        operations: Counter[str] = Counter()
        for path_id in base.LINE_PATHS:
            for direction in ("Down", "Up"):
                required_support = max(
                    min_station_support,
                    math.ceil(0.25 * len(base.LINE_PATHS[path_id]["nodes"])),
                )
                path_candidates, path_ops = discover_path_candidates(
                    events,
                    path_id,
                    direction,
                    offsets_by_pd[(path_id, direction)],
                    station_phase,
                    cfg,
                    required_support,
                    complexity_penalty,
                )
                all_candidates.extend(path_candidates)
                operations.update(path_ops)

        candidates, dedup_ops = base.deduplicate_path_candidates(
            all_candidates, merge_time_s=cfg.merge_time_s
        )
        operations.update(dedup_ops)
        operation_total.update(operations)
        new_phase = base.estimate_station_phase(candidates, events, offsets_by_pd)
        max_phase_change = max(
            [
                abs(new_phase.get(station, 0.0) - station_phase.get(station, 0.0))
                for station in set(new_phase) | set(station_phase)
            ]
            or [0.0]
        )
        station_phase = new_phase
        iteration_summaries.append({
            "iteration": iteration + 1,
            "candidate_trajectory_count": len(candidates),
            "station_phase_count": len(station_phase),
            "max_station_phase_change_s": float(max_phase_change),
            "operations": dict(operations),
        })

    trajectories = base.materialize_trajectories(candidates, offsets_by_pd, station_phase)
    by_pd = Counter(f"{row['path_id']}:{row['direction']}" for row in trajectories)
    edge_sources = Counter(value["source"] for value in edge_lags.values())
    return {
        "schema": SCHEMA_LEVEL,
        "status": "MULTIRES_COUNT_FREE_SERVICE_LEVEL_ESTIMATED_REQUIRES_PASSENGER_JOINT_SCORING",
        "structural_resolution_s": cfg.structural_resolution_s,
        "observation_bin_s": OBSERVATION_BIN_S,
        "resolution_config": asdict(cfg),
        "semantics": {
            "train_count_is_input": False,
            "service_count_inherited_from_previous_level": False,
            "previous_level_is_constraint": False,
            "planned_timetable_used": False,
            "legacy_candidate_roots_used_as_input": False,
            "same_raw_afc_reestimated_at_every_resolution": True,
            "operations_reopened_each_level": ["birth", "death", "split", "merge", "path_reassignment"],
            "five_second_level_is_service_structure_inference_not_event_time_refinement": cfg.structural_resolution_s == 5,
        },
        "event_detection": {
            "total_station_events": sum(len(value) for value in events.values()),
            "stations_with_events": len(events),
        },
        "edge_lag_sources": dict(edge_sources),
        "iterations": iteration_summaries,
        "operation_totals": dict(operation_total),
        "objective": {
            "complexity_penalty_per_trajectory": complexity_penalty,
            "retained_total_objective_gain": float(
                sum(max(0.0, float(row.get("objective_gain", 0.0))) for row in candidates)
            ),
        },
        "inferred_service_trajectory_count": len(trajectories),
        "trajectory_counts": dict(by_pd),
        "station_phase_nuisance_s": {str(key): value for key, value in sorted(station_phase.items())},
        "trajectories": trajectories,
    }


def run_ladder(
    counts: np.ndarray,
    output_dir: Path,
    *,
    iterations: int = 3,
    min_station_support: int = 4,
    complexity_penalty: float = 0.60,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = []
    for cfg in resolution_configs():
        result = discover_level(
            counts,
            cfg,
            iterations=iterations,
            min_station_support=min_station_support,
            complexity_penalty=complexity_penalty,
        )
        path = output_dir / f"R1_SERVICE_LEVEL_{cfg.structural_resolution_s}S.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        levels.append({
            "structural_resolution_s": cfg.structural_resolution_s,
            "inferred_service_trajectory_count": result["inferred_service_trajectory_count"],
            "event_detection": result["event_detection"],
            "operation_totals": result["operation_totals"],
            "trajectory_counts": result["trajectory_counts"],
            "artifact": path.name,
        })

    return {
        "schema": SCHEMA_LADDER,
        "status": "MULTIRES_SERVICE_COUNT_LADDER_ESTIMATED_REQUIRES_PASSENGER_CLOSURE_COMPARISON",
        "observation_bin_s": OBSERVATION_BIN_S,
        "levels": levels,
        "service_count_sequence": {
            f"N_{row['structural_resolution_s']}s": row["inferred_service_trajectory_count"]
            for row in levels
        },
        "semantics": {
            "resolution_sequence_s": [cfg.structural_resolution_s for cfg in resolution_configs()],
            "all_levels_use_same_five_second_raw_afc_grid": True,
            "every_level_reestimates_service_count_from_raw_afc": True,
            "no_level_fixes_service_count_to_previous_level": True,
            "planned_timetable_used": False,
            "one_second_refinement_starts_only_after_five_second_structure_stabilizes": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--service-date", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--min-station-support", type=int, default=4)
    parser.add_argument("--complexity-penalty", type=float, default=0.60)
    args = parser.parse_args()

    counts, source_profile = base.load_exit_counts(args.input, args.service_date, OBSERVATION_BIN_S)
    result = run_ladder(
        counts,
        args.output_dir,
        iterations=args.iterations,
        min_station_support=args.min_station_support,
        complexity_penalty=args.complexity_penalty,
    )
    result["service_date"] = args.service_date
    result["source_profile"] = source_profile
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "service_date": args.service_date,
        "service_count_sequence": result["service_count_sequence"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
