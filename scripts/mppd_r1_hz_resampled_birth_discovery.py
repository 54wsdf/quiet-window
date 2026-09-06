from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import scripts.mppd_r1_hz_count_free_service as base
import scripts.mppd_r1_hz_count_free_service_v2 as cf2  # noqa: F401 - patches v2 competition
import scripts.mppd_r1_hz_multires_service_count_v2 as mr2
import scripts.mppd_r1_hz_progressive_birth_refinement as birth

SCHEMA = "mppd.r1-hz-resampled-birth-discovery.v1"
OBSERVATION_BIN_S = 5
FRONTIER_POINTS: tuple[tuple[str, float, float], ...] = (
    ("n60e100", 60.0, 1.00),
    ("n45e090", 45.0, 0.90),
    ("n45e080", 45.0, 0.80),
    ("n30e100", 30.0, 1.00),
    ("n30e090", 30.0, 0.90),
    ("n30e080", 30.0, 0.80),
)


def load_exit_counts_poisson_bootstrap(
    paths: list[Path],
    service_date: str,
    *,
    bin_s: int = OBSERVATION_BIN_S,
    bootstrap_seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load AFC exit counts, optionally applying a Poisson(1) row bootstrap.

    Poisson bootstrap preserves expected intensity while perturbing the empirical
    passenger sample. A fixed seed and stable input ordering make every replicate
    exactly reproducible. The realized service count remains unconstrained.
    """
    nbins = base.DAY_S // bin_s
    counts = np.zeros((81, nbins), dtype=np.uint32)
    rng = np.random.default_rng(bootstrap_seed) if bootstrap_seed is not None else None
    rows = exits = retained_rows = 0
    bootstrap_weight_sum = 0
    nonzero_bootstrap_rows = 0
    by_station: Counter[int] = Counter()

    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"time", "stationID", "status"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise SystemExit(f"missing AFC columns in {path}: {sorted(missing)}")
            for row in reader:
                rows += 1
                if str(row["status"]).strip() != "0":
                    continue
                exits += 1
                try:
                    station = int(row["stationID"])
                    t = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    continue
                if not 0 <= station < 81:
                    continue
                rel = base.service_day_seconds(t, service_date)
                if rel is None:
                    continue
                k = int(rel) // bin_s
                if not 0 <= k < nbins:
                    continue
                retained_rows += 1
                by_station[station] += 1
                w = 1 if rng is None else int(rng.poisson(1.0))
                bootstrap_weight_sum += w
                if w > 0:
                    nonzero_bootstrap_rows += 1
                    counts[station, k] += np.uint32(w)

    return counts, {
        "raw_rows_scanned": rows,
        "exit_rows_scanned": exits,
        "retained_service_day_exit_rows_before_resampling": retained_rows,
        "stations_with_exit_rows": len(by_station),
        "bootstrap_mode": "NONE_FULL_DATA" if bootstrap_seed is None else "POISSON_1_ROW_BOOTSTRAP",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_weight_sum": bootstrap_weight_sum,
        "nonzero_bootstrap_rows": nonzero_bootstrap_rows,
    }


def compact_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": str(row.get("trajectory_id")),
        "afc_line": str(row.get("afc_line")),
        "direction": str(row.get("direction")),
        "path_id": str(row.get("path_id")),
        "path_ambiguous": bool(row.get("path_ambiguous", False)),
        "path_alternatives": row.get("path_alternatives", []),
        "anchor_time_s": birth.anchor_time(row),
        "reference_time_s": float(row.get("reference_time_s", 0.0)),
        "support_station_count": int(row.get("support_station_count", 0)),
        "support_event_count": int(row.get("support_event_count", 0)),
        "support_event_ids": list(row.get("support_event_ids", [])),
        "support_weight": float(row.get("support_weight", 0.0)),
        "evidence_score": float(row.get("evidence_score", 0.0)),
    }


def discover_from_counts(
    counts: np.ndarray,
    *,
    iterations: int = 3,
    min_station_support: int = 4,
    complexity_penalty: float = 0.60,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    coarse = base.discover_count_free(
        counts,
        bin_s=OBSERVATION_BIN_S,
        iterations=iterations,
        cluster_radius_s=55.0,
        split_std_s=55.0,
        min_station_support=min_station_support,
        complexity_penalty=complexity_penalty,
    )
    fine_cfg = mr2.mr.resolution_configs()[-1]
    fine = mr2.discover_level_with_v2_contract(
        counts,
        fine_cfg,
        iterations=iterations,
        min_station_support=min_station_support,
        complexity_penalty=complexity_penalty,
    )

    frontier: dict[str, Any] = {}
    for label, novelty, evidence in FRONTIER_POINTS:
        births, excluded = birth.select_residual_births(
            coarse,
            fine,
            novelty_radius_s=novelty,
            minimum_evidence_score=evidence,
            minimum_support_stations=6,
        )
        frontier[label] = {
            "novelty_radius_s": novelty,
            "minimum_evidence_score": evidence,
            "birth_count": len(births),
            "births_by_line_direction": dict(Counter(f"{x['afc_line']}:{x['direction']}" for x in births)),
            "excluded_counts": excluded,
            "births": [compact_trajectory(x) for x in births],
        }
    return coarse, fine, frontier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, action="append", required=True)
    p.add_argument("--service-date", required=True)
    p.add_argument("--bootstrap-seed", type=int)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--complexity-penalty", type=float, default=0.60)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--coarse-output", type=Path)
    p.add_argument("--fine-output", type=Path)
    a = p.parse_args()

    counts, source_profile = load_exit_counts_poisson_bootstrap(
        a.input,
        a.service_date,
        bootstrap_seed=a.bootstrap_seed,
    )
    coarse, fine, frontier = discover_from_counts(
        counts,
        iterations=a.iterations,
        complexity_penalty=a.complexity_penalty,
    )

    result = {
        "schema": SCHEMA,
        "status": "RESAMPLED_COUNT_FREE_SERVICE_BIRTH_DISCOVERY_COMPLETED_REQUIRES_STABILITY_MATCHING",
        "service_date": a.service_date,
        "source_profile": source_profile,
        "coarse_service_count": int(coarse["inferred_service_trajectory_count"]),
        "fine_five_second_service_count": int(fine["inferred_service_trajectory_count"]),
        "coarse_support_trajectories": [compact_trajectory(x) for x in coarse.get("trajectories", [])],
        "frontier": frontier,
        "semantics": {
            "planned_timetable_used": False,
            "service_count_is_inferred_in_every_replicate": True,
            "coarse_and_fine_worlds_are_reestimated_in_every_replicate": True,
            "poisson_bootstrap_preserves_expected_afc_intensity": a.bootstrap_seed is not None,
            "five_second_grid_is_observation_and_structure_resolution_not_minimum_headway": True,
            "support_event_ids_are_preserved_for_cross_resample_evidence_matching": True,
            "full_data_birth_can_be_recovered_as_coarse_service_in_a_resample": True,
        },
    }
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.coarse_output:
        a.coarse_output.write_text(json.dumps(coarse, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.fine_output:
        a.fine_output.write_text(json.dumps(fine, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "bootstrap_seed": a.bootstrap_seed,
        "coarse_service_count": result["coarse_service_count"],
        "fine_five_second_service_count": result["fine_five_second_service_count"],
        "frontier_birth_counts": {k: v["birth_count"] for k, v in frontier.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
