from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import scripts.mppd_r1_hz_count_free_passenger_coverage as cov

SCHEMA = "mppd.r1-hz-directional-transfer-evidence.v1"


def option_set(leg: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(x["path_id"]), str(x["direction"]))
        for x in leg.get("compatible_service_options", [])
    }


def weighted_quantile(hist: Counter[float], q: float) -> float | None:
    total = float(sum(hist.values()))
    if total <= 0:
        return None
    target = q * total
    acc = 0.0
    for value, mass in sorted(hist.items()):
        acc += float(mass)
        if acc >= target:
            return float(value)
    return float(max(hist))


def authority_maps(authority: dict[str, Any]) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str, str, str, str], dict[str, Any]]]:
    option_line: dict[tuple[str, str], str] = {}
    for row in authority.get("line_paths", []):
        option_line[(str(row["path_id"]), str(row["direction_id"]))] = str(row["line_id"])
    moves = {}
    for row in authority.get("transfer_movements", []):
        key = (
            str(row["station_id"]),
            str(row["from_line_id"]),
            str(row["from_direction_id"]),
            str(row["to_line_id"]),
            str(row["to_direction_id"]),
        )
        moves[key] = row
    return option_line, moves


def matching_service_options(service: dict[str, Any], leg: dict[str, Any]) -> set[tuple[str, str]]:
    allowed = option_set(leg)
    matches = set(service.get("options", set())) & allowed
    if matches:
        return matches
    canonical = (str(service["path_id"]), str(service["direction"]))
    return {canonical} if canonical in allowed else set()


def relation_candidates(
    station: int,
    left_service: dict[str, Any],
    right_service: dict[str, Any],
    left_leg: dict[str, Any],
    right_leg: dict[str, Any],
    option_line: dict[tuple[str, str], str],
    authority_moves: dict[tuple[str, str, str, str, str], dict[str, Any]],
) -> set[tuple[str, str, str, str, str]]:
    left = matching_service_options(left_service, left_leg)
    right = matching_service_options(right_service, right_leg)
    out: set[tuple[str, str, str, str, str]] = set()
    for p1, d1 in left:
        for p2, d2 in right:
            l1 = option_line.get((p1, d1))
            l2 = option_line.get((p2, d2))
            if l1 is None or l2 is None:
                continue
            key = (str(station), l1, d1, l2, d2)
            if key in authority_moves:
                out.add(key)
    return out


def run(
    cohorts_path: Path,
    routes_path: Path,
    services_path: Path,
    authority_path: Path,
    output: Path,
) -> dict[str, Any]:
    routes = cov.load_routes(routes_path)
    services, service_raw = cov.load_services(services_path)
    index = cov.LegIndex(services)
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    option_line, authority_moves = authority_maps(authority)

    evidence_mass: Counter[tuple[str, str, str, str, str]] = Counter()
    ambiguous_mass: Counter[tuple[str, str, str, str, str]] = Counter()
    budget_hist: dict[tuple[str, str, str, str, str], Counter[float]] = defaultdict(Counter)
    resolved_mass = 0.0
    transfer_passenger_mass = 0.0
    transfer_episode_mass = 0.0
    no_relation_candidate_mass = 0.0

    pf = pq.ParquetFile(cohorts_path)
    for batch in pf.iter_batches(batch_size=50000):
        for row in batch.to_pylist():
            mass = float(row["passenger_mass"])
            rr = routes.get(cov.route_key(row))
            if not rr:
                continue
            chosen = None
            result = None
            for route in rr:
                r = cov.evaluate_route(row, route, index)
                if r["ok"]:
                    chosen = route
                    result = r
                    break
            if chosen is None or result is None:
                continue
            resolved_mass += mass
            legs = chosen.get("ride_legs", [])
            chain = result["chain"]
            if len(legs) <= 1:
                continue
            transfer_passenger_mass += mass
            for j in range(len(legs) - 1):
                left_ride = chain[j]
                right_ride = chain[j + 1]
                left_service = services[str(left_ride["trajectory_id"])]
                right_service = services[str(right_ride["trajectory_id"])]
                station = int(legs[j]["to_station"])
                candidates = relation_candidates(
                    station,
                    left_service,
                    right_service,
                    legs[j],
                    legs[j + 1],
                    option_line,
                    authority_moves,
                )
                if not candidates:
                    no_relation_candidate_mass += mass
                    continue
                transfer_episode_mass += mass
                responsibility = 1.0 / len(candidates)
                budget = float(right_ride["departure_s"]) - float(left_ride["arrival_s"])
                budget_rounded = float(round(budget))
                for key in candidates:
                    w = mass * responsibility
                    evidence_mass[key] += w
                    budget_hist[key][budget_rounded] += w
                    if len(candidates) > 1:
                        ambiguous_mass[key] += w

    rows = []
    for key, meta in sorted(authority_moves.items()):
        mass = float(evidence_mass[key])
        hist = budget_hist[key]
        rows.append(
            {
                "movement_id": str(meta.get("movement_id", "")),
                "station_id": key[0],
                "from_line_id": key[1],
                "from_direction_id": key[2],
                "to_line_id": key[3],
                "to_direction_id": key[4],
                "identification": "DIRECT_PASSENGER_CHAIN_EVIDENCE" if mass > 0 else "UNRESOLVED_IN_CURRENT_DAY",
                "evidence_mass": mass,
                "ambiguous_option_responsibility_mass": float(ambiguous_mass[key]),
                "selected_service_gap_q05_s": weighted_quantile(hist, 0.05),
                "selected_service_gap_q50_s": weighted_quantile(hist, 0.50),
                "selected_service_gap_q95_s": weighted_quantile(hist, 0.95),
                "gap_semantics": "UPSTREAM_ARRIVAL_TO_SELECTED_DOWNSTREAM_DEPARTURE_EQUALS_TRANSFER_MOVEMENT_PLUS_WAITING; NOT A PURE_TRANSFER_TIME_ESTIMATE",
            }
        )

    result = {
        "schema": SCHEMA,
        "status": "DIRECTIONAL_TRANSFER_EVIDENCE_EXTRACTED_REQUIRES_JOINT_MOVEMENT_POSTERIOR",
        "service_date": service_raw.get("service_date", "2019-01-04"),
        "service_trajectory_count": int(service_raw["inferred_service_trajectory_count"]),
        "resolved_passenger_mass": resolved_mass,
        "transfer_passenger_mass": transfer_passenger_mass,
        "transfer_episode_mass": transfer_episode_mass,
        "no_authority_relation_candidate_mass": no_relation_candidate_mass,
        "authority_transfer_movement_count": len(authority_moves),
        "movement_with_direct_evidence_count": sum(x["evidence_mass"] > 0 for x in rows),
        "movements": rows,
        "semantics": {
            "planned_timetable_used": False,
            "service_count_is_input": False,
            "service_world_is_unshifted_count_free_warm_start": True,
            "direction_and_line_are_explicit": True,
            "service_option_ambiguity_is_fractionally_retained": True,
            "selected_service_gap_is_not_pure_transfer_time": True,
            "pure_transfer_posterior_pending": True,
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "status",
        "service_trajectory_count",
        "resolved_passenger_mass",
        "transfer_passenger_mass",
        "transfer_episode_mass",
        "no_authority_relation_candidate_mass",
        "authority_transfer_movement_count",
        "movement_with_direct_evidence_count",
    )}, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohorts", type=Path, required=True)
    p.add_argument("--routes", type=Path, required=True)
    p.add_argument("--services", type=Path, required=True)
    p.add_argument("--authority", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.cohorts, a.routes, a.services, a.authority, a.output)


if __name__ == "__main__":
    main()
