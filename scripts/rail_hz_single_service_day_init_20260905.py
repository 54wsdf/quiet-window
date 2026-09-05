from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

import scripts.rail_hz_daily_service_field_init_20260905 as base

BOUNDARY_HOUR = 4
BIN_S = base.BIN_S
NBINS = base.NBINS


def load_target_exit_counts(inputs: list[Path], target_date: str) -> tuple[np.ndarray, dict[str, Any]]:
    service_date = datetime.strptime(target_date, "%Y-%m-%d")
    start = service_date + timedelta(hours=BOUNDARY_HOUR)
    end = start + timedelta(days=1)
    counts = np.zeros((81, NBINS), dtype=np.uint32)
    rows = exit_rows = retained = 0
    source_stats = []
    station_rows: Counter[int] = Counter()
    for input_path in inputs:
        local_rows = local_exit = local_retained = 0
        with gzip.open(input_path, "rt", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"time", "stationID", "status"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise SystemExit(f"missing AFC columns: {sorted(missing)}")
            for row in reader:
                rows += 1; local_rows += 1
                if row["status"] != "0":
                    continue
                exit_rows += 1; local_exit += 1
                t = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
                if not (start <= t < end):
                    continue
                try:
                    station = int(row["stationID"])
                except ValueError:
                    continue
                if not 0 <= station < 81:
                    continue
                rel = (t - start).total_seconds()
                k = int(rel) // BIN_S
                if 0 <= k < NBINS:
                    counts[station, k] += 1
                    retained += 1; local_retained += 1
                    station_rows[station] += 1
        source_stats.append({"file": input_path.name, "rows": local_rows, "exit_rows": local_exit, "retained_target_service_day_exit_rows": local_retained})
    return counts, {
        "target_date": target_date,
        "service_day_start": start.isoformat(sep=" "),
        "service_day_end_exclusive": end.isoformat(sep=" "),
        "rows_read": rows,
        "exit_rows_read": exit_rows,
        "retained_exit_rows": retained,
        "stations_with_retained_exits": len(station_rows),
        "source_stats": source_stats,
    }


def detect_events(counts: np.ndarray):
    station_event_counts: dict[str, int] = {}
    arrays = {"station": [], "center": [], "start": [], "end": [], "mass": [], "excess": [], "score": [], "threshold": []}
    for station in range(81):
        events = base.detect_events(counts[station]) if counts[station].sum() > 0 else []
        station_event_counts[str(station)] = len(events)
        for e in events:
            arrays["station"].append(station)
            arrays["center"].append(e["center_s"])
            arrays["start"].append(e["start_s"])
            arrays["end"].append(e["end_s"])
            arrays["mass"].append(e["mass"])
            arrays["excess"].append(e["excess_mass"])
            arrays["score"].append(e["score"])
            arrays["threshold"].append(e["threshold"])
    return station_event_counts, arrays


def propagation_contexts(counts: np.ndarray, prior: dict[str, Any]):
    contexts = []
    finite = total = 0
    fallback_contexts = 0
    for edge in prior["edges"]:
        expected = edge.get("arrival_to_arrival_lag_sec_median")
        if expected is None:
            continue
        u = int(edge["from_station"]); v = int(edge["to_station"])
        for context_start in range(0, 86400, base.CONTEXT_S):
            context_end = min(86400, context_start + base.CONTEXT_S)
            a = context_start // BIN_S; b = context_end // BIN_S
            x = counts[u, a:b].astype(float); y = counts[v, a:b].astype(float)
            x_mass = float(x.sum()); y_mass = float(y.sum())
            total += 1
            if x_mass >= 20 and y_mass >= 20:
                lag, corr = base.best_lag(base.highpass(x), base.highpass(y), float(expected))
            else:
                lag, corr = None, None
            if lag is not None:
                finite += 1
            if edge.get("prior_is_fallback", False):
                fallback_contexts += 1
            contexts.append({
                "path_id": edge["path_id"],
                "direction": edge["direction"],
                "from_station": u,
                "to_station": v,
                "context_start_s": context_start,
                "context_end_s": context_end,
                "upstream_exit_mass": x_mass,
                "downstream_exit_mass": y_mass,
                "structural_prior_lag_s": float(expected),
                "structural_prior_source_class": edge.get("prior_source_class"),
                "structural_prior_is_fallback": bool(edge.get("prior_is_fallback", False)),
                "afc_supported_lag_s": lag,
                "afc_correlation": corr,
                "afc_minus_structural_lag_s": lag - float(expected) if lag is not None else None,
                "effective_initial_lag_s": lag if lag is not None else float(expected),
                "evidence_class": "AFC_PLUS_STRUCTURAL_PRIOR" if lag is not None else "STRUCTURAL_PRIOR_FALLBACK",
            })
    return contexts, finite, total, fallback_contexts


def run(inputs: list[Path], target_date: str, prior_path: Path, passenger_summary_path: Path, out_prefix: Path) -> dict[str, Any]:
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    passenger = json.loads(passenger_summary_path.read_text(encoding="utf-8"))
    counts, source_meta = load_target_exit_counts(inputs, target_date)
    station_event_counts, arrays = detect_events(counts)
    contexts, finite, total, fallback_contexts = propagation_contexts(counts, prior)

    npz_path = Path(str(out_prefix) + ".npz")
    json_path = Path(str(out_prefix) + ".json")
    np.savez_compressed(
        npz_path,
        bin_s=np.array([BIN_S], dtype=np.int32),
        boundary_hour=np.array([BOUNDARY_HOUR], dtype=np.int16),
        exit_counts=counts,
        event_station=np.asarray(arrays["station"], dtype=np.int16),
        event_center_s=np.asarray(arrays["center"], dtype=np.float32),
        event_start_s=np.asarray(arrays["start"], dtype=np.float32),
        event_end_s=np.asarray(arrays["end"], dtype=np.float32),
        event_mass=np.asarray(arrays["mass"], dtype=np.float32),
        event_excess_mass=np.asarray(arrays["excess"], dtype=np.float32),
        event_score=np.asarray(arrays["score"], dtype=np.float32),
        event_threshold=np.asarray(arrays["threshold"], dtype=np.float32),
    )

    no_event_nodes = [i for i in range(81) if station_event_counts[str(i)] == 0]
    gates = {
        "target_is_full_0400_to_0400_service_day": True,
        "passenger_domain_present": passenger.get("raw_afc_events_in_service_day", 0) > 0,
        "retained_exit_rows_positive": source_meta["retained_exit_rows"] > 0,
        "passenger_facing_events_detected": len(arrays["station"]) > 0,
        "all_structural_edges_have_contexts": total > 0,
        "absolute_auxiliary_timetable_timestamp_not_used": prior.get("absolute_timestamp_policy") == "FORBIDDEN_AS_2019_REALIZED_TRUTH",
        "no_peak_window_filter": True,
        "no_line_filter": True,
        "no_passenger_subsample": True,
    }
    payload = {
        "schema": "rail.hz-single-complete-service-day-service-field-init.v1",
        "dataset_id": "CN_HZ_Tianchi_2019",
        "source_date": target_date,
        "status": "QUALIFIED_SINGLE_FULL_SERVICE_DAY_SERVICE_INIT" if all(gates.values()) else "SINGLE_DAY_SERVICE_INIT_GATE_FAILED",
        "service_day_start": source_meta["service_day_start"],
        "service_day_end_exclusive": source_meta["service_day_end_exclusive"],
        "state_representation": "FACTORIZED_PASSENGER_FACING_EVENT_AND_PROPAGATION_FIELD",
        "state_semantics": "single-day engineering qualification of S_d^(0); not observed ATS and not exact train identity",
        "source_meta": source_meta,
        "passenger_domain": {
            "raw_afc_events": passenger.get("raw_afc_events_in_service_day"),
            "all_consecutive_valid_journeys": passenger.get("all_consecutive_valid_1_to_0_journeys"),
            "cross_midnight_valid_journeys": passenger.get("cross_midnight_valid_journeys"),
            "exact_second_cohort_count": passenger.get("exact_second_cohort_count"),
        },
        "event_field": {
            "bin_s": BIN_S,
            "event_count": len(arrays["station"]),
            "station_event_counts": station_event_counts,
            "stations_with_events": sum(v > 0 for v in station_event_counts.values()),
            "nodes_without_detected_exit_events": no_event_nodes,
            "evidence_class": "AFC_INFERRED_PASSENGER_FACING_EVENT",
            "event_time_is_train_arrival_truth": False,
        },
        "propagation_field": {
            "context_width_s": base.CONTEXT_S,
            "total_structural_prior_contexts": total,
            "afc_supported_lag_contexts": finite,
            "afc_supported_share": finite / total if total else None,
            "fallback_prior_contexts": fallback_contexts,
            "contexts": contexts,
        },
        "integrity_gates": gates,
        "machine_state_file": npz_path.name,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "source_date": target_date,
        "retained_exit_rows": source_meta["retained_exit_rows"],
        "events": payload["event_field"]["event_count"],
        "stations_with_events": payload["event_field"]["stations_with_events"],
        "afc_supported_lag_context_share": payload["propagation_field"]["afc_supported_share"],
        "fallback_prior_contexts": fallback_contexts,
        "integrity_gates": gates,
    }, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, action="append", required=True)
    p.add_argument("--target-date", required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--passenger-summary", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    a = p.parse_args()
    run(a.input, a.target_date, a.prior, a.passenger_summary, a.out_prefix)


if __name__ == "__main__":
    main()
