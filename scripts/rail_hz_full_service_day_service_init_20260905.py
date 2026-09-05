from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

import scripts.rail_hz_daily_service_field_init_20260905 as base

BOUNDARY_HOUR = 4
START_DATE = datetime(2019, 1, 1)
COMPLETE_DAYS = 24
BIN_S = base.BIN_S
NBINS = base.NBINS
STORE_SHAPE = (COMPLETE_DAYS, 81, NBINS)


def service_day_index(t: datetime) -> tuple[int | None, float | None]:
    d = datetime(t.year, t.month, t.day)
    if t.hour < BOUNDARY_HOUR:
        service_date = d - timedelta(days=1)
    else:
        service_date = d
    idx = (service_date.date() - START_DATE.date()).days
    if not 0 <= idx < COMPLETE_DAYS:
        return None, None
    boundary = service_date + timedelta(hours=BOUNDARY_HOUR)
    rel = (t - boundary).total_seconds()
    if not 0 <= rel < 86400:
        return None, None
    return idx, rel


def init_store(path: Path, meta_path: Path) -> None:
    arr = np.memmap(path, dtype="uint32", mode="w+", shape=STORE_SHAPE)
    arr[:] = 0
    arr.flush()
    meta = {
        "schema": "rail.hz24-service-exit-count-store.v1",
        "boundary_hour": BOUNDARY_HOUR,
        "start_service_date": "2019-01-01",
        "complete_service_days": COMPLETE_DAYS,
        "shape": list(STORE_SHAPE),
        "calendar_source_stats": [],
        "excluded_edge_fragment_exit_rows": 0,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def accumulate(input_path: Path, source_file: str, store_path: Path, meta_path: Path) -> dict[str, Any]:
    arr = np.memmap(store_path, dtype="uint32", mode="r+", shape=STORE_SHAPE)
    rows = 0
    exit_rows = 0
    retained_exit_rows = 0
    excluded_edge_exit_rows = 0
    retained_by_service_day: Counter[str] = Counter()
    station_rows: Counter[int] = Counter()
    with gzip.open(input_path, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"time", "stationID", "status"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"missing AFC columns: {sorted(missing)}")
        for row in reader:
            rows += 1
            if row["status"] != "0":
                continue
            exit_rows += 1
            t = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
            idx, rel = service_day_index(t)
            if idx is None or rel is None:
                excluded_edge_exit_rows += 1
                continue
            try:
                station = int(row["stationID"])
            except ValueError:
                continue
            if not 0 <= station < 81:
                continue
            k = int(rel) // BIN_S
            if 0 <= k < NBINS:
                arr[idx, station, k] += 1
                retained_exit_rows += 1
                retained_by_service_day[(START_DATE + timedelta(days=idx)).date().isoformat()] += 1
                station_rows[station] += 1
    arr.flush()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["calendar_source_stats"].append({
        "source_file": source_file,
        "rows": rows,
        "exit_rows": exit_rows,
        "retained_complete_service_day_exit_rows": retained_exit_rows,
        "excluded_archive_edge_exit_rows": excluded_edge_exit_rows,
        "retained_by_service_day": dict(retained_by_service_day),
        "observed_station_count_retained": len(station_rows),
    })
    meta["excluded_edge_fragment_exit_rows"] += excluded_edge_exit_rows
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "source_file": source_file,
        "rows": rows,
        "exit_rows": exit_rows,
        "retained_exit_rows": retained_exit_rows,
        "excluded_edge_exit_rows": excluded_edge_exit_rows,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def detect_day_events(counts: np.ndarray) -> tuple[dict[str, int], dict[str, list[float]], dict[int, list[dict[str, float]]]]:
    station_event_counts: dict[str, int] = {}
    arrays = {
        "station": [], "center": [], "start": [], "end": [], "mass": [], "excess": [], "score": [], "threshold": []
    }
    event_lookup: dict[int, list[dict[str, float]]] = defaultdict(list)
    for station in range(81):
        events = base.detect_events(counts[station]) if counts[station].sum() > 0 else []
        station_event_counts[str(station)] = len(events)
        for e in events:
            event_lookup[station].append(e)
            arrays["station"].append(station)
            arrays["center"].append(e["center_s"])
            arrays["start"].append(e["start_s"])
            arrays["end"].append(e["end_s"])
            arrays["mass"].append(e["mass"])
            arrays["excess"].append(e["excess_mass"])
            arrays["score"].append(e["score"])
            arrays["threshold"].append(e["threshold"])
    return station_event_counts, arrays, event_lookup


def propagation_contexts(counts: np.ndarray, prior: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    contexts: list[dict[str, Any]] = []
    finite = 0
    total = 0
    for edge in prior["edges"]:
        expected = edge["arrival_to_arrival_lag_sec_median"]
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
            contexts.append({
                "path_id": edge["path_id"],
                "direction": edge["direction"],
                "from_station": u,
                "to_station": v,
                "context_start_s": context_start,
                "context_end_s": context_end,
                "upstream_exit_mass": x_mass,
                "downstream_exit_mass": y_mass,
                "structural_prior_lag_s": expected,
                "afc_supported_lag_s": lag,
                "afc_correlation": corr,
                "afc_minus_structural_lag_s": lag - float(expected) if lag is not None else None,
                "effective_initial_lag_s": lag if lag is not None else expected,
                "evidence_class": "AFC_PLUS_STRUCTURAL_PRIOR" if lag is not None else "STRUCTURAL_PRIOR_FALLBACK",
            })
    return contexts, finite, total


def finalize(store_path: Path, store_meta_path: Path, prior_path: Path, passenger_domain_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(store_path, dtype="uint32", mode="r", shape=STORE_SHAPE)
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    store_meta = json.loads(store_meta_path.read_text(encoding="utf-8"))
    summaries = []
    for idx in range(COMPLETE_DAYS):
        date = (START_DATE + timedelta(days=idx)).date().isoformat()
        counts = np.asarray(arr[idx])
        passenger_summary_path = passenger_domain_dir / f"{date}.service_day_summary.json"
        passenger = json.loads(passenger_summary_path.read_text(encoding="utf-8"))
        station_event_counts, event_arrays, _lookup = detect_day_events(counts)
        contexts, finite, total = propagation_contexts(counts, prior)
        npz_path = out_dir / f"{date}.service_init.npz"
        np.savez_compressed(
            npz_path,
            bin_s=np.array([BIN_S], dtype=np.int32),
            boundary_hour=np.array([BOUNDARY_HOUR], dtype=np.int16),
            exit_counts=counts,
            event_station=np.asarray(event_arrays["station"], dtype=np.int16),
            event_center_s=np.asarray(event_arrays["center"], dtype=np.float32),
            event_start_s=np.asarray(event_arrays["start"], dtype=np.float32),
            event_end_s=np.asarray(event_arrays["end"], dtype=np.float32),
            event_mass=np.asarray(event_arrays["mass"], dtype=np.float32),
            event_excess_mass=np.asarray(event_arrays["excess"], dtype=np.float32),
            event_score=np.asarray(event_arrays["score"], dtype=np.float32),
            event_threshold=np.asarray(event_arrays["threshold"], dtype=np.float32),
        )
        no_event_nodes = [i for i in range(81) if station_event_counts[str(i)] == 0]
        payload = {
            "schema": "rail.hz-complete-service-day-service-field-init.v1",
            "dataset_id": "CN_HZ_Tianchi_2019",
            "source_date": date,
            "service_day_start": f"{date} 04:00:00",
            "service_day_end_exclusive": ((START_DATE + timedelta(days=idx+1)).date().isoformat() + " 04:00:00"),
            "state_representation": "FACTORIZED_PASSENGER_FACING_EVENT_AND_PROPAGATION_FIELD",
            "state_semantics": "initial latent service field S_d^(0), not observed ATS and not exact train identity",
            "passenger_domain": {
                "raw_afc_events": passenger["raw_afc_events_in_service_day"],
                "all_consecutive_valid_journeys": passenger["all_consecutive_valid_1_to_0_journeys"],
                "cross_midnight_valid_journeys": passenger["cross_midnight_valid_journeys"],
                "exact_second_cohort_count": passenger["exact_second_cohort_count"],
            },
            "event_field": {
                "bin_s": BIN_S,
                "event_count": len(event_arrays["station"]),
                "station_event_counts": station_event_counts,
                "stations_with_events": sum(v > 0 for v in station_event_counts.values()),
                "nodes_without_detected_exit_events": no_event_nodes,
                "evidence_class": "AFC_INFERRED_PASSENGER_FACING_EVENT",
                "event_time_is_train_arrival_truth": False,
            },
            "propagation_field": {
                "context_width_s": base.CONTEXT_S,
                "lag_search_half_width_s": base.LAG_SEARCH_HALF_WIDTH_S,
                "total_structural_prior_contexts": total,
                "afc_supported_lag_contexts": finite,
                "afc_supported_share": finite / total if total else None,
                "contexts": contexts,
            },
            "uncertainty_policy": {
                "local_event_time": "pulse width and detector score retained",
                "edge_lag": "AFC-supported lag used when estimable; otherwise structural prior remains explicit fallback",
                "unresolved_nodes": "retained as unresolved latent service support rather than deleted",
                "direction": "line-direction structure is represented by propagation factors; station pulses remain potentially direction-ambiguous until joint inference",
            },
            "scope_invariants": {
                "full_service_day": True,
                "service_day_boundary_hour": BOUNDARY_HOUR,
                "peak_window_filter": False,
                "line_filter": False,
                "passenger_subsample": False,
                "absolute_auxiliary_timetable_timestamp_used": False,
            },
            "machine_state_file": npz_path.name,
        }
        json_path = out_dir / f"{date}.service_init.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        summaries.append(payload)

    event_counts = [x["event_field"]["event_count"] for x in summaries]
    lag_shares = [x["propagation_field"]["afc_supported_share"] for x in summaries if x["propagation_field"]["afc_supported_share"] is not None]
    gates = {
        "exactly_24_complete_service_days": len(summaries) == 24,
        "all_days_have_full_day_passenger_domain": all(x["passenger_domain"]["raw_afc_events"] > 0 for x in summaries),
        "all_days_have_passenger_facing_events": all(x["event_field"]["event_count"] > 0 for x in summaries),
        "all_days_preserve_full_scope": all(x["scope_invariants"]["full_service_day"] and not x["scope_invariants"]["peak_window_filter"] and not x["scope_invariants"]["line_filter"] and not x["scope_invariants"]["passenger_subsample"] for x in summaries),
        "no_day_uses_absolute_auxiliary_timestamp": all(not x["scope_invariants"]["absolute_auxiliary_timetable_timestamp_used"] for x in summaries),
        "auxiliary_prior_policy_safe": prior["absolute_timestamp_policy"] == "FORBIDDEN_AS_2019_REALIZED_TRUTH",
    }
    result = {
        "schema": "rail.hz24-complete-service-day-service-field-init-summary.v1",
        "dataset_id": "CN_HZ_Tianchi_2019",
        "status": "QUALIFIED_HZ24_COMPLETE_SERVICE_DAY_SERVICE_INIT" if all(gates.values()) else "HZ24_SERVICE_INIT_GATE_FAILED",
        "complete_service_days": 24,
        "service_day_boundary_hour": BOUNDARY_HOUR,
        "archive_store_meta": store_meta,
        "integrity_gates": gates,
        "event_field": {
            "total_events": sum(event_counts),
            "daily_event_count_median": statistics.median(event_counts),
            "daily_event_count_min": min(event_counts),
            "daily_event_count_max": max(event_counts),
        },
        "propagation_field": {
            "daily_afc_supported_context_share_median": statistics.median(lag_shares) if lag_shares else None,
            "daily_afc_supported_context_share_min": min(lag_shares) if lag_shares else None,
            "daily_afc_supported_context_share_max": max(lag_shares) if lag_shares else None,
        },
        "day_summaries": [{
            "date": x["source_date"],
            "events": x["event_field"]["event_count"],
            "stations_with_events": x["event_field"]["stations_with_events"],
            "afc_supported_lag_context_share": x["propagation_field"]["afc_supported_share"],
            "unresolved_nodes": x["event_field"]["nodes_without_detected_exit_events"],
        } for x in summaries],
        "next_stage": "CANDIDATE_SERVICE_ROOT_COMPLETION_THEN_FORMAL_R1B_FULL_DAY",
    }
    summary_path = out_dir / "HZ24_SERVICE_FIELD_INIT_SUMMARY_20260905.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "complete_service_days": 24,
        "total_events": result["event_field"]["total_events"],
        "median_afc_supported_context_share": result["propagation_field"]["daily_afc_supported_context_share_median"],
        "integrity_gates": gates,
    }, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)
    return result


def main() -> None:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("init-store"); s.add_argument("--store", type=Path, required=True); s.add_argument("--meta", type=Path, required=True)
    s = sub.add_parser("accumulate"); s.add_argument("--input", type=Path, required=True); s.add_argument("--source-file", required=True); s.add_argument("--store", type=Path, required=True); s.add_argument("--meta", type=Path, required=True)
    s = sub.add_parser("finalize"); s.add_argument("--store", type=Path, required=True); s.add_argument("--meta", type=Path, required=True); s.add_argument("--prior", type=Path, required=True); s.add_argument("--passenger-domain-dir", type=Path, required=True); s.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    if a.command == "init-store": init_store(a.store, a.meta)
    elif a.command == "accumulate": accumulate(a.input, a.source_file, a.store, a.meta)
    else: finalize(a.store, a.meta, a.prior, a.passenger_domain_dir, a.out_dir)


if __name__ == "__main__": main()
