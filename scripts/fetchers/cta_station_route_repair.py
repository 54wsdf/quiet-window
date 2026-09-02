#!/usr/bin/env python3
"""Finalize CTA station-by-route allocation with CTA's common station IDs.

The City of Chicago rail-station ridership feed exposes station_id values such
as 40010, 40020, ... that coincide with CTA GTFS parent-station IDs. This
validator uses that numeric identity first and computes a mass-conserving local
station-by-route scheduled-service allocation. It consumes only files already
retrieved by the public source runner and updates the CTA audit in place.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Root rail-source-refresh output directory")
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    cfg = request["cta"]
    cta_dir = args.output_dir / "cta"
    raw = cta_dir / "raw"
    processed = cta_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    entries_path = raw / "cta_station_entries_query.json"
    station_route_path = processed / "cta_station_route_scheduled_departures.csv"
    gtfs_path = raw / "google_transit.zip"
    if not entries_path.exists() or not station_route_path.exists() or not gtfs_path.exists():
        raise RuntimeError("CTA repair requires the retrieved ridership JSON, GTFS ZIP and station-route schedule table")

    demand_raw = pd.DataFrame(json.loads(entries_path.read_text(encoding="utf-8")))
    if demand_raw.empty:
        raise RuntimeError("CTA ridership JSON is empty")
    demand_raw["station_id"] = pd.to_numeric(demand_raw["station_id"], errors="raise").astype(int)
    demand_raw["rides"] = pd.to_numeric(demand_raw["rides"], errors="coerce")
    demand_raw = demand_raw[demand_raw["rides"].notna()].copy()
    demand = demand_raw.groupby(["station_id", "stationname"], as_index=False).agg(
        avg_weekday_entries=("rides", "mean"),
        weekday_days=("rides", "size"),
    )
    demand["station_key"] = demand["station_id"].astype(str)

    station_route = pd.read_csv(station_route_path, dtype={"station_key": str, "route_label": str})
    station_route["station_key"] = station_route["station_key"].astype(str).str.replace(r"\.0$", "", regex=True)
    station_route["scheduled_departures"] = pd.to_numeric(station_route["scheduled_departures"], errors="raise")
    gtfs_keys = set(station_route["station_key"].dropna())

    demand["mapped_numeric_id"] = demand["station_key"].isin(gtfs_keys)
    diagnostics = demand[["station_id", "stationname", "station_key", "avg_weekday_entries", "weekday_days", "mapped_numeric_id"]].copy()
    diagnostics.to_csv(processed / "cta_station_gtfs_id_crosswalk.csv", index=False)

    mapped = demand[demand["mapped_numeric_id"]].copy()
    unmapped = demand[~demand["mapped_numeric_id"]].copy()
    unmapped.to_csv(processed / "cta_unmapped_station_ids.csv", index=False)
    if mapped.empty:
        raise RuntimeError("No CTA ridership station_id matched a GTFS parent station ID")

    source_mass = float(demand["avg_weekday_entries"].sum())
    mapped_mass = float(mapped["avg_weekday_entries"].sum())
    unmapped_mass = float(unmapped["avg_weekday_entries"].sum()) if len(unmapped) else 0.0

    route_by_station = {
        str(station): group[["route_label", "scheduled_departures"]].copy()
        for station, group in station_route.groupby("station_key")
    }

    allocation_rows: list[dict[str, object]] = []
    station_checks: list[dict[str, object]] = []
    for row in mapped.itertuples(index=False):
        key = str(row.station_key)
        routes_here = route_by_station[key].copy()
        routes_here = routes_here[pd.to_numeric(routes_here["scheduled_departures"], errors="coerce") > 0]
        routes_here = routes_here.groupby("route_label", as_index=False)["scheduled_departures"].sum()
        if routes_here.empty:
            continue
        entries = float(row.avg_weekday_entries)
        route_count = int(len(routes_here))
        total_weight = float(routes_here["scheduled_departures"].sum())
        equal_sum = 0.0
        local_sum = 0.0
        for rr in routes_here.itertuples(index=False):
            equal_alloc = entries / route_count
            local_alloc = entries * float(rr.scheduled_departures) / total_weight
            equal_sum += equal_alloc
            local_sum += local_alloc
            allocation_rows.append({
                "station_id": int(row.station_id),
                "stationname": row.stationname,
                "station_key": key,
                "route": str(rr.route_label),
                "route_count": route_count,
                "station_entries": entries,
                "scheduled_departures": int(rr.scheduled_departures),
                "equal_alloc": equal_alloc,
                "local_gtfs_alloc": local_alloc,
            })
        station_checks.append({
            "station_id": int(row.station_id),
            "stationname": row.stationname,
            "station_key": key,
            "route_count": route_count,
            "station_entries": entries,
            "equal_sum": equal_sum,
            "local_gtfs_sum": local_sum,
        })

    allocation = pd.DataFrame(allocation_rows)
    checks = pd.DataFrame(station_checks)
    if allocation.empty or checks.empty:
        raise RuntimeError("CTA ID-first station-route allocation produced no rows")

    max_equal_error = float((checks["equal_sum"] - checks["station_entries"]).abs().max())
    max_local_error = float((checks["local_gtfs_sum"] - checks["station_entries"]).abs().max())
    if max_equal_error > 1e-6 or max_local_error > 1e-6:
        raise RuntimeError(f"CTA station-level conservation failed: equal={max_equal_error}, local={max_local_error}")

    allocated_mass = float(checks["station_entries"].sum())
    if not np.isclose(allocated_mass, mapped_mass, rtol=0, atol=1e-6):
        raise RuntimeError(f"CTA mapped demand mass mismatch: mapped={mapped_mass}, allocated={allocated_mass}")

    route = allocation.groupby("route", as_index=False).agg(
        equal_alloc=("equal_alloc", "sum"),
        local_gtfs_alloc=("local_gtfs_alloc", "sum"),
    )
    equal_total = float(route["equal_alloc"].sum())
    local_total = float(route["local_gtfs_alloc"].sum())
    route["equal_share"] = route["equal_alloc"] / equal_total
    route["local_gtfs_share"] = route["local_gtfs_alloc"] / local_total
    route["share_diff_pp"] = 100.0 * (route["local_gtfs_share"] - route["equal_share"])
    route["abs_share_diff_pp"] = route["share_diff_pp"].abs()
    route["equal_rank"] = route["equal_share"].rank(method="min", ascending=False).astype(int)
    route["local_gtfs_rank"] = route["local_gtfs_share"].rank(method="min", ascending=False).astype(int)

    allocation.to_csv(processed / "cta_station_route_local_allocation.csv", index=False)
    checks.to_csv(processed / "cta_station_mass_checks.csv", index=False)
    route.to_csv(processed / "cta_route_comparison.csv", index=False)

    total_variation = 0.5 * float((route["local_gtfs_share"] - route["equal_share"]).abs().sum())
    spearman = float(route[["equal_share", "local_gtfs_share"]].corr(method="spearman").iloc[0, 1])
    max_row = route.loc[route["abs_share_diff_pp"].idxmax()]
    shared = checks[checks["route_count"] > 1]
    full_copy_total = float((checks["station_entries"] * checks["route_count"]).sum())
    full_copy_inflation = 100.0 * (full_copy_total / mapped_mass - 1.0)

    contrib = allocation.copy()
    contrib["gross_abs_difference"] = (contrib["local_gtfs_alloc"] - contrib["equal_alloc"]).abs()
    station_contrib = contrib.groupby(["station_id", "stationname"], as_index=False)["gross_abs_difference"].sum()
    station_contrib = station_contrib.sort_values("gross_abs_difference", ascending=False)
    gross_total = float(station_contrib["gross_abs_difference"].sum())
    station_contrib["gross_difference_share"] = station_contrib["gross_abs_difference"] / gross_total if gross_total else 0.0
    station_contrib.to_csv(processed / "cta_station_difference_contributions.csv", index=False)

    audit = {
        "dataset_id": cfg.get("dataset_id", "cta_station_route_local_service"),
        "provider": "Chicago Transit Authority / City of Chicago",
        "retrieved_utc": utc_now(),
        "local_state": "VALIDATED",
        "gtfs_download_url": cfg.get("gtfs_url"),
        "gtfs_producer_url": cfg.get("producer_url"),
        "demand_period": [cfg["demand_start"], cfg["demand_end"]],
        "gtfs_service_date": cfg["service_date"],
        "crosswalk_rule": "City ridership station_id equals CTA GTFS parent_station ID; numeric ID is used directly",
        "source_station_rows": int(len(demand)),
        "mapped_station_rows": int(len(mapped)),
        "unmapped_station_rows": int(len(unmapped)),
        "source_station_mass": source_mass,
        "mapped_station_mass": mapped_mass,
        "unmapped_station_mass": unmapped_mass,
        "mapped_mass_share": (mapped_mass / source_mass) if source_mass else None,
        "shared_stations": int(len(shared)),
        "shared_station_entry_share": float(shared["station_entries"].sum() / mapped_mass) if mapped_mass else None,
        "route_labels": sorted(route["route"].astype(str).tolist()),
        "total_variation_distance": total_variation,
        "spearman_route_share": spearman,
        "max_abs_share_diff_pp": float(max_row["abs_share_diff_pp"]),
        "max_abs_share_diff_route": str(max_row["route"]),
        "equal_rank_order": route.sort_values("equal_share", ascending=False)["route"].astype(str).tolist(),
        "local_gtfs_rank_order": route.sort_values("local_gtfs_share", ascending=False)["route"].astype(str).tolist(),
        "max_station_mass_error_equal": max_equal_error,
        "max_station_mass_error_local_gtfs": max_local_error,
        "full_copy_total": full_copy_total,
        "full_copy_inflation_pct": full_copy_inflation,
        "top5_station_gross_difference_share": float(station_contrib.head(5)["gross_difference_share"].sum()) if gross_total else 0.0,
        "interpretation": "Observed station entries are conserved and allocated in proportion to scheduled route departure opportunities at the same parent station. This is an allocation sensitivity result, not observed passenger route choice.",
        "terms_url": cfg.get("terms_url"),
    }
    audit_path = processed / "cta_station_route_gtfs_sensitivity.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    failure_path = processed / "failure.audit.json"
    if failure_path.exists():
        failure_path.unlink()

    summary_path = args.output_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {"run_id": request.get("run_id"), "tasks": {}}
    summary.setdefault("tasks", {})["cta"] = audit
    summary["cta_repair_completed_utc"] = utc_now()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "cta_state": audit["local_state"],
        "stations": audit["source_station_rows"],
        "mapped": audit["mapped_station_rows"],
        "mapped_mass_share": audit["mapped_mass_share"],
        "shared_stations": audit["shared_stations"],
        "tv": audit["total_variation_distance"],
        "max_abs_share_diff_pp": audit["max_abs_share_diff_pp"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
