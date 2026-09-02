#!/usr/bin/env python3
"""Fetch and validate selected public rail data products.

This module is intentionally limited to public-source acquisition and technical
validation. It writes raw provider bytes separately from processed validation
outputs and never requires credentials other than the workflow's external
storage configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_session() -> requests.Session:
    retry = Retry(
        total=7,
        connect=7,
        read=7,
        backoff_factor=1.0,
        status_forcelist=[408, 425, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "quiet-window/public-rail-source-validation"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def download_file(session: requests.Session, url: str, path: Path, min_bytes: int = 1) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=(30, 360)) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        content_type = response.headers.get("Content-Type", "")
    size = path.stat().st_size
    if size < min_bytes:
        raise RuntimeError(f"Downloaded file too small: {path.name} ({size} bytes)")
    return {
        "url": url,
        "filename": path.name,
        "bytes": size,
        "sha256": sha256_file(path),
        "content_type": content_type,
    }


def write_sha256s(root: Path, records: list[dict[str, Any]]) -> None:
    lines = [f"{record['sha256']}  {record['filename']}" for record in records]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_mbta_url(base_root: str, base_family: str, value: str) -> str:
    value = str(value).strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("/"):
        return urljoin(base_root, value)
    if "/" in value:
        return urljoin(base_root, value)
    return urljoin(base_family.rstrip("/") + "/", value)


def mbta_range_stats(groups: pd.core.groupby.generic.DataFrameGroupBy, column: str) -> tuple[int, float | None, float | None]:
    values: list[float] = []
    for _, group in groups:
        x = pd.to_numeric(group[column], errors="coerce").dropna()
        if len(x) >= 2:
            values.append(float(x.max() - x.min()))
    if not values:
        return 0, None, None
    series = pd.Series(values, dtype=float)
    return len(series), float(series.median()), float(series.quantile(0.90))


def run_mbta(session: requests.Session, cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    raw = out / "raw"
    processed = out / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    index_url = cfg["index_url"]
    index_path = raw / "index.csv"
    raw_records = [download_file(session, index_url, index_path, min_bytes=100)]
    index_bytes = index_path.read_bytes()
    index = pd.read_csv(io.BytesIO(index_bytes))
    index.columns = [str(column).strip() for column in index.columns]
    date_col = next((column for column in index.columns if "date" in column.lower()), None)
    path_col = next((column for column in index.columns if any(token in column.lower() for token in ["path", "file", "url"])), None)
    if not date_col:
        raise RuntimeError(f"MBTA index has no date column: {list(index.columns)}")
    index["_date"] = pd.to_datetime(index[date_col], errors="coerce")
    index = index[index["_date"].notna()].copy().sort_values("_date", ascending=False)

    cutoff = pd.Timestamp(cfg["cutoff_date"]).normalize()
    excluded = {pd.Timestamp(value).normalize() for value in cfg.get("exclude_dates", [])}
    days = int(cfg.get("days", 20))
    family = cfg.get("family_base_url") or index_url.rsplit("/", 1)[0]
    root = f"{urlparse(index_url).scheme}://{urlparse(index_url).netloc}/"

    selected: list[tuple[pd.Timestamp, str]] = []
    for _, row in index.iterrows():
        day = pd.Timestamp(row["_date"]).normalize()
        if day >= cutoff or day.weekday() >= 5 or day in excluded:
            continue
        if path_col:
            url = resolve_mbta_url(root, family, row[path_col])
        else:
            url = f"{family.rstrip('/')}/{day.strftime('%Y-%m-%d')}-subway-on-time-performance-v1.parquet"
        selected.append((day, url))
        if len(selected) >= days:
            break
    if len(selected) != days:
        raise RuntimeError(f"MBTA index supplied {len(selected)} eligible weekdays; requested {days}")

    coarse = ["service_date", "trip_id", "stop_id"]
    route_aware = ["service_date", "route_id", "trip_id", "stop_id"]
    keep = ["service_date", "route_id", "trip_id", "stop_id", "travel_time_seconds", "headway_trunk_seconds"]
    date_rows: list[dict[str, Any]] = []
    collision_frames: list[pd.DataFrame] = []
    total_rows = 0
    total_removed = 0
    route_unique_all = True

    for day, url in selected:
        filename = f"{day.date().isoformat()}-subway-on-time-performance-v1.parquet"
        parquet_path = raw / filename
        record = download_file(session, url, parquet_path, min_bytes=1024)
        try:
            frame = pd.read_parquet(parquet_path, columns=keep)
        except Exception as exc:
            raise RuntimeError(f"Invalid MBTA Parquet {filename}: {exc}") from exc
        if len(frame) < 1000:
            raise RuntimeError(f"MBTA {filename} has only {len(frame)} rows")
        raw_records.append(record)
        frame["service_date"] = frame["service_date"].astype(str)
        duplicated = frame.duplicated(coarse, keep=False)
        dup = frame.loc[duplicated].copy()
        if len(dup):
            grouped = dup.groupby(coarse, dropna=False)
            groups = int(grouped.ngroups)
            cross_route = int((grouped["route_id"].nunique(dropna=True) > 1).sum())
            removed = int(len(dup) - groups)
            dup["_requested_date"] = day.date().isoformat()
            collision_frames.append(dup)
        else:
            groups = 0
            cross_route = 0
            removed = 0
        route_unique = not frame.duplicated(route_aware).any()
        route_unique_all = route_unique_all and route_unique
        total_rows += len(frame)
        total_removed += removed
        date_rows.append({
            "service_date": day.date().isoformat(),
            "rows": int(len(frame)),
            "coarse_collision_rows": int(duplicated.sum()),
            "coarse_collision_groups": groups,
            "cross_route_collision_groups": cross_route,
            "cross_route_group_share": (cross_route / groups) if groups else None,
            "naive_dedup_removed_rows": removed,
            "naive_dedup_removed_share": (removed / len(frame)) if len(frame) else None,
            "route_aware_key_unique": bool(route_unique),
            "source_url": url,
            "sha256": record["sha256"],
        })

    dates = pd.DataFrame(date_rows)
    dates.to_csv(processed / "mbta_temporal_robustness_by_date.csv", index=False)
    collisions = pd.concat(collision_frames, ignore_index=True) if collision_frames else pd.DataFrame(columns=keep)
    collision_groups = collisions.groupby(coarse, dropna=False)
    total_groups = int(collision_groups.ngroups)
    total_cross = int((collision_groups["route_id"].nunique(dropna=True) > 1).sum()) if total_groups else 0
    travel_n, travel_med, travel_p90 = mbta_range_stats(collision_groups, "travel_time_seconds") if total_groups else (0, None, None)
    head_n, head_med, head_p90 = mbta_range_stats(collision_groups, "headway_trunk_seconds") if total_groups else (0, None, None)
    collision_routes = sorted(str(value) for value in collisions["route_id"].dropna().unique()) if len(collisions) else []

    audit = {
        "dataset_id": cfg.get("dataset_id", "mbta_lamp_temporal_window"),
        "provider": "MBTA LAMP",
        "retrieved_utc": utc_now(),
        "source_index_url": index_url,
        "local_state": "VALIDATED",
        "selected_dates": dates["service_date"].tolist(),
        "days": int(len(dates)),
        "days_with_collision": int((dates["coarse_collision_groups"] > 0).sum()),
        "share_days_with_collision": float((dates["coarse_collision_groups"] > 0).mean()),
        "rows": int(total_rows),
        "collision_groups": total_groups,
        "cross_route_collision_groups": total_cross,
        "cross_route_collision_group_share": (total_cross / total_groups) if total_groups else None,
        "collision_route_ids": collision_routes,
        "naive_dedup_removed_rows": int(total_removed),
        "naive_dedup_removed_share": (total_removed / total_rows) if total_rows else None,
        "route_aware_key_unique_all_days": bool(route_unique_all),
        "travel_time_groups_with_2plus_values": int(travel_n),
        "travel_time_range_median_s": travel_med,
        "travel_time_range_p90_s": travel_p90,
        "trunk_headway_groups_with_2plus_values": int(head_n),
        "trunk_headway_range_median_s": head_med,
        "trunk_headway_range_p90_s": head_p90,
        "coarse_key": coarse,
        "route_aware_key": route_aware,
        "raw_files": raw_records,
        "terms_url": cfg.get("terms_url"),
    }
    (processed / "mbta_temporal_robustness.audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_sha256s(raw, raw_records)
    return audit


def normalized_name(value: object) -> str:
    text = str(value).lower().replace("&", "and")
    text = text.replace("ohare", "o hare").replace("harlem lake", "harlem")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(station|terminal|cta|line|l)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def parse_gtfs_time(value: str) -> float:
    hour, minute, second = (int(part) for part in str(value).split(":"))
    return hour * 3600 + minute * 60 + second


def cta_active_services(calendar: pd.DataFrame, exceptions: pd.DataFrame, service_date: pd.Timestamp) -> set[str]:
    cal = calendar.copy()
    cal["_start"] = pd.to_datetime(cal["start_date"], format="%Y%m%d", errors="coerce")
    cal["_end"] = pd.to_datetime(cal["end_date"], format="%Y%m%d", errors="coerce")
    weekday = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][service_date.weekday()]
    active = set(cal.loc[
        (cal[weekday] == "1") & (cal["_start"] <= service_date) & (cal["_end"] >= service_date),
        "service_id",
    ].dropna())
    day = exceptions[exceptions["date"] == service_date.strftime("%Y%m%d")]
    active.update(day.loc[day["exception_type"] == "1", "service_id"].dropna())
    active.difference_update(day.loc[day["exception_type"] == "2", "service_id"].dropna())
    return active


def run_cta(session: requests.Session, cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    raw = out / "raw"
    processed = out / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    gtfs_path = raw / "google_transit.zip"
    raw_records = [download_file(session, cfg["gtfs_url"], gtfs_path, min_bytes=10_000)]
    with zipfile.ZipFile(gtfs_path) as archive:
        required = ["calendar.txt", "calendar_dates.txt", "routes.txt", "trips.txt", "stops.txt", "stop_times.txt"]
        missing = [name for name in required if name not in archive.namelist()]
        if missing:
            raise RuntimeError(f"CTA GTFS missing required files: {missing}")
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"CTA GTFS failed ZIP integrity at {bad}")
        tables = {name[:-4]: pd.read_csv(archive.open(name), dtype=str) for name in required}

    demand_start = pd.Timestamp(cfg["demand_start"]).strftime("%Y-%m-%dT00:00:00")
    demand_end = pd.Timestamp(cfg["demand_end"]).strftime("%Y-%m-%dT23:59:59")
    ridership_params = {
        "$select": "station_id,stationname,date,daytype,rides",
        "$where": f"date between '{demand_start}' and '{demand_end}' and daytype='W'",
        "$limit": "50000",
    }
    response = session.get(cfg["ridership_api"], params=ridership_params, timeout=(30, 180))
    response.raise_for_status()
    ridership_rows = response.json()
    if not ridership_rows:
        raise RuntimeError("CTA ridership query returned no weekday rows")
    ridership_raw = raw / "cta_station_entries_query.json"
    ridership_raw.write_text(json.dumps(ridership_rows, ensure_ascii=False), encoding="utf-8")
    raw_records.append({
        "url": response.url,
        "filename": ridership_raw.name,
        "bytes": ridership_raw.stat().st_size,
        "sha256": sha256_file(ridership_raw),
        "content_type": response.headers.get("Content-Type", ""),
    })

    meta_response = session.get(cfg["station_metadata_api"], params={"$limit": 500}, timeout=(30, 120))
    meta_response.raise_for_status()
    meta_rows = meta_response.json()
    if not meta_rows:
        raise RuntimeError("CTA station metadata query returned no rows")
    meta_raw = raw / "cta_station_metadata.json"
    meta_raw.write_text(json.dumps(meta_rows, ensure_ascii=False), encoding="utf-8")
    raw_records.append({
        "url": meta_response.url,
        "filename": meta_raw.name,
        "bytes": meta_raw.stat().st_size,
        "sha256": sha256_file(meta_raw),
        "content_type": meta_response.headers.get("Content-Type", ""),
    })

    demand = pd.DataFrame(ridership_rows)
    demand["station_id"] = pd.to_numeric(demand["station_id"], errors="raise").astype(int)
    demand["rides"] = pd.to_numeric(demand["rides"], errors="coerce")
    demand = demand[demand["rides"].notna()].copy()
    station_demand = demand.groupby(["station_id", "stationname"], as_index=False).agg(
        avg_weekday_entries=("rides", "mean"),
        weekday_days=("rides", "size"),
    )

    metadata = pd.DataFrame(meta_rows)
    metadata["station_id"] = pd.to_numeric(metadata["station_id"], errors="raise").astype(int)
    metadata["station_normalized"] = metadata["longname"].map(normalized_name)
    station_demand = station_demand.merge(
        metadata[["station_id", "longname", "station_normalized"]],
        on="station_id", how="left", validate="one_to_one",
    )

    service_date = pd.Timestamp(cfg["service_date"]).normalize()
    active = cta_active_services(tables["calendar"], tables["calendar_dates"], service_date)
    if not active:
        start = pd.to_datetime(tables["calendar"]["start_date"], format="%Y%m%d", errors="coerce").min()
        end = pd.to_datetime(tables["calendar"]["end_date"], format="%Y%m%d", errors="coerce").max()
        raise RuntimeError(f"CTA GTFS has no active service on {service_date.date()} (calendar span {start.date()} to {end.date()})")

    routes = tables["routes"].copy()
    rail = routes[pd.to_numeric(routes["route_type"], errors="coerce") == 1].copy()
    rail["route_label"] = rail["route_short_name"].fillna("").str.strip()
    rail.loc[rail["route_label"] == "", "route_label"] = rail.loc[rail["route_label"] == "", "route_id"]
    rail = rail[["route_id", "route_label"]].drop_duplicates()

    trips = tables["trips"].merge(rail, on="route_id", how="inner", validate="many_to_one")
    trips = trips[trips["service_id"].isin(active)][["trip_id", "route_id", "route_label"]].drop_duplicates("trip_id")
    if trips.empty:
        raise RuntimeError("CTA GTFS has no active rail trips for requested service date")

    stops = tables["stops"].copy()
    stops["station_key"] = stops["parent_station"].where(
        stops["parent_station"].notna() & (stops["parent_station"].str.strip() != ""),
        stops["stop_id"],
    )
    stop_to_station = stops.set_index("stop_id")["station_key"].to_dict()
    parent_name = stops.set_index("stop_id")["stop_name"].to_dict()

    stop_times = tables["stop_times"]
    stop_times = stop_times[stop_times["trip_id"].isin(set(trips["trip_id"]))][["trip_id", "stop_id", "departure_time"]].copy()
    stop_times["station_key"] = stop_times["stop_id"].map(stop_to_station)
    stop_times = stop_times[stop_times["station_key"].notna()].merge(trips, on="trip_id", how="left", validate="many_to_one")
    stop_times["departure_seconds"] = stop_times["departure_time"].map(parse_gtfs_time)
    stop_times["is_am"] = stop_times["departure_seconds"].between(6 * 3600, 9 * 3600, inclusive="left")
    stop_times["is_pm"] = stop_times["departure_seconds"].between(15 * 3600, 19 * 3600, inclusive="left")
    unique = stop_times.drop_duplicates(["station_key", "route_label", "trip_id"])
    station_route = unique.groupby(["station_key", "route_label"], as_index=False).agg(
        scheduled_departures=("trip_id", "nunique"),
        scheduled_departures_am=("is_am", "sum"),
        scheduled_departures_pm=("is_pm", "sum"),
    )
    station_route["gtfs_station_name"] = station_route["station_key"].map(parent_name)
    station_route["station_normalized"] = station_route["gtfs_station_name"].map(normalized_name)
    station_route.to_csv(processed / "cta_station_route_scheduled_departures.csv", index=False)

    station_lookup = station_route[["station_key", "gtfs_station_name", "station_normalized"]].drop_duplicates()
    duplicate_norms = station_lookup.groupby("station_normalized")["station_key"].nunique()
    ambiguous = set(duplicate_norms[duplicate_norms > 1].index)
    if ambiguous:
        station_lookup = station_lookup[~station_lookup["station_normalized"].isin(ambiguous)].copy()
    station_lookup = station_lookup.drop_duplicates("station_normalized")
    mapped = station_demand.merge(station_lookup, on="station_normalized", how="left", validate="many_to_one")
    mapped_mass = float(mapped.loc[mapped["station_key"].notna(), "avg_weekday_entries"].sum())
    source_mass = float(mapped["avg_weekday_entries"].sum())
    if mapped_mass <= 0:
        raise RuntimeError("CTA station demand did not map to GTFS parent stations")

    route_by_station = {
        str(station): group[["route_label", "scheduled_departures"]].copy()
        for station, group in station_route.groupby("station_key")
    }
    allocation_rows: list[dict[str, Any]] = []
    station_checks: list[dict[str, Any]] = []
    for row in mapped[mapped["station_key"].notna()].itertuples(index=False):
        key = str(row.station_key)
        routes_here = route_by_station.get(key)
        if routes_here is None or routes_here.empty:
            continue
        entries = float(row.avg_weekday_entries)
        routes_here = routes_here[routes_here["scheduled_departures"] > 0].copy()
        weights = routes_here["scheduled_departures"].astype(float)
        total_weight = float(weights.sum())
        route_count = len(routes_here)
        equal_sum = 0.0
        local_sum = 0.0
        for route_row in routes_here.itertuples(index=False):
            equal_alloc = entries / route_count
            local_alloc = entries * float(route_row.scheduled_departures) / total_weight
            equal_sum += equal_alloc
            local_sum += local_alloc
            allocation_rows.append({
                "station_id": int(row.station_id),
                "stationname": row.stationname,
                "station_key": key,
                "route": str(route_row.route_label),
                "route_count": route_count,
                "station_entries": entries,
                "scheduled_departures": int(route_row.scheduled_departures),
                "equal_alloc": equal_alloc,
                "local_gtfs_alloc": local_alloc,
            })
        station_checks.append({
            "station_id": int(row.station_id),
            "stationname": row.stationname,
            "route_count": route_count,
            "station_entries": entries,
            "equal_sum": equal_sum,
            "local_gtfs_sum": local_sum,
        })

    allocation = pd.DataFrame(allocation_rows)
    checks = pd.DataFrame(station_checks)
    if allocation.empty:
        raise RuntimeError("CTA local station-route allocation produced no rows")
    max_equal_error = float((checks["equal_sum"] - checks["station_entries"]).abs().max())
    max_local_error = float((checks["local_gtfs_sum"] - checks["station_entries"]).abs().max())
    if max_equal_error > 1e-6 or max_local_error > 1e-6:
        raise RuntimeError(f"CTA station mass conservation failed: equal={max_equal_error}, local={max_local_error}")

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
    route.to_csv(processed / "cta_route_comparison.csv", index=False)
    checks.to_csv(processed / "cta_station_mass_checks.csv", index=False)

    total_variation = 0.5 * float((route["local_gtfs_share"] - route["equal_share"]).abs().sum())
    spearman = float(route[["equal_share", "local_gtfs_share"]].corr(method="spearman").iloc[0, 1])
    max_row = route.loc[route["abs_share_diff_pp"].idxmax()]
    shared = checks[checks["route_count"] > 1]
    full_copy_total = float((checks["station_entries"] * checks["route_count"]).sum())
    full_copy_inflation = 100.0 * (full_copy_total / mapped_mass - 1.0)
    contribution = allocation.copy()
    contribution["gross_abs_difference"] = (contribution["local_gtfs_alloc"] - contribution["equal_alloc"]).abs()
    station_contrib = contribution.groupby(["station_id", "stationname"], as_index=False)["gross_abs_difference"].sum()
    station_contrib = station_contrib.sort_values("gross_abs_difference", ascending=False)
    gross_total = float(station_contrib["gross_abs_difference"].sum())
    station_contrib["gross_difference_share"] = station_contrib["gross_abs_difference"] / gross_total if gross_total else 0.0
    station_contrib.to_csv(processed / "cta_station_difference_contributions.csv", index=False)

    audit = {
        "dataset_id": cfg.get("dataset_id", "cta_station_route_local_weighting"),
        "provider": "Chicago Transit Authority / City of Chicago",
        "retrieved_utc": utc_now(),
        "local_state": "VALIDATED",
        "demand_period": [cfg["demand_start"], cfg["demand_end"]],
        "gtfs_service_date": service_date.date().isoformat(),
        "source_station_rows": int(len(station_demand)),
        "mapped_station_rows": int(mapped["station_key"].notna().sum()),
        "mapped_station_mass": mapped_mass,
        "source_station_mass": source_mass,
        "mapped_mass_share": mapped_mass / source_mass if source_mass else None,
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
        "ambiguous_normalized_station_names_excluded": sorted(ambiguous),
        "raw_files": raw_records,
        "terms_url": cfg.get("terms_url"),
    }
    (processed / "cta_station_route_gtfs_sensitivity.audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_sha256s(raw, raw_records)
    return audit


def write_failure(out: Path, dataset_id: str, provider: str, exc: Exception) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    processed = out / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    audit = {
        "dataset_id": dataset_id,
        "provider": provider,
        "retrieved_utc": utc_now(),
        "local_state": "FAILED",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    (processed / "failure.audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()
    summary: dict[str, Any] = {
        "run_id": request.get("run_id"),
        "started_utc": utc_now(),
        "tasks": {},
    }

    if request.get("mbta", {}).get("enabled", True):
        try:
            summary["tasks"]["mbta"] = run_mbta(session, request["mbta"], args.output_dir / "mbta")
        except Exception as exc:  # noqa: BLE001
            summary["tasks"]["mbta"] = write_failure(
                args.output_dir / "mbta",
                request.get("mbta", {}).get("dataset_id", "mbta_lamp_temporal_window"),
                "MBTA LAMP",
                exc,
            )

    if request.get("cta", {}).get("enabled", True):
        try:
            summary["tasks"]["cta"] = run_cta(session, request["cta"], args.output_dir / "cta")
        except Exception as exc:  # noqa: BLE001
            summary["tasks"]["cta"] = write_failure(
                args.output_dir / "cta",
                request.get("cta", {}).get("dataset_id", "cta_station_route_local_weighting"),
                "Chicago Transit Authority / City of Chicago",
                exc,
            )

    summary["finished_utc"] = utc_now()
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "run_id": summary["run_id"],
        "mbta_state": summary.get("tasks", {}).get("mbta", {}).get("local_state"),
        "cta_state": summary.get("tasks", {}).get("cta", {}).get("local_state"),
    }, indent=2))
    # Task failures are recorded in machine-readable audits. The workflow can still
    # persist partial diagnostics without converting a network error into an
    # implicit success state.
    return 0


if __name__ == "__main__":
    sys.exit(main())
