#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import pandas as pd
import requests

HOURLY_ID = "5wq4-mkjj"
OD_ID = "28vm-gjqr"
HOURLY_ROWS_UPDATED_AT = 1787757107
HOURLY_ARCHIVE_ROWS = 43_835_841
HOURLY_ARCHIVE_PARTS = 439
OD_ROWS_UPDATED_AT = 1787243523

HERE = Path(__file__).resolve().parent
T8_CORE = HERE / "t8_mta_public_registry.py"
spec = importlib.util.spec_from_file_location("t8_core", T8_CORE)
t8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t8)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def get_json_retry(session: requests.Session, url: str, *, params=None, timeout=180, tries=8):
    last = None
    for n in range(tries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            if n + 1 < tries:
                time.sleep(min(2 ** n, 30))
    raise RuntimeError(f"GET failed after {tries} tries: {url} params={params}: {last}")


def metadata(session: requests.Session, dataset: str) -> dict:
    return get_json_retry(session, f"https://data.ny.gov/api/views/{dataset}", timeout=120)


def quote_in(values) -> str:
    vals = []
    for x in values:
        s = str(x).replace("'", "''")
        vals.append(f"'{s}'")
    return "(" + ",".join(vals) + ")"


def fetch_grouped_json(session, dataset: str, *, select: str, where: str, group: str, order: str, page_size=50000):
    url = f"https://data.ny.gov/resource/{dataset}.json"
    offset = 0
    out = []
    pages = []
    while True:
        params = {
            "$select": select,
            "$where": where,
            "$group": group,
            "$order": order,
            "$limit": str(page_size),
            "$offset": str(offset),
        }
        data = get_json_retry(session, url, params=params, timeout=600)
        out.extend(data)
        pages.append({"offset": offset, "returned_rows": len(data)})
        if len(data) < page_size:
            break
        offset += page_size
    return out, pages


def require_frozen_hourly(hourly_path: Path, source_manifest_path: Path, start: str, end_exclusive: str) -> dict:
    if not hourly_path.is_file():
        raise RuntimeError(f"missing frozen hourly slice: {hourly_path}")
    if not source_manifest_path.is_file():
        raise RuntimeError(f"missing frozen hourly derivation manifest: {source_manifest_path}")
    d = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if d.get("state") != "DERIVED_FROM_FROZEN_ARCHIVE":
        raise RuntimeError(f"unexpected frozen hourly state: {d.get('state')}")
    if d.get("dataset_id") != HOURLY_ID:
        raise RuntimeError(f"unexpected frozen hourly dataset: {d.get('dataset_id')}")
    src = d.get("source_archive") or {}
    if int(src.get("row_count") or -1) != HOURLY_ARCHIVE_ROWS:
        raise RuntimeError(f"unexpected frozen hourly archive rows: {src.get('row_count')}")
    if int(src.get("part_count") or -1) != HOURLY_ARCHIVE_PARTS:
        raise RuntimeError(f"unexpected frozen hourly archive parts: {src.get('part_count')}")
    if int(src.get("verified_part_count") or -1) != HOURLY_ARCHIVE_PARTS:
        raise RuntimeError(f"frozen hourly parts not fully verified: {src.get('verified_part_count')}")
    if int(src.get("verified_row_count") or -1) != HOURLY_ARCHIVE_ROWS:
        raise RuntimeError(f"frozen hourly rows not fully verified: {src.get('verified_row_count')}")
    if int(src.get("rowsUpdatedAt") or -1) != HOURLY_ROWS_UPDATED_AT:
        raise RuntimeError(f"unexpected frozen hourly rowsUpdatedAt: {src.get('rowsUpdatedAt')}")
    period = d.get("period") or {}
    if period.get("start") != start or period.get("end_exclusive") != end_exclusive:
        raise RuntimeError(f"frozen hourly period mismatch: {period}")
    expected_sha = (d.get("output") or {}).get("sha256")
    actual_sha = sha256(hourly_path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"frozen hourly slice hash mismatch: {actual_sha} != {expected_sha}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtfs", type=Path, required=True)
    ap.add_argument("--hourly", type=Path, required=True)
    ap.add_argument("--hourly-source-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--route-id", default="7")
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end-exclusive", default="2026-07-01")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--month", type=int, default=6)
    ap.add_argument("--mapping-tolerance-m", type=float, default=200.0)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    hourly_source = require_frozen_hourly(
        args.hourly, args.hourly_source_manifest, args.start, args.end_exclusive
    )
    hourly = pd.read_csv(args.hourly, dtype=str)
    required_hourly_cols = {
        "transit_timestamp", "station_complex_id", "station_complex",
        "latitude", "longitude", "ridership",
    }
    missing_hourly = sorted(required_hourly_cols - set(hourly.columns))
    if missing_hourly:
        raise RuntimeError(f"frozen hourly slice missing columns: {missing_hourly}")
    if len(hourly) < 1000:
        raise RuntimeError(f"unexpectedly small frozen 30-day hourly aggregate: {len(hourly)}")

    s = requests.Session()
    s.headers["User-Agent"] = "OpenCity-T8-public-registry-rehearsal/20260903"
    om = metadata(s, OD_ID)
    oru = int(om.get("rowsUpdatedAt") or 0)
    if oru != OD_ROWS_UPDATED_AT:
        raise RuntimeError(f"OD rowsUpdatedAt changed: {oru} != {OD_ROWS_UPDATED_AT}")

    gtfs = t8.read_gtfs(args.gtfs)
    stop_to_top, top_stops = t8.top_level_stop_map(gtfs["stops"])
    route_id = t8.select_route_id(gtfs["routes"], str(args.route_id))
    route_seq, _ = t8.canonical_route_sequence(gtfs, route_id, stop_to_top)
    mapping = t8.map_route_stations_to_complexes(route_seq, top_stops, hourly, args.mapping_tolerance_m)
    mapping.to_csv(out / "route_station_complex_mapping_preflight.csv", index=False)
    route_complexes = list(dict.fromkeys(mapping["station_complex_id"].astype(str)))
    if len(route_complexes) < 10:
        raise RuntimeError(f"unexpectedly small route mapping: {len(route_complexes)}")

    ids = quote_in(route_complexes)
    base_where = f"year={args.year} AND month={args.month} AND origin_station_complex_id in {ids}"
    denom_select = (
        "day_of_week,hour_of_day,origin_station_complex_id,"
        "sum(estimated_average_ridership) as total_ridership"
    )
    denom_group = "day_of_week,hour_of_day,origin_station_complex_id"
    denom_order = "day_of_week,hour_of_day,origin_station_complex_id"
    denom, denom_pages = fetch_grouped_json(
        s, OD_ID, select=denom_select, where=base_where,
        group=denom_group, order=denom_order,
    )
    if not denom:
        raise RuntimeError("OD denominator query returned zero rows")

    corridor_where = base_where + f" AND destination_station_complex_id in {ids}"
    corridor_select = (
        "day_of_week,hour_of_day,origin_station_complex_id,destination_station_complex_id,"
        "sum(estimated_average_ridership) as estimated_average_ridership"
    )
    corridor_group = "day_of_week,hour_of_day,origin_station_complex_id,destination_station_complex_id"
    corridor_order = corridor_group
    corridor, corridor_pages = fetch_grouped_json(
        s, OD_ID, select=corridor_select, where=corridor_where,
        group=corridor_group, order=corridor_order,
    )
    if not corridor:
        raise RuntimeError("OD corridor query returned zero rows")

    key_cols = ["day_of_week", "hour_of_day", "origin_station_complex_id"]
    denom_df = pd.DataFrame(denom)
    corr_df = pd.DataFrame(corridor)
    denom_df["total_ridership"] = pd.to_numeric(denom_df["total_ridership"], errors="coerce").fillna(0.0)
    corr_df["estimated_average_ridership"] = pd.to_numeric(
        corr_df["estimated_average_ridership"], errors="coerce"
    ).fillna(0.0)
    corr_sum = corr_df.groupby(key_cols, as_index=False)["estimated_average_ridership"].sum().rename(
        columns={"estimated_average_ridership": "corridor_total"}
    )
    residual = denom_df.merge(corr_sum, on=key_cols, how="left")
    residual["corridor_total"] = residual["corridor_total"].fillna(0.0)
    residual["estimated_average_ridership"] = (
        residual["total_ridership"] - residual["corridor_total"]
    ).clip(lower=0.0)
    residual = residual[residual["estimated_average_ridership"] > 0].copy()
    residual["destination_station_complex_id"] = "OUTSIDE_CORRIDOR"

    od_cols = [
        "day_of_week", "hour_of_day", "origin_station_complex_id",
        "destination_station_complex_id", "estimated_average_ridership",
    ]
    od_slice = pd.concat([corr_df[od_cols], residual[od_cols]], ignore_index=True)
    od_path = out / "mta_od_2026-06_route7_denominator_preserving.csv"
    od_slice.to_csv(od_path, index=False)

    hourly_copy = out / "mta_hourly_2026-06_aggregated.csv"
    if args.hourly.resolve() != hourly_copy.resolve():
        hourly_copy.write_bytes(args.hourly.read_bytes())
    hourly_manifest_copy = out / "HOURLY_FROZEN_SLICE_MANIFEST.json"
    if args.hourly_source_manifest.resolve() != hourly_manifest_copy.resolve():
        hourly_manifest_copy.write_bytes(args.hourly_source_manifest.read_bytes())

    manifest = {
        "state": "PREPARED",
        "evidence_class": "public_data registry / official public source derived slice",
        "period": {
            "start": args.start, "end_exclusive": args.end_exclusive,
            "year": args.year, "month": args.month,
        },
        "route_id": route_id,
        "route_complex_ids": route_complexes,
        "route_station_count": int(len(mapping)),
        "max_gtfs_to_complex_match_m": float(mapping["match_distance_m"].max()),
        "source_versions": {
            "hourly": {
                "dataset_id": HOURLY_ID,
                "rowsUpdatedAt": HOURLY_ROWS_UPDATED_AT,
                "row_count": HOURLY_ARCHIVE_ROWS,
                "archive_parts": HOURLY_ARCHIVE_PARTS,
                "mode": "frozen_drive_archive_all_parts_hash_verified",
                "source_manifest_sha256": hourly_source["source_archive"]["manifest_sha256"],
                "parts_folder_id": hourly_source["source_archive"]["parts_folder_id"],
            },
            "od": {
                "dataset_id": OD_ID,
                "rowsUpdatedAt": oru,
                "expected_rowsUpdatedAt": OD_ROWS_UPDATED_AT,
                "mode": "official_socrata_grouped_query_under_canonical_snapshot_version_gate",
            },
        },
        "queries": {
            "hourly": {
                "mode": "frozen_archive_filter_and_group",
                "period": hourly_source["period"],
                "verified_part_count": hourly_source["source_archive"]["verified_part_count"],
                "verified_row_count": hourly_source["source_archive"]["verified_row_count"],
                "ordered_part_sha256_aggregate": hourly_source["source_archive"]["ordered_part_sha256_aggregate"],
            },
            "od_denominator": {
                "select": denom_select, "where": base_where,
                "group": denom_group, "order": denom_order, "pages": denom_pages,
            },
            "od_corridor": {
                "select": corridor_select, "where": corridor_where,
                "group": corridor_group, "order": corridor_order, "pages": corridor_pages,
            },
        },
        "derived_rows": {
            "hourly_station_hour_rows": int(len(hourly)),
            "hourly_selected_raw_rows": int(hourly_source["selected_raw_rows"]),
            "od_corridor_rows": int(len(corr_df)),
            "od_outside_residual_rows": int(len(residual)),
            "od_slice_rows": int(len(od_slice)),
        },
        "files": {
            hourly_copy.name: {
                "bytes": hourly_copy.stat().st_size,
                "sha256": sha256(hourly_copy),
            },
            hourly_manifest_copy.name: {
                "bytes": hourly_manifest_copy.stat().st_size,
                "sha256": sha256(hourly_manifest_copy),
            },
            od_path.name: {"bytes": od_path.stat().st_size, "sha256": sha256(od_path)},
            "route_station_complex_mapping_preflight.csv": {
                "bytes": (out / "route_station_complex_mapping_preflight.csv").stat().st_size,
                "sha256": sha256(out / "route_station_complex_mapping_preflight.csv"),
            },
        },
        "derivation_note": "Hourly June 2026 demand is derived exclusively from the closed 43,835,841-row Drive archive after all 439 source parts pass the frozen per-part SHA-256 manifest. OD corridor queries are allowed only while the official provider metadata still matches the canonical OD snapshot rowsUpdatedAt=1787243523. One OUTSIDE_CORRIDOR residual row per origin/day/hour preserves the official all-destination denominator exactly.",
    }
    (out / "PREP_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
