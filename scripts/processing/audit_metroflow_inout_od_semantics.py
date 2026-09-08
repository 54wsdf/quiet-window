#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
import zipfile
from datetime import date as Date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

INOUT_MEMBER = "MetroFlow/metroData_InOutFlow.csv"
EXPECTED_DATES = tuple((Date(2017, 5, 1) + timedelta(days=i)).isoformat() for i in range(123))
EXPECTED_STATIONS = 302
SLOTS_PER_DAY = 102
START_MINUTE = 6 * 60
SOURCE_ZIP_SHA256 = "427e7c8e3021d4007114db876db9fb6d1b35674b29ad950ebc64cf1a838cab88"
SOURCE_ZIP_BYTES = 1211783206
REQUIRED = (
    "date", "timeslot", "startTime", "endTime", "station", "inFlow", "outFlow",
    "CinFlow", "HBOinFlow", "NHBinFlow", "CoutFlow", "HBOoutFlow", "NHBoutFlow",
)
LOW_FLOW_REFERENCE = {
    "2017-05-04", "2017-05-08", "2017-05-09", "2017-06-16", "2017-06-27", "2017-06-28"
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "quiet-window-metroflow-data-audit/1.0"
    for attempt in range(8):
        try:
            with session.get(url, stream=True, timeout=(30, 7200), allow_redirects=True) as response:
                response.raise_for_status()
                with dest.open("wb") as out:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            out.write(block)
            require(dest.stat().st_size > 0, "zero-byte source archive")
            return
        except Exception:
            dest.unlink(missing_ok=True)
            if attempt == 7:
                raise
            time.sleep(min(120, 2 ** attempt))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_clock(local_slot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start_min = START_MINUTE + local_slot * 10
    end_min = start_min + 10
    start = (start_min // 60) * 10000 + (start_min % 60) * 100
    end = (end_min // 60) * 10000 + (end_min % 60) * 100
    return start.astype(np.int64), end.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-url", required=True)
    ap.add_argument("--derived-manifest", required=True)
    ap.add_argument("--date-quality", required=True)
    ap.add_argument("--station-index", required=True)
    ap.add_argument("--marginals-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    derived = load_json(Path(args.derived_manifest))
    quality = load_json(Path(args.date_quality))
    require(derived.get("state") == "QUALIFIED_CANONICAL_Q6_Q10", "derived views not qualified")
    require(tuple(derived.get("dates", [])) == EXPECTED_DATES, "derived date authority mismatch")
    require(derived.get("source_zip_sha256") == SOURCE_ZIP_SHA256, "derived source ZIP mismatch")
    require(len(quality.get("dates", [])) == 123, "date quality coverage mismatch")
    quality_by_date = {row["date"]: row for row in quality["dates"]}

    station_index = pd.read_csv(args.station_index)
    require(list(station_index.columns) == ["station_index", "station_id"], "station index schema mismatch")
    require(len(station_index) == EXPECTED_STATIONS, "station count mismatch")
    require(np.array_equal(station_index["station_index"].to_numpy(), np.arange(EXPECTED_STATIONS)), "station index is not contiguous")
    station_ids = station_index["station_id"].astype(np.int64).to_numpy()
    require(len(set(station_ids.tolist())) == EXPECTED_STATIONS, "station IDs not unique")
    station_to_idx = {int(s): i for i, s in enumerate(station_ids.tolist())}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="metroflow-inout-audit-") as td_text:
        td = Path(td_text)
        zip_path = td / "MetroFlow.zip"
        print(json.dumps({"state": "DOWNLOADING_SOURCE"}), flush=True)
        download(args.source_url, zip_path)
        require(zip_path.stat().st_size == SOURCE_ZIP_BYTES, "source ZIP byte mismatch")
        require(sha256_file(zip_path) == SOURCE_ZIP_SHA256, "source ZIP SHA-256 mismatch")
        with zipfile.ZipFile(zip_path) as zf:
            info = zf.getinfo(INOUT_MEMBER)
            with zf.open(info) as src:
                x = pd.read_csv(src, low_memory=False, skipinitialspace=True)

    x.columns = x.columns.str.strip()
    missing = [c for c in REQUIRED if c not in x.columns]
    require(not missing, f"missing InOut columns: {missing}")
    x = x.loc[:, list(REQUIRED)].copy()
    for c in REQUIRED:
        x[c] = pd.to_numeric(x[c], errors="raise")
    require(not x[list(REQUIRED)].isna().any().any(), "InOut contains null values")
    for c in REQUIRED:
        arr = x[c].to_numpy(dtype=float)
        require(np.isfinite(arr).all() and np.equal(arr, np.floor(arr)).all(), f"noninteger/nonfinite InOut column: {c}")
        x[c] = x[c].astype(np.int64)

    expected_rows = len(EXPECTED_DATES) * EXPECTED_STATIONS * SLOTS_PER_DAY
    require(len(x) == expected_rows, f"InOut row count mismatch {len(x)} != {expected_rows}")
    require(x["station"].nunique() == EXPECTED_STATIONS, "InOut station count mismatch")
    require(set(x["station"].astype(int).unique()) == set(map(int, station_ids)), "InOut station authority mismatch")
    duplicate_keys = int(x.duplicated(["date", "timeslot", "station"]).sum())
    require(duplicate_keys == 0, "InOut duplicate date/timeslot/station keys")

    flow_cols = ["inFlow", "outFlow", "CinFlow", "HBOinFlow", "NHBinFlow", "CoutFlow", "HBOoutFlow", "NHBoutFlow"]
    negative_rows = int((x[flow_cols] < 0).any(axis=1).sum())
    in_decomp_fail = int((x["inFlow"] != x["CinFlow"] + x["HBOinFlow"] + x["NHBinFlow"]).sum())
    out_decomp_fail = int((x["outFlow"] != x["CoutFlow"] + x["HBOoutFlow"] + x["NHBoutFlow"]).sum())
    require(negative_rows == in_decomp_fail == out_decomp_fail == 0, "InOut flow quality failure")

    timeslot = x["timeslot"].to_numpy(dtype=np.int64)
    day_idx = timeslot // SLOTS_PER_DAY
    local_slot = timeslot % SLOTS_PER_DAY
    require(day_idx.min() == 0 and day_idx.max() == len(EXPECTED_DATES) - 1, "InOut timeslot day range mismatch")
    expected_dates_int = np.asarray([int(d.replace("-", "")) for d in EXPECTED_DATES], dtype=np.int64)[day_idx]
    date_timeslot_failures = int(np.count_nonzero(x["date"].to_numpy(dtype=np.int64) != expected_dates_int))
    exp_start, exp_end = expected_clock(local_slot)
    start_failures = int(np.count_nonzero(x["startTime"].to_numpy(dtype=np.int64) != exp_start))
    end_failures = int(np.count_nonzero(x["endTime"].to_numpy(dtype=np.int64) != exp_end))
    require(date_timeslot_failures == start_failures == end_failures == 0, "InOut date/time authority failure")

    x["iso_date"] = pd.to_datetime(x["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    x["local_slot"] = local_slot
    x["station_idx"] = x["station"].map(station_to_idx).astype(np.int64)
    rows_per_day = x.groupby("iso_date", sort=False).size().reindex(EXPECTED_DATES)
    grid_fail_dates = int((rows_per_day != EXPECTED_STATIONS * SLOTS_PER_DAY).sum())
    require(grid_fail_dates == 0, "InOut day grid incomplete")

    daily_io = x.groupby("iso_date", sort=False)[["inFlow", "outFlow"]].sum().reindex(EXPECTED_DATES)
    daily_balance_fail_dates = int((daily_io["inFlow"] != daily_io["outFlow"]).sum())
    require(daily_balance_fail_dates == 0, "daily InOut in/out totals do not balance")

    derived_daily = {row["date"]: row for row in derived["daily"]}
    daily_records: list[dict] = []
    all_origin_gap_nonnegative = True
    all_daily_station_gap_equal = True
    max_daily_station_gap_abs = 0
    same_slot_destination_abs_error = 0
    same_slot_destination_den = 0

    for date in EXPECTED_DATES:
        compact = date.replace("-", "")
        day = x.loc[x["iso_date"] == date]
        in_mat = np.zeros((SLOTS_PER_DAY, EXPECTED_STATIONS), dtype=np.int64)
        out_mat = np.zeros_like(in_mat)
        rr = day["local_slot"].to_numpy(dtype=np.int64)
        cc = day["station_idx"].to_numpy(dtype=np.int64)
        in_mat[rr, cc] = day["inFlow"].to_numpy(dtype=np.int64)
        out_mat[rr, cc] = day["outFlow"].to_numpy(dtype=np.int64)

        marginal_path = Path(args.marginals_dir) / f"entry_clock-{compact}.npz"
        require(marginal_path.is_file(), f"missing marginal file: {date}")
        with np.load(marginal_path, allow_pickle=False) as z:
            origin = z["origin"].astype(np.int64, copy=False)
            destination = z["destination"].astype(np.int64, copy=False)
        require(origin.shape == destination.shape == (SLOTS_PER_DAY, EXPECTED_STATIONS), f"marginal shape mismatch: {date}")
        od_total = int(origin.sum(dtype=np.int64))
        require(od_total == int(destination.sum(dtype=np.int64)), f"origin/destination OD total mismatch: {date}")
        require(od_total == int(derived_daily[date]["Flow"]), f"derived daily Flow mismatch: {date}")

        origin_gap = in_mat - origin
        negative_gap_cells = int(np.count_nonzero(origin_gap < 0))
        all_origin_gap_nonnegative &= negative_gap_cells == 0
        station_origin_gap = origin_gap.sum(axis=0, dtype=np.int64)
        station_destination_gap = out_mat.sum(axis=0, dtype=np.int64) - destination.sum(axis=0, dtype=np.int64)
        station_gap_delta = station_origin_gap - station_destination_gap
        gap_mismatch_stations = int(np.count_nonzero(station_gap_delta))
        max_gap = int(np.max(np.abs(station_gap_delta), initial=0))
        max_daily_station_gap_abs = max(max_daily_station_gap_abs, max_gap)
        all_daily_station_gap_equal &= gap_mismatch_stations == 0
        require(int(station_origin_gap.sum(dtype=np.int64)) == int(in_mat.sum(dtype=np.int64)) - od_total, f"origin deficit total mismatch: {date}")
        require(int(station_destination_gap.sum(dtype=np.int64)) == int(out_mat.sum(dtype=np.int64)) - od_total, f"destination deficit total mismatch: {date}")

        slot_dest_abs = int(np.abs(out_mat - destination).sum(dtype=np.int64))
        same_slot_destination_abs_error += slot_dest_abs
        same_slot_destination_den += int(out_mat.sum(dtype=np.int64))
        q = quality_by_date[date]
        daily_records.append({
            "date": date,
            "weekday": pd.Timestamp(date).day_name(),
            "inFlow": int(in_mat.sum(dtype=np.int64)),
            "outFlow": int(out_mat.sum(dtype=np.int64)),
            "odFlow": od_total,
            "implied_omitted_diagonal": int(station_origin_gap.sum(dtype=np.int64)),
            "origin_gap_negative_cells": negative_gap_cells,
            "daily_station_origin_destination_gap_mismatch_stations": gap_mismatch_stations,
            "max_daily_station_gap_abs": max_gap,
            "same_slot_destination_vs_exit_abs_error": slot_dest_abs,
            "derived_rows": int(q["rows"]),
            "derived_observed_slots": len(q["observed_slots"]),
            "derived_unobserved_slots": q["unobserved_slots"],
        })

    require(all_origin_gap_nonnegative, "derived origin marginal exceeds InOut entry count in at least one cell")
    require(all_daily_station_gap_equal, "daily station origin/destination diagonal deficits do not match")

    daily_df = pd.DataFrame(daily_records)
    daily_df["date_dt"] = pd.to_datetime(daily_df["date"])
    # Weekday medians are robust because only six source-level collapses are present.
    med = daily_df.groupby("weekday")["inFlow"].median().to_dict()
    daily_df["inflow_weekday_median"] = daily_df["weekday"].map(med).astype(float)
    daily_df["inflow_ratio_to_weekday_median"] = daily_df["inFlow"] / daily_df["inflow_weekday_median"]
    flagged = set(daily_df.loc[daily_df["inflow_ratio_to_weekday_median"] < 0.05, "date"])
    low_df = daily_df.loc[daily_df["date"].isin(sorted(flagged))].copy().sort_values("inFlow")

    audit = {
        "schema": "shanghai-metroflow-inout-od-semantics-audit-v1",
        "state": "QUALIFIED_DATA_SEMANTICS_AUDIT",
        "source_zip_sha256": SOURCE_ZIP_SHA256,
        "derived_manifest_sha256": hashlib.sha256(Path(args.derived_manifest).read_bytes()).hexdigest(),
        "inout": {
            "rows": int(len(x)), "dates": int(x["iso_date"].nunique()), "stations": int(x["station"].nunique()),
            "duplicate_keys": duplicate_keys, "negative_rows": negative_rows,
            "in_decomposition_failures": in_decomp_fail, "out_decomposition_failures": out_decomp_fail,
            "date_timeslot_failures": date_timeslot_failures, "start_time_failures": start_failures,
            "end_time_failures": end_failures, "grid_fail_dates": grid_fail_dates,
            "daily_total_balance_fail_dates": daily_balance_fail_dates,
        },
        "od_inout_semantics": {
            "origin_clock": "origin_entry_time",
            "inout_inflow_clock": "entry_gate_time",
            "inout_outflow_clock": "exit_gate_time",
            "same_slot_origin_le_inflow_all_cells": bool(all_origin_gap_nonnegative),
            "daily_station_origin_gap_equals_destination_gap_all_dates": bool(all_daily_station_gap_equal),
            "max_daily_station_gap_abs": int(max_daily_station_gap_abs),
            "interpretation": "The nonnegative origin-side deficit is the implied same-station-ID diagonal omitted by corrected OD. Destination marginal is indexed by origin entry time, so same-slot destination vs exit-gate equality is not a qualification condition.",
            "same_slot_destination_vs_exit_wape_diagnostic": float(same_slot_destination_abs_error / same_slot_destination_den) if same_slot_destination_den else None,
        },
        "low_flow": {
            "threshold_ratio_to_weekday_median": 0.05,
            "flagged_dates": sorted(flagged),
            "matches_previously_flagged_six": flagged == LOW_FLOW_REFERENCE,
            "reference_six": sorted(LOW_FLOW_REFERENCE),
            "policy": "retain source rows unchanged; quarantine these dates from default estimation/training and report inclusion as a sensitivity until an external source explains them",
            "source_level_evidence": "InOut remains a complete 302x102 grid on every date, while counts collapse on the flagged dates; the OD-derived views inherit the same collapse rather than creating it.",
        },
        "derived": {
            "date_count": len(EXPECTED_DATES),
            "flow_total": int(daily_df["odFlow"].sum()),
            "manifest_flow_total": int(derived["flow_totals"]["Flow"]),
            "all_dates_checked_against_marginals": True,
        },
        "runtime": {"github_run_id": os.environ.get("GITHUB_RUN_ID"), "github_sha": os.environ.get("GITHUB_SHA")},
    }
    require(audit["derived"]["flow_total"] == audit["derived"]["manifest_flow_total"], "full-period derived Flow mismatch")

    daily_df.drop(columns=["date_dt"]).to_csv(out_dir / "daily_quality.csv", index=False)
    low_df.drop(columns=["date_dt"]).to_csv(out_dir / "low_flow_days.csv", index=False)
    (out_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"state": audit["state"], "flagged_dates": sorted(flagged), "same_slot_destination_vs_exit_wape_diagnostic": audit["od_inout_semantics"]["same_slot_destination_vs_exit_wape_diagnostic"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
