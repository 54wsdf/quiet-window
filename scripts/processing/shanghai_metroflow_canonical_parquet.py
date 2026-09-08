#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

OD_MEMBER = "MetroFlow/metroData_ODFlow_10.csv"
STATION_MEMBER = "MetroFlow/stationInfo.csv"
CANONICAL_COLUMNS = (
    "date",
    "startTime",
    "originStation",
    "destinationStation",
    "Flow",
    "CFlow",
    "HBOFlow",
    "NHBFlow",
)
FLOW_COLUMNS = ("Flow", "CFlow", "HBOFlow", "NHBFlow")
EXPECTED_DATES = tuple(
    d.strftime("%Y-%m-%d")
    for d in pd.date_range("2017-05-01", "2017-08-31", freq="D")
)
EXPECTED_DATE_SET = set(EXPECTED_DATES)
EXPECTED_STATIONS = 302
EXPECTED_SLOTS = 102
START_MINUTE = 6 * 60
END_MINUTE = 23 * 60

ARROW_SCHEMA = pa.schema(
    [
        pa.field("date", pa.string(), nullable=False),
        pa.field("startTime", pa.int32(), nullable=False),
        pa.field("originStation", pa.int32(), nullable=False),
        pa.field("destinationStation", pa.int32(), nullable=False),
        pa.field("Flow", pa.int64(), nullable=False),
        pa.field("CFlow", pa.int64(), nullable=False),
        pa.field("HBOFlow", pa.int64(), nullable=False),
        pa.field("NHBFlow", pa.int64(), nullable=False),
    ]
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "quiet-window-metroflow-canonical-builder/1.0"
    for attempt in range(8):
        try:
            with session.get(url, stream=True, timeout=(30, 7200), allow_redirects=True) as response:
                response.raise_for_status()
                with dest.open("wb") as out:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            out.write(block)
            if dest.stat().st_size <= 0:
                raise RuntimeError("download produced a zero-byte archive")
            return
        except Exception:
            dest.unlink(missing_ok=True)
            if attempt == 7:
                raise
            time.sleep(min(120, 2**attempt))


def _station_ids(station_bytes: bytes) -> tuple[list[int], str, list[str]]:
    digest = hashlib.sha256(station_bytes).hexdigest()
    frame = pd.read_csv(io.BytesIO(station_bytes), low_memory=False)
    frame.columns = frame.columns.str.strip()
    candidates = [
        "stationID",
        "stationId",
        "station_id",
        "station",
        "StationID",
        "id",
    ]
    chosen = None
    for column in candidates:
        if column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().all() and numeric.nunique() == EXPECTED_STATIONS:
                chosen = column
                break
    if chosen is None:
        for column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().all() and numeric.nunique() == EXPECTED_STATIONS:
                chosen = column
                break
    if chosen is None:
        raise ValueError("unable to identify the 302-station ID authority in stationInfo.csv")
    numeric = pd.to_numeric(frame[chosen], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("station authority contains non-finite or non-integer IDs")
    ids = sorted({int(x) for x in numeric.tolist()})
    if len(ids) != EXPECTED_STATIONS:
        raise ValueError(f"station authority count mismatch: {len(ids)} != {EXPECTED_STATIONS}")
    return ids, digest, list(frame.columns)


def _normalize_dates(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    compact = s.str.fullmatch(r"\d{8}").fillna(False)
    parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if bool(compact.any()):
        parsed.loc[compact] = pd.to_datetime(s.loc[compact], format="%Y%m%d", errors="coerce")
    if bool((~compact).any()):
        parsed.loc[~compact] = pd.to_datetime(s.loc[~compact], errors="coerce")
    if parsed.isna().any():
        bad = s.loc[parsed.isna()].drop_duplicates().head(10).tolist()
        raise ValueError(f"unparseable corrected OD date values: {bad}")
    out = parsed.dt.strftime("%Y-%m-%d")
    if not out.isin(EXPECTED_DATE_SET).all():
        bad = sorted(out.loc[~out.isin(EXPECTED_DATE_SET)].unique().tolist())[:10]
        raise ValueError(f"date outside corrected 2017-05-01..2017-08-31 authority: {bad}")
    return out.astype("string")


def _normalize_start_time(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    if np.isnan(arr).any() or not np.isfinite(arr).all() or not np.equal(arr, np.floor(arr)).all():
        raise ValueError("corrected OD startTime contains non-finite or non-integer values")
    integer = numeric.astype("int64")
    s = integer.astype(str).str.zfill(6)
    hour = pd.to_numeric(s.str[:2], errors="coerce")
    minute = pd.to_numeric(s.str[2:4], errors="coerce")
    second = pd.to_numeric(s.str[4:6], errors="coerce")
    absolute = hour * 60 + minute
    delta = absolute - START_MINUTE
    slot = delta // 10
    valid = (
        s.str.fullmatch(r"\d{6}").fillna(False)
        & hour.between(0, 23)
        & minute.between(0, 59)
        & second.eq(0)
        & delta.ge(0)
        & absolute.lt(END_MINUTE)
        & delta.mod(10).eq(0)
        & slot.ge(0)
        & slot.lt(EXPECTED_SLOTS)
    )
    if not bool(valid.all()):
        bad = integer.loc[~valid].drop_duplicates().head(10).tolist()
        raise ValueError(f"invalid corrected OD 10-minute clock values: {bad}")
    if (integer < np.iinfo(np.int32).min).any() or (integer > np.iinfo(np.int32).max).any():
        raise ValueError("startTime exceeds int32 range")
    return integer.astype("int32")


def _validated_chunk(chunk: pd.DataFrame, station_set: set[int]) -> pd.DataFrame:
    work = chunk.copy()
    work.columns = work.columns.str.strip()
    missing = set(CANONICAL_COLUMNS).difference(work.columns)
    if missing:
        raise KeyError(f"corrected OD source missing columns: {sorted(missing)}")
    work = work.loc[:, list(CANONICAL_COLUMNS)]
    work["date"] = _normalize_dates(work["date"])
    work["startTime"] = _normalize_start_time(work["startTime"])

    for column in ("originStation", "destinationStation", *FLOW_COLUMNS):
        values = pd.to_numeric(work[column], errors="coerce")
        arr = values.to_numpy(dtype=float)
        if np.isnan(arr).any() or not np.isfinite(arr).all() or not np.equal(arr, np.floor(arr)).all():
            raise ValueError(f"{column} contains non-finite or non-integer values")
        if (np.abs(arr) >= 2**53).any():
            raise ValueError(f"{column} exceeds exact integer validation range")
        work[column] = values.astype("int64")

    if (work[list(FLOW_COLUMNS)] < 0).any().any():
        raise ValueError("corrected OD contains negative flow")
    if (work["Flow"] != work["CFlow"] + work["HBOFlow"] + work["NHBFlow"]).any():
        raise ValueError("corrected OD purpose decomposition violation")
    if (work["originStation"] == work["destinationStation"]).any():
        raise ValueError("corrected OD unexpectedly contains same-station diagonal rows")

    observed = set(work["originStation"].unique().tolist()) | set(work["destinationStation"].unique().tolist())
    outside = sorted(observed.difference(station_set))
    if outside:
        raise ValueError(f"corrected OD contains station IDs outside stationInfo authority: {outside[:20]}")
    if (work["originStation"] < np.iinfo(np.int32).min).any() or (work["originStation"] > np.iinfo(np.int32).max).any():
        raise ValueError("originStation exceeds int32 range")
    if (work["destinationStation"] < np.iinfo(np.int32).min).any() or (work["destinationStation"] > np.iinfo(np.int32).max).any():
        raise ValueError("destinationStation exceeds int32 range")
    work["originStation"] = work["originStation"].astype("int32")
    work["destinationStation"] = work["destinationStation"].astype("int32")
    for column in FLOW_COLUMNS:
        work[column] = work[column].astype("int64")
    return work


def _table(block: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(block, schema=ARROW_SCHEMA, preserve_index=False, safe=True)


def _close_writers(writers: dict[str, pq.ParquetWriter]) -> None:
    errors = []
    for date, writer in list(writers.items()):
        try:
            writer.close()
        except Exception as exc:  # pragma: no cover - defensive close path
            errors.append((date, repr(exc)))
    writers.clear()
    if errors:
        raise RuntimeError(f"failed closing Parquet writers: {errors[:5]}")


def build(
    source_url: str,
    output_dir: Path,
    *,
    expected_zip_sha256: str,
    expected_zip_bytes: int,
    expected_od_bytes: int,
    chunksize: int,
) -> dict:
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="metroflow-canonical-") as td_text:
        td = Path(td_text)
        zip_path = td / "MetroFlow.zip"
        print(json.dumps({"state": "DOWNLOADING_SOURCE", "url": source_url}), flush=True)
        download(source_url, zip_path)
        zip_bytes = zip_path.stat().st_size
        zip_sha = sha256_file(zip_path)
        if zip_bytes != expected_zip_bytes:
            raise ValueError(f"corrected ZIP byte mismatch: {zip_bytes} != {expected_zip_bytes}")
        if zip_sha != expected_zip_sha256:
            raise ValueError(f"corrected ZIP SHA-256 mismatch: {zip_sha} != {expected_zip_sha256}")

        with zipfile.ZipFile(zip_path) as zf:
            try:
                od_info = zf.getinfo(OD_MEMBER)
                station_info = zf.getinfo(STATION_MEMBER)
            except KeyError as exc:
                raise KeyError(f"corrected archive missing required member: {exc}") from exc
            if od_info.file_size != expected_od_bytes:
                raise ValueError(f"corrected OD member byte mismatch: {od_info.file_size} != {expected_od_bytes}")
            station_bytes = zf.read(station_info)
            station_ids, station_sha, station_columns = _station_ids(station_bytes)
            station_set = set(station_ids)

            writers: dict[str, pq.ParquetWriter] = {}
            total_rows = 0
            totals = {column: 0 for column in FLOW_COLUMNS}
            date_stats = {
                date: {"rows": 0, **{column: 0 for column in FLOW_COLUMNS}}
                for date in EXPECTED_DATES
            }
            observed_stations: set[int] = set()
            chunk_index = 0
            try:
                with zf.open(od_info) as src:
                    reader = pd.read_csv(
                        src,
                        chunksize=chunksize,
                        low_memory=False,
                        skipinitialspace=True,
                        dtype={"date": "string", "startTime": "string"},
                    )
                    for chunk in reader:
                        chunk_index += 1
                        work = _validated_chunk(chunk, station_set)
                        total_rows += int(len(work))
                        observed_stations.update(work["originStation"].unique().tolist())
                        observed_stations.update(work["destinationStation"].unique().tolist())
                        for column in FLOW_COLUMNS:
                            totals[column] += int(work[column].sum())

                        for date, positions in work.groupby("date", sort=False).indices.items():
                            date = str(date)
                            pos = np.asarray(positions, dtype=np.int64)
                            block = work.iloc[pos]
                            date_dir = output_dir / f"date={date}"
                            date_dir.mkdir(parents=True, exist_ok=True)
                            path = date_dir / "od.parquet"
                            if date not in writers:
                                writers[date] = pq.ParquetWriter(
                                    path,
                                    ARROW_SCHEMA,
                                    compression="zstd",
                                    compression_level=6,
                                    use_dictionary=["date", "startTime"],
                                    write_statistics=True,
                                )
                            writers[date].write_table(_table(block), row_group_size=250_000)
                            stats = date_stats[date]
                            stats["rows"] += int(len(block))
                            for column in FLOW_COLUMNS:
                                stats[column] += int(block[column].sum())
                        if chunk_index % 10 == 0:
                            print(
                                json.dumps(
                                    {
                                        "state": "STREAMING_OD",
                                        "chunks": chunk_index,
                                        "source_rows": total_rows,
                                        "dates_seen": sum(1 for x in date_stats.values() if x["rows"] > 0),
                                        "Flow": totals["Flow"],
                                    }
                                ),
                                flush=True,
                            )
                # Reading the ZipExtFile through EOF also validates the member CRC.
                _close_writers(writers)
            except Exception:
                for writer in list(writers.values()):
                    try:
                        writer.close()
                    except Exception:
                        pass
                writers.clear()
                raise

    if set(date for date, stats in date_stats.items() if stats["rows"] > 0) != EXPECTED_DATE_SET:
        missing = sorted(date for date, stats in date_stats.items() if stats["rows"] == 0)
        raise ValueError(f"canonical date coverage incomplete: missing={missing}")
    if totals["Flow"] != totals["CFlow"] + totals["HBOFlow"] + totals["NHBFlow"]:
        raise ValueError("global corrected OD purpose decomposition failed")
    if sum(stats["rows"] for stats in date_stats.values()) != total_rows:
        raise ValueError("date row counts do not close to source row count")
    for column in FLOW_COLUMNS:
        if sum(stats[column] for stats in date_stats.values()) != totals[column]:
            raise ValueError(f"date {column} totals do not close to source total")

    date_entries = []
    verified_rows = 0
    verified_totals = {column: 0 for column in FLOW_COLUMNS}
    for date in EXPECTED_DATES:
        path = output_dir / f"date={date}" / "od.parquet"
        parquet_file = pq.ParquetFile(path)
        rows = int(parquet_file.metadata.num_rows)
        if rows != date_stats[date]["rows"]:
            raise ValueError(f"{date}: Parquet row count mismatch {rows} != {date_stats[date]['rows']}")
        flow_table = pq.read_table(path, columns=list(FLOW_COLUMNS))
        output_sums = {
            column: int(pc.sum(flow_table[column]).as_py() or 0)
            for column in FLOW_COLUMNS
        }
        if output_sums != {column: date_stats[date][column] for column in FLOW_COLUMNS}:
            raise ValueError(f"{date}: independent Parquet flow verification failed")
        verified_rows += rows
        for column in FLOW_COLUMNS:
            verified_totals[column] += output_sums[column]
        date_entries.append(
            {
                "date": date,
                "path": f"date={date}/od.parquet",
                "rows": rows,
                **output_sums,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "row_groups": int(parquet_file.metadata.num_row_groups),
            }
        )

    if verified_rows != total_rows or verified_totals != totals:
        raise ValueError("independent canonical output verification does not close to source")
    if not observed_stations.issubset(station_set):
        raise ValueError("observed station set escaped stationInfo authority")

    date_index = output_dir / "date_index.csv"
    with date_index.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(date_entries[0].keys()))
        writer.writeheader()
        writer.writerows(date_entries)

    manifest = {
        "schema": "shanghai-metroflow-canonical-parquet-v1",
        "state": "QUALIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "url": source_url,
            "zip_bytes": expected_zip_bytes,
            "zip_sha256": expected_zip_sha256,
            "od_member": OD_MEMBER,
            "od_member_uncompressed_bytes": expected_od_bytes,
            "od_member_crc32": f"{od_info.CRC:08x}",
            "station_member": STATION_MEMBER,
            "station_member_sha256": station_sha,
            "station_info_columns": station_columns,
        },
        "physical_layout": "hive_date_partitioned_parquet_one_file_per_date",
        "canonical_columns": list(CANONICAL_COLUMNS),
        "date_start": EXPECTED_DATES[0],
        "date_end": EXPECTED_DATES[-1],
        "date_count": len(EXPECTED_DATES),
        "station_authority_count": len(station_ids),
        "observed_station_count": len(observed_stations),
        "station_ids": station_ids,
        "source_rows": total_rows,
        "flow_totals": totals,
        "purpose_decomposition_equal": totals["Flow"] == totals["CFlow"] + totals["HBOFlow"] + totals["NHBFlow"],
        "same_station_diagonal_rows": 0,
        "negative_flow_rows": 0,
        "time_authority": {
            "slots_per_day": EXPECTED_SLOTS,
            "service_window": "06:00:00-23:00:00",
            "slot_minutes": 10,
            "all_rows_valid": True,
        },
        "independent_output_verification": {
            "rows": verified_rows,
            "flow_totals": verified_totals,
            "exact_equal_to_source_accumulators": True,
        },
        "dates": date_entries,
        "contract": {
            "raw_transport_parts_are_not_scientific_units": True,
            "downstream_access": "read date=YYYY-MM-DD/od.parquet; never depend on raw part numbers",
            "raw_corrected_zip_remains_source_authority": True,
        },
        "runtime": {
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "chunksize": chunksize,
            "pandas_version": pd.__version__,
            "pyarrow_version": pa.__version__,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    qualification = {
        "state": "QUALIFIED_CANONICAL_DATASET",
        "manifest": "manifest.json",
        "manifest_sha256": sha256_file(manifest_path),
        "date_count": len(EXPECTED_DATES),
        "source_rows": total_rows,
        "Flow": totals["Flow"],
        "source_zip_sha256": expected_zip_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "QUALIFIED.json").write_text(
        json.dumps(qualification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qualification, ensure_ascii=False), flush=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a qualified date-native Parquet view of corrected Shanghai MetroFlow OD")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-zip-sha256", required=True)
    parser.add_argument("--expected-zip-bytes", type=int, required=True)
    parser.add_argument("--expected-od-bytes", type=int, required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args(argv)
    build(
        args.source_url,
        Path(args.output_dir),
        expected_zip_sha256=args.expected_zip_sha256,
        expected_zip_bytes=args.expected_zip_bytes,
        expected_od_bytes=args.expected_od_bytes,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
