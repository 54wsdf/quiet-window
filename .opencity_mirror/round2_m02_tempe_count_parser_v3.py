#!/usr/bin/env python3
"""Round-2 Tempe count parser v3 targeted validation compatibility layer.

v3 builds on the byte-qualified v2 parser and adds only the public workbook
families needed to construct a high-coverage, content-dated Tempe validation set:

D. Day-volume 15-minute sheets (``Day 1`` / ``DAY 1`` / ``DAY 2``).  These
   contain a ``Volumes for:`` date and an ``AM Period`` direction block.  Derived
   ``2-Day Average`` sheets are intentionally excluded from primary observations;
   original daily sheets are emitted separately with their own dates.
E. Single-direction 15-minute Morning/Afternoon reports used on Price Road.
F. Single-direction hourly 24-hour reports used on a small Price Road subset.

Filename year tokens and directory timestamps are never used as observation dates.
This module does not perform StreetNode extraction, map matching, GEH/NMAE, V, or
OpenScore calculation.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

V2_PATH = Path(__file__).with_name("round2_m02_tempe_count_parser_v2.py")
V2_MODULE_NAME = "_opencity_tempe_count_parser_v2"


def _load_v2():
    spec = importlib.util.spec_from_file_location(V2_MODULE_NAME, V2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v2 parser: {V2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[V2_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


v2 = _load_v2()
base = v2.base
ORIGINAL_PARSE_FILE = base.parse_file


def _exact_label_row(df: pd.DataFrame, label: str) -> int | None:
    target = base.norm(label)
    for i in range(len(df)):
        if any(base.norm(v) == target for v in df.iloc[i].tolist()):
            return i
    return None


def _first_date_in_rows(df: pd.DataFrame, rows: range) -> date | None:
    for i in rows:
        if i < 0 or i >= len(df):
            continue
        d = base.row_first_date(df, i)
        if d:
            return d
    return None


def _active_direction_columns(df: pd.DataFrame, header_row: int, time_col: int, start_col: int, end_col: int) -> dict[int, str]:
    header = df.iloc[header_row].tolist()
    candidates = {
        col: base.txt(header[col]).upper()
        for col in range(max(0, start_col), min(len(header), end_col))
        if base.txt(header[col]).upper() in base.DIRECTIONS
    }
    active: dict[int, str] = {}
    scan_end = min(len(df), header_row + 60)
    for col, direction in candidates.items():
        nonzero = 0
        numeric = 0
        for i in range(header_row + 1, scan_end):
            row = df.iloc[i].tolist()
            minute = base.parse_clock_minutes(row[time_col] if time_col < len(row) else None)
            if minute is None:
                continue
            value = base.to_float(row[col]) if col < len(row) else None
            if value is None:
                continue
            numeric += 1
            if abs(value) > 0:
                nonzero += 1
        if numeric and nonzero:
            active[col] = direction
    return active


def _detect_family_d(sheets: dict[str, pd.DataFrame]) -> bool:
    for sheet_name, df in sheets.items():
        if "average" in sheet_name.lower():
            continue
        if _exact_label_row(df, "Volumes for") is None:
            continue
        header_row = _exact_label_row(df, "AM Period")
        if header_row is None:
            continue
        row = [base.txt(v).upper() for v in df.iloc[header_row].tolist()]
        if any(v in base.DIRECTIONS for v in row):
            return True
    return False


def parse_family_d(path: Path, sheets: dict[str, pd.DataFrame]) -> list[Any]:
    records = []
    parsed_daily_sheets = 0
    for sheet_name, df in sheets.items():
        if "average" in sheet_name.lower():
            continue
        date_row = _exact_label_row(df, "Volumes for")
        header_row = _exact_label_row(df, "AM Period")
        if date_row is None or header_row is None:
            continue
        obs_date = _first_date_in_rows(df, range(date_row, min(len(df), date_row + 3)))
        if not obs_date:
            raise ValueError(f"{sheet_name}: Volumes for row has no parseable observation date")
        header = df.iloc[header_row].tolist()
        pm_cols = [j for j, v in enumerate(header) if base.norm(v) == "pm period"]
        am_end_col = pm_cols[0] if pm_cols else min(len(header), 10)
        active = _active_direction_columns(df, header_row, time_col=0, start_col=1, end_col=am_end_col)
        if not active:
            raise ValueError(f"{sheet_name}: no active AM direction columns")
        location = base.metadata_value(df, ["Location", "Location "])
        parsed_daily_sheets += 1
        for i in range(header_row + 1, len(df)):
            row = df.iloc[i].tolist()
            minute = base.parse_clock_minutes(row[0] if row else None)
            if minute is None:
                continue
            if minute >= 12 * 60:
                continue
            for col, direction in active.items():
                value = base.to_float(row[col]) if col < len(row) else None
                if value is None:
                    continue
                if value < 0:
                    raise ValueError("negative observed volume")
                records.append(base.IntervalRecord(
                    source_file=str(path), schema_family="D_DAY_VOLUME_15MIN",
                    source_sheet=sheet_name, observation_date=obs_date.isoformat(), route="",
                    location=location, direction=direction, minute_of_day=minute,
                    clock=base.clock_str(minute), interval_minutes=15, volume=value))
    if not parsed_daily_sheets:
        raise ValueError("day-volume workbook has no original daily sheet")
    return records


def _find_single_direction_header(df: pd.DataFrame) -> tuple[int, str] | None:
    for i in range(min(len(df) - 1, 25)):
        row = df.iloc[i].tolist()
        dirs = [base.txt(v).upper() for v in row if base.txt(v).upper() in base.DIRECTIONS]
        if len(set(dirs)) != 1:
            continue
        if not base.row_first_date(df, i):
            continue
        next_norm = [base.norm(v) for v in df.iloc[i + 1].tolist()]
        if "morning" in next_norm and "afternoon" in next_norm:
            return i, dirs[0]
    return None


def _detect_family_e(sheets: dict[str, pd.DataFrame]) -> bool:
    if len(sheets) != 1:
        return False
    return _find_single_direction_header(next(iter(sheets.values()))) is not None


def parse_family_e(path: Path, sheets: dict[str, pd.DataFrame]) -> list[Any]:
    sheet_name, df = next(iter(sheets.items()))
    found = _find_single_direction_header(df)
    if found is None:
        raise ValueError("single-direction Morning/Afternoon header not found")
    header_row, direction = found
    obs_date = base.row_first_date(df, header_row)
    if not obs_date:
        raise ValueError("single-direction report has no observation date")
    sub = [base.norm(v) for v in df.iloc[header_row + 1].tolist()]
    morning_cols = [i for i, v in enumerate(sub) if v == "morning"]
    afternoon_cols = [i for i, v in enumerate(sub) if v == "afternoon"]
    if not morning_cols or not afternoon_cols:
        raise ValueError("single-direction report missing Morning/Afternoon columns")
    mcol, pcol = morning_cols[0], afternoon_cols[0]
    if pcol != mcol + 1:
        raise ValueError("single-direction Morning/Afternoon observation columns are not adjacent")
    location = ""
    for i in range(max(0, header_row - 8), header_row):
        row = df.iloc[i].tolist()
        for value in row:
            s = base.txt(value)
            if direction in s.upper() and "PRICE" in s.upper():
                location = s
                break
        if location:
            break
    records = []
    for i in range(header_row + 2, len(df)):
        row = df.iloc[i].tolist()
        raw_minute = base.parse_clock_minutes(row[0] if row else None)
        if raw_minute is None:
            continue
        for col, half in ((mcol, "MORNING"), (pcol, "AFTERNOON")):
            value = base.to_float(row[col]) if col < len(row) else None
            if value is None:
                continue
            if value < 0:
                raise ValueError("negative observed volume")
            minute = base.halfday_to_minutes(raw_minute, half)
            records.append(base.IntervalRecord(
                source_file=str(path), schema_family="E_SINGLE_DIRECTION_15MIN",
                source_sheet=sheet_name, observation_date=obs_date.isoformat(), route="",
                location=location, direction=direction, minute_of_day=minute,
                clock=base.clock_str(minute), interval_minutes=15, volume=value))
    return records


def _parse_ampm_clock(value: Any, half: str | None) -> tuple[int | None, str | None]:
    s = base.txt(value).upper()
    m = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)$", s)
    if m:
        hh, mm, marker = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (1 <= hh <= 12 and 0 <= mm <= 59):
            return None, half
        h24 = (0 if hh == 12 else hh) + (12 if marker == "PM" else 0)
        return h24 * 60 + mm, marker
    raw = base.parse_clock_minutes(value)
    if raw is None or half not in {"AM", "PM"}:
        return None, half
    hh, mm = raw // 60, raw % 60
    if hh > 11:
        return None, half
    h24 = hh + (12 if half == "PM" else 0)
    return h24 * 60 + mm, half


def _find_hourly_single_direction_header(df: pd.DataFrame) -> tuple[int, str] | None:
    for i in range(min(len(df), 25)):
        row = df.iloc[i].tolist()
        dirs = [base.txt(v).upper() for v in row if base.txt(v).upper() in base.DIRECTIONS]
        if len(set(dirs)) == 1 and i > 0 and base.row_first_date(df, i - 1):
            return i, dirs[0]
    return None


def _detect_family_f(sheets: dict[str, pd.DataFrame]) -> bool:
    if len(sheets) != 1:
        return False
    df = next(iter(sheets.values()))
    found = _find_hourly_single_direction_header(df)
    if found is None:
        return False
    header_row, _ = found
    for i in range(header_row + 1, min(len(df), header_row + 30)):
        s = base.txt(df.iloc[i, 0] if len(df.columns) else None).upper()
        if s in {"12:00 AM", "12:00 PM"}:
            return True
    return False


def parse_family_f(path: Path, sheets: dict[str, pd.DataFrame]) -> list[Any]:
    sheet_name, df = next(iter(sheets.items()))
    found = _find_hourly_single_direction_header(df)
    if found is None:
        raise ValueError("single-direction hourly header not found")
    header_row, direction = found
    obs_date = base.row_first_date(df, header_row - 1)
    if not obs_date:
        raise ValueError("single-direction hourly report has no observation date")
    direction_cols = [j for j, v in enumerate(df.iloc[header_row].tolist()) if base.txt(v).upper() == direction]
    if len(direction_cols) != 1:
        raise ValueError(f"single-direction hourly report has ambiguous direction columns: {direction_cols}")
    value_col = direction_cols[0]
    location = ""
    for i in range(max(0, header_row - 8), header_row):
        for value in df.iloc[i].tolist():
            s = base.txt(value)
            if direction in s.upper() and "PRICE" in s.upper():
                location = s
                break
        if location:
            break
    records = []
    half: str | None = None
    seen_minutes: set[int] = set()
    for i in range(header_row + 1, len(df)):
        row = df.iloc[i].tolist()
        minute, half = _parse_ampm_clock(row[0] if row else None, half)
        if minute is None:
            continue
        value = base.to_float(row[value_col]) if value_col < len(row) else None
        if value is None:
            continue
        if value < 0:
            raise ValueError("negative observed volume")
        if minute in seen_minutes:
            raise ValueError(f"duplicate hourly minute {base.clock_str(minute)}")
        seen_minutes.add(minute)
        records.append(base.IntervalRecord(
            source_file=str(path), schema_family="F_SINGLE_DIRECTION_HOURLY_24H",
            source_sheet=sheet_name, observation_date=obs_date.isoformat(), route="",
            location=location, direction=direction, minute_of_day=minute,
            clock=base.clock_str(minute), interval_minutes=60, volume=value))
    return records


def _audit_from_records(path: Path, family: str, records: list[Any]):
    if not records:
        raise ValueError("schema detected but no interval records parsed")
    dates = sorted({r.observation_date for r in records})
    dirs = sorted({r.direction for r in records})
    resolutions = sorted({r.interval_minutes for r in records})
    keys = [(r.observation_date, r.direction, r.minute_of_day) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate observation_date/direction/interval keys")
    return base.FileAudit(str(path), path.suffix.lower(), "PARSED", family, len(records),
                          "|".join(dates), "|".join(dirs), "|".join(map(str, resolutions)), "")


def parse_file(path: Path):
    records, audit = ORIGINAL_PARSE_FILE(path)
    if audit.status != "UNSUPPORTED_SCHEMA":
        return records, audit
    try:
        sheets = base.load_sheets(path)
        if _detect_family_d(sheets):
            family = "D_DAY_VOLUME_15MIN"
            records = parse_family_d(path, sheets)
        elif _detect_family_e(sheets):
            family = "E_SINGLE_DIRECTION_15MIN"
            records = parse_family_e(path, sheets)
        elif _detect_family_f(sheets):
            family = "F_SINGLE_DIRECTION_HOURLY_24H"
            records = parse_family_f(path, sheets)
        else:
            return records, audit
        return records, _audit_from_records(path, family, records)
    except Exception as e:
        return [], base.FileAudit(str(path), path.suffix.lower(), "ERROR", "", 0, "", "", "", f"{type(e).__name__}: {e}")


base.parse_file = parse_file


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
