#!/usr/bin/env python3
"""Parse City of Tempe historical traffic-count workbooks into auditable intervals.

This is a source parser only. It does not map observations to OpenCity model links
and it does not calculate V/OpenScore. Raw source bytes stay outside knowledge-hub.

Supported schema families are based on independent 2026-09-08 probes:
  A. legacy single-sheet hourly TRAFFIC COUNT SUMMARY (e.g. 2008)
  B. one direction per sheet, 15-minute AM/PM columns (e.g. 2010/2016)
  C. combined-direction 15-minute Morning/Afternoon report (e.g. 2019)

The current Tempe reference assigns AM = 06:00-09:00 and converts period demand to
an average hourly vehicle rate by dividing by 3 hours. Therefore AM observed vph is
computed as vehicles observed in [06:00,09:00) divided by 3, only when all expected
intervals are present.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DIRECTIONS = {"NB", "SB", "EB", "WB"}
SUPPORTED_EXT = {".xls", ".xlsx"}
AM_START_MIN = 6 * 60
AM_END_MIN = 9 * 60
AM_HOURS = 3.0


@dataclass
class IntervalRecord:
    source_file: str
    schema_family: str
    source_sheet: str
    observation_date: str
    route: str
    location: str
    direction: str
    minute_of_day: int
    clock: str
    interval_minutes: int
    volume: float


@dataclass
class FileAudit:
    source_file: str
    extension: str
    status: str
    schema_family: str
    record_count: int
    observation_dates: str
    directions: str
    resolution_minutes: str
    error: str


def txt(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", txt(value)).strip().lower().rstrip(":")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            if math.isnan(float(value)):
                return None
        except Exception:
            pass
        return float(value)
    s = txt(value).replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_date_value(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = txt(value)
    if not s:
        return None
    # Prefer explicit date substrings over free-form parser guesses.
    candidates = [s]
    m = re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", s)
    if m:
        candidates.insert(0, m.group(0))
    m = re.search(r"\b\d{1,2}-[A-Za-z]{3}-\d{2,4}\b", s)
    if m:
        candidates.insert(0, m.group(0))
    for candidate in candidates:
        try:
            parsed = pd.to_datetime(candidate, errors="raise")
            return parsed.date()
        except Exception:
            continue
    return None


def parse_clock_minutes(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.hour * 60 + value.minute
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute
    if isinstance(value, time):
        return value.hour * 60 + value.minute
    # Excel time can arrive as fraction of a day.
    if isinstance(value, (int, float)) and 0 <= float(value) < 1:
        return int(round(float(value) * 24 * 60)) % (24 * 60)
    s = txt(value)
    m = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", s)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if hh > 23 or mm > 59:
        return None
    return hh * 60 + mm


def clock_str(minute_of_day: int) -> str:
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"


def halfday_to_minutes(raw_minutes: int, halfday: str) -> int:
    hh = raw_minutes // 60
    mm = raw_minutes % 60
    h = halfday.upper()
    if h in {"AM", "MORNING"}:
        hh = 0 if hh == 12 else hh
    elif h in {"PM", "AFTERNOON"}:
        hh = 12 if hh == 12 else hh + 12
    else:
        raise ValueError(f"unknown half-day label: {halfday}")
    return hh * 60 + mm


def load_sheets(path: Path) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path, engine="calamine")
    return {name: pd.read_excel(path, sheet_name=name, header=None, engine="calamine") for name in xl.sheet_names}


def find_label_row(df: pd.DataFrame, label: str) -> int | None:
    target = norm(label)
    for i in range(len(df)):
        for value in df.iloc[i].tolist():
            if norm(value) == target:
                return i
    return None


def row_first_date(df: pd.DataFrame, row_index: int) -> date | None:
    for value in df.iloc[row_index].tolist():
        d = parse_date_value(value)
        if d:
            return d
    return None


def metadata_value(df: pd.DataFrame, labels: Iterable[str]) -> str:
    wanted = {norm(x) for x in labels}
    for i in range(len(df)):
        row = df.iloc[i].tolist()
        for j, value in enumerate(row[:-1]):
            if norm(value) in wanted:
                for candidate in row[j + 1 :]:
                    s = txt(candidate)
                    if s:
                        return s
    return ""


def detect_family(sheets: dict[str, pd.DataFrame]) -> str:
    for df in sheets.values():
        for i in range(min(len(df), 20)):
            if any("traffic count summary" in norm(v) for v in df.iloc[i].tolist()):
                return "A_LEGACY_HOURLY_SUMMARY"
    direction_sheets = [name.upper().strip() for name in sheets]
    if direction_sheets and all(name in DIRECTIONS for name in direction_sheets):
        if all(find_label_row(df, "Count Date") is not None for df in sheets.values()):
            return "B_DIRECTION_SHEET_15MIN_AMPM"
    for df in sheets.values():
        for i in range(min(len(df) - 1, 20)):
            dirs = [txt(v).upper() for v in df.iloc[i].tolist() if txt(v).upper() in DIRECTIONS]
            next_text = [norm(v) for v in df.iloc[i + 1].tolist()]
            if len(set(dirs)) >= 2 and "morning" in next_text and "afternoon" in next_text:
                return "C_COMBINED_15MIN_MORNING_AFTERNOON"
    return "UNSUPPORTED"


def parse_family_a(path: Path, sheets: dict[str, pd.DataFrame]) -> list[IntervalRecord]:
    sheet_name, df = next(iter(sheets.items()))
    start_row = find_label_row(df, "Start Date")
    direction_row = find_label_row(df, "Direction")
    if start_row is None or direction_row is None:
        raise ValueError("legacy summary missing Start Date/Direction rows")
    obs_date = row_first_date(df, start_row)
    if not obs_date:
        raise ValueError("legacy summary has no parseable Start Date")
    route = metadata_value(df, ["Location 1"])
    location = metadata_value(df, ["Location 2"])
    header = df.iloc[direction_row].tolist()
    direction_cols = {j: txt(v).upper() for j, v in enumerate(header) if txt(v).upper() in DIRECTIONS}
    if not direction_cols:
        raise ValueError("legacy summary has no EB/WB/NB/SB direction columns")
    records: list[IntervalRecord] = []
    for i in range(direction_row + 1, len(df)):
        row = df.iloc[i].tolist()
        first = norm(row[0] if row else None)
        if first in {"total", "totals"}:
            break
        minute = parse_clock_minutes(row[0] if row else None)
        if minute is None:
            continue
        # Family A values are hourly totals, not 15-minute counts.
        for col, direction in direction_cols.items():
            if col >= len(row):
                continue
            volume = to_float(row[col])
            if volume is None:
                continue
            if volume < 0:
                raise ValueError("negative observed volume")
            records.append(IntervalRecord(
                source_file=str(path), schema_family="A_LEGACY_HOURLY_SUMMARY",
                source_sheet=sheet_name, observation_date=obs_date.isoformat(), route=route,
                location=location, direction=direction, minute_of_day=minute,
                clock=clock_str(minute), interval_minutes=60, volume=volume))
    return records


def active_halfday_columns(df: pd.DataFrame, header_row: int) -> dict[int, str]:
    header = df.iloc[header_row].tolist()
    candidates: dict[int, str] = {}
    end = min(len(df), header_row + 55)
    for col, value in enumerate(header):
        h = txt(value).upper()
        if h not in {"AM", "PM"}:
            continue
        numeric_nonzero = 0
        for i in range(header_row + 1, end):
            row = df.iloc[i].tolist()
            minute = parse_clock_minutes(row[0] if row else None)
            if minute is None:
                continue
            v = to_float(row[col]) if col < len(row) else None
            if v is not None and abs(v) > 0:
                numeric_nonzero += 1
        if numeric_nonzero:
            candidates[col] = h
    return candidates


def parse_family_b(path: Path, sheets: dict[str, pd.DataFrame]) -> list[IntervalRecord]:
    records: list[IntervalRecord] = []
    for sheet_name, df in sheets.items():
        direction = sheet_name.upper().strip()
        if direction not in DIRECTIONS:
            raise ValueError(f"unexpected direction sheet: {sheet_name}")
        date_row = find_label_row(df, "Count Date")
        header_row = find_label_row(df, "Count Time")
        if date_row is None or header_row is None:
            raise ValueError(f"{sheet_name}: missing Count Date/Count Time")
        obs_date = row_first_date(df, date_row)
        if not obs_date:
            raise ValueError(f"{sheet_name}: no parseable Count Date")
        active = active_halfday_columns(df, header_row)
        by_half = Counter(active.values())
        if by_half.get("AM", 0) != 1 or by_half.get("PM", 0) != 1:
            raise ValueError(f"{sheet_name}: ambiguous active AM/PM columns: {active}")
        route = metadata_value(df, ["Route"])
        location = metadata_value(df, ["Location"])
        for i in range(header_row + 1, len(df)):
            row = df.iloc[i].tolist()
            if norm(row[0] if row else None) in {"totals", "total", "day total"}:
                break
            raw_minute = parse_clock_minutes(row[0] if row else None)
            if raw_minute is None:
                continue
            for col, half in active.items():
                if col >= len(row):
                    continue
                volume = to_float(row[col])
                if volume is None:
                    continue
                if volume < 0:
                    raise ValueError("negative observed volume")
                minute = halfday_to_minutes(raw_minute, half)
                records.append(IntervalRecord(
                    source_file=str(path), schema_family="B_DIRECTION_SHEET_15MIN_AMPM",
                    source_sheet=sheet_name, observation_date=obs_date.isoformat(), route=route,
                    location=location, direction=direction, minute_of_day=minute,
                    clock=clock_str(minute), interval_minutes=15, volume=volume))
    return records


def find_combined_header(df: pd.DataFrame) -> int | None:
    for i in range(min(len(df) - 1, 25)):
        row = [txt(v).upper() for v in df.iloc[i].tolist()]
        next_row = [txt(v).upper() for v in df.iloc[i + 1].tolist()]
        dirs = [v for v in row if v in DIRECTIONS]
        if len(set(dirs)) >= 2 and "MORNING" in next_row and "AFTERNOON" in next_row:
            return i
    return None


def parse_family_c(path: Path, sheets: dict[str, pd.DataFrame]) -> list[IntervalRecord]:
    if len(sheets) != 1:
        raise ValueError("combined report expected one sheet")
    sheet_name, df = next(iter(sheets.items()))
    header_row = find_combined_header(df)
    if header_row is None:
        raise ValueError("combined report direction/Morning/Afternoon header not found")
    obs_date = None
    for i in range(max(0, header_row - 8), min(len(df), header_row + 2)):
        obs_date = row_first_date(df, i)
        if obs_date:
            break
    if not obs_date:
        raise ValueError("combined report has no parseable observation date")
    header = [txt(v).upper() for v in df.iloc[header_row].tolist()]
    sub = [txt(v).upper() for v in df.iloc[header_row + 1].tolist()]
    direction_cols: dict[int, tuple[str, str]] = {}
    for col, value in enumerate(header):
        if value not in DIRECTIONS:
            continue
        if col >= len(sub) or sub[col] != "MORNING":
            raise ValueError(f"{value}: Morning column not aligned")
        if col + 1 >= len(sub) or sub[col + 1] != "AFTERNOON":
            raise ValueError(f"{value}: Afternoon column not adjacent")
        direction_cols[col] = (value, "MORNING")
        direction_cols[col + 1] = (value, "AFTERNOON")
    if len({d for d, _ in direction_cols.values()}) < 2:
        raise ValueError("combined report has fewer than two directions")
    records: list[IntervalRecord] = []
    route = ""
    location = ""
    for i in range(header_row + 2, len(df)):
        row = df.iloc[i].tolist()
        first = norm(row[0] if row else None)
        if first in {"total", "totals", "grand total"}:
            break
        raw_minute = parse_clock_minutes(row[0] if row else None)
        if raw_minute is None:
            continue
        for col, (direction, half) in direction_cols.items():
            if col >= len(row):
                continue
            volume = to_float(row[col])
            if volume is None:
                continue
            if volume < 0:
                raise ValueError("negative observed volume")
            minute = halfday_to_minutes(raw_minute, half)
            records.append(IntervalRecord(
                source_file=str(path), schema_family="C_COMBINED_15MIN_MORNING_AFTERNOON",
                source_sheet=sheet_name, observation_date=obs_date.isoformat(), route=route,
                location=location, direction=direction, minute_of_day=minute,
                clock=clock_str(minute), interval_minutes=15, volume=volume))
    return records


def parse_file(path: Path) -> tuple[list[IntervalRecord], FileAudit]:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        return [], FileAudit(str(path), ext, "SKIPPED", "UNSUPPORTED_EXTENSION", 0, "", "", "", "")
    try:
        sheets = load_sheets(path)
        family = detect_family(sheets)
        if family == "A_LEGACY_HOURLY_SUMMARY":
            records = parse_family_a(path, sheets)
        elif family == "B_DIRECTION_SHEET_15MIN_AMPM":
            records = parse_family_b(path, sheets)
        elif family == "C_COMBINED_15MIN_MORNING_AFTERNOON":
            records = parse_family_c(path, sheets)
        else:
            return [], FileAudit(str(path), ext, "UNSUPPORTED_SCHEMA", family, 0, "", "", "", "")
        if not records:
            raise ValueError("schema detected but no interval records parsed")
        dates = sorted({r.observation_date for r in records})
        dirs = sorted({r.direction for r in records})
        resolutions = sorted({r.interval_minutes for r in records})
        # Duplicate interval keys inside one source file make scoring ambiguous.
        keys = [(r.observation_date, r.direction, r.minute_of_day) for r in records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate observation_date/direction/interval keys")
        audit = FileAudit(str(path), ext, "PARSED", family, len(records), "|".join(dates),
                          "|".join(dirs), "|".join(map(str, resolutions)), "")
        return records, audit
    except Exception as e:
        return [], FileAudit(str(path), ext, "ERROR", "", 0, "", "", "", f"{type(e).__name__}: {e}")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def derive_am(records: list[IntervalRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[IntervalRecord]] = {}
    for r in records:
        if AM_START_MIN <= r.minute_of_day < AM_END_MIN:
            key = (r.source_file, r.schema_family, r.observation_date, r.direction)
            grouped.setdefault(key, []).append(r)
    out = []
    for (source_file, family, obs_date, direction), items in sorted(grouped.items()):
        resolutions = {r.interval_minutes for r in items}
        resolution = next(iter(resolutions)) if len(resolutions) == 1 else None
        expected = int(AM_HOURS * 60 / resolution) if resolution else None
        unique_minutes = {r.minute_of_day for r in items}
        complete = bool(resolution and len(items) == expected and len(unique_minutes) == expected)
        total = sum(r.volume for r in items)
        out.append({
            "source_file": source_file,
            "schema_family": family,
            "observation_date": obs_date,
            "direction": direction,
            "resolution_minutes": resolution or "",
            "am_start": "06:00",
            "am_end": "09:00",
            "observed_intervals": len(items),
            "expected_intervals": expected or "",
            "complete_am": complete,
            "am_total_vehicles": round(total, 6),
            "am_average_vph": round(total / AM_HOURS, 6) if complete else "",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--interval-output", required=True)
    ap.add_argument("--am-output", required=True)
    ap.add_argument("--audit-output", required=True)
    ap.add_argument("--summary-output", required=True)
    args = ap.parse_args()

    root = Path(args.input_root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".xls", ".xlsx", ".pdf"})
    all_records: list[IntervalRecord] = []
    audits: list[FileAudit] = []
    for path in files:
        records, audit = parse_file(path)
        all_records.extend(records)
        audits.append(audit)

    interval_rows = [asdict(r) for r in all_records]
    write_csv(Path(args.interval_output), interval_rows, [f.name for f in IntervalRecord.__dataclass_fields__.values()])
    am_rows = derive_am(all_records)
    write_csv(Path(args.am_output), am_rows, [
        "source_file", "schema_family", "observation_date", "direction", "resolution_minutes",
        "am_start", "am_end", "observed_intervals", "expected_intervals", "complete_am",
        "am_total_vehicles", "am_average_vph"])
    audit_rows = [asdict(a) for a in audits]
    write_csv(Path(args.audit_output), audit_rows, [f.name for f in FileAudit.__dataclass_fields__.values()])

    status_counts = Counter(a.status for a in audits)
    family_counts = Counter(a.schema_family for a in audits if a.schema_family)
    summary = {
        "input_root": str(root),
        "source_files_seen": len(files),
        "audit_status_counts": dict(status_counts),
        "schema_family_counts": dict(family_counts),
        "interval_records": len(all_records),
        "am_direction_records": len(am_rows),
        "complete_am_direction_records": sum(1 for r in am_rows if r["complete_am"]),
        "am_definition": "06:00-09:00; sum observed vehicles then divide by 3 hours for average vph",
        "scoring_boundary": "Parser output is not model-link agreement. Map matching and held-out validation selection remain separate gates."
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
