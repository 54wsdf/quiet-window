#!/usr/bin/env python3
"""Round-2 Tempe count parser v2 compatibility layer.

This module keeps the qualified v1 parser byte-for-byte intact and replaces only
Family-B AM/PM column selection. Some Tempe 2010/2016 workbooks repeat the same
AM/PM observations in trailing ``Average`` columns. Multiple active columns are
accepted only when their timed numeric series are exactly identical; in that case
the leftmost source column is selected deterministically. Any disagreement remains
a hard error.

The wrapper exists so the schema fix can be independently qualified before the
small change is folded back into the canonical standalone parser.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

BASE_PATH = Path(__file__).with_name("round2_m02_tempe_count_parser.py")
BASE_MODULE_NAME = "_opencity_tempe_count_parser_v1"


def _load_base():
    spec = importlib.util.spec_from_file_location(BASE_MODULE_NAME, BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base parser: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[BASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()


def _timed_numeric_series(df: pd.DataFrame, header_row: int, col: int) -> tuple[tuple[int, float | None], ...]:
    """Return the timed numeric series used to decide whether columns are true duplicates."""
    end = min(len(df), header_row + 55)
    out: list[tuple[int, float | None]] = []
    for i in range(header_row + 1, end):
        row = df.iloc[i].tolist()
        minute = base.parse_clock_minutes(row[0] if row else None)
        if minute is None:
            continue
        value = base.to_float(row[col]) if col < len(row) else None
        out.append((minute, value))
    return tuple(out)


def active_halfday_columns(df: pd.DataFrame, header_row: int) -> dict[int, str]:
    """Select one AM and one PM column, collapsing only exact mirrored duplicates."""
    header = df.iloc[header_row].tolist()
    candidates: dict[str, list[int]] = {"AM": [], "PM": []}
    series: dict[int, tuple[tuple[int, float | None], ...]] = {}

    for col, raw_header in enumerate(header):
        half = base.txt(raw_header).upper()
        if half not in candidates:
            continue
        values = _timed_numeric_series(df, header_row, col)
        if not any(v is not None and abs(v) > 0 for _, v in values):
            continue
        candidates[half].append(col)
        series[col] = values

    selected: dict[int, str] = {}
    for half in ("AM", "PM"):
        cols = candidates[half]
        if not cols:
            continue
        primary = cols[0]
        for duplicate in cols[1:]:
            if series[duplicate] != series[primary]:
                raise ValueError(
                    f"conflicting active {half} columns: {cols}; "
                    "duplicate collapse is allowed only for exactly identical timed series"
                )
        selected[primary] = half
    return selected


base.active_halfday_columns = active_halfday_columns


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
