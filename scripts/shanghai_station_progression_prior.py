#!/usr/bin/env python3
"""Build a high-confidence station-progression prior from the pinned SODA snapshot.

FIRST_T_TIME and LAST_T_TIME are source-documented first/last service clock
seconds since midnight. This script never calls them observed runtimes. It
retains only adjacent revenue-station pairs within one DRAWING_NO+DESTINATION
whose first- and last-service progression increments agree within a tolerance.

The result is a scheduled station-progression prior for service-aware screening,
not a 2016 timetable and not ATS truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_code(x: str, by_code: dict[str, dict]):
    digits = "".join(c for c in (x or "") if c.isdigit())
    if len(digits) >= 4 and digits[-4:] in by_code:
        return digits[-4:]
    return None


def parse_clock(x: str):
    x = (x or "").strip()
    return int(x) if x.isdigit() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-first-last-disagreement-s", type=int, default=30)
    ap.add_argument("--max-segment-progression-s", type=int, default=600)
    args = ap.parse_args()

    base = args.snapshot_root / "train" / "base"
    stations = read_csv(base / "station.txt")
    timetable = read_csv(base / "timetable.txt")
    by_code = {r["ST_NO"].zfill(4): r for r in stations}

    line_stations = defaultdict(list)
    for code, r in by_code.items():
        line = str(r["LINE_NO"]).zfill(2)
        line_stations[line].append((int(r["SERIAL_NO"]), code))

    adjacent = {}
    for line, rows in line_stations.items():
        rows = sorted(rows)
        adjacent[line] = [
            (a[1], b[1])
            for a, b in zip(rows, rows[1:])
            if b[0] - a[0] == 1
        ]

    # Preserve all source variants before normalisation. If collapsing an
    # operational-row prefix produces conflicting clocks, that station/profile
    # observation is excluded rather than resolved heuristically.
    grouped = defaultdict(lambda: defaultdict(set))
    for r in timetable:
        st = norm_code(r["ST_NO"], by_code)
        dst = norm_code(r["DESTINATION"], by_code)
        if not st or not dst:
            continue
        if str(by_code[st]["LINE_NO"]).zfill(2) != str(by_code[dst]["LINE_NO"]).zfill(2):
            continue
        grouped[(r["DRAWING_NO"], dst)][st].add(
            (parse_clock(r["FIRST_T_TIME"]), parse_clock(r["LAST_T_TIME"]))
        )

    accepted = defaultdict(list)
    rejects = Counter()
    for (drawing, dst), station_rows in grouped.items():
        unique = {s: next(iter(v)) for s, v in station_rows.items() if len(v) == 1}
        line = str(by_code[dst]["LINE_NO"]).zfill(2)
        for a, b in adjacent.get(line, []):
            if a not in unique or b not in unique:
                continue
            f1, l1 = unique[a]
            f2, l2 = unique[b]
            if None in (f1, l1, f2, l2):
                rejects["missing_numeric_first_or_last"] += 1
                continue
            first_delta = abs(f2 - f1)
            last_delta = abs(l2 - l1)
            if abs(first_delta - last_delta) > args.max_first_last_disagreement_s:
                rejects["first_last_progression_disagreement"] += 1
                continue
            value = (first_delta + last_delta) / 2
            if not (0 < value <= args.max_segment_progression_s):
                rejects["outside_progression_bound"] += 1
                continue
            accepted[(line, a, b)].append(value)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {}
    for line in sorted(adjacent):
        covered = 0
        observations = 0
        for a, b in adjacent[line]:
            vals = accepted.get((line, a, b), [])
            if not vals:
                continue
            covered += 1
            observations += len(vals)
            ordered = sorted(vals)
            q10 = ordered[int((len(ordered) - 1) * 0.10)]
            q90 = ordered[int((len(ordered) - 1) * 0.90)]
            rows.append(
                {
                    "line": line,
                    "a_code": a,
                    "a_name": by_code[a]["ST_NAME"],
                    "b_code": b,
                    "b_name": by_code[b]["ST_NAME"],
                    "accepted_observations": len(vals),
                    "median_progression_s": statistics.median(vals),
                    "p10_progression_s": q10,
                    "p90_progression_s": q90,
                }
            )
        total = len(adjacent[line])
        summary[line] = {
            "physical_adjacent_segments": total,
            "covered_segments": covered,
            "coverage_share": covered / total if total else None,
            "accepted_observations": observations,
        }

    csv_path = args.out_dir / "segment_station_progression_prior.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    result = {
        "schema": "mppd.shanghai.soda2015.station-progression-prior.audit.v1",
        "source": {
            "station": "train/base/station.txt",
            "timetable": "train/base/timetable.txt",
            "field_semantics": "FIRST_T_TIME/LAST_T_TIME are source-documented first/last service times in seconds since 00:00:00",
        },
        "acceptance": {
            "same_drawing_and_destination": True,
            "adjacent_revenue_stations": True,
            "normalised_station_row_must_be_unique": True,
            "both_first_and_last_numeric": True,
            "max_first_last_progression_disagreement_s": args.max_first_last_disagreement_s,
            "max_segment_progression_s": args.max_segment_progression_s,
        },
        "line_summary": summary,
        "rejects": dict(rejects),
        "boundaries": [
            "Scheduled station-progression prior only; not observed ATS.",
            "Do not forward-fill 2015 departure clocks into 2016.",
            "L4 ring closure/direction requires separate treatment.",
            "L13 and L16 have partial coverage and require additional service-specific evidence.",
            "Progression increments can include source timetable conventions beyond pure interstation running time.",
        ],
    }
    (args.out_dir / "qualification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
