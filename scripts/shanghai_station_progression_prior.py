#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {
    "01": {"physical_adjacent_segments": 27, "covered_segments": 27, "accepted_observations": 183},
    "02": {"physical_adjacent_segments": 29, "covered_segments": 29, "accepted_observations": 370},
    "03": {"physical_adjacent_segments": 28, "covered_segments": 28, "accepted_observations": 695},
    "04": {"physical_adjacent_segments": 25, "covered_segments": 0, "accepted_observations": 0},
    "05": {"physical_adjacent_segments": 10, "covered_segments": 9, "accepted_observations": 30},
    "06": {"physical_adjacent_segments": 27, "covered_segments": 27, "accepted_observations": 398},
    "07": {"physical_adjacent_segments": 32, "covered_segments": 31, "accepted_observations": 115},
    "08": {"physical_adjacent_segments": 29, "covered_segments": 29, "accepted_observations": 592},
    "09": {"physical_adjacent_segments": 25, "covered_segments": 25, "accepted_observations": 134},
    "10": {"physical_adjacent_segments": 30, "covered_segments": 27, "accepted_observations": 107},
    "11": {"physical_adjacent_segments": 30, "covered_segments": 26, "accepted_observations": 384},
    "12": {"physical_adjacent_segments": 15, "covered_segments": 15, "accepted_observations": 56},
    "13": {"physical_adjacent_segments": 9, "covered_segments": 5, "accepted_observations": 14},
    "16": {"physical_adjacent_segments": 12, "covered_segments": 7, "accepted_observations": 54},
}


def suffix4(value: str) -> str | None:
    value = (value or "").strip().strip('"')
    if not value or value == "—":
        return None
    return value[-4:].zfill(4)


def numeric(value: str) -> int | None:
    value = (value or "").strip().strip('"')
    if not value or value == "—":
        return None
    return int(float(value))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build(station_path: Path, timetable_path: Path, connections_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    with station_path.open(encoding="utf-8-sig", newline="") as f:
        station_rows = list(csv.DictReader(f))
    by_code = {
        r["ST_NO"].zfill(4): {"line": str(r["LINE_NO"]).zfill(2), "name": r["ST_NAME"].strip()}
        for r in station_rows
    }
    by_line_name = {(v["line"], v["name"]): code for code, v in by_code.items()}

    connections = json.loads(connections_path.read_text(encoding="utf-8"))
    adjacency = []
    physical_counts = Counter()
    for a, b, line in connections:
        line = str(line).zfill(2)
        a = a.strip()
        b = b.strip()
        ca = by_line_name.get((line, a))
        cb = by_line_name.get((line, b))
        if ca is None or cb is None:
            raise ValueError(f"adjacency station missing from station.txt: {(a, b, line, ca, cb)}")
        adjacency.append((line, a, b, ca, cb))
        physical_counts[line] += 1

    rows_by_drawing_dest = defaultdict(lambda: defaultdict(list))
    with timetable_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sc = suffix4(row["ST_NO"])
            dc = suffix4(row["DESTINATION"])
            if sc is None or dc is None:
                continue
            rows_by_drawing_dest[(row["DRAWING_NO"], dc)][sc].append(row)

    accepted = []
    rejects = Counter()
    covered = defaultdict(set)

    for (drawing, destination), station_map in rows_by_drawing_dest.items():
        destination_name = by_code.get(destination, {}).get("name")
        for line, a, b, ca, cb in adjacency:
            rows_a = station_map.get(ca, [])
            rows_b = station_map.get(cb, [])
            if len(rows_a) != 1 or len(rows_b) != 1:
                continue

            ra = rows_a[0]
            rb = rows_b[0]
            af = numeric(ra["FIRST_T_TIME"])
            al = numeric(ra["LAST_T_TIME"])
            bf = numeric(rb["FIRST_T_TIME"])
            bl = numeric(rb["LAST_T_TIME"])
            if None in (af, al, bf, bl):
                rejects["missing_numeric_first_or_last"] += 1
                continue

            first_delta = bf - af
            last_delta = bl - al
            if first_delta == 0 or last_delta == 0 or (first_delta > 0) != (last_delta > 0):
                rejects["direction_disagreement"] += 1
                continue

            if first_delta > 0:
                from_station, to_station = a, b
                first_prog, last_prog = first_delta, last_delta
            else:
                from_station, to_station = b, a
                first_prog, last_prog = -first_delta, -last_delta

            if not (1 <= first_prog <= 600 and 1 <= last_prog <= 600):
                rejects["outside_progression_bound"] += 1
                continue
            if abs(first_prog - last_prog) > 30:
                rejects["first_last_progression_disagreement"] += 1
                continue

            accepted.append(
                {
                    "line": line,
                    "from_station": from_station,
                    "to_station": to_station,
                    "drawing_no": drawing,
                    "destination_code": destination,
                    "destination_station": destination_name,
                    "first_progression_s": first_prog,
                    "last_progression_s": last_prog,
                    "consistency_delta_s": abs(first_prog - last_prog),
                    "progression_lower_s": min(first_prog, last_prog),
                    "progression_upper_s": max(first_prog, last_prog),
                }
            )
            covered[line].add(tuple(sorted((a, b))))

    line_summary = {}
    for line in sorted(physical_counts):
        line_summary[line] = {
            "physical_adjacent_segments": physical_counts[line],
            "covered_segments": len(covered[line]),
            "accepted_observations": sum(1 for x in accepted if x["line"] == line),
        }

    by_directed_segment = defaultdict(list)
    for obs in accepted:
        by_directed_segment[(obs["line"], obs["from_station"], obs["to_station"])].append(obs)

    directed = []
    for (line, from_station, to_station), observations in sorted(by_directed_segment.items()):
        centers = [(x["first_progression_s"] + x["last_progression_s"]) / 2 for x in observations]
        directed.append(
            {
                "line": line,
                "from_station": from_station,
                "to_station": to_station,
                "observations": len(observations),
                "lower_s": min(x["progression_lower_s"] for x in observations),
                "upper_s": max(x["progression_upper_s"] for x in observations),
                "median_center_s": statistics.median(centers),
                "min_center_s": min(centers),
                "max_center_s": max(centers),
            }
        )

    matches = {line: line_summary.get(line) == expected for line, expected in EXPECTED.items()}
    missing = sorted(set(EXPECTED) - set(line_summary))
    extra = sorted(set(line_summary) - set(EXPECTED))

    qualification = {
        "schema": "mppd.shanghai.soda2015.station-progression-prior.materialized.v1",
        "upstream_repo": "jeevanyue/metro",
        "upstream_commit": "bb78b72ecadc57e0c8deedf9731a78a65ab62ef4",
        "inputs": {
            "station.txt": {"sha256": sha256(station_path)},
            "timetable.txt": {"sha256": sha256(timetable_path)},
            "connections_by_station_name.json": {"sha256": sha256(connections_path)},
        },
        "acceptance": {
            "same_drawing_and_destination": True,
            "adjacent_revenue_stations": True,
            "normalized_station_row_must_be_unique": True,
            "both_first_and_last_numeric": True,
            "max_abs_first_last_progression_difference_s": 30,
            "accepted_progression_range_s": [1, 600],
        },
        "line_summary": line_summary,
        "frozen_expected_line_summary": EXPECTED,
        "frozen_line_summary_exact_match": matches,
        "missing_expected_lines": missing,
        "extra_lines": extra,
        "accepted_observation_count": len(accepted),
        "directed_segment_count": len(directed),
        "reject_counts": dict(sorted(rejects.items())),
        "claim_boundary": [
            "Scheduled station-progression prior only; not observed ATS.",
            "No 2015 departure clock is forward-filled into 2016.",
            "Segment lower/upper values are extrema of accepted historical scheduled progression observations, not independently calibrated physical runtime bounds.",
        ],
    }

    (outdir / "qualification.json").write_text(
        json.dumps(qualification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "directed_segment_prior.json").write_text(
        json.dumps(directed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "accepted_observations.json").write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(qualification, ensure_ascii=False, indent=2))
    if missing or extra or not all(matches.values()):
        raise SystemExit("frozen station-progression audit was not reproduced exactly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", type=Path, required=True)
    ap.add_argument("--timetable", type=Path, required=True)
    ap.add_argument("--connections", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    build(args.station, args.timetable, args.connections, args.outdir)
