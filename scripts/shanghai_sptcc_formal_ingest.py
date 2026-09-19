#!/usr/bin/env python3
"""Privacy-preserving Shanghai SPTCC formal journey worker.

This worker implements the qualified MPPD ingestion semantics for one raw day:
- stable hashed card identity and deterministic train/validation/test split;
- all timestamped card transactions preserved in chronology;
- non-metro/rejected events act as barriers;
- duplicate timestamps act as barriers;
- metro entry is fare==0, exit is fare>0, negative/unparsed fares are barriers;
- one gate segment is capped at 6 hours;
- strict virtual-transfer stitching is date-aware and requires source-event
  adjacency, so no intervening card event can be crossed.

Raw card IDs never leave runner-local processing.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

FIELDS = ["card_id", "date", "time", "station", "vehicle", "money", "property"]
ALIASES = {
    "上海大学站": "上海大学",
    "淞浜路": "淞滨路",
    "李子园路": "李子园",
    "外高桥保税区北": "外高桥保税区北站",
    "外高桥保税区南": "外高桥保税区南站",
    "上海野生动物园": "野生动物园",
}
RULES = [
    {
        "policy_id": "SH_RAILWAY_STATION_L1_L34_VIRTUAL",
        "station": "上海火车站",
        "pairs": [("01", "03"), ("01", "04")],
        "valid_from": "2008-06-01",
        "valid_to": None,
        "max_gap_s": 1800,
    },
    {
        "policy_id": "HONGQIAO_T2_L2_L10_VIRTUAL",
        "station": "虹桥2号航站楼",
        "pairs": [("02", "10")],
        "valid_from": "2010-11-30",
        "valid_to": "2017-12-29",
        "max_gap_s": 1800,
    },
    {
        "policy_id": "NANJING_WEST_L2_L12_L13_VIRTUAL",
        "station": "南京西路",
        "pairs": [("02", "12"), ("02", "13"), ("12", "13")],
        "valid_from": "2015-12-19",
        "valid_to": None,
        "max_gap_s": 1800,
    },
    {
        "policy_id": "LONGHUA_L11_L12_VIRTUAL",
        "station": "龙华",
        "pairs": [("11", "12")],
        "valid_from": "2015-12-19",
        "valid_to": "2018-12-29",
        "max_gap_s": 1800,
    },
]


def open_text(path: Path):
    raw = gzip.open if path.suffix.lower() == ".gz" else open
    last = None
    for enc in ("utf-8-sig", "gb18030", "gbk"):
        try:
            f = raw(path, "rt", encoding=enc, newline="")
            f.read(8192)
            f.seek(0)
            return f
        except UnicodeDecodeError as e:
            last = e
            try:
                f.close()
            except Exception:
                pass
    raise last


def iter_rows(path: Path):
    f = open_text(path)
    try:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return
        headers = {
            "卡ID", "卡号", "刷卡日期", "交易日期", "刷卡时间",
            "刷卡站点", "刷卡乘车类型", "card_id",
        }
        if not headers.intersection(x.strip() for x in first):
            yield dict(zip(FIELDS, first))
        for row in reader:
            if row:
                yield dict(zip(FIELDS, row))
    finally:
        f.close()


def clean(x):
    return (x or "").replace("\ufeff", "").strip()


def parse_station_token(raw):
    token = clean(raw)
    m = re.match(r"^(\d+)号线(.+)$", token)
    if not m:
        return None
    line = str(int(m.group(1))).zfill(2)
    station = clean(m.group(2))
    station = ALIASES.get(station, station)
    return line, station, f"{line}::{station}"


def card_token(card, salt):
    return hashlib.sha256((salt + str(card)).encode()).hexdigest()[:24]


def split_name(token):
    x = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 10
    return "validation" if x == 0 else ("test" if x in (1, 2) else "train")


def date_ok(d, rule):
    return d >= rule["valid_from"] and (
        rule["valid_to"] is None or d <= rule["valid_to"]
    )


def virtual_rule(exit_line, exit_station, entry_line, entry_station, gap, date_text):
    if exit_station != entry_station or gap <= 0:
        return None
    pair = frozenset((exit_line, entry_line))
    for rule in RULES:
        if exit_station != rule["station"]:
            continue
        if gap > rule["max_gap_s"] or not date_ok(date_text, rule):
            continue
        if any(pair == frozenset(x) for x in rule["pairs"]):
            return rule
    return None


def ingest_file(
    input_path: Path,
    db_path: Path,
    output_path: Path,
    audit_path: Path,
    source_date: str,
    expected_strict: int | None,
    salt: str = "SHANGHAI_MPPD_SPLIT_V1",
    max_gate_duration_s: int = 21600,
):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript(
        "PRAGMA journal_mode=WAL;"
        "PRAGMA synchronous=NORMAL;"
        "PRAGMA temp_store=FILE;"
        "DROP TABLE IF EXISTS events;"
        "CREATE TABLE events("
        "card TEXT, ts INTEGER, datestr TEXT, line TEXT, station TEXT, "
        "money REAL, raw_row INTEGER, node TEXT, barrier TEXT"
        ");"
    )
    audit = Counter()
    rejects = Counter()
    batch = []
    blocked_cards = set()
    epoch = datetime(1970, 1, 1)

    def flush(force=False):
        if batch and (force or len(batch) >= 50000):
            cur.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)", batch)
            con.commit()
            batch.clear()

    for raw_row, r in enumerate(iter_rows(input_path), start=1):
        audit["input_rows"] += 1
        raw_card = clean(r.get("card_id"))
        token = card_token(raw_card, salt) if raw_card else None
        vehicle = clean(r.get("vehicle"))
        try:
            datestr = clean(r.get("date"))
            dt = datetime.strptime(datestr + " " + clean(r.get("time")), "%Y-%m-%d %H:%M:%S")
        except Exception:
            rejects["datetime"] += 1
            if token:
                blocked_cards.add(token)
            continue
        if datestr != source_date:
            rejects["unexpected_date"] += 1
            if token:
                blocked_cards.add(token)
            continue
        ts = int((dt - epoch).total_seconds())
        if not token:
            rejects["missing_card_id"] += 1
            continue

        if vehicle != "地铁":
            audit["non_metro_rows"] += 1
            audit["non_metro_barrier_events"] += 1
            audit["timestamped_card_events"] += 1
            batch.append((token, ts, datestr, "", "", None, raw_row, "", "NON_METRO"))
            flush()
            continue

        audit["metro_source_rows"] += 1
        try:
            money = float(r.get("money"))
        except Exception:
            rejects["money"] += 1
            audit["money_barrier_events"] += 1
            audit["metro_events"] += 1
            audit["timestamped_card_events"] += 1
            batch.append((token, ts, datestr, "", "", None, raw_row, "", "MONEY_PARSE"))
            flush()
            continue

        parsed = parse_station_token(r.get("station"))
        if parsed is None:
            rejects["station_parse"] += 1
            audit["station_barrier_events"] += 1
            audit["metro_events"] += 1
            audit["timestamped_card_events"] += 1
            batch.append((token, ts, datestr, "", "", None, raw_row, "", "STATION_PARSE"))
            flush()
            continue

        line, station, node = parsed
        if money < 0:
            rejects["negative_money"] += 1
            audit["negative_money_barrier_events"] += 1
            audit["metro_events"] += 1
            audit["timestamped_card_events"] += 1
            batch.append((token, ts, datestr, line, station, None, raw_row, node, "NEGATIVE_MONEY"))
            flush()
            continue

        audit["metro_events"] += 1
        audit["timestamped_card_events"] += 1
        batch.append((token, ts, datestr, line, station, money, raw_row, node, ""))
        flush()

    flush(force=True)
    cur.execute("CREATE INDEX idx_events_card_ts_row ON events(card,ts,raw_row)")
    con.commit()

    q = cur.execute(
        "SELECT card,ts,datestr,line,station,money,raw_row,node,barrier "
        "FROM events ORDER BY card,ts,raw_row"
    )

    def groups():
        current = None
        group = []
        for row in q:
            if current is not None and row[0] != current:
                yield current, group
                group = []
            current = row[0]
            group.append(row)
        if current is not None:
            yield current, group

    opener = gzip.open if output_path.suffix.lower() == ".gz" else open
    fields = [
        "card_token","split","entry_time","exit_time",
        "origin_line","origin_station","origin_node",
        "destination_line","destination_station","destination_node",
        "duration_s","virtual_transfer_count","virtual_transfer_stations",
        "virtual_transfer_policy_ids",
    ]
    with opener(output_path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for card, events in groups():
            audit["unique_staged_cards"] += 1
            if any(e[3] for e in events):
                audit["cards_with_metro_event"] += 1
            if card in blocked_cards:
                audit["cards_blocked_by_unordered_reject"] += 1
                audit["events_on_blocked_cards"] += len(events)
                continue

            ts_counts = Counter(e[1] for e in events)
            segments = []
            open_entry = None
            open_pos = None

            for pos, e in enumerate(events):
                _card, ts, datestr, line, station, money, raw_row, node, barrier = e
                if ts_counts[ts] > 1:
                    audit["duplicate_timestamp_events"] += 1
                    open_entry = None
                    open_pos = None
                    continue
                if barrier or money is None:
                    audit["barrier_events"] += 1
                    if barrier:
                        audit["barrier_" + barrier.lower()] += 1
                    open_entry = None
                    open_pos = None
                    continue
                if money == 0:
                    if open_entry is not None:
                        audit["entry_before_exit"] += 1
                    open_entry = e
                    open_pos = pos
                elif money > 0:
                    if open_entry is None:
                        audit["exit_without_entry"] += 1
                        continue
                    if ts <= open_entry[1]:
                        audit["nonpositive_duration"] += 1
                        open_entry = None
                        open_pos = None
                        continue
                    if ts - open_entry[1] > max_gate_duration_s:
                        audit["gate_segment_over_max_duration"] += 1
                        open_entry = None
                        open_pos = None
                        continue
                    segments.append((open_entry, e, open_pos, pos))
                    open_entry = None
                    open_pos = None

            if open_entry is not None:
                audit["open_entry_at_end"] += 1
            audit["gate_segments"] += len(segments)

            i = 0
            while i < len(segments):
                a, b, a_pos, b_pos = segments[i]
                transfers = []
                policies = []
                j = i + 1
                while j < len(segments):
                    na, nb, na_pos, nb_pos = segments[j]
                    gap = na[1] - b[1]
                    rule = virtual_rule(b[3], b[4], na[3], na[4], gap, na[2])
                    if rule is None:
                        break
                    audit["virtual_transfer_candidates"] += 1
                    audit["vt_candidate_policy_" + rule["policy_id"]] += 1
                    if na_pos != b_pos + 1:
                        audit["virtual_transfer_candidate_blocked_by_intervening_event"] += 1
                        break
                    transfers.append(rule["station"])
                    policies.append(rule["policy_id"])
                    audit["virtual_transfer_stitches"] += 1
                    audit["virtual_transfer_source_event_adjacency_verified"] += 1
                    audit["vt_" + rule["station"]] += 1
                    audit["vt_policy_" + rule["policy_id"]] += 1
                    b = nb
                    b_pos = nb_pos
                    j += 1

                duration = b[1] - a[1]
                rec = {
                    "card_token": card,
                    "split": split_name(card),
                    "entry_time": (epoch + timedelta(seconds=a[1])).isoformat(sep=" "),
                    "exit_time": (epoch + timedelta(seconds=b[1])).isoformat(sep=" "),
                    "origin_line": a[3],
                    "origin_station": a[4],
                    "origin_node": a[7],
                    "destination_line": b[3],
                    "destination_station": b[4],
                    "destination_node": b[7],
                    "duration_s": duration,
                    "virtual_transfer_count": len(transfers),
                    "virtual_transfer_stations": "|".join(transfers),
                    "virtual_transfer_policy_ids": "|".join(policies),
                }
                w.writerow(rec)
                audit["journeys"] += 1
                audit["split_" + rec["split"]] += 1
                if transfers:
                    audit["journeys_with_virtual_transfer"] += 1
                if duration < 120:
                    audit["duration_under_120s"] += 1
                if duration > 14400:
                    audit["duration_over_4h"] += 1
                i = j

    audit_out = dict(sorted(audit.items()))
    audit_out["rejects"] = dict(sorted(rejects.items()))
    audit_out["blocked_card_hashes_count"] = len(blocked_cards)
    audit_out["source_date"] = source_date
    audit_out["expected_strict_journeys"] = expected_strict
    if expected_strict is not None:
        audit_out["gate_segment_delta_vs_strict"] = audit_out.get("gate_segments", 0) - expected_strict
        audit_out["gate_segments_equal_strict"] = audit_out.get("gate_segments", 0) == expected_strict
    audit_out["journey_mass_invariant"] = (
        audit_out.get("journeys", 0)
        == audit_out.get("gate_segments", 0) - audit_out.get("virtual_transfer_stitches", 0)
    )
    audit_out["split_mass_invariant"] = (
        audit_out.get("journeys", 0)
        == audit_out.get("split_train", 0)
        + audit_out.get("split_validation", 0)
        + audit_out.get("split_test", 0)
    )
    audit_out["worker_schema"] = "mppd.shanghai.formal-card-chronology-worker.v1"
    audit_out["source_authority_commit"] = "d0455d314aa55a44cdb05fc3e382f26062919f51"
    audit_out["privacy"] = "Raw card IDs are hashed before persistent runner staging; output contains only 96-bit stable card tokens."
    audit_out["qualification_state"] = (
        "FORMAL_CANDIDATE_BUILT_RECONCILIATION_REQUIRED"
        if audit_out["journey_mass_invariant"] and audit_out["split_mass_invariant"]
        else "FAILED_INTERNAL_MASS_INVARIANT"
    )
    audit_path.write_text(json.dumps(audit_out, ensure_ascii=False, indent=2), encoding="utf-8")
    con.close()
    return audit_out


def _count_rows(path):
    op = gzip.open if path.suffix.lower() == ".gz" else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def self_test():
    header = "卡ID,刷卡日期,刷卡时间,刷卡站点,刷卡乘车类型,刷卡扣钱,是否优惠\n"
    cases = [
        (
            "normal_virtual",
            [
                "a,2016-09-01,07:50:00,2号线陆家嘴,地铁,0.00,非优惠",
                "a,2016-09-01,08:10:00,2号线南京西路,地铁,4.00,非优惠",
                "a,2016-09-01,08:15:00,12号线南京西路,地铁,0.00,非优惠",
                "a,2016-09-01,08:40:00,12号线七莘路,地铁,5.00,非优惠",
            ],
            1,
            {"virtual_transfer_stitches": 1},
        ),
        (
            "nonmetro_pair_barrier",
            [
                "b,2016-09-01,08:00:00,2号线陆家嘴,地铁,0.00,非优惠",
                "b,2016-09-01,08:10:00,公交测试站,公交,2.00,非优惠",
                "b,2016-09-01,08:20:00,2号线南京西路,地铁,4.00,非优惠",
            ],
            0,
            {"non_metro_barrier_events": 1},
        ),
        (
            "nonmetro_stitch_barrier",
            [
                "c,2016-09-01,07:50:00,2号线陆家嘴,地铁,0.00,非优惠",
                "c,2016-09-01,08:10:00,2号线南京西路,地铁,4.00,非优惠",
                "c,2016-09-01,08:12:00,公交测试站,公交,2.00,非优惠",
                "c,2016-09-01,08:15:00,12号线南京西路,地铁,0.00,非优惠",
                "c,2016-09-01,08:40:00,12号线七莘路,地铁,5.00,非优惠",
            ],
            2,
            {"virtual_transfer_candidate_blocked_by_intervening_event": 1},
        ),
        (
            "over_6h",
            [
                "d,2016-09-01,01:00:00,2号线陆家嘴,地铁,0.00,非优惠",
                "d,2016-09-01,08:00:01,2号线南京西路,地铁,4.00,非优惠",
            ],
            0,
            {"gate_segment_over_max_duration": 1},
        ),
        (
            "duplicate_timestamp",
            [
                "e,2016-09-01,08:00:00,2号线陆家嘴,地铁,0.00,非优惠",
                "e,2016-09-01,08:20:00,2号线南京西路,地铁,4.00,非优惠",
                "e,2016-09-01,08:20:00,2号线陆家嘴,地铁,0.00,非优惠",
                "e,2016-09-01,09:00:00,2号线南京西路,地铁,4.00,非优惠",
            ],
            0,
            {"duplicate_timestamp_events": 2},
        ),
    ]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name, lines, expected_rows, expected_audit in cases:
            raw = root / f"{name}.csv"
            raw.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
            out = root / f"{name}.csv.gz"
            auditp = root / f"{name}.json"
            db = root / f"{name}.sqlite"
            audit = ingest_file(raw, db, out, auditp, "2016-09-01", None, salt="TEST_SALT")
            assert _count_rows(out) == expected_rows, (name, _count_rows(out), expected_rows)
            for k, v in expected_audit.items():
                assert audit.get(k, 0) == v, (name, k, audit.get(k, 0), v)
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(db) + suffix)
                if p.exists():
                    p.unlink()
    print(json.dumps({"self_test":"PASS","cases":len(cases)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--input", type=Path)
    ap.add_argument("--db", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--source-date")
    ap.add_argument("--expected-strict", type=int, default=None)
    ap.add_argument("--salt", default="SHANGHAI_MPPD_SPLIT_V1")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    for x in (a.input, a.db, a.out, a.audit):
        if x is None:
            raise SystemExit("missing required execution path")
    if not a.source_date:
        raise SystemExit("missing --source-date")
    result = ingest_file(
        a.input, a.db, a.out, a.audit, a.source_date, a.expected_strict, a.salt
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
