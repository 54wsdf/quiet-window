"""Verified, date-native readers and lossless time-bin views of public OD counts.

No forecasting or causal models are implemented here. This file is also vendored
unchanged into the private research repository. A whole source release is required;
missing date files are never interpreted as zero demand.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import date as Date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

FLOW_COLUMNS = ("Flow", "CFlow", "HBOFlow", "NHBFlow")
COLUMNS = ("date", "startTime", "originStation", "destinationStation", *FLOW_COLUMNS)
DATES = tuple((Date(2017, 5, 1) + timedelta(days=i)).isoformat() for i in range(123))
SOURCE_SHA = "427e7c8e3021d4007114db876db9fb6d1b35674b29ad950ebc64cf1a838cab88"
BIN_LABELS = ("06-10", "10-13", "13-16", "16-19", "19-21", "21-23")
BIN_EDGES = (0, 24, 42, 60, 78, 90, 102)
VIEW_STATE = "QUALIFIED_CANONICAL_Q6_Q10"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def safe_path(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    require(p.is_relative_to(root) and p != root, "artifact path escapes dataset root")
    return p


def normalize_date(value):
    s = str(value)
    if len(s) == 8 and s.isdigit():
        s = s[:4] + "-" + s[4:6] + "-" + s[6:]
    require(len(s) == 10 and Date.fromisoformat(s).isoformat() == s, "invalid date")
    return s


def verify_file(root, entry):
    p = safe_path(root, entry["path"])
    require(p.is_file(), "missing artifact: " + str(p))
    require(p.stat().st_size == entry["bytes"], "artifact byte mismatch: " + str(p))
    require(sha256_file(p) == entry["sha256"], "artifact SHA-256 mismatch: " + str(p))
    return p


def file_entry(root, p):
    p = Path(p)
    return {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
            "sha256": sha256_file(p)}


def station_authority(manifest, count):
    ids = manifest["station_ids"]
    require(all(type(x) is int and 0 < x <= np.iinfo(np.int32).max for x in ids),
            "station IDs must be positive int32 integers")
    require(len(ids) == count and ids == sorted(set(ids)), "invalid ordered station authority")
    return ids


class QualifiedOD:
    """Production reader. Overrides of coverage are intended only for synthetic tests."""
    def __init__(self, root, *, expected_manifest_sha256=None,
                 expected_dates=DATES, expected_station_count=302):
        self.root = Path(root).resolve()
        mp = self.root / "manifest.json"
        self.manifest_sha256 = sha256_file(mp)
        if expected_manifest_sha256 is not None:
            require(self.manifest_sha256 == expected_manifest_sha256, "pinned manifest SHA mismatch")
        q = read_json(self.root / "QUALIFIED.json")
        m = read_json(mp)
        require(q.get("state") == "QUALIFIED_CANONICAL_DATASET", "canonical dataset is unqualified")
        require(q.get("manifest_sha256") == self.manifest_sha256, "qualification manifest SHA mismatch")
        require(m.get("schema") == "shanghai-metroflow-canonical-parquet-v1" and m.get("state") == "QUALIFIED",
                "unsupported or unqualified canonical manifest")
        require(m["source"]["zip_sha256"] == SOURCE_SHA and q["source_zip_sha256"] == SOURCE_SHA,
                "unexpected corrected source ZIP")
        self.dates = tuple(expected_dates)
        require(self.dates and self.dates == tuple(sorted(set(self.dates))), "invalid expected dates")
        require(m["date_count"] == q["date_count"] == len(self.dates), "date count mismatch")
        require(tuple(e["date"] for e in m["dates"]) == self.dates, "date coverage or order mismatch")
        self.station_ids = station_authority(m, expected_station_count)
        require(m["station_authority_count"] == expected_station_count, "station count mismatch")
        require(m["time_authority"]["slots_per_day"] == 102 and m["time_authority"]["all_rows_valid"] is True,
                "time authority is unqualified")
        require(m["source_rows"] == q["source_rows"] == sum(e["rows"] for e in m["dates"]), "row count mismatch")
        for c in FLOW_COLUMNS:
            require(m["flow_totals"][c] == sum(e[c] for e in m["dates"]), "source daily totals mismatch")
        require(q["Flow"] == m["flow_totals"]["Flow"], "qualification Flow mismatch")
        require(m["flow_totals"]["Flow"] == sum(m["flow_totals"][c] for c in FLOW_COLUMNS[1:]),
                "source purpose totals mismatch")
        self.entries = {e["date"]: e for e in m["dates"]}
        for d, e in self.entries.items():
            require(e["path"] == f"date={d}/od.parquet", "noncanonical date path")
            p = safe_path(self.root, e["path"])
            require(p.is_file() and p.stat().st_size == e["bytes"], "missing or truncated date: " + d)
        self.manifest = m

    def iter_batches(self, date, batch_size=250000):
        d = normalize_date(date)
        require(type(batch_size) is int and batch_size > 0, "batch_size must be a positive integer")
        require(d in self.entries, "date outside qualified coverage")
        e = self.entries[d]
        p = verify_file(self.root, e)
        pf = pq.ParquetFile(p)  # Never infer a second date column from Hive directory names.
        require(pf.metadata.num_rows == e["rows"], "Parquet row count mismatch")
        require(tuple(pf.schema_arrow.names) == COLUMNS, "Parquet schema column mismatch")
        require(pa.types.is_string(pf.schema_arrow.field("date").type), "date must be string")
        for c in COLUMNS[1:]:
            require(pa.types.is_integer(pf.schema_arrow.field(c).type), "noninteger Parquet column: " + c)
        rows = 0
        for b in pf.iter_batches(batch_size=batch_size, columns=list(COLUMNS)):
            require(all(b.column(c).null_count == 0 for c in COLUMNS), "null canonical values")
            require(pc.all(pc.equal(b.column("date"), d)).as_py() is True, "row belongs to another date")
            rows += b.num_rows
            yield b
        require(rows == e["rows"], "streamed row count mismatch")
        require(sha256_file(p) == e["sha256"], "source mutated during read")

    def load_date(self, date, usecols=None):
        columns = list(COLUMNS if usecols is None else usecols)
        require(len(columns) == len(set(columns)) and set(columns).issubset(COLUMNS), "invalid projected columns")
        table = pa.Table.from_batches(list(self.iter_batches(date)))
        return table.select(columns).to_pandas()


def integer_values(batch, column):
    values = batch.column(column).to_numpy(zero_copy_only=False)
    require(np.issubdtype(values.dtype, np.integer), "noninteger values: " + column)
    require(not np.any(values < 0) and not np.any(values >= 2**53), "out-of-range values: " + column)
    return values.astype(np.int64, copy=False)


def reduce_date(dataset, date):
    n = len(dataset.station_ids)
    pairs = n * n
    ids = np.asarray(dataset.station_ids, dtype=np.int64)
    q10 = np.zeros((102, pairs), dtype=np.int32)
    q6 = {c: np.zeros((6, pairs), dtype=np.int64) for c in FLOW_COLUMNS}
    seen = np.zeros(102 * pairs, dtype=bool)
    slot_rows = np.zeros(102, dtype=np.int64)
    totals = {c: 0 for c in FLOW_COLUMNS}
    rows = 0
    for b in dataset.iter_batches(date):
        clock = integer_values(b, "startTime")
        hour, minute, second = clock // 10000, clock // 100 % 100, clock % 100
        absolute = hour * 60 + minute
        require(np.all((hour >= 6) & (hour < 23) & (minute < 60) & (minute % 10 == 0) & (second == 0)),
                "invalid ten-minute startTime")
        slots = (absolute - 360) // 10
        orig, dest = integer_values(b, "originStation"), integer_values(b, "destinationStation")
        oi, di = np.searchsorted(ids, orig), np.searchsorted(ids, dest)
        require(np.all(oi < n) and np.all(di < n), "unknown station ID")
        require(np.array_equal(ids[oi], orig) and np.array_equal(ids[di], dest), "unknown station ID")
        require(np.all(oi != di), "same-station diagonal row")
        v = {c: integer_values(b, c) for c in FLOW_COLUMNS}
        require(np.array_equal(v["Flow"], v["CFlow"] + v["HBOFlow"] + v["NHBFlow"]), "purpose decomposition failure")
        require(np.all(v["Flow"] <= np.iinfo(np.int32).max), "Q10 int32 overflow")
        pair = oi * n + di
        key = slots * pairs + pair
        require(len(np.unique(key)) == len(key) and not np.any(seen[key]), "duplicate date/time/OD key")
        seen[key] = True
        q10.ravel()[key] = v["Flow"].astype(np.int32)
        bins = np.searchsorted(np.asarray(BIN_EDGES[1:]), slots, side="right")
        bin_key = bins * pairs + pair
        for c in FLOW_COLUMNS:
            np.add.at(q6[c].ravel(), bin_key, v[c])
            totals[c] += int(v[c].sum(dtype=np.int64))
        slot_rows += np.bincount(slots, minlength=102)
        rows += b.num_rows
    e = dataset.entries[normalize_date(date)]
    require(rows == e["rows"] and all(totals[c] == e[c] for c in FLOW_COLUMNS), "decoded source totals mismatch")
    folded = np.stack([q10[a:z].sum(axis=0, dtype=np.int64) for a, z in zip(BIN_EDGES[:-1], BIN_EDGES[1:])])
    require(np.array_equal(folded, q6["Flow"]), "Q10 to Q6 cell mismatch")
    require(np.array_equal(q6["Flow"], sum(q6[c] for c in FLOW_COLUMNS[1:])), "Q6 purpose cell mismatch")
    require(all(int(q6[c].max(initial=0)) <= np.iinfo(np.int32).max for c in FLOW_COLUMNS), "Q6 int32 overflow")
    require(int(q10.sum(dtype=np.int64)) == totals["Flow"], "Q10 total mismatch")
    report = {"date": normalize_date(date), "rows": rows, **totals,
              "duplicate_keys": 0, "q10_q6_cell_mismatches": 0,
              "observed_slots": np.flatnonzero(slot_rows).tolist(),
              "unobserved_slots": np.flatnonzero(slot_rows == 0).tolist()}
    return q10, q6, report


def build_views(dataset, output):
    output = Path(output).resolve()
    building = output.with_name(output.name + ".building")
    require(not output.exists() and not building.exists(), "output already exists; never overwrite a build")
    (building / "q6").mkdir(parents=True)
    (building / "q10").mkdir()
    (building / "marginals").mkdir()
    n = len(dataset.station_ids)
    tensors = {c: np.lib.format.open_memmap(building / "q6" / f"{c}.npy", mode="w+", dtype=np.int32,
               shape=(len(dataset.dates) * 6, n * n)) for c in FLOW_COLUMNS}
    reports = []
    for idx, d in enumerate(dataset.dates):
        q10, q6, report = reduce_date(dataset, d)
        compact = d.replace("-", "")
        path = building / "q10" / f"q10.date-{compact}.npz"
        matrix = sparse.csr_matrix(q10)
        sparse.save_npz(path, matrix, compressed=True)
        restored = sparse.load_npz(path)
        require(restored.shape == matrix.shape and (restored != matrix).nnz == 0, "Q10 write/read cell mismatch")
        for c in FLOW_COLUMNS:
            tensors[c][idx * 6:(idx + 1) * 6] = q6[c].astype(np.int32)
        cube = q10.reshape(102, n, n)
        np.savez_compressed(building / "marginals" / f"entry_clock-{compact}.npz",
                            origin=cube.sum(axis=2, dtype=np.int64), destination=cube.sum(axis=1, dtype=np.int64))
        report["q10"] = file_entry(building, path)
        reports.append(report)
        if (idx + 1) % 10 == 0 or idx + 1 == len(dataset.dates):
            print(json.dumps({"state": "DERIVING_DATE_VIEWS", "dates": idx + 1,
                              "Flow": sum(r["Flow"] for r in reports)}), flush=True)
    q6_entries = {}
    for c, tensor in tensors.items():
        tensor.flush()
        path = building / "q6" / f"{c}.npy"
        restored = np.load(path, mmap_mode="r", allow_pickle=False)
        for idx, r in enumerate(reports):
            require(np.array_equal(restored[idx*6:(idx+1)*6], tensor[idx*6:(idx+1)*6]), "Q6 write/read cell mismatch")
            require(int(restored[idx*6:(idx+1)*6].sum(dtype=np.int64)) == r[c], "Q6 daily total mismatch")
        require(int(restored.sum(dtype=np.int64)) == dataset.manifest["flow_totals"][c], "Q6 full-period total mismatch")
        q6_entries[c] = file_entry(building, path)
    require(sum(r["rows"] for r in reports) == dataset.manifest["source_rows"], "full-period row mismatch")
    require(all(sum(r[c] for r in reports) == dataset.manifest["flow_totals"][c] for c in FLOW_COLUMNS), "global count mismatch")
    with (building / "station_index.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["station_index", "station_id"]); w.writerows(enumerate(dataset.station_ids))
    with (building / "snapshot_index.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["snapshot_index", "date", "time_bin"])
        w.writerows((i*6+b, d.replace("-", ""), label) for i, d in enumerate(dataset.dates) for b, label in enumerate(BIN_LABELS))
    write_json(building / "date_quality.json", {"dates": reports, "policy": "retain all source dates; unobserved slots are quality flags, not imputed observations"})
    manifest = {"schema": "metroflow-canonical-derived-views-v1", "state": VIEW_STATE,
                "source_manifest_sha256": dataset.manifest_sha256, "source_zip_sha256": SOURCE_SHA,
                "source_run_id": dataset.manifest.get("runtime", {}).get("github_run_id"),
                "source_rows": dataset.manifest["source_rows"], "dates": list(dataset.dates),
                "station_ids": dataset.station_ids, "flow_totals": dataset.manifest["flow_totals"],
                "bin_labels": list(BIN_LABELS), "bin_edges_slots": list(BIN_EDGES), "slots_per_day": 102,
                "q6": q6_entries, "daily": reports, "q6_shape": [len(dataset.dates)*6, n*n],
                "q10_shape": [102, n*n], "source_files_sha256_verified": len(reports),
                "q10_q6_cell_mismatches": 0, "duplicate_keys": 0,
                "clock_semantics": "origin entry clock; destination marginal is not contemporaneous exit-gate count",
                "runtime": {"github_sha": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                            "producer_sha256": sha256_file(Path(__file__))}}
    write_json(building / "manifest.json", manifest)
    write_json(building / "QUALIFIED.json", {"state": VIEW_STATE, "manifest_sha256": sha256_file(building / "manifest.json")})
    with (building / "SHA256SUMS").open("w") as f:
        for p in sorted(building.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS":
                f.write(sha256_file(p) + "  " + p.relative_to(building).as_posix() + "\n")
    os.replace(building, output)
    summary = {"state": VIEW_STATE, "dates": len(reports), "source_rows": manifest["source_rows"],
               "flow_totals": manifest["flow_totals"], "q6_shape": manifest["q6_shape"], "q10_shape": manifest["q10_shape"],
               "q10_q6_cell_mismatches": 0, "duplicate_keys": 0, "manifest_sha256": sha256_file(output / "manifest.json")}
    print(json.dumps(summary), flush=True)
    return manifest


def load_view_manifest(path, *, expected_dates=DATES, expected_station_count=302):
    path = Path(path)
    m = read_json(path)
    q = read_json(path.parent / "QUALIFIED.json")
    require(m.get("schema") == "metroflow-canonical-derived-views-v1" and m.get("state") == VIEW_STATE,
            "unsupported or unqualified derived views")
    require(q.get("state") == VIEW_STATE and q.get("manifest_sha256") == sha256_file(path), "view manifest hash mismatch")
    require(tuple(m["dates"]) == tuple(expected_dates), "view date coverage mismatch")
    station_authority(m, expected_station_count)
    require(m["source_zip_sha256"] == SOURCE_SHA, "view source identity mismatch")
    require(m["q6_shape"] == [len(expected_dates)*6, expected_station_count**2] and
            m["q10_shape"] == [102, expected_station_count**2], "view shape mismatch")
    require(m["bin_edges_slots"] == list(BIN_EDGES) and m["bin_labels"] == list(BIN_LABELS), "view bin definition mismatch")
    require(m["q10_q6_cell_mismatches"] == 0 and m["duplicate_keys"] == 0, "view validation failed")
    require(tuple(r["date"] for r in m["daily"]) == tuple(expected_dates), "daily coverage mismatch")
    require(sum(r["rows"] for r in m["daily"]) == m["source_rows"], "view row total mismatch")
    for c in FLOW_COLUMNS:
        require(sum(r[c] for r in m["daily"]) == m["flow_totals"][c], "view daily count mismatch")
    for e in [*m["q6"].values(), *(r["q10"] for r in m["daily"])]:
        p = safe_path(path.parent, e["path"])
        require(p.is_file() and p.stat().st_size == e["bytes"], "view artifact missing or truncated")
    return m


def load_view_map(path, dates, *, resolution):
    path = Path(path)
    m = load_view_manifest(path)
    require(resolution in (6, 10), "resolution must be 6 or 10")
    out = {}
    if resolution == 6:
        p = verify_file(path.parent, m["q6"]["Flow"])
        tensor = np.load(p, mmap_mode="r", allow_pickle=False)
        require(list(tensor.shape) == m["q6_shape"], "Q6 tensor shape mismatch")
    daily = {r["date"]: r for r in m["daily"]}
    for value in dates:
        d = normalize_date(value)
        require(d in daily, "requested date is unavailable")
        if resolution == 6:
            idx = m["dates"].index(d)
            matrix = sparse.csr_matrix(np.asarray(tensor[idx*6:(idx+1)*6], dtype=np.float64))
        else:
            p = verify_file(path.parent, daily[d]["q10"])
            matrix = sparse.load_npz(p).tocsr()
            require(list(matrix.shape) == m["q10_shape"], "Q10 matrix shape mismatch")
        require(not np.any(matrix.data < 0) and float(matrix.sum()) == daily[d]["Flow"], "loaded daily counts mismatch")
        out[d.replace("-", "")] = matrix
    return out, m["station_ids"]


def main():
    p = argparse.ArgumentParser(description="Verify public canonical OD and derive date-native count views")
    p.add_argument("source_root"); p.add_argument("output_root")
    p.add_argument("--manifest-sha256", required=True)
    args = p.parse_args()
    source = QualifiedOD(args.source_root, expected_manifest_sha256=args.manifest_sha256)
    build_views(source, args.output_root)


if __name__ == "__main__":
    main()
