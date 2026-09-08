"""Synthetic regression tests; these fixtures are not real passenger results."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

P = Path(os.environ.get("OD_VIEWS_MODULE", "scripts/processing/qualified_od_views.py"))
spec = importlib.util.spec_from_file_location("od_views_under_test", P)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
DAYS = ("2017-05-01", "2017-05-02")
IDS = [11, 22, 33]


def rows(d):
    clocks = [60000, 61000, 100000, 130000, 160000, 190000, 210000, 225000]
    return [{"date": d, "startTime": t, "originStation": 11, "destinationStation": 22,
             "Flow": i + 3, "CFlow": i + 1, "HBOFlow": 1, "NHBFlow": 1}
            for i, t in enumerate(clocks)]


def make_source(tmp_path, mutate=None):
    root = tmp_path / "canonical"
    root.mkdir()
    entries = []
    for d in DAYS:
        r = rows(d)
        if mutate is not None and d == DAYS[0]:
            mutate(r)
        path = root / f"date={d}" / "od.parquet"
        path.parent.mkdir()
        pq.write_table(pa.Table.from_pylist(r), path, row_group_size=2)
        e = {"date": d, "rows": len(r), **v.file_entry(root, path)}
        e.update({c: sum(x[c] or 0 for x in r) for c in v.FLOW_COLUMNS})
        entries.append(e)
    m = {"schema": "shanghai-metroflow-canonical-parquet-v1", "state": "QUALIFIED",
         "source": {"zip_sha256": v.SOURCE_SHA}, "dates": entries,
         "date_count": 2, "station_ids": IDS, "station_authority_count": 3,
         "source_rows": sum(e["rows"] for e in entries),
         "flow_totals": {c: sum(e[c] for e in entries) for c in v.FLOW_COLUMNS},
         "time_authority": {"slots_per_day": 102, "all_rows_valid": True},
         "runtime": {"github_run_id": "SYNTHETIC_TEST_ONLY"}}
    repin(root, m)
    return root


def repin(root, m):
    v.write_json(root / "manifest.json", m)
    v.write_json(root / "QUALIFIED.json", {"state": "QUALIFIED_CANONICAL_DATASET",
        "manifest_sha256": v.sha256_file(root / "manifest.json"), "date_count": m["date_count"],
        "source_rows": m["source_rows"], "Flow": m["flow_totals"]["Flow"], "source_zip_sha256": v.SOURCE_SHA})


def source(root):
    return v.QualifiedOD(root, expected_dates=DAYS, expected_station_count=3)


def test_hive_directory_string_date_is_readable(tmp_path):
    s = source(make_source(tmp_path))
    actual = s.load_date("20170501")
    assert list(actual.columns) == list(v.COLUMNS)
    assert set(actual.date) == {DAYS[0]}
    assert len(actual) == 8
    assert list(s.load_date(DAYS[0], ["Flow"]).columns) == ["Flow"]


def test_production_rejects_synthetic_coverage(tmp_path):
    root = make_source(tmp_path)
    with pytest.raises(ValueError, match="date count"):
        v.QualifiedOD(root)


def test_pin_and_qualification(tmp_path):
    root = make_source(tmp_path)
    with pytest.raises(ValueError, match="pinned manifest"):
        v.QualifiedOD(root, expected_manifest_sha256="0" * 64)
    q = v.read_json(root / "QUALIFIED.json"); q["manifest_sha256"] = "0" * 64
    v.write_json(root / "QUALIFIED.json", q)
    with pytest.raises(ValueError, match="manifest SHA"):
        source(root)


def test_missing_qualification(tmp_path):
    root = make_source(tmp_path); (root / "QUALIFIED.json").unlink()
    with pytest.raises(FileNotFoundError):
        source(root)


def test_missing_date_is_not_zero(tmp_path):
    root = make_source(tmp_path); (root / f"date={DAYS[1]}" / "od.parquet").unlink()
    with pytest.raises(ValueError, match="missing or truncated"):
        source(root)


def test_content_corruption_even_when_size_equal(tmp_path):
    root = make_source(tmp_path); p = root / f"date={DAYS[0]}" / "od.parquet"
    b = bytearray(p.read_bytes()); b[10] ^= 1; p.write_bytes(b)
    with pytest.raises(ValueError, match="SHA-256"):
        source(root).load_date(DAYS[0])


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_batch_size_validation(tmp_path, size):
    s = source(make_source(tmp_path))
    with pytest.raises(ValueError, match="batch_size"):
        list(s.iter_batches(DAYS[0], size))


@pytest.mark.parametrize("clock", [50000, 230000, 106000, 61001, 60500])
def test_invalid_clock(tmp_path, clock):
    s = source(make_source(tmp_path, lambda r: r[0].update(startTime=clock)))
    with pytest.raises(ValueError, match="startTime"):
        v.reduce_date(s, DAYS[0])


@pytest.mark.parametrize("station", [1, 12, 44])
def test_unknown_station(tmp_path, station):
    s = source(make_source(tmp_path, lambda r: r[0].update(originStation=station)))
    with pytest.raises(ValueError, match="unknown station"):
        v.reduce_date(s, DAYS[0])


def test_diagonal(tmp_path):
    s = source(make_source(tmp_path, lambda r: r[0].update(destinationStation=11)))
    with pytest.raises(ValueError, match="diagonal"):
        v.reduce_date(s, DAYS[0])


def test_duplicate_key(tmp_path):
    s = source(make_source(tmp_path, lambda r: r.append(dict(r[0]))))
    with pytest.raises(ValueError, match="duplicate"):
        v.reduce_date(s, DAYS[0])


def test_duplicate_across_batches(tmp_path):
    s = source(make_source(tmp_path, lambda r: r.append(dict(r[0]))))
    old = s.iter_batches
    s.iter_batches = lambda d: old(d, batch_size=1)
    with pytest.raises(ValueError, match="duplicate"):
        v.reduce_date(s, DAYS[0])


def test_wrong_date_inside_file(tmp_path):
    s = source(make_source(tmp_path, lambda r: r[0].update(date=DAYS[1])))
    with pytest.raises(ValueError, match="another date"):
        v.reduce_date(s, DAYS[0])


def test_fractional_integer_column(tmp_path):
    s = source(make_source(tmp_path, lambda r: r[0].update(originStation=11.5)))
    with pytest.raises(ValueError, match="noninteger"):
        v.reduce_date(s, DAYS[0])


def test_null_value(tmp_path):
    s = source(make_source(tmp_path, lambda r: r[0].update(startTime=None)))
    with pytest.raises(ValueError, match="null"):
        v.reduce_date(s, DAYS[0])


def test_negative_counts(tmp_path):
    s = source(make_source(tmp_path, lambda r: r[0].update(Flow=-1, CFlow=-3)))
    with pytest.raises(ValueError, match="out-of-range"):
        v.reduce_date(s, DAYS[0])


def test_row_purpose_mismatch_even_if_global_totals_match(tmp_path):
    def mutate(r):
        r[0]["Flow"] += 1; r[1]["Flow"] -= 1
    s = source(make_source(tmp_path, mutate))
    with pytest.raises(ValueError, match="purpose decomposition"):
        v.reduce_date(s, DAYS[0])


def test_cell_identity_and_all_six_bins(tmp_path):
    s = source(make_source(tmp_path))
    q10, q6, report = v.reduce_date(s, DAYS[0])
    assert q10.shape == (102, 9)
    assert q10[0, 1] == 3 and q10[1, 1] == 4
    assert q10[101, 1] == 10
    assert q6["Flow"][:, 1].tolist() == [7, 5, 6, 7, 8, 19]
    assert report["duplicate_keys"] == report["q10_q6_cell_mismatches"] == 0
    assert len(report["unobserved_slots"]) == 94


def test_q6_overflow_is_explicit(tmp_path):
    def mutate(r):
        for row in r[:2]:
            row.update(Flow=1500000000, CFlow=1499999998)
    s = source(make_source(tmp_path, mutate))
    with pytest.raises(ValueError, match="Q6 int32 overflow"):
        v.reduce_date(s, DAYS[0])


def test_build_qualification_readback_and_no_overwrite(tmp_path):
    s = source(make_source(tmp_path)); out = tmp_path / "views"
    m = v.build_views(s, out)
    loaded = v.load_view_manifest(out / "manifest.json", expected_dates=DAYS, expected_station_count=3)
    assert m["state"] == loaded["state"] == v.VIEW_STATE
    assert m["q6_shape"] == [12, 9]
    assert m["flow_totals"]["Flow"] == 104
    assert (out / "SHA256SUMS").exists()
    assert np.load(out / "q6" / "Flow.npy").sum() == 104
    with pytest.raises(ValueError, match="already exists"):
        v.build_views(s, out)
    q = v.read_json(out / "QUALIFIED.json"); q["manifest_sha256"] = "0" * 64
    v.write_json(out / "QUALIFIED.json", q)
    with pytest.raises(ValueError, match="view manifest hash"):
        v.load_view_manifest(out / "manifest.json", expected_dates=DAYS, expected_station_count=3)


def test_manifest_path_escape(tmp_path):
    root = make_source(tmp_path); m = v.read_json(root / "manifest.json")
    m["dates"][0]["path"] = "../outside.parquet"; repin(root, m)
    with pytest.raises(ValueError, match="noncanonical date path"):
        source(root)


def test_coverage_order(tmp_path):
    root = make_source(tmp_path); m = v.read_json(root / "manifest.json")
    m["dates"].reverse(); repin(root, m)
    with pytest.raises(ValueError, match="coverage or order"):
        source(root)
