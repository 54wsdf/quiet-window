from __future__ import annotations
import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "mppd_r1_joint_reconstruction_validate.py"
spec = importlib.util.spec_from_file_location("r1v", MODULE)
r1v = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(r1v)

def assert_bad(doc, code):
    report = r1v.validate(doc, r1v.authority_fixture())
    assert report["status"] == "R1_JOINT_RECONSTRUCTION_NOT_QUALIFIED", report
    assert report["violation_counts"].get(code, 0) > 0, report

def test_valid_joint_world():
    report = r1v.validate(r1v.fixture(), r1v.authority_fixture())
    assert report["status"] == "QUALIFIED_R1_JOINT_RECONSTRUCTION", report
    assert all(report["qualification_gates"].values())

def test_rejects_sub_five_second_transfer_interval():
    doc = r1v.fixture()
    doc["transfer_path_intervals"][0]["paths"][0]["transfer_interval_s"][0] = 1.0
    assert_bad(doc, "physical_lower_bound")

def test_rejects_single_transfer_path():
    doc = r1v.fixture()
    doc["transfer_path_intervals"][0]["paths"] = doc["transfer_path_intervals"][0]["paths"][:1]
    assert_bad(doc, "transfer_path_multiplicity")

def test_rejects_missing_realized_service_event():
    doc = r1v.fixture()
    doc["realized_service_timetable"] = doc["realized_service_timetable"][:-1]
    assert_bad(doc, "service_coverage")

def test_rejects_negative_wait():
    doc = r1v.fixture()
    doc["passenger_chains"][0]["access_time_s"] = 35.0
    assert_bad(doc, "negative_wait")

def test_rejects_transfer_outside_selected_path_interval():
    doc = r1v.fixture()
    doc["passenger_chains"][0]["transfers"][0]["transfer_time_s"] = 25.0
    assert_bad(doc, "transfer_membership")

def test_rejects_self_declared_incomplete_authority():
    authority = r1v.authority_fixture()
    authority["schema"] = "wrong"
    report = r1v.validate(r1v.fixture(), authority)
    assert report["status"] == "R1_JOINT_RECONSTRUCTION_NOT_QUALIFIED", report
    assert report["violation_counts"].get("authority", 0) > 0, report

if __name__ == "__main__":
    tests=[v for k,v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
