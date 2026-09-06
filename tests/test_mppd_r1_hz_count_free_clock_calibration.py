import json
from pathlib import Path

import pytest

import scripts.mppd_r1_hz_count_free_clock_calibration as cal
import scripts.mppd_r1_hz_count_free_passenger_coverage as cov


def service(sid, times):
    return {
        "trajectory_id": sid,
        "path_id": "P",
        "direction": "Down",
        "options": {("P", "Down")},
        "events": {str(s): {"time_s": float(t)} for s, t in times.items()},
        "support_weight": 1.0,
        "evidence_score": 1.0,
        "path_ambiguous": False,
    }


def leg(u, v):
    return {"from_station": u, "to_station": v, "compatible_service_options": [{"path_id": "P", "direction": "Down"}]}


def test_small_shift_can_repair_boundary_without_large_jump():
    services = {"t1": service("t1", {1: 130, 2: 230})}
    idx = cov.LegIndex(services)
    route = {"ride_legs": [leg(1, 2)]}
    row = {"entry_sec": 100.0, "exit_sec": 244.0, "destination_station": 2}
    assert cal.evaluate_row(row, [route], idx, 0.0) == "FINAL_EGRESS_HORIZON"
    assert cal.evaluate_row(row, [route], idx, 5.0) == "FEASIBLE_COMPLETE_CHAIN"


def test_too_large_shift_can_make_first_service_unreachable():
    services = {"t1": service("t1", {1: 130, 2: 230})}
    idx = cov.LegIndex(services)
    route = {"ride_legs": [leg(1, 2)]}
    row = {"entry_sec": 110.0, "exit_sec": 300.0, "destination_station": 2}
    assert cal.evaluate_row(row, [route], idx, 0.0) == "FEASIBLE_COMPLETE_CHAIN"
    assert cal.evaluate_row(row, [route], idx, 10.0) == "FIRST_LEG_NO_SERVICE"


def test_shift_grid_is_five_second_trust_region():
    assert cal.shift_values(-5, 5, 5) == [-5.0, 0.0, 5.0]
    with pytest.raises(ValueError):
        cal.shift_values(0, 60, 5)


def test_apply_refuses_old_plus_60_solution(tmp_path: Path):
    services = {
        "inferred_service_trajectory_count": 1,
        "trajectories": [{
            "reference_time_s": 100.0,
            "events": [{"time_s": 100.0}],
        }],
        "station_phase_nuisance_s": {},
        "semantics": {},
    }
    calibration = {"best_clock_shift_s": 60.0}
    sp = tmp_path / "services.json"
    cp = tmp_path / "calibration.json"
    op = tmp_path / "out.json"
    sp.write_text(json.dumps(services), encoding="utf-8")
    cp.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(SystemExit):
        cal.apply_shift(sp, cp, op)


def test_apply_accepts_plus_5_and_marks_reinference(tmp_path: Path):
    services = {
        "inferred_service_trajectory_count": 1,
        "trajectories": [{
            "reference_time_s": 100.0,
            "events": [{"time_s": 100.0}],
        }],
        "station_phase_nuisance_s": {"1": 20.0},
        "semantics": {},
    }
    calibration = {"best_clock_shift_s": 5.0}
    sp = tmp_path / "services.json"
    cp = tmp_path / "calibration.json"
    op = tmp_path / "out.json"
    sp.write_text(json.dumps(services), encoding="utf-8")
    cp.write_text(json.dumps(calibration), encoding="utf-8")
    out = cal.apply_shift(sp, cp, op)
    assert out["trajectories"][0]["reference_time_s"] == 95.0
    assert out["trajectories"][0]["events"][0]["time_s"] == 95.0
    assert out["absolute_clock_calibration"]["single_update_trust_region_s"] == 5.0
    assert "REQUIRES_FULL_JOINT_REINFERENCE" in out["status"]
