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


def test_common_shift_moves_service_earlier_without_changing_transfer_gap():
    services = {
        "t1": service("t1", {1: 130, 2: 230}),
        "t2": service("t2", {2: 240, 3: 330}),
    }
    idx = cov.LegIndex(services)
    route = {"ride_legs": [leg(1, 2), leg(2, 3)]}
    row = {"entry_sec": 80.0, "exit_sec": 330.0, "destination_station": 3}
    assert cal.evaluate_row(row, [route], idx, 0.0) == "FINAL_EGRESS_HORIZON"
    assert cal.evaluate_row(row, [route], idx, 30.0) == "FEASIBLE_COMPLETE_CHAIN"


def test_too_large_shift_can_make_first_service_unreachable():
    services = {"t1": service("t1", {1: 130, 2: 230})}
    idx = cov.LegIndex(services)
    route = {"ride_legs": [leg(1, 2)]}
    row = {"entry_sec": 100.0, "exit_sec": 300.0, "destination_station": 2}
    assert cal.evaluate_row(row, [route], idx, 0.0) == "FEASIBLE_COMPLETE_CHAIN"
    assert cal.evaluate_row(row, [route], idx, 30.0) == "FIRST_LEG_NO_SERVICE"


def test_shift_grid_is_inclusive():
    assert cal.shift_values(0, 60, 15) == [0.0, 15.0, 30.0, 45.0, 60.0]
