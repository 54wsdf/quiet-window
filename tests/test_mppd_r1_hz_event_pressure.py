import scripts.mppd_r1_hz_count_free_passenger_coverage as cov
from scripts.mppd_r1_hz_event_pressure import detailed_evaluate, previous_service


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
    return {
        "from_station": u,
        "to_station": v,
        "compatible_service_options": [{"path_id": "P", "direction": "Down"}],
    }


def test_final_egress_failure_retains_concrete_chain_for_event_pressure():
    services = {"q1": service("q1", {1: 120, 2: 200})}
    index = cov.LegIndex(services)
    row = {"entry_sec": 100.0, "exit_sec": 210.0, "destination_station": 2}
    r = detailed_evaluate(row, {"ride_legs": [leg(1, 2)]}, index)
    assert r["ok"] is False
    assert r["reason"] == "FINAL_EGRESS_HORIZON"
    assert r["excess_s"] == 5.0
    assert r["chain"][-1]["trajectory_id"] == "q1"


def test_resolved_chain_is_available_for_counter_pressure():
    services = {"q1": service("q1", {1: 120, 2: 180})}
    index = cov.LegIndex(services)
    row = {"entry_sec": 100.0, "exit_sec": 210.0, "destination_station": 2}
    r = detailed_evaluate(row, {"ride_legs": [leg(1, 2)]}, index)
    assert r["ok"] is True
    assert r["chain"][0]["departure_s"] == 120.0
    assert r["chain"][0]["arrival_s"] == 180.0


def test_previous_service_can_support_plus_five_rescue_pressure():
    services = {
        "q1": service("q1", {1: 120, 2: 180}),
        "q2": service("q2", {1: 300, 2: 360}),
    }
    index = cov.LegIndex(services)
    p = previous_service(index, leg(1, 2), 124.0)
    assert p is not None
    assert p[0] == 120.0
    assert p[2] == "q1"
