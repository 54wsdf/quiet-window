import scripts.mppd_r1_hz_count_free_passenger_coverage as m


def service(sid, path, direction, times):
    return {
        "trajectory_id": sid,
        "path_id": path,
        "direction": direction,
        "options": {(path, direction)},
        "events": {str(s): {"time_s": float(t)} for s, t in times.items()},
        "support_weight": 1.0,
        "evidence_score": 1.0,
        "path_ambiguous": False,
    }


def leg(u, v, path="P", direction="Down"):
    return {
        "from_station": u,
        "to_station": v,
        "compatible_service_options": [{"path_id": path, "direction": direction}],
    }


def test_complete_chain_obeys_15_15_5_physical_gates():
    services = {
        "t1": service("t1", "P", "Down", {1: 100, 2: 200}),
        "t2": service("t2", "Q", "Down", {2: 210, 3: 300}),
    }
    idx = m.LegIndex(services)
    route = {"ride_legs": [leg(1, 2), leg(2, 3, "Q")]}
    row = {"entry_sec": 85.0, "exit_sec": 315.0, "destination_station": 3}
    out = m.evaluate_route(row, route, idx)
    assert out["ok"] is True
    assert out["access_budget_s"] == 15.0
    assert out["egress_budget_s"] == 15.0
    assert out["chain"][1]["departure_s"] - out["chain"][0]["arrival_s"] == 10.0


def test_access_below_15_rejects_first_train():
    services = {"t1": service("t1", "P", "Down", {1: 100, 2: 200})}
    idx = m.LegIndex(services)
    route = {"ride_legs": [leg(1, 2)]}
    row = {"entry_sec": 90.0, "exit_sec": 240.0, "destination_station": 2}
    out = m.evaluate_route(row, route, idx)
    assert out["ok"] is False
    assert out["reason"] == "FIRST_LEG_NO_SERVICE"


def test_transfer_below_5_rejects_downstream_train():
    services = {
        "t1": service("t1", "P", "Down", {1: 100, 2: 200}),
        "too_early": service("too_early", "Q", "Down", {2: 203, 3: 290}),
    }
    idx = m.LegIndex(services)
    route = {"ride_legs": [leg(1, 2), leg(2, 3, "Q")]}
    row = {"entry_sec": 80.0, "exit_sec": 330.0, "destination_station": 3}
    out = m.evaluate_route(row, route, idx)
    assert out["ok"] is False
    assert out["reason"] == "TRANSFER_NO_DOWNSTREAM_SERVICE"


def test_egress_below_15_rejects_chain():
    services = {"t1": service("t1", "P", "Down", {1: 100, 2: 200})}
    idx = m.LegIndex(services)
    route = {"ride_legs": [leg(1, 2)]}
    row = {"entry_sec": 80.0, "exit_sec": 210.0, "destination_station": 2}
    out = m.evaluate_route(row, route, idx)
    assert out["ok"] is False
    assert out["reason"] == "FINAL_EGRESS_HORIZON"
    assert out["excess_s"] == 5.0


def test_earliest_arrival_not_first_departure_is_used():
    services = {
        "slow": service("slow", "P", "Down", {1: 100, 2: 240}),
        "fast": service("fast", "P", "Down", {1: 110, 2: 210}),
    }
    idx = m.LegIndex(services)
    route = {"ride_legs": [leg(1, 2)]}
    row = {"entry_sec": 80.0, "exit_sec": 230.0, "destination_station": 2}
    out = m.evaluate_route(row, route, idx)
    assert out["ok"] is True
    assert out["chain"][0]["trajectory_id"] == "fast"
