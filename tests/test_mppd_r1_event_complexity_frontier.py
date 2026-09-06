from scripts.mppd_r1_event_complexity_frontier import build_frontier, filter_proposal


def proposal():
    return {
        "schema": "mppd.r1-hz-event-update-proposal.v1",
        "updates": [
            {"trajectory_id": "a", "delta_s": -5.0, "direct_rescue_mass": 10.0, "direct_damage_mass": 0.0, "direct_net_mass": 10.0},
            {"trajectory_id": "b", "delta_s": -5.0, "direct_rescue_mass": 6.0, "direct_damage_mass": 1.0, "direct_net_mass": 5.0},
            {"trajectory_id": "c", "delta_s": 5.0, "direct_rescue_mass": 2.0, "direct_damage_mass": 1.0, "direct_net_mass": 1.0},
        ],
        "proposed_event_update_count": 3,
        "proposed_trajectory_update_count": 3,
        "sum_direct_rescue_mass_not_deduplicated": 18.0,
        "sum_direct_damage_mass_not_deduplicated": 2.0,
        "semantics": {},
    }


def test_frontier_is_induced_by_observed_net_rescue_values():
    f = build_frontier(proposal())
    counts = [x["update_count"] for x in f["frontier"]]
    assert counts == [3, 2, 1]
    assert f["semantics"]["frontier_does_not_select_final_lambda"] is True


def test_complexity_filter_keeps_only_net_gain_above_lambda():
    p = filter_proposal(proposal(), 4.0)
    assert p["proposed_event_update_count"] == 2
    assert {x["trajectory_id"] for x in p["updates"]} == {"a", "b"}
    assert p["semantics"]["lambda_event"] == 4.0
