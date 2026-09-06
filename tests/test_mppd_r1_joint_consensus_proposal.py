from scripts.mppd_r1_joint_consensus_proposal import build_consensus


def proposal():
    return {
        "proposed_event_update_count": 3,
        "updates": [
            {"trajectory_id": "a", "station_id": 1, "delta_s": -5.0, "direct_rescue_mass": 10.0, "direct_damage_mass": 1.0, "direct_net_mass": 9.0},
            {"trajectory_id": "b", "station_id": 2, "delta_s": -5.0, "direct_rescue_mass": 8.0, "direct_damage_mass": 1.0, "direct_net_mass": 7.0},
            {"trajectory_id": "c", "station_id": 3, "delta_s": -5.0, "direct_rescue_mass": 6.0, "direct_damage_mass": 1.0, "direct_net_mass": 5.0},
        ],
    }


def afc_frontier():
    return {
        "event_audit": [
            {"trajectory_id": "a", "station_id": 1, "delta_s": -5.0, "afc_nearest_distance_change_s": -5.0, "afc_weighted_distance_change": -10.0, "nearest_afc_pulse_before_distance_s": 8.0, "nearest_afc_pulse_after_distance_s": 3.0, "afc_supported_before_under_trajectory_residual_p90": True},
            {"trajectory_id": "b", "station_id": 2, "delta_s": -5.0, "afc_nearest_distance_change_s": 5.0, "afc_weighted_distance_change": 10.0, "nearest_afc_pulse_before_distance_s": 3.0, "nearest_afc_pulse_after_distance_s": 8.0, "afc_supported_before_under_trajectory_residual_p90": True},
            {"trajectory_id": "c", "station_id": 3, "delta_s": -5.0, "afc_nearest_distance_change_s": 0.0, "afc_weighted_distance_change": 0.0, "nearest_afc_pulse_before_distance_s": 10.0, "nearest_afc_pulse_after_distance_s": 10.0, "afc_supported_before_under_trajectory_residual_p90": False},
        ]
    }


def test_consensus_keeps_only_direct_supported_nonworsening_afc_moves():
    x = build_consensus(proposal(), afc_frontier(), lambda_event=0.0, require_direct_afc_support=True)
    assert x["consensus_event_update_count"] == 1
    assert x["updates"][0]["trajectory_id"] == "a"
    assert x["excluded_counts"]["afc_anchor_worsens"] == 1
    assert x["excluded_counts"]["not_directly_afc_supported_under_trajectory_envelope"] == 1


def test_lambda_is_applied_before_consensus():
    x = build_consensus(proposal(), afc_frontier(), lambda_event=9.0, require_direct_afc_support=True)
    assert x["consensus_event_update_count"] == 0
    assert x["excluded_counts"]["below_event_complexity"] == 3


def test_propagated_event_can_be_retained_only_when_explicitly_allowed():
    x = build_consensus(proposal(), afc_frontier(), lambda_event=0.0, require_direct_afc_support=False)
    assert {u["trajectory_id"] for u in x["updates"]} == {"a", "c"}
