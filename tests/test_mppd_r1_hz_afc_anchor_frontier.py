from scripts.mppd_r1_hz_afc_anchor_frontier import aggregate, audit_updates, nearest_event
import scripts.mppd_r1_hz_count_free_service as afc


def services():
    return {
        "inferred_service_trajectory_count": 1,
        "trajectories": [
            {
                "trajectory_id": "q1",
                "path_id": "A_main",
                "direction": "Down",
                "path_ambiguous": False,
                "residual_p90_abs_s": 20.0,
                "events": [
                    {"station": 1, "sequence_index": 0, "time_s": 100.0, "station_phase_nuisance_s": 10.0},
                    {"station": 2, "sequence_index": 1, "time_s": 160.0, "station_phase_nuisance_s": 10.0},
                ],
            }
        ],
    }


def proposal(delta=-5.0, net=6.0):
    return {
        "updates": [
            {
                "trajectory_id": "q1",
                "station_id": 2,
                "delta_s": delta,
                "direct_net_mass": net,
                "direct_rescue_mass": net + 1,
                "direct_damage_mass": 1.0,
            }
        ]
    }


def event(station, center, score=2.0, excess=9.0):
    return afc.Event(f"{station}@{center}", station, float(center), float(score), float(excess))


def test_nearest_event_is_deterministic():
    e, d = nearest_event([event(2, 160), event(2, 180)], 171)
    assert e is not None
    assert e.center_s == 180.0
    assert d == 9.0


def test_minus_five_can_improve_raw_afc_anchor_distance_after_station_phase():
    # Predicted passenger pulse is event_time 160 + station phase 10 = 170.
    # Moving the service event -5 moves the predicted pulse to 165, closer to AFC pulse 164.
    rows = audit_updates(services(), proposal(-5), {2: [event(2, 164)]})
    r = rows[0]
    assert r["nearest_afc_pulse_before_distance_s"] == 6.0
    assert r["nearest_afc_pulse_after_distance_s"] == 1.0
    assert r["afc_nearest_distance_change_s"] == -5.0


def test_lambda_aggregate_uses_same_nested_direct_net_rule_as_passenger_frontier():
    rows = audit_updates(services(), proposal(-5, net=6), {2: [event(2, 164)]})
    a0 = aggregate(rows, 0.0)
    a6 = aggregate(rows, 6.0)
    assert a0["modified_event_count"] == 1
    assert a0["afc_distance_improved_count"] == 1
    assert a6["modified_event_count"] == 0
