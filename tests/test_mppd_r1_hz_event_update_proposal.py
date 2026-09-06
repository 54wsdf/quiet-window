from scripts.mppd_r1_hz_event_update_proposal import apply_proposal, build_proposal


def services(residual=20.0):
    return {
        "inferred_service_trajectory_count": 1,
        "semantics": {
            "train_count_is_input": False,
            "planned_timetable_used": False,
            "legacy_candidate_roots_used_as_input": False,
            "service_trajectory_count_is_inferred": True,
        },
        "trajectories": [
            {
                "trajectory_id": "q1",
                "afc_line": "A",
                "path_id": "A_main",
                "direction": "Down",
                "residual_p90_abs_s": residual,
                "events": [
                    {"station": 1, "sequence_index": 0, "time_s": 100.0},
                    {"station": 2, "sequence_index": 1, "time_s": 160.0},
                    {"station": 3, "sequence_index": 2, "time_s": 220.0},
                ],
            }
        ],
    }


def pressure(rescue_minus=3.0, risk_minus=0.0, rescue_plus=0.0, risk_plus=0.0):
    return {
        "service_trajectory_count": 1,
        "event_pressure_count": 1,
        "event_pressure": [
            {
                "trajectory_id": "q1",
                "station_id": 2,
                "anchor_time_s": 160.0,
                "path_ambiguous": False,
                "resolvable_if_minus5_mass": rescue_minus,
                "resolvable_if_plus5_mass": rescue_plus,
                "protect_against_minus5_mass": risk_minus,
                "protect_against_plus5_mass": risk_plus,
                "minus5_pressure_mass": 100.0,
                "plus5_pressure_mass": 0.0,
                "minus5_residual_reduction_mass_seconds": 500.0,
            }
        ],
    }


def test_direct_one_step_rescue_can_propose_minus_five():
    p = build_proposal(services(), pressure())
    assert p["proposed_event_update_count"] == 1
    assert p["updates"][0]["delta_s"] == -5.0
    out = apply_proposal(services(), p)
    assert out["trajectories"][0]["events"][1]["time_s"] == 155.0
    assert out["joint_local_update"]["applied_event_update_count"] == 1


def test_direct_damage_dominating_rescue_rejects_move():
    p = build_proposal(services(), pressure(rescue_minus=3.0, risk_minus=4.0))
    assert p["proposed_event_update_count"] == 0


def test_five_second_move_outside_afc_residual_envelope_is_rejected():
    p = build_proposal(services(residual=4.0), pressure())
    assert p["proposed_event_update_count"] == 0


def test_positive_plus_five_rescue_is_allowed_when_safe():
    p = build_proposal(services(), pressure(rescue_minus=0, risk_minus=0, rescue_plus=2, risk_plus=0))
    assert p["proposed_event_update_count"] == 1
    assert p["updates"][0]["delta_s"] == 5.0


def test_only_one_event_per_trajectory_is_selected():
    p0 = pressure()
    second = dict(p0["event_pressure"][0])
    second["station_id"] = 3
    second["anchor_time_s"] = 220.0
    second["resolvable_if_minus5_mass"] = 2.0
    p0["event_pressure"].append(second)
    p0["event_pressure_count"] = 2
    p = build_proposal(services(), p0)
    assert p["proposed_event_update_count"] == 1
    assert p["updates"][0]["station_id"] == 2
