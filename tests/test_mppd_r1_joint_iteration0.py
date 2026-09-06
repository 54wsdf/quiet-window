import pytest

from scripts.mppd_r1_joint_iteration0 import assemble_iteration0


def warm_state():
    return {
        "schema": "mppd.r1-joint-state.v2",
        "stage": "COARSE_5S",
        "iteration": 0,
        "config": {
            "schema": "mppd.r1-joint-config.v2",
            "train_count_is_input": False,
            "planned_timetable_used_in_primary_inference": False,
            "legacy_candidate_roots_used_as_input": False,
            "access_min_s": 15.0,
            "egress_min_s": 15.0,
            "transfer_min_s": 5.0,
            "event_trust_region_s": 5.0,
            "coarse_update_step_s": 5.0,
            "fine_update_step_s": 1.0,
            "planned_timetable_role": "POST_HOC_EXTERNAL_COMPARISON_ONLY",
            "legacy_1673_role": "HISTORICAL_BASELINE_ONLY",
            "legacy_plus_60_clock_role": "DIAGNOSTIC_ONLY_NOT_ADMISSIBLE_STATE_UPDATE",
        },
        "services": [
            {
                "trajectory_id": "q1",
                "line_id": "A",
                "direction_id": "Down",
                "path_id": "A_main",
                "events": [
                    {"station_id": "1", "sequence_index": 0, "anchor_time_s": 100.0, "arrival_time_s": None, "departure_time_s": None, "evidence_class": "AFC_PASSENGER_FACING_RIDGE"},
                    {"station_id": "2", "sequence_index": 1, "anchor_time_s": 160.0, "arrival_time_s": None, "departure_time_s": None, "evidence_class": "AFC_PASSENGER_FACING_RIDGE"},
                ],
            }
        ],
        "metadata": {"warm_start_only": True},
    }


def movement_fit(shift=0.0):
    def x(q05, med, q95, evidence):
        return {"q05_s": q05, "median_s": med, "q95_s": q95, "evidence_mass": evidence, "identified_from_current_day": evidence > 0}
    return {
        "service_clock_shift_s": shift,
        "service_trajectory_count": 1,
        "access_hazard": 0.9,
        "transfer_hazard": 0.8,
        "access_intervals": {"1": x(15.5, 25, 50, 10), "2": x(16, 26, 52, 0)},
        "egress_intervals": {"1": x(20, 40, 80, 10), "2": x(21, 42, 82, 0)},
        "transfer_path_intervals": {"A:1->B:1": {"paths": []}},
    }


def passenger_feedback():
    return {
        "count_free_service_trajectory_count": 1,
        "passenger_mass": 100.0,
        "resolved_mass": 60.0,
        "unresolved_route_exists_mass": 35.0,
        "failure_mass": {"NO_ROUTE_SUPPORT": 5.0},
    }


def test_iteration0_keeps_unshifted_service_and_adds_station_passenger_state():
    s = assemble_iteration0(warm_state(), movement_fit(), passenger_feedback())
    assert s.inferred_service_count == 1
    assert len(s.station_movements) == 2
    assert s.station_movements[0].identification == "DIRECTLY_IDENTIFIED"
    assert s.station_movements[1].identification == "HIERARCHICALLY_INFERRED"
    assert s.passenger_summary.resolved_share == 0.6
    assert s.transfer_movements == []
    assert s.metadata["direction_specific_transfer_posterior_pending"] is True


def test_plus_60_or_any_nonzero_global_shift_is_rejected_from_iteration0():
    with pytest.raises(ValueError, match="unshifted"):
        assemble_iteration0(warm_state(), movement_fit(60.0), passenger_feedback())
