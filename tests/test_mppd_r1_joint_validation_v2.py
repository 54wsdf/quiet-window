from scripts.mppd_r1_joint_validation_v2 import NETWORK_AUTHORITY_SCHEMA, validate


def fine_state():
    return {
        "schema": "mppd.r1-joint-state.v2",
        "stage": "FINE_1S",
        "iteration": 8,
        "config": {
            "train_count_is_input": False,
            "planned_timetable_used_in_primary_inference": False,
            "legacy_candidate_roots_used_as_input": False,
            "access_min_s": 15.0,
            "egress_min_s": 15.0,
            "transfer_min_s": 5.0,
            "event_trust_region_s": 5.0,
        },
        "services": [
            {
                "trajectory_id": "q1",
                "line_id": "A",
                "direction_id": "Down",
                "path_id": "A_main",
                "events": [
                    {"station_id": "1", "sequence_index": 0, "anchor_time_s": 100.0, "arrival_time_s": 99.0, "departure_time_s": 101.0},
                    {"station_id": "2", "sequence_index": 1, "anchor_time_s": 160.0, "arrival_time_s": 159.0, "departure_time_s": 161.0},
                ],
            }
        ],
        "station_movements": [
            {"station_id": "1", "identification": "DIRECTLY_IDENTIFIED", "access_q05_s": 15.0, "access_q50_s": 25.0, "access_q95_s": 50.0, "egress_q05_s": 20.0, "egress_q50_s": 40.0, "egress_q95_s": 80.0},
            {"station_id": "2", "identification": "DIRECTLY_IDENTIFIED", "access_q05_s": 16.0, "access_q50_s": 26.0, "access_q95_s": 52.0, "egress_q05_s": 21.0, "egress_q50_s": 42.0, "egress_q95_s": 82.0},
        ],
        "transfer_movements": [],
        "passenger_summary": {
            "passenger_mass": 100.0,
            "resolved_mass": 90.0,
            "max_time_closure_error_s": 0.0,
        },
        "metadata": {"service_count_is_fixed": False},
    }


def authority():
    return {
        "schema": NETWORK_AUTHORITY_SCHEMA,
        "station_ids": ["1", "2"],
        "line_paths": [
            {"path_id": "A_main", "line_id": "A", "direction_id": "Down", "station_ids": ["1", "2"]}
        ],
        "transfer_movements": [],
    }


def test_count_free_network_authority_can_qualify_without_service_event_inventory():
    r = validate(fine_state(), authority())
    assert r["status"] == "QUALIFIED_R1_JOINT_RECONSTRUCTION_V2"
    assert r["count_free"]["service_count_is_authority_input"] is False
    assert r["count_free"]["inferred_service_count"] == 1


def test_authority_is_not_allowed_to_leak_service_count_or_event_keys():
    a = authority()
    a["expected_service_count"] = 1
    a["service_event_keys"] = [{"service_id": "q1", "station_id": "1"}]
    r = validate(fine_state(), a)
    assert r["status"] != "QUALIFIED_R1_JOINT_RECONSTRUCTION_V2"
    assert r["violation_counts"]["authority_service_leakage"] == 2


def test_transfer_path_count_is_inferred_and_one_path_is_not_automatically_invalid():
    a = authority()
    a["transfer_movements"] = [
        {"station_id": "2", "from_line_id": "A", "from_direction_id": "Down", "to_line_id": "B", "to_direction_id": "Up"}
    ]
    s = fine_state()
    s["transfer_movements"] = [
        {
            "movement_id": "A:2->B:2",
            "station_id": "2",
            "from_line_id": "A",
            "from_direction_id": "Down",
            "to_line_id": "B",
            "to_direction_id": "Up",
            "paths": [{"path_id": "latent:0", "q05_s": 6.0, "q50_s": 25.0, "q95_s": 70.0, "weight": 1.0}],
        }
    ]
    r = validate(s, a)
    assert r["status"] == "QUALIFIED_R1_JOINT_RECONSTRUCTION_V2"
    assert r["coverage"]["transfer_path_count"] == 1


def test_coarse_state_is_valid_working_state_but_not_final_qualification():
    s = fine_state()
    s["stage"] = "COARSE_5S"
    for e in s["services"][0]["events"]:
        e["arrival_time_s"] = None
        e["departure_time_s"] = None
    r = validate(s, authority())
    assert r["status"] != "QUALIFIED_R1_JOINT_RECONSTRUCTION_V2"
    assert r["violation_counts"]["final_resolution"] == 1
