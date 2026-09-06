from scripts.mppd_r1_joint_model import JOINT_STATE_SCHEMA
from scripts.mppd_r1_joint_validation_v2 import validate


def authority():
    return {
        "schema": "mppd.r1-network-authority.v2",
        "station_ids": ["1", "2"],
        "line_paths": [
            {
                "path_id": "A_main",
                "line_id": "A",
                "direction_id": "Down",
                "station_ids": ["1", "2"],
            },
            {
                "path_id": "A_main",
                "line_id": "A",
                "direction_id": "Up",
                "station_ids": ["2", "1"],
            },
        ],
        "transfer_movements": [],
    }


def state(with_circulation=True):
    doc = {
        "schema": JOINT_STATE_SCHEMA,
        "stage": "FINE_1S",
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
                "trajectory_id": "d1",
                "path_id": "A_main",
                "line_id": "A",
                "direction_id": "Down",
                "events": [
                    {
                        "station_id": "1",
                        "sequence_index": 0,
                        "anchor_time_s": 100.0,
                        "arrival_time_s": 100.0,
                        "departure_time_s": 105.0,
                    },
                    {
                        "station_id": "2",
                        "sequence_index": 1,
                        "anchor_time_s": 160.0,
                        "arrival_time_s": 160.0,
                        "departure_time_s": 165.0,
                    },
                ],
            },
            {
                "trajectory_id": "u1",
                "path_id": "A_main",
                "line_id": "A",
                "direction_id": "Up",
                "events": [
                    {
                        "station_id": "2",
                        "sequence_index": 0,
                        "anchor_time_s": 220.0,
                        "arrival_time_s": 220.0,
                        "departure_time_s": 225.0,
                    },
                    {
                        "station_id": "1",
                        "sequence_index": 1,
                        "anchor_time_s": 280.0,
                        "arrival_time_s": 280.0,
                        "departure_time_s": 285.0,
                    },
                ],
            },
        ],
        "service_circulations": [],
        "station_movements": [
            {
                "station_id": "1",
                "identification": "DIRECTLY_IDENTIFIED",
                "access_q05_s": 15.0,
                "access_q50_s": 20.0,
                "access_q95_s": 30.0,
                "egress_q05_s": 15.0,
                "egress_q50_s": 20.0,
                "egress_q95_s": 30.0,
            },
            {
                "station_id": "2",
                "identification": "DIRECTLY_IDENTIFIED",
                "access_q05_s": 15.0,
                "access_q50_s": 20.0,
                "access_q95_s": 30.0,
                "egress_q05_s": 15.0,
                "egress_q50_s": 20.0,
                "egress_q95_s": 30.0,
            },
        ],
        "transfer_movements": [],
        "passenger_summary": {
            "passenger_mass": 100.0,
            "resolved_mass": 80.0,
            "max_time_closure_error_s": 0.0,
        },
        "metadata": {
            "service_count_is_fixed": False,
            "physical_vehicle_identity_claimed": False,
        },
    }
    if with_circulation:
        doc["service_circulations"] = [
            {
                "circulation_id": "c1",
                "ordered_trajectory_ids": ["d1", "u1"],
                "chain_type": "TURNBACK",
                "start_boundary": "UNRESOLVED",
                "end_boundary": "UNRESOLVED",
                "physical_vehicle_identity_claimed": False,
            }
        ]
    return doc


def test_valid_service_circulation_is_audited_without_claiming_physical_fleet():
    report = validate(state(), authority())
    assert report["status"] == "QUALIFIED_R1_JOINT_RECONSTRUCTION_V2"
    assert report["violation_counts"] == {}
    assert report["count_free"]["physical_vehicle_identity_is_claimed"] is False
    assert report["coverage"]["service_circulation_count"] == 1
    assert report["coverage"]["service_circulation_membership_count"] == 2
    assert report["coverage"]["service_circulation_link_count"] == 1


def test_circulation_remains_optional_until_production_turnback_window_is_calibrated():
    report = validate(state(with_circulation=False), authority())
    assert report["status"] == "QUALIFIED_R1_JOINT_RECONSTRUCTION_V2"
    assert report["coverage"]["service_circulation_count"] == 0


def test_validator_rejects_physical_vehicle_identity_claim():
    doc = state()
    doc["service_circulations"][0]["physical_vehicle_identity_claimed"] = True
    doc["service_circulations"][0]["vehicle_id"] = "real-trainset-1"
    report = validate(doc, authority())
    assert report["status"] == "R1_JOINT_RECONSTRUCTION_V2_NOT_QUALIFIED"
    assert report["violation_counts"]["circulation_physical_identity"] == 2


def test_validator_rejects_bad_turnback_station_and_direction():
    doc = state()
    doc["services"][1]["direction_id"] = "Down"
    doc["services"][1]["events"][0]["station_id"] = "1"
    doc["services"][1]["events"][1]["station_id"] = "2"
    report = validate(doc, authority())
    assert report["status"] == "R1_JOINT_RECONSTRUCTION_V2_NOT_QUALIFIED"
    assert report["violation_counts"]["service_circulation"] >= 1
