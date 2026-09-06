import pytest

from scripts.mppd_r1_rttp_bridge import (
    COUNT_FREE,
    FINE_STAGE,
    PLAN_ANCHORED,
    RTTP_PROVIDER_CONTRACT,
    RTTP_PROVIDER_REPOSITORY,
    RTTP_PROVIDER_SHA,
    joint_state_to_provider_payload,
    network_authority_to_provider_payload,
    normalize_r1_operations_for_rttp,
    plan_anchored_reference_contract,
    provider_lock,
    verify_observed_provider_sha,
)


def authority():
    return {
        "dataset_id": "CN_HZ_Tianchi_2019",
        "authority_role": "NETWORK_AND_TRANSFER_PHYSICAL_DOMAIN_ONLY",
        "line_paths": [
            {
                "path_id": "A_main",
                "line_id": "A",
                "direction_id": "Down",
                "station_ids": ["1", "2", "3"],
            },
            {
                "path_id": "A_main",
                "line_id": "A",
                "direction_id": "Up",
                "station_ids": ["3", "2", "1"],
            },
        ],
        "semantics": {
            "contains_service_event_inventory": False,
            "contains_expected_service_count": False,
            "contains_planned_timetable": False,
            "service_count_must_be_inferred": True,
            "transfer_path_component_count_must_be_inferred": True,
        },
    }


def fine_state():
    return {
        "schema": "mppd.r1-joint-state.v2",
        "stage": FINE_STAGE,
        "iteration": 3,
        "config": {
            "train_count_is_input": False,
            "planned_timetable_used_in_primary_inference": False,
            "planned_timetable_role": "POST_HOC_EXTERNAL_COMPARISON_ONLY",
        },
        "services": [
            {
                "trajectory_id": "q1",
                "line_id": "A",
                "direction_id": "Down",
                "path_id": "A_main",
                "events": [
                    {
                        "station_id": "1",
                        "sequence_index": 0,
                        "anchor_time_s": 102.5,
                        "arrival_time_s": 100,
                        "departure_time_s": 110,
                        "evidence_class": "AFC_PASSENGER_FACING_RIDGE",
                    },
                    {
                        "station_id": "2",
                        "sequence_index": 1,
                        "anchor_time_s": 172.5,
                        "arrival_time_s": 170,
                        "departure_time_s": 180,
                        "evidence_class": "AFC_PASSENGER_FACING_RIDGE",
                    },
                ],
            }
        ],
    }


def test_provider_lock_is_exact_and_contract_versioned():
    lock = provider_lock()
    assert lock["repository"] == RTTP_PROVIDER_REPOSITORY
    assert lock["commit_sha"] == RTTP_PROVIDER_SHA
    assert len(lock["commit_sha"]) == 40
    assert lock["provider_contract"] == RTTP_PROVIDER_CONTRACT
    assert lock["module"] == "rttp.integrations.mppd_r1"


def test_provider_sha_verification_fails_closed():
    verify_observed_provider_sha(RTTP_PROVIDER_SHA)
    with pytest.raises(ValueError, match="provider SHA mismatch"):
        verify_observed_provider_sha("0" * 40)


def test_network_authority_projection_has_zero_service_answer_information():
    payload = network_authority_to_provider_payload(
        authority(),
        network_authority_ref="quiet-window:network-authority:test",
    )
    assert payload["provider_contract"] == RTTP_PROVIDER_CONTRACT
    assert len(payload["routes"]) == 2
    assert payload["semantics"] == {
        "canonical_timetable_materialized": False,
        "service_pattern_materialized": False,
        "service_instance_count_materialized": False,
        "physical_route_structure_only": True,
    }
    assert payload["metadata"]["planned_absolute_times_used"] is False
    assert payload["metadata"]["planned_trip_count_used"] is False
    assert payload["metadata"]["planned_trip_ids_used"] is False
    assert all("time" not in item for item in payload["routes"])
    assert all("trip" not in item for item in payload["routes"])


def test_network_authority_projection_rejects_plan_or_count_contamination():
    doc = authority()
    doc["semantics"]["contains_planned_timetable"] = True
    with pytest.raises(ValueError, match="not count-free structural authority"):
        network_authority_to_provider_payload(
            doc,
            network_authority_ref="quiet-window:network-authority:test",
        )
    doc = authority()
    doc["semantics"]["contains_expected_service_count"] = True
    with pytest.raises(ValueError, match="not count-free structural authority"):
        network_authority_to_provider_payload(
            doc,
            network_authority_ref="quiet-window:network-authority:test",
        )


def test_coarse_joint_state_cannot_be_exported_as_canonical_arrival_departure():
    state = fine_state()
    state["stage"] = "COARSE_5S"
    with pytest.raises(ValueError, match="coarse AFC ridge anchors stay noncanonical"):
        joint_state_to_provider_payload(state)


def test_fine_joint_state_exports_arrival_departure_but_never_coarse_anchor_as_event():
    payload = joint_state_to_provider_payload(fine_state())
    assert payload["mode"] == COUNT_FREE
    assert payload["stage"] == FINE_STAGE
    event = payload["services"][0]["events"][0]
    assert event["arrival_time_s"] == 100
    assert event["departure_time_s"] == 110
    assert "anchor_time_s" not in event
    assert payload["semantics"]["coarse_anchor_time_exported_as_canonical_event"] is False
    assert payload["semantics"]["service_count_is_inferred"] is True


def test_fine_joint_state_rejects_subsecond_canonical_event():
    state = fine_state()
    state["services"][0]["events"][0]["arrival_time_s"] = 100.5
    with pytest.raises(ValueError, match="1-second grid"):
        joint_state_to_provider_payload(state)


def test_circulation_annotation_carries_chain_identity_without_vehicle_identity():
    payload = joint_state_to_provider_payload(
        fine_state(),
        service_annotations={
            "q1": {
                "circulation_id": "inferred-chain-7",
                "terminal_stop_type": "turnback",
            }
        },
    )
    service = payload["services"][0]
    assert service["circulation_id"] == "inferred-chain-7"
    assert service["terminal_stop_type"] == "turnback"
    assert "vehicle_id" not in service
    assert payload["semantics"]["physical_vehicle_identity_claimed"] is False


def test_service_annotations_fail_closed_on_physical_or_planned_identity_claims():
    with pytest.raises(ValueError, match="forbidden/noncanonical fields"):
        joint_state_to_provider_payload(
            fine_state(),
            service_annotations={"q1": {"vehicle_id": "trainset-1"}},
        )
    with pytest.raises(ValueError, match="forbidden/noncanonical fields"):
        joint_state_to_provider_payload(
            fine_state(),
            service_annotations={"q1": {"train_no_display": "001"}},
        )


def test_split_merge_require_explicit_lowering_before_rttp_handoff():
    with pytest.raises(ValueError, match="requires explicit lowered_operations"):
        normalize_r1_operations_for_rttp(
            [{"operation_id": "s1", "operation_type": "split"}]
        )
    lowered = normalize_r1_operations_for_rttp(
        [
            {
                "operation_id": "s1",
                "operation_type": "split",
                "lowered_operations": [
                    {
                        "operation_id": "kill-parent",
                        "operation_type": "death",
                        "target_object_ids": ["mppd-r1:q1"],
                        "payload": {},
                    },
                    {
                        "operation_id": "birth-child",
                        "operation_type": "birth",
                        "target_object_ids": [],
                        "payload": {"trip": {}, "stop_events": []},
                    },
                ],
            }
        ]
    )
    assert [item["operation_type"] for item in lowered] == ["death", "birth"]


def test_plan_anchored_route_is_explicitly_separate_from_count_free():
    contract = plan_anchored_reference_contract()
    assert contract["mode"] == PLAN_ANCHORED
    assert contract["planned_timetable_may_be_input"] is True
    assert contract["planned_trip_count_may_be_input"] is True
    assert contract["must_not_feed_count_free_route"] is True
