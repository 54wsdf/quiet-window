import pytest

from scripts.mppd_r1_joint_model import (
    FINE_STAGE,
    JOINT_STATE_SCHEMA,
    JointState,
    R1Config,
    ServiceCirculationState,
    ServiceEventState,
    ServiceTrajectoryState,
)
from scripts.mppd_r1_rttp_projection import (
    circulation_annotations_from_joint_state,
    joint_state_to_rttp_provider_payload,
)


def service(tid, direction, stations, start):
    return ServiceTrajectoryState(
        trajectory_id=tid,
        line_id="A",
        direction_id=direction,
        path_id="A_main",
        events=[
            ServiceEventState(
                stations[0],
                0,
                start,
                arrival_time_s=start,
                departure_time_s=start + 5,
            ),
            ServiceEventState(
                stations[1],
                1,
                start + 60,
                arrival_time_s=start + 60,
                departure_time_s=start + 65,
            ),
        ],
    )


def state():
    return JointState(
        schema=JOINT_STATE_SCHEMA,
        stage=FINE_STAGE,
        iteration=4,
        config=R1Config(),
        services=[
            service("d1", "Down", ("1", "2"), 100),
            service("u1", "Up", ("2", "1"), 200),
            service("d2", "Down", ("1", "2"), 300),
        ],
        service_circulations=[
            ServiceCirculationState(
                circulation_id="chain-1",
                ordered_trajectory_ids=["d1", "u1", "d2"],
                chain_type="TURNBACK",
                start_boundary="DEPOT_EXIT",
                end_boundary="DEPOT_ENTRY",
                physical_vehicle_identity_claimed=False,
            )
        ],
    )


def test_turnback_chain_projects_membership_and_boundary_semantics_only():
    annotations = circulation_annotations_from_joint_state(state())
    assert annotations == {
        "d1": {"circulation_id": "chain-1", "origin_stop_type": "depot_exit", "terminal_stop_type": "turnback"},
        "u1": {"circulation_id": "chain-1", "terminal_stop_type": "turnback"},
        "d2": {"circulation_id": "chain-1", "terminal_stop_type": "depot_entry"},
    }
    assert all("vehicle_id" not in row for row in annotations.values())
    assert all("train_no_display" not in row for row in annotations.values())


def test_high_level_provider_payload_automatically_carries_inferred_circulation():
    payload = joint_state_to_rttp_provider_payload(state())
    assert payload["semantics"]["service_circulation_count"] == 1
    assert payload["semantics"]["service_circulation_projected"] is True
    assert payload["semantics"]["physical_vehicle_identity_claimed"] is False
    by_id = {row["trajectory_id"]: row for row in payload["services"]}
    assert by_id["d1"]["circulation_id"] == "chain-1"
    assert by_id["d1"]["origin_stop_type"] == "depot_exit"
    assert by_id["u1"]["terminal_stop_type"] == "turnback"
    assert by_id["d2"]["terminal_stop_type"] == "depot_entry"


def test_projection_rejects_physical_vehicle_claim_even_for_raw_mapping_state():
    raw = state().to_dict()
    raw["service_circulations"][0]["physical_vehicle_identity_claimed"] = True
    with pytest.raises(ValueError, match="forbids physical vehicle identity"):
        circulation_annotations_from_joint_state(raw)


def test_projection_rejects_same_service_in_two_circulations():
    raw = state().to_dict()
    raw["service_circulations"].append(
        {
            "circulation_id": "chain-2",
            "ordered_trajectory_ids": ["d1"],
            "chain_type": "TURNBACK",
            "start_boundary": "UNRESOLVED",
            "end_boundary": "UNRESOLVED",
            "physical_vehicle_identity_claimed": False,
        }
    )
    with pytest.raises(ValueError, match="multiple service circulations"):
        circulation_annotations_from_joint_state(raw)


def test_through_running_chain_does_not_falsely_label_intermediate_trip_as_turnback():
    raw = state().to_dict()
    raw["service_circulations"][0]["chain_type"] = "THROUGH_RUNNING"
    raw["service_circulations"][0]["start_boundary"] = "UNRESOLVED"
    raw["service_circulations"][0]["end_boundary"] = "UNRESOLVED"
    annotations = circulation_annotations_from_joint_state(raw)
    assert annotations["d1"] == {"circulation_id": "chain-1"}
    assert annotations["u1"] == {"circulation_id": "chain-1"}
    assert annotations["d2"] == {"circulation_id": "chain-1"}
