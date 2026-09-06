import pytest

from scripts.mppd_r1_joint_model import (
    COARSE_STAGE,
    FINE_STAGE,
    JOINT_STATE_SCHEMA,
    JointState,
    R1Config,
    ServiceCirculationState,
    ServiceEventState,
    ServiceTrajectoryState,
)
from scripts.mppd_r1_joint_solver import initialize_fine_arrival_departure
from scripts.mppd_r1_service_circulation import (
    INFERENCE_METHOD,
    TurnbackRule,
    infer_service_circulations,
    turnback_rules_from_network_authority,
)


def service(tid, direction, start, end):
    stations = ("1", "2") if direction == "Down" else ("2", "1")
    return ServiceTrajectoryState(
        trajectory_id=tid,
        line_id="A",
        direction_id=direction,
        path_id="A_main",
        events=[
            ServiceEventState(stations[0], 0, float(start)),
            ServiceEventState(stations[1], 1, float(end)),
        ],
    )


def state():
    return JointState(
        schema=JOINT_STATE_SCHEMA,
        stage=COARSE_STAGE,
        iteration=0,
        config=R1Config(),
        services=[
            service("d1", "Down", 0, 100),
            service("u1", "Up", 140, 240),
            service("d2", "Down", 280, 380),
            service("u2", "Up", 430, 530),
            service("d-orphan", "Down", 900, 1000),
        ],
    )


def authority():
    return {
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
        "semantics": {
            "contains_service_event_inventory": False,
            "contains_expected_service_count": False,
            "contains_planned_timetable": False,
            "service_count_must_be_inferred": True,
        },
    }


def rules():
    return turnback_rules_from_network_authority(
        authority(), min_turnback_s=20, max_turnback_s=80
    )


def test_turnback_rules_use_network_geometry_without_planned_service_answers():
    rows = rules()
    assert len(rows) == 2
    assert {(x.from_direction_id, x.to_direction_id, x.station_id) for x in rows} == {
        ("Down", "Up", "2"),
        ("Up", "Down", "1"),
    }
    assert all(x.min_turnback_s == 20 for x in rows)
    assert all(x.max_turnback_s == 80 for x in rows)


def test_turnback_rule_derivation_rejects_planned_or_count_contamination():
    doc = authority()
    doc["semantics"]["contains_planned_timetable"] = True
    with pytest.raises(ValueError, match="plan-free"):
        turnback_rules_from_network_authority(doc, min_turnback_s=20, max_turnback_s=80)
    doc = authority()
    doc["semantics"]["contains_expected_service_count"] = True
    with pytest.raises(ValueError, match="expected service count"):
        turnback_rules_from_network_authority(doc, min_turnback_s=20, max_turnback_s=80)


def test_turnback_rule_requires_explicit_valid_physical_window():
    with pytest.raises(ValueError, match="non-negative"):
        TurnbackRule("r", "A_main", "A", "Down", "Up", "2", -1, 80)
    with pytest.raises(ValueError, match=">="):
        TurnbackRule("r", "A_main", "A", "Down", "Up", "2", 90, 80)


def test_count_free_inference_builds_long_alternating_chain_and_singleton_without_vehicle_claim():
    out, audit = infer_service_circulations(state(), rules())
    assert out.validate() == []
    chains = [row.ordered_trajectory_ids for row in out.service_circulations]
    assert ["d1", "u1", "d2", "u2"] in chains
    assert ["d-orphan"] in chains
    assert audit.service_count == 5
    assert audit.matched_link_count == 3
    assert audit.circulation_count == 2
    assert audit.singleton_circulation_count == 1
    assert audit.physical_vehicle_identity_claimed is False
    assert audit.planned_timetable_used is False
    assert out.metadata["service_circulation_method"] == INFERENCE_METHOD
    assert out.metadata["physical_vehicle_identity_claimed"] is False


def test_inference_does_not_change_service_count_or_event_times():
    base = state()
    before = base.to_dict()
    out, _ = infer_service_circulations(base, rules())
    assert out.inferred_service_count == base.inferred_service_count
    assert [s.to_dict() if hasattr(s, "to_dict") else None for s in []] == []
    assert [
        [(e.station_id, e.anchor_time_s) for e in s.events] for s in out.services
    ] == [
        [(e.station_id, e.anchor_time_s) for e in s.events] for s in base.services
    ]
    assert before["services"] == out.to_dict()["services"]


def test_joint_state_rejects_one_trajectory_in_multiple_circulations():
    base = state()
    base.service_circulations = [
        ServiceCirculationState("c1", ["d1", "u1"]),
        ServiceCirculationState("c2", ["u1", "d2"]),
    ]
    assert any("appears in both" in error for error in base.validate())


def test_joint_state_rejects_turnback_chain_that_keeps_direction():
    base = state()
    base.service_circulations = [ServiceCirculationState("bad", ["d1", "d2"])]
    errors = base.validate()
    assert any("successor station mismatch" in error for error in errors)
    assert any("turnback successor keeps direction" in error for error in errors)


def test_joint_state_rejects_physical_vehicle_identity_claim_at_service_chain_layer():
    base = state()
    base.service_circulations = [
        ServiceCirculationState(
            "bad", ["d1"], physical_vehicle_identity_claimed=True
        )
    ]
    assert any("may not claim physical vehicle identity" in error for error in base.validate())


def test_fine_stage_matching_uses_terminal_arrival_to_next_origin_departure():
    fine = initialize_fine_arrival_departure(state(), initial_dwell_s=5)
    assert fine.stage == FINE_STAGE
    out, audit = infer_service_circulations(fine, rules())
    assert audit.matched_link_count == 3
    assert out.validate() == []
