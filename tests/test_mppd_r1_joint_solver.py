import pytest

from scripts.mppd_r1_joint_model import (
    COARSE_STAGE,
    JOINT_STATE_SCHEMA,
    JointState,
    R1Config,
    ServiceEventState,
    ServiceTrajectoryState,
)
from scripts.mppd_r1_joint_solver import (
    EventUpdate,
    IterationMetrics,
    allowed_event_deltas,
    apply_event_updates,
    coarse_structure_is_stable,
    initialize_fine_arrival_departure,
    validate_event_delta,
    validate_structure_operations,
)


def state():
    return JointState(
        schema=JOINT_STATE_SCHEMA,
        stage=COARSE_STAGE,
        iteration=0,
        config=R1Config(),
        services=[
            ServiceTrajectoryState(
                trajectory_id="q1",
                line_id="A",
                direction_id="Down",
                path_id="A_main",
                events=[
                    ServiceEventState("1", 0, 100.0),
                    ServiceEventState("2", 1, 160.0),
                    ServiceEventState("3", 2, 220.0),
                ],
            )
        ],
    )


def test_coarse_and_fine_resolution_are_distinct():
    assert allowed_event_deltas("COARSE_5S") == (-5.0, 0.0, 5.0)
    assert allowed_event_deltas("FINE_1S") == tuple(float(x) for x in range(-5, 6))
    validate_event_delta(5, "COARSE_5S")
    with pytest.raises(ValueError):
        validate_event_delta(1, "COARSE_5S")
    validate_event_delta(1, "FINE_1S")
    with pytest.raises(ValueError, match="trust region"):
        validate_event_delta(6, "FINE_1S")


def test_one_coarse_update_is_bounded_and_requires_posterior_reinference():
    s = apply_event_updates(
        state(),
        [EventUpdate("q1", "2", 1, 5.0, "anchor")],
    )
    assert s.services[0].events[1].anchor_time_s == 165.0
    assert s.metadata["last_update"]["posterior_reinference_required"] is True


def test_fine_stage_uses_one_second_updates_but_keeps_five_second_trust_region():
    s = initialize_fine_arrival_departure(state(), initial_dwell_s=0.0)
    s2 = apply_event_updates(
        s,
        [EventUpdate("q1", "2", 1, 2.0, "both")],
    )
    assert s2.services[0].events[1].arrival_time_s == 162.0
    assert s2.services[0].events[1].departure_time_s == 162.0


def test_fine_stage_cannot_hide_structure_changes():
    validate_structure_operations("COARSE_5S", ["birth", "split", "merge"])
    with pytest.raises(ValueError, match="return to coarse"):
        validate_structure_operations("FINE_1S", ["birth"])


def test_three_stable_coarse_iterations_can_open_fine_stage():
    history = [
        IterationMetrics(1, 1845, 0.985, 0.018, 4.0, 1000.0, 12),
        IterationMetrics(2, 1842, 0.989, 0.012, 3.0, 1001.0, 5),
        IterationMetrics(3, 1841, 0.992, 0.009, 2.0, 1001.5, 2),
    ]
    assert coarse_structure_is_stable(history)
