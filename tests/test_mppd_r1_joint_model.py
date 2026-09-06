import pytest

from scripts.mppd_r1_joint_model import (
    COARSE_STAGE,
    FINE_STAGE,
    JOINT_STATE_SCHEMA,
    JointState,
    R1Config,
    ServiceEventState,
    ServiceTrajectoryState,
    bootstrap_from_count_free_discovery,
)


def sample_service():
    return ServiceTrajectoryState(
        trajectory_id="q1",
        line_id="A",
        direction_id="Down",
        path_id="A_main",
        events=[
            ServiceEventState("1", 0, 100.0),
            ServiceEventState("2", 1, 160.0),
        ],
    )


def test_new_r1_contract_is_count_free_and_excludes_planned_absolute_timetable():
    cfg = R1Config()
    cfg.validate()
    assert cfg.train_count_is_input is False
    assert cfg.planned_timetable_used_in_primary_inference is False
    assert cfg.legacy_candidate_roots_used_as_input is False
    assert cfg.access_min_s == 15.0
    assert cfg.egress_min_s == 15.0
    assert cfg.transfer_min_s == 5.0


def test_fixed_train_count_semantics_are_rejected():
    with pytest.raises(ValueError, match="count-free"):
        R1Config(train_count_is_input=True).validate()


def test_coarse_state_accepts_ridge_anchors_but_fine_requires_arrival_departure():
    coarse = JointState(
        schema=JOINT_STATE_SCHEMA,
        stage=COARSE_STAGE,
        iteration=0,
        config=R1Config(),
        services=[sample_service()],
    )
    assert coarse.validate() == []
    coarse.stage = FINE_STAGE
    assert any("separate arrival/departure" in x for x in coarse.validate())


def test_count_free_discovery_is_only_a_warm_start_and_count_remains_free():
    doc = {
        "schema": "mppd.r1-hz-count-free-service-discovery.v2",
        "semantics": {
            "train_count_is_input": False,
            "planned_timetable_used": False,
            "legacy_candidate_roots_used_as_input": False,
        },
        "legacy_baseline": {"legacy_candidate_root_count": 1673},
        "trajectories": [
            {
                "trajectory_id": "cf:a",
                "afc_line": "A",
                "direction": "Down",
                "path_id": "A_main",
                "path_ambiguous": False,
                "direction_ambiguous": False,
                "support_weight": 3.0,
                "evidence_score": 1.2,
                "events": [
                    {"station": 1, "sequence_index": 0, "time_s": 100.0},
                    {"station": 2, "sequence_index": 1, "time_s": 180.0},
                ],
            },
            {
                "trajectory_id": "cf:b",
                "afc_line": "A",
                "direction": "Down",
                "path_id": "A_main",
                "path_ambiguous": False,
                "direction_ambiguous": False,
                "support_weight": 2.0,
                "evidence_score": 1.0,
                "events": [
                    {"station": 1, "sequence_index": 0, "time_s": 300.0},
                    {"station": 2, "sequence_index": 1, "time_s": 380.0},
                ],
            },
        ],
    }
    state = bootstrap_from_count_free_discovery(doc)
    assert state.inferred_service_count == 2
    assert state.metadata["warm_start_only"] is True
    assert state.metadata["service_count_is_free_after_bootstrap"] is True
    assert state.metadata["legacy_baseline"]["legacy_candidate_root_count"] == 1673
    assert state.validate() == []
