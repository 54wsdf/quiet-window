import json
from pathlib import Path

import pytest

from scripts.mppd_r1_rttp_bridge import (
    RTTP_PROVIDER_CONTRACT,
    RTTP_PROVIDER_REPOSITORY,
    RTTP_PROVIDER_SHA,
    joint_state_to_provider_payload,
    normalize_r1_operations_for_rttp,
)


def _fine_state():
    return {
        "stage": "FINE_1S",
        "config": {
            "train_count_is_input": False,
            "planned_timetable_used_in_primary_inference": False,
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
                        "arrival_time_s": 100,
                        "departure_time_s": 110,
                    },
                    {
                        "station_id": "2",
                        "sequence_index": 1,
                        "arrival_time_s": 170,
                        "departure_time_s": 180,
                    },
                ],
            }
        ],
    }


def test_provider_lock_file_matches_bridge_constants():
    lock = json.loads(
        Path("contracts/MPPD_R1_RTTP_PROVIDER_LOCK.json").read_text(encoding="utf-8")
    )
    provider = lock["provider"]
    assert provider["repository"] == RTTP_PROVIDER_REPOSITORY
    assert provider["commit_sha"] == RTTP_PROVIDER_SHA
    assert provider["contract_version"] == RTTP_PROVIDER_CONTRACT


def test_mapping_fine_state_cannot_bypass_interstation_time_validation():
    state = _fine_state()
    state["services"][0]["events"][1]["arrival_time_s"] = 105
    with pytest.raises(ValueError, match="non-positive interstation running time"):
        joint_state_to_provider_payload(state)


def test_split_lowering_rejects_non_mapping_members_instead_of_silently_dropping_them():
    with pytest.raises(ValueError, match="must contain only mappings"):
        normalize_r1_operations_for_rttp(
            [
                {
                    "operation_id": "split-1",
                    "operation_type": "split",
                    "lowered_operations": [
                        {
                            "operation_id": "death-parent",
                            "operation_type": "death",
                            "target_object_ids": ["mppd-r1:q1"],
                            "payload": {},
                        },
                        "not-an-operation",
                    ],
                }
            ]
        )
