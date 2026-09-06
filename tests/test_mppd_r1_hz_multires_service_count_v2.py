from __future__ import annotations

import numpy as np

import scripts.mppd_r1_hz_multires_service_count_v2 as mr2


def candidate(path_id: str, direction: str, events: list[str], score: float = 2.0):
    return {
        "path_id": path_id,
        "afc_line": "B",
        "direction": direction,
        "reference_time_s": 100.0,
        "event_ids": events,
        "evidence_score": score,
        "station_count": len(events),
        "event_count": len(events),
        "support_weight": 10.0,
        "objective_gain": 1.0,
        "residual_median_abs_s": 1.0,
        "residual_p90_abs_s": 2.0,
    }


def test_v2_shared_event_competition_collapses_path_rivals():
    a = candidate("B_main", "Up", ["0@100", "1@200", "2@300"])
    b = candidate("B_branch", "Up", ["0@100", "1@200", "2@300"], score=1.8)
    out, ops = mr2.mr.base.deduplicate_path_candidates([a, b])
    assert len(out) == 1
    assert ops["merge_shared_event_duplicate"] == 1
    assert out[0]["path_ambiguous"] is True


def test_v2_level_declares_count_inferred_for_passenger_engine():
    counts = np.zeros((81, 300), dtype=np.uint32)
    result = mr2.discover_level_with_v2_contract(
        counts,
        mr2.mr.resolution_configs()[-1],
        iterations=1,
    )
    sem = result["semantics"]
    assert sem["train_count_is_input"] is False
    assert sem["service_trajectory_count_is_inferred"] is True
    assert sem["shared_event_representation_competition"] is True
    assert sem["shared_event_overlap_threshold"] == 0.60
