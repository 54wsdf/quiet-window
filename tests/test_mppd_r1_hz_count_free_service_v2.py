from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.mppd_r1_hz_count_free_service_v2 as v2


def candidate(path_id, direction, reference_time, event_ids, evidence=1.5, stations=8):
    return {
        "path_id": path_id,
        "afc_line": "B",
        "direction": direction,
        "reference_time_s": reference_time,
        "station_count": stations,
        "event_count": len(event_ids),
        "event_ids": list(event_ids),
        "support_weight": 30.0,
        "evidence_score": evidence,
        "objective_gain": evidence - 0.6,
        "residual_p90_abs_s": 15.0,
        "residual_median_abs_s": 7.0,
    }


def test_reference_time_is_not_used_to_keep_main_branch_duplicates():
    shared = [f"{s}@{100+s*100}" for s in range(10)]
    main = candidate("B_main", "Up", 1000.0, shared + ["27@4000"], evidence=1.8, stations=11)
    branch = candidate("B_branch", "Up", 1700.0, shared + ["33@4700"], evidence=1.7, stations=11)
    final, ops = v2.deduplicate_shared_event_candidates([main, branch], overlap_threshold=0.60)
    assert len(final) == 1
    assert final[0]["path_ambiguous"] is True
    assert ops["merge_shared_event_duplicate"] == 1
    assert ops["path_reassignment_competition"] == 1


def test_opposite_direction_reuse_becomes_direction_competition_not_two_trains():
    shared = [f"{s}@{500+s*120}" for s in range(8)]
    down = candidate("B_main", "Down", 1000.0, shared, evidence=1.9, stations=8)
    up = candidate("B_main", "Up", 4000.0, shared, evidence=1.4, stations=8)
    final, ops = v2.deduplicate_shared_event_candidates([down, up], overlap_threshold=0.60)
    assert len(final) == 1
    assert final[0]["direction_ambiguous"] is True
    assert set(final[0]["direction_alternatives"]) == {"Down", "Up"}
    assert ops["direction_reassignment_competition"] == 1


def test_distinct_event_ridges_remain_distinct_services():
    a = candidate("B_main", "Up", 1000.0, [f"{s}@{100+s*100}" for s in range(8)])
    b = candidate("B_branch", "Up", 1700.0, [f"{s}@{200+s*100}" for s in range(8)])
    final, _ops = v2.deduplicate_shared_event_candidates([a, b], overlap_threshold=0.60)
    assert len(final) == 2
