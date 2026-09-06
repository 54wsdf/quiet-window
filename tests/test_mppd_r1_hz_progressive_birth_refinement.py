from __future__ import annotations

import scripts.mppd_r1_hz_progressive_birth_refinement as p


def tr(tid: str, line: str, direction: str, t: float, evidence: float = 1.0, support: int = 8):
    station = p.LINE_ANCHOR_STATION[line]
    return {
        "trajectory_id": tid,
        "afc_line": line,
        "path_id": f"{line}_main",
        "direction": direction,
        "reference_time_s": t,
        "evidence_score": evidence,
        "support_station_count": support,
        "support_weight": 10.0,
        "events": [{"station": station, "sequence_index": 0, "time_s": t}],
    }


def world(rows):
    return {
        "schema": "x",
        "semantics": {
            "train_count_is_input": False,
            "planned_timetable_used": False,
            "legacy_candidate_roots_used_as_input": False,
        },
        "inferred_service_trajectory_count": len(rows),
        "trajectories": rows,
    }


def test_five_second_candidate_near_parent_is_not_a_birth():
    parent = world([tr("p", "A", "Down", 100.0)])
    fine = world([tr("f", "A", "Down", 110.0, 2.0)])
    births, excluded = p.select_residual_births(parent, fine, novelty_radius_s=30.0, minimum_evidence_score=0.6)
    assert births == []
    assert excluded["explained_by_parent_service"] == 1


def test_distinct_fine_candidate_can_birth_without_fixing_count():
    parent = world([tr("p", "A", "Down", 100.0)])
    fine = world([tr("f", "A", "Down", 170.0, 1.3)])
    births, excluded = p.select_residual_births(parent, fine, novelty_radius_s=30.0, minimum_evidence_score=0.8)
    assert len(births) == 1
    out = p.build_augmented_world(parent, births, novelty_radius_s=30.0, minimum_evidence_score=0.8, minimum_support_stations=6)
    assert out["inferred_service_trajectory_count"] == 2
    assert out["semantics"]["train_count_is_input"] is False
    assert out["semantics"]["service_trajectory_count_is_inferred"] is True
    assert out["semantics"]["five_second_grid_is_time_resolution_not_minimum_train_headway"] is True


def test_stronger_birth_wins_local_fine_candidate_competition():
    parent = world([tr("p", "A", "Down", 100.0)])
    fine = world([
        tr("weak", "A", "Down", 200.0, 0.9),
        tr("strong", "A", "Down", 210.0, 1.4),
    ])
    births, excluded = p.select_residual_births(parent, fine, novelty_radius_s=30.0, minimum_evidence_score=0.6)
    assert len(births) == 1
    assert births[0]["trajectory_id"] == "strong"
    assert excluded["competes_with_stronger_birth"] == 1
