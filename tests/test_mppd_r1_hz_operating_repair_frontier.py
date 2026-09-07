from scripts.mppd_r1_hz_operating_authority import clock_to_service_seconds
from scripts.mppd_r1_hz_operating_repair_frontier import (
    _solve_weighted_vertex_cover,
    build_frontier,
)


A_NODES = [67, 68, 69, 70, 71, 72, 73, 74, 5, 75, 76, 77, 46, 78, 79, 80, 15, 16]


def trajectory(tid, start_s, evidence=1.0):
    return {
        "trajectory_id": tid,
        "afc_line": "A",
        "path_id": "A_main",
        "direction": "Down",
        "station_support": len(A_NODES),
        "support_weight": 10.0,
        "evidence_score": evidence,
        "path_ambiguous": False,
        "direction_ambiguous": False,
        "events": [
            {"station": station, "sequence_index": i, "time_s": start_s + i * 100.0}
            for i, station in enumerate(A_NODES)
        ],
    }


def discovery(rows):
    return {
        "schema": "mppd.r1-hz-count-free-service-discovery.v2",
        "inferred_service_trajectory_count": len(rows),
        "trajectories": rows,
        "semantics": {
            "train_count_is_input": False,
            "planned_timetable_used": False,
            "legacy_candidate_roots_used_as_input": False,
            "service_trajectory_count_is_inferred": True,
        },
    }


def passenger(assignments):
    return {"trajectory_assignment_mass": assignments}


def test_exact_cover_uses_one_middle_vertex_for_two_edge_path_when_costs_equal():
    rows = [
        {"trajectory_id": "a", "path_id": "A_main", "direction": "Down", "evidence_score": 1.0},
        {"trajectory_id": "b", "path_id": "A_main", "direction": "Down", "evidence_score": 1.0},
        {"trajectory_id": "c", "path_id": "A_main", "direction": "Down", "evidence_score": 1.0},
    ]
    selected, audit = _solve_weighted_vertex_cover(
        rows,
        [("a", "b"), ("b", "c")],
        assignment={},
        passenger_weight=0.0,
    )
    assert selected == {"b"}
    assert audit["selected_count"] == 1


def test_high_passenger_middle_can_be_preserved_by_deleting_two_low_use_neighbors():
    rows = [
        {"trajectory_id": "a", "path_id": "A_main", "direction": "Down", "evidence_score": 1.0},
        {"trajectory_id": "b", "path_id": "A_main", "direction": "Down", "evidence_score": 1.0},
        {"trajectory_id": "c", "path_id": "A_main", "direction": "Down", "evidence_score": 1.0},
    ]
    selected, _audit = _solve_weighted_vertex_cover(
        rows,
        [("a", "b"), ("b", "c")],
        assignment={"a": 0.0, "b": 100000.0, "c": 0.0},
        passenger_weight=4.0,
    )
    assert selected == {"a", "c"}


def test_frontier_keeps_all_candidates_headway_legal_and_count_free():
    start = clock_to_service_seconds(7, 40)
    rows = [
        trajectory("a", start),
        trajectory("b", start + 100),
        trajectory("c", start + 250),
    ]
    candidates, frontier = build_frontier(
        discovery(rows),
        passenger({"a": 10.0, "b": 1000.0, "c": 10.0}),
        passenger_weights=(0.0, 4.0),
    )
    assert frontier["status"].endswith("NO_CANDIDATE_SELECTED")
    low = candidates["pw_0p0"]
    high = candidates["pw_4p0"]
    assert low["inferred_service_trajectory_count"] == 2
    assert high["inferred_service_trajectory_count"] == 1
    for world in candidates.values():
        assert world["semantics"]["service_count_is_fixed"] is False
        assert world["semantics"]["planned_timetable_used"] is False
        assert world["operating_repair_frontier"]["candidate_world_is_final_r1"] is False
