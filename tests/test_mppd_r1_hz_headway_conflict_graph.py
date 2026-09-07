from scripts.mppd_r1_hz_headway_conflict_graph import (
    analyze_conflicts,
    exact_minimum_vertex_cover,
)
from scripts.mppd_r1_hz_operating_authority import clock_to_service_seconds


A_NODES = [67, 68, 69, 70, 71, 72, 73, 74, 5, 75, 76, 77, 46, 78, 79, 80, 15, 16]
B_MAIN = list(range(0, 28))
B_BRANCH = list(range(0, 21)) + list(range(28, 34))


def trajectory(tid, line, path, direction, nodes, start_s, step_s=100.0, evidence=1.0):
    ordered = nodes if direction == "Down" else list(reversed(nodes))
    return {
        "trajectory_id": tid,
        "afc_line": line,
        "path_id": path,
        "direction": direction,
        "path_ambiguous": False,
        "direction_ambiguous": False,
        "support_station_count": 8,
        "support_event_count": 8,
        "support_weight": 10.0,
        "evidence_score": evidence,
        "objective_gain": evidence,
        "events": [
            {"station": station, "sequence_index": i, "time_s": start_s + i * step_s}
            for i, station in enumerate(ordered)
        ],
    }


def discovery(rows):
    return {
        "schema": "mppd.r1-hz-count-free-service-discovery.v2",
        "inferred_service_trajectory_count": len(rows),
        "trajectories": rows,
    }


def test_exact_vertex_cover_minimizes_cardinality_before_afc_evidence_loss():
    nodes = {
        "a": {"evidence_score": 100.0, "support_station_count": 8, "support_weight": 10.0},
        "b": {"evidence_score": 1.0, "support_station_count": 8, "support_weight": 10.0},
        "c": {"evidence_score": 100.0, "support_station_count": 8, "support_weight": 10.0},
    }
    cover, _states = exact_minimum_vertex_cover({("a", "b"), ("b", "c")}, nodes)
    assert cover == frozenset({"b"})


def test_three_mutually_conflicting_services_require_two_removals():
    start = clock_to_service_seconds(7, 40)
    rows = [
        trajectory("a1", "A", "A_main", "Down", A_NODES, start, evidence=2.0),
        trajectory("a2", "A", "A_main", "Down", A_NODES, start + 60, evidence=1.0),
        trajectory("a3", "A", "A_main", "Down", A_NODES, start + 120, evidence=3.0),
    ]
    result = analyze_conflicts(discovery(rows))
    assert result["conflict_graph"]["union_conflict_edge_count"] == 3
    assert result["conflict_graph"]["physical_90s_conflict_edge_count"] == 2
    assert result["repair_bounds"]["all_components_solved_exactly"] is True
    assert result["repair_bounds"]["exact_minimum_removals"] == 2
    assert result["repair_bounds"]["exact_maximum_retained_count"] == 1


def test_shared_trunk_main_branch_conflict_is_one_graph_edge():
    start = clock_to_service_seconds(8, 0)
    rows = [
        trajectory("m1", "B", "B_main", "Down", B_MAIN, start, evidence=4.0),
        trajectory("b1", "B", "B_branch", "Down", B_BRANCH, start + 120, evidence=1.0),
    ]
    result = analyze_conflicts(discovery(rows))
    assert result["conflict_graph"]["union_conflict_edge_count"] == 1
    assert result["conflict_graph"]["physical_90s_conflict_edge_count"] == 0
    assert result["conflict_graph"]["operating_only_conflict_edge_count"] == 1
    assert result["repair_bounds"]["exact_minimum_removals"] == 1
    assert result["diagnostic_repair"]["removed_trajectory_ids"] == ["b1"]


def test_nonconflicting_services_remain_outside_conflict_graph():
    start = clock_to_service_seconds(12, 0)
    rows = [
        trajectory("a1", "A", "A_main", "Down", A_NODES, start),
        trajectory("a2", "A", "A_main", "Down", A_NODES, start + 400),
    ]
    result = analyze_conflicts(discovery(rows))
    assert result["conflict_graph"]["union_conflict_edge_count"] == 0
    assert result["conflict_graph"]["incident_service_count"] == 0
    assert result["repair_bounds"]["exact_minimum_removals"] == 0
    assert result["diagnostic_repair"]["retained_service_count"] == 2
