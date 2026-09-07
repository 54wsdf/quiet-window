from scripts.mppd_r1_hz_operating_authority import clock_to_service_seconds
from scripts.mppd_r1_hz_operating_repair import repair_service_world


A_NODES = [67, 68, 69, 70, 71, 72, 73, 74, 5, 75, 76, 77, 46, 78, 79, 80, 15, 16]
B_MAIN = list(range(0, 28))
B_BRANCH = list(range(0, 21)) + list(range(28, 34))


def trajectory(
    tid,
    line,
    path,
    direction,
    nodes,
    start_s,
    *,
    evidence_score=1.0,
    station_support=None,
    support_weight=10.0,
    residual_p90=20.0,
    path_ambiguous=False,
):
    ordered = nodes if direction == "Down" else list(reversed(nodes))
    if station_support is None:
        station_support = len(ordered)
    return {
        "trajectory_id": tid,
        "afc_line": line,
        "path_id": path,
        "direction": direction,
        "station_support": station_support,
        "support_weight": support_weight,
        "evidence_score": evidence_score,
        "support_event_count": station_support,
        "support_event_ids": [f"{tid}:e{i}" for i in range(station_support)],
        "trajectory_residual_p90_s": residual_p90,
        "path_ambiguous": path_ambiguous,
        "direction_ambiguous": False,
        "events": [
            {"station": station, "sequence_index": i, "time_s": start_s + i * 100.0}
            for i, station in enumerate(ordered)
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
        },
    }


def ids(repaired):
    return {row["trajectory_id"] for row in repaired["trajectories"]}


def test_lower_evidence_service_is_removed_inside_one_conflict():
    start = clock_to_service_seconds(7, 40)
    rows = [
        trajectory("strong", "A", "A_main", "Down", A_NODES, start, evidence_score=5.0),
        trajectory("weak", "A", "A_main", "Down", A_NODES, start + 150, evidence_score=0.2),
    ]
    repaired, report = repair_service_world(discovery(rows))
    assert ids(repaired) == {"strong"}
    assert report["removed_trajectory_ids"] == ["weak"]
    assert report["after"]["headway"]["total_normal_floor_violation_pair_count"] == 0


def test_shared_trunk_main_and_branch_compete_before_path_labels_can_duplicate_service():
    start = clock_to_service_seconds(8, 0)
    rows = [
        trajectory("main", "B", "B_main", "Down", B_MAIN, start, evidence_score=4.0),
        trajectory("branch", "B", "B_branch", "Down", B_BRANCH, start + 120, evidence_score=0.5),
    ]
    repaired, report = repair_service_world(discovery(rows))
    assert ids(repaired) == {"main"}
    op = report["operations"][0]
    assert op["trajectory_id"] == "branch"
    assert op["representative_conflict"]["resource_id"] == "B_SHARED_TRUNK"
    assert op["representative_conflict"]["required_floor_s"] == 130.0


def test_greedy_cover_removes_middle_fragment_instead_of_both_outer_services():
    start = clock_to_service_seconds(7, 40)
    rows = [
        trajectory("left", "A", "A_main", "Down", A_NODES, start, evidence_score=2.0),
        trajectory("middle", "A", "A_main", "Down", A_NODES, start + 100, evidence_score=3.0),
        trajectory("right", "A", "A_main", "Down", A_NODES, start + 250, evidence_score=2.0),
    ]
    repaired, report = repair_service_world(discovery(rows))
    assert ids(repaired) == {"left", "right"}
    assert report["removed_trajectory_ids"] == ["middle"]
    assert report["round_count"] == 1
    assert report["after"]["headway"]["total_normal_floor_violation_pair_count"] == 0


def test_nonconflicting_direction_imbalance_is_reported_but_not_deleted_to_force_symmetry():
    start = clock_to_service_seconds(10, 0)
    rows = [
        trajectory("d1", "A", "A_main", "Down", A_NODES, start),
        trajectory("d2", "A", "A_main", "Down", A_NODES, start + 500),
        trajectory("u1", "A", "A_main", "Up", A_NODES, start + 1000),
    ]
    repaired, report = repair_service_world(discovery(rows))
    assert len(repaired["trajectories"]) == 3
    assert report["removed_service_count"] == 0
    assert report["after"]["direction_balance"]["by_path"]["A_main"]["absolute_imbalance"] == 1
    assert report["semantics"]["direction_balance_forced_by_nonconflict_deletion"] is False


def test_repaired_discovery_remains_count_free_and_marks_candidate_not_final_truth():
    start = clock_to_service_seconds(7, 40)
    rows = [
        trajectory("a", "A", "A_main", "Down", A_NODES, start),
        trajectory("b", "A", "A_main", "Down", A_NODES, start + 80),
    ]
    repaired, report = repair_service_world(discovery(rows))
    assert repaired["schema"] == "mppd.r1-hz-count-free-service-discovery.v2"
    assert repaired["semantics"]["service_count_is_fixed"] is False
    assert repaired["semantics"]["planned_timetable_used"] is False
    assert repaired["operating_constraint_repair"]["candidate_world_is_final_r1"] is False
    assert report["semantics"]["birth_operations_applied"] == 0
    assert report["semantics"]["merge_operations_applied"] == 0
