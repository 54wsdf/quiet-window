from scripts.mppd_r1_hz_operating_audit import audit_service_world
from scripts.mppd_r1_hz_operating_authority import clock_to_service_seconds


A_NODES = [67, 68, 69, 70, 71, 72, 73, 74, 5, 75, 76, 77, 46, 78, 79, 80, 15, 16]
B_MAIN = list(range(0, 28))
B_BRANCH = list(range(0, 21)) + list(range(28, 34))


def trajectory(tid, line, path, direction, nodes, start_s, step_s=100.0, **extra):
    ordered = nodes if direction == "Down" else list(reversed(nodes))
    return {
        "trajectory_id": tid,
        "afc_line": line,
        "path_id": path,
        "direction": direction,
        "events": [
            {"station": station, "sequence_index": i, "time_s": start_s + i * step_s}
            for i, station in enumerate(ordered)
        ],
        **extra,
    }


def discovery(rows):
    return {
        "schema": "mppd.r1-hz-count-free-service-discovery.v2",
        "inferred_service_trajectory_count": len(rows),
        "trajectories": rows,
    }


def test_audit_counts_one_close_pair_once_across_many_stations():
    start = clock_to_service_seconds(7, 40)
    rows = [
        trajectory("a1", "A", "A_main", "Down", A_NODES, start),
        trajectory("a2", "A", "A_main", "Down", A_NODES, start + 150),
        trajectory("a3", "A", "A_main", "Down", A_NODES, start + 230),
    ]
    result = audit_service_world(discovery(rows))
    headway = result["headway"]
    # a1-a2 violates the 160 s Line-4 peak envelope; a2-a3 also violates the
    # absolute 90 s physical floor.  Each pair is counted once, not once per stop.
    assert headway["unique_adjacent_service_pair_count"] == 2
    assert headway["absolute_physical_violation_pair_count"] == 1
    assert headway["verified_operating_envelope_violation_pair_count"] == 2
    assert headway["by_resource"]["A_MAIN"]["minimum_gap_s"] == 80.0


def test_shared_trunk_competes_main_and_branch_as_one_physical_resource():
    start = clock_to_service_seconds(8, 0)
    rows = [
        trajectory("m1", "B", "B_main", "Down", B_MAIN, start),
        trajectory("b1", "B", "B_branch", "Down", B_BRANCH, start + 120),
    ]
    result = audit_service_world(discovery(rows))
    shared = result["headway"]["by_resource"]["B_SHARED_TRUNK"]
    assert shared["adjacent_pair_count"] == 1
    assert shared["minimum_gap_s"] == 120.0
    assert shared["absolute_physical_violation_pair_count"] == 0
    assert shared["operating_envelope_violation_pair_count"] == 1
    violation = result["headway"]["worst_violation_pairs"][0]
    assert violation["required_floor_s"] == 130.0
    assert {violation["left_path_id"], violation["right_path_id"]} == {
        "B_main",
        "B_branch",
    }


def test_branch_exclusive_segment_uses_430_second_special_peak_floor():
    start = clock_to_service_seconds(8, 0)
    rows = [
        trajectory("b1", "B", "B_branch", "Down", B_BRANCH, start),
        trajectory("b2", "B", "B_branch", "Down", B_BRANCH, start + 420),
    ]
    result = audit_service_world(discovery(rows))
    branch = result["headway"]["by_resource"]["B_BRANCH_ONLY"]
    assert branch["minimum_gap_s"] == 420.0
    assert branch["absolute_physical_violation_pair_count"] == 0
    assert branch["operating_envelope_violation_pair_count"] == 1


def test_direction_balance_is_reported_without_fake_depot_resolution():
    start = clock_to_service_seconds(10, 0)
    rows = [
        trajectory("a-down-1", "A", "A_main", "Down", A_NODES, start),
        trajectory("a-down-2", "A", "A_main", "Down", A_NODES, start + 400),
        trajectory("a-up-1", "A", "A_main", "Up", A_NODES, start + 800),
    ]
    result = audit_service_world(discovery(rows))
    balance = result["direction_balance"]["by_path"]["A_main"]
    assert balance["down"] == 2
    assert balance["up"] == 1
    assert balance["absolute_imbalance"] == 1
    assert balance["balance_is_not_depot_resolved"] is True
    assert result["semantics"]["depot_network_station_mapping_resolved"] is False
