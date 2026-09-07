from scripts.mppd_r1_hz_lineage_timing_map import solve_timing_map
from scripts.mppd_r1_hz_operating_authority import clock_to_service_seconds


A_NODES = [67, 68, 69, 70, 71, 72, 73, 74, 5, 75, 76, 77, 46, 78, 79, 80, 15, 16]


def tr(tid, start, evidence=1.0):
    return {
        "trajectory_id": tid,
        "afc_line": "A",
        "path_id": "A_main",
        "direction": "Down",
        "evidence_score": evidence,
        "events": [
            {"station": s, "sequence_index": i, "time_s": start + i * 100.0}
            for i, s in enumerate(A_NODES)
        ],
    }


def world(rows):
    return {
        "schema": "mppd.r1-hz-count-free-service-discovery.v2",
        "inferred_service_trajectory_count": len(rows),
        "trajectories": rows,
        "semantics": {"service_count_is_fixed": False, "planned_timetable_used": False},
    }


def lineage(rows):
    return {"lineages": rows}


def feedback(**mass):
    return {"trajectory_assignment_mass": mass}


def test_single_group_can_choose_removed_high_passenger_timing_hypothesis():
    start = clock_to_service_seconds(7, 40)
    keep = tr("keep", start, evidence=1.0)
    drop = tr("drop", start + 80, evidence=1.0)
    result, report = solve_timing_map(
        world([keep, drop]),
        world([keep]),
        lineage([{
            "removed_trajectory_id": "drop",
            "survivor_trajectory_id": "keep",
            "lineage_class": "PARALLEL_SAME_PATH_FRAGMENT",
        }]),
        feedback(keep=1, drop=1000),
        passenger_weight=4.0,
    )
    row = result["trajectories"][0]
    assert row["trajectory_id"] == "keep"
    assert row["lineage_timing_map"]["source_hypothesis_trajectory_id"] == "drop"
    assert row["lineage_timing_map"]["whole_trajectory_shift_from_survivor_s"] == 80.0
    assert report["non_survivor_hypothesis_selected_count"] == 1
    assert report["service_count"] == 1


def test_two_latent_services_never_choose_illegal_close_hypotheses_together():
    start = clock_to_service_seconds(7, 40)
    base1 = tr("g1", start, evidence=1.0)
    alt1 = tr("g1alt", start + 80, evidence=1.0)
    base2 = tr("g2", start + 300, evidence=1.0)
    alt2 = tr("g2alt", start + 180, evidence=1.0)
    original = world([base1, alt1, base2, alt2])
    candidate = world([base1, base2])
    lin = lineage([
        {"removed_trajectory_id": "g1alt", "survivor_trajectory_id": "g1", "lineage_class": "PARALLEL_SAME_PATH_FRAGMENT"},
        {"removed_trajectory_id": "g2alt", "survivor_trajectory_id": "g2", "lineage_class": "PARALLEL_SAME_PATH_FRAGMENT"},
    ])
    result, report = solve_timing_map(
        original,
        candidate,
        lin,
        feedback(g1=1, g1alt=100, g2=1, g2alt=100),
        passenger_weight=4.0,
    )
    assert result["inferred_service_trajectory_count"] == 2
    assert report["remaining_headway_conflicts"] == 0
    starts = sorted(row["events"][0]["time_s"] for row in result["trajectories"])
    assert starts[1] - starts[0] >= 160.0


def test_zero_passenger_weight_keeps_survivor_on_exact_evidence_tie():
    start = clock_to_service_seconds(7, 40)
    keep = tr("keep", start, evidence=1.0)
    drop = tr("drop", start + 80, evidence=1.0)
    result, report = solve_timing_map(
        world([keep, drop]), world([keep]),
        lineage([{"removed_trajectory_id":"drop","survivor_trajectory_id":"keep","lineage_class":"PARALLEL_SAME_PATH_FRAGMENT"}]),
        feedback(keep=1, drop=999), passenger_weight=0.0,
    )
    assert result["trajectories"][0]["lineage_timing_map"]["source_hypothesis_trajectory_id"] == "keep"
    assert report["non_survivor_hypothesis_selected_count"] == 0
