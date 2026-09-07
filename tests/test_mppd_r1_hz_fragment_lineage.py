from scripts.mppd_r1_hz_fragment_lineage import audit_fragment_lineage


def tr(tid, path, direction, start, stations=(0, 1, 2, 3, 4, 5)):
    return {
        "trajectory_id": tid,
        "afc_line": "B" if path.startswith("B_") else "A",
        "path_id": path,
        "direction": direction,
        "events": [
            {"station": s, "sequence_index": i, "time_s": start + i * 100.0}
            for i, s in enumerate(stations)
        ],
    }


def world(rows):
    return {
        "inferred_service_trajectory_count": len(rows),
        "trajectories": rows,
    }


def feedback(**mass):
    return {"trajectory_assignment_mass": mass}


def test_parallel_same_path_removed_fragment_links_to_survivor():
    original = world([tr("keep", "A_main", "Down", 100), tr("drop", "A_main", "Down", 150)])
    candidate = world([tr("keep", "A_main", "Down", 100)])
    result = audit_fragment_lineage(original, candidate, feedback(keep=10, drop=20))
    row = result["lineages"][0]
    assert row["lineage_class"] == "PARALLEL_SAME_PATH_FRAGMENT"
    assert row["survivor_trajectory_id"] == "keep"
    assert row["median_offset_s_removed_minus_survivor"] == 50.0
    assert row["p90_parallel_residual_s"] == 0.0
    assert result["parallel_lineage_assignment_mass_share_of_removed"] == 1.0


def test_b_main_branch_can_share_lineage_only_on_shared_stations():
    original = world([
        tr("keep", "B_main", "Down", 100, stations=(0,1,2,3,4,5,21,22)),
        tr("drop", "B_branch", "Down", 220, stations=(0,1,2,3,4,5,28,29)),
    ])
    candidate = world([tr("keep", "B_main", "Down", 100, stations=(0,1,2,3,4,5,21,22))])
    result = audit_fragment_lineage(original, candidate, feedback(keep=3, drop=7))
    row = result["lineages"][0]
    assert row["lineage_class"] == "PARALLEL_SHARED_TRUNK_ALTERNATE_PATH"
    assert row["shared_station_count"] == 6
    assert row["cross_path"] is True


def test_nonparallel_geometry_stays_unresolved():
    keep = tr("keep", "A_main", "Down", 100)
    drop = tr("drop", "A_main", "Down", 150)
    # Distort downstream timing instead of creating a parallel offset.
    for i, event in enumerate(drop["events"]):
        event["time_s"] += i * 30
    result = audit_fragment_lineage(world([keep, drop]), world([keep]), feedback(drop=9))
    assert result["lineages"][0]["lineage_class"] == "UNRESOLVED_NONPARALLEL_FRAGMENT"
    assert result["parallel_lineage_count"] == 0


def test_lineage_does_not_mutate_candidate_or_claim_final_truth():
    original = world([tr("keep", "A_main", "Down", 100), tr("drop", "A_main", "Down", 180)])
    candidate = world([tr("keep", "A_main", "Down", 100)])
    result = audit_fragment_lineage(original, candidate, feedback())
    assert result["service_count_candidate"] == 1
    assert result["semantics"]["event_times_changed"] is False
    assert result["semantics"]["lineage_is_evidence_association_not_proven_same_vehicle"] is True
    assert result["semantics"]["candidate_world_is_final_r1"] is False
