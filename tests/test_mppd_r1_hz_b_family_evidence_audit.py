from __future__ import annotations

import scripts.mppd_r1_hz_b_family_evidence_audit as audit


def b_row(t: float, path: str, score: float = 1.0):
    return {
        "trajectory_id": f"{path}:{t}",
        "afc_line": "B",
        "direction": "Down",
        "path_id": path,
        "path_ambiguous": path == "B_branch",
        "support_station_count": 6,
        "support_event_count": 6,
        "support_event_ids": [
            f"0@{t:.1f}",
            f"1@{t+50:.1f}",
            f"2@{t+100:.1f}",
            f"3@{t+150:.1f}",
            f"4@{t+200:.1f}",
            f"5@{t+250:.1f}",
        ],
        "evidence_score": score,
        "events": [{"station": 5, "time_s": t + 250.0}],
    }


def test_shared_trunk_duplicate_is_detected_across_path_labels():
    a = b_row(1000.0, "B_main")
    b = b_row(1005.0, "B_branch")
    x = audit.aligned_on_b_trunk(a, b)
    assert x["shared_trunk_duplicate_evidence"] is True
    assert x["matched_support_station_count"] == 6


def test_distinct_parallel_train_survives_shared_trunk_gate():
    parent = {"trajectories": [b_row(1000.0, "B_main")]}
    candidate = b_row(1070.0, "B_branch", score=1.2)
    survivors, rows = audit.audit_b_births(parent, [candidate])
    assert len(survivors) == 1
    assert rows[0]["classification"] == "SURVIVES_B_SHARED_TRUNK_EVIDENCE_GATE"


def test_duplicate_birth_is_rejected_against_parent_even_with_branch_path():
    parent = {"trajectories": [b_row(1000.0, "B_main")]}
    candidate = b_row(1005.0, "B_branch", score=1.2)
    survivors, rows = audit.audit_b_births(parent, [candidate])
    assert survivors == []
    assert rows[0]["classification"] == "REJECT_BIRTH_SHARED_TRUNK_DUPLICATE_OF_PARENT"
