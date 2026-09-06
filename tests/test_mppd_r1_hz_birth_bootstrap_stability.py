from __future__ import annotations

import gzip

import scripts.mppd_r1_hz_birth_bootstrap_stability as stab
import scripts.mppd_r1_hz_resampled_birth_discovery as discovery


def row(t: float, *, line: str = "A", direction: str = "Down", path: str = "A_main"):
    return {
        "trajectory_id": f"x:{t}",
        "afc_line": line,
        "direction": direction,
        "path_id": path,
        "path_ambiguous": False,
        "anchor_time_s": t,
        "support_station_count": 5,
        "support_event_ids": [
            f"5@{t:.1f}",
            f"67@{t-100:.1f}",
            f"68@{t-50:.1f}",
            f"75@{t+50:.1f}",
            f"76@{t+100:.1f}",
        ],
        "evidence_score": 1.1,
    }


def test_event_alignment_recovers_same_cross_station_pulse():
    a = row(1000.0)
    b = row(1005.0)
    x = stab.event_alignment(a, b, tolerance_s=15.0)
    assert x["matched_support_station_count"] == 5
    assert x["matched_support_fraction"] == 1.0
    assert x["median_nearest_event_delta_s"] == 5.0


def test_pair_score_rejects_different_pulse_even_if_anchor_is_near():
    a = row(1000.0)
    b = row(1035.0)
    assert stab.pair_score(a, b) is None


def test_stability_counts_recovery_as_coarse_service():
    full_birth = row(1000.0)
    full = {
        "coarse_service_count": 1841,
        "fine_five_second_service_count": 2709,
        "frontier": {
            "n60e100": {
                "novelty_radius_s": 60.0,
                "minimum_evidence_score": 1.0,
                "births": [full_birth],
            }
        },
    }
    reps = []
    for seed, shift in [(1, 5.0), (2, -5.0), (3, 10.0)]:
        reps.append({
            "source_profile": {"bootstrap_seed": seed},
            "coarse_service_count": 1842,
            "fine_five_second_service_count": 2700,
            "coarse_support_trajectories": [row(1000.0 + shift)],
            "frontier": {"n60e100": {"birth_count": 0, "births": []}},
        })
    out = stab.summarize(full, reps)
    p = out["frontier"]["n60e100"]
    assert p["stable_birth_count_two_thirds"] == 1
    assert p["strong_stable_birth_count_five_sixths"] == 1
    assert p["full_data_stable_augmented_count_two_thirds"] == 1842
    c = p["candidates"][0]
    assert c["recovery_source_counts"]["RECOVERED_AS_COARSE_SERVICE"] == 3


def test_full_data_exit_loader_has_no_bootstrap_perturbation(tmp_path):
    p = tmp_path / "afc.csv.gz"
    with gzip.open(p, "wt", encoding="utf-8", newline="") as f:
        f.write("time,stationID,status\n")
        f.write("2019-01-04 04:00:05,0,0\n")
        f.write("2019-01-04 04:00:10,0,0\n")
        f.write("2019-01-04 03:59:59,0,0\n")
        f.write("2019-01-04 04:00:15,0,1\n")
    counts, profile = discovery.load_exit_counts_poisson_bootstrap(
        [p], "2019-01-04", bootstrap_seed=None
    )
    assert int(counts.sum()) == 2
    assert profile["bootstrap_mode"] == "NONE_FULL_DATA"
    assert profile["retained_service_day_exit_rows_before_resampling"] == 2
