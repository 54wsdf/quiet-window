from collections import defaultdict
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mppd_r1_hz_count_free_service.py"
SPEC = importlib.util.spec_from_file_location("mppd_r1_hz_count_free_service", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def make_events(path_id, direction, starts, station_count=8, step=120.0):
    base = list(m.LINE_PATHS[path_id]["nodes"])
    nodes = base if direction == "Down" else list(reversed(base))
    nodes = nodes[:station_count]
    events = defaultdict(list)
    offsets = {s: i * step for i, s in enumerate(nodes)}
    for t0 in starts:
        for s in nodes:
            t = t0 + offsets[s]
            events[s].append(m.Event(f"{s}@{t}", s, t, 4.0, 25.0))
    return events, offsets


def test_count_free_discovers_number_without_count_input():
    events, offsets = make_events("C_main", "Down", [1000.0, 1070.0, 1400.0], station_count=9)
    out, ops = m.discover_path_candidates(
        events, "C_main", "Down", offsets, {},
        cluster_radius_s=25.0, split_std_s=20.0,
        min_station_support=5, complexity_penalty=0.10,
    )
    assert len(out) == 3
    assert [round(x["reference_time_s"]) for x in out] == [1000, 1070, 1400]
    assert ops["birth_retained"] == 3


def test_low_support_candidate_dies():
    events, offsets = make_events("C_main", "Down", [1000.0], station_count=2)
    out, ops = m.discover_path_candidates(
        events, "C_main", "Down", offsets, {},
        cluster_radius_s=30.0, split_std_s=20.0,
        min_station_support=4, complexity_penalty=0.10,
    )
    assert out == []
    assert ops["death_low_support"] >= 1


def test_cross_path_duplicate_becomes_one_service_with_path_competition():
    shared = ["1@100", "2@200", "3@300", "4@400"]
    a = {
        "path_id": "B_main", "afc_line": "B", "direction": "Up", "reference_time_s": 1000.0,
        "station_count": 8, "event_count": 8,
        "event_ids": shared + ["5@500", "6@600", "7@700", "8@800"],
        "support_weight": 30.0, "residual_p90_abs_s": 12.0, "residual_median_abs_s": 5.0,
    }
    b = {
        "path_id": "B_branch", "afc_line": "B", "direction": "Up", "reference_time_s": 1010.0,
        "station_count": 7, "event_count": 7,
        "event_ids": shared + ["5@500", "6@600", "7@700"],
        "support_weight": 28.0, "residual_p90_abs_s": 10.0, "residual_median_abs_s": 4.0,
    }
    final, ops = m.deduplicate_path_candidates([a, b], merge_time_s=30.0)
    assert len(final) == 1
    assert final[0]["path_ambiguous"] is True
    assert {x["path_id"] for x in final[0]["path_alternatives"]} == {"B_main", "B_branch"}
    assert ops["merge_cross_path_duplicate"] == 1
    assert ops["path_reassignment_competition"] == 1


def test_split_operation_is_available_for_bimodal_cluster():
    events, offsets = make_events("A_main", "Down", [1000.0, 1060.0], station_count=8)
    out, ops = m.discover_path_candidates(
        events, "A_main", "Down", offsets, {},
        cluster_radius_s=80.0, split_std_s=15.0,
        min_station_support=5, complexity_penalty=0.10,
    )
    assert len(out) == 2
    assert ops["split"] >= 1


def test_source_contract_contains_count_free_semantics():
    source = open(m.__file__, encoding="utf-8").read()
    assert '"train_count_is_input": False' in source
    assert '"planned_timetable_used": False' in source
    assert '"legacy_candidate_roots_used_as_input": False' in source
