from __future__ import annotations

import numpy as np

import scripts.mppd_r1_hz_multires_service_count as mr


def test_resolution_ladder_refines_structure_without_changing_observation_grid():
    cfgs = mr.resolution_configs()
    assert [x.structural_resolution_s for x in cfgs] == [60, 30, 15, 5]
    assert mr.OBSERVATION_BIN_S == 5
    assert cfgs[0].cluster_radius_s == 55.0
    assert cfgs[-1].cluster_radius_s == 5.0
    assert cfgs[-1].split_minimum_separation_s == 5.0
    assert cfgs[-1].event_minimum_separation_s == 5


def test_five_second_event_detector_can_retain_closer_peaks_than_sixty_second_level():
    counts = np.zeros((81, 240), dtype=np.uint32)
    # Two sharp exit pulses 10 s apart on the same station.
    counts[0, 80] = 40
    counts[0, 82] = 35
    coarse = mr.detect_station_events(counts, 5, mr.resolution_configs()[0], threshold_quantile=0.90)
    fine = mr.detect_station_events(counts, 5, mr.resolution_configs()[-1], threshold_quantile=0.90)
    assert len(fine.get(0, [])) >= len(coarse.get(0, []))


def test_split_minimum_separation_is_resolution_specific():
    event = lambda i, t: mr.base.Event(str(i), i, float(t), 3.0, 20.0)
    items = [
        (0.0, event(0, 0)),
        (1.0, event(1, 1)),
        (2.0, event(2, 2)),
        (10.0, event(3, 10)),
        (11.0, event(4, 11)),
        (12.0, event(5, 12)),
    ]
    assert mr.two_means_split(items, 25.0) is None
    assert mr.two_means_split(items, 5.0) is not None


def test_level_semantics_keep_service_count_free():
    cfg = mr.resolution_configs()[-1]
    # Synthetic no-signal field is sufficient to inspect the scientific contract.
    counts = np.zeros((81, 300), dtype=np.uint32)
    result = mr.discover_level(counts, cfg, iterations=1)
    sem = result["semantics"]
    assert sem["train_count_is_input"] is False
    assert sem["service_count_inherited_from_previous_level"] is False
    assert sem["previous_level_is_constraint"] is False
    assert sem["planned_timetable_used"] is False
    assert sem["same_raw_afc_reestimated_at_every_resolution"] is True
    assert sem["five_second_level_is_service_structure_inference_not_event_time_refinement"] is True
