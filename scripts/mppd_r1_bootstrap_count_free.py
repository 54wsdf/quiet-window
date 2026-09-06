from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.mppd_r1_joint_model import load_count_free_bootstrap


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()

    state = load_count_free_bootstrap(a.input, a.output)
    event_count = sum(len(s.events) for s in state.services)
    path_ambiguous = sum(s.path_ambiguous for s in state.services)
    direction_ambiguous = sum(s.direction_ambiguous for s in state.services)
    summary = {
        "schema": "mppd.r1-joint-bootstrap-summary.v1",
        "status": "R1_JOINT_COUNT_FREE_WARM_START_READY",
        "stage": state.stage,
        "iteration": state.iteration,
        "inferred_service_count": state.inferred_service_count,
        "service_event_anchor_count": event_count,
        "path_ambiguous_service_count": path_ambiguous,
        "direction_ambiguous_service_count": direction_ambiguous,
        "semantics": {
            "service_count_is_fixed": False,
            "train_count_is_input": state.config.train_count_is_input,
            "planned_timetable_used_in_primary_inference": state.config.planned_timetable_used_in_primary_inference,
            "legacy_candidate_roots_used_as_input": state.config.legacy_candidate_roots_used_as_input,
            "coarse_resolution_s": state.config.coarse_update_step_s,
            "fine_resolution_s": state.config.fine_update_step_s,
            "single_event_update_trust_region_s": state.config.event_trust_region_s,
            "access_min_s": state.config.access_min_s,
            "egress_min_s": state.config.egress_min_s,
            "transfer_min_s": state.config.transfer_min_s,
            "arrival_departure_separation_pending": True,
            "passenger_posterior_reinference_pending": True,
        },
        "scientific_boundary": (
            "This artifact is a warm start for one count-free joint R1 posterior. "
            "Its service count and coarse AFC ridge times are not qualified realized timetable truth."
        ),
    }
    a.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
