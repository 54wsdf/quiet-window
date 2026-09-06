from __future__ import annotations

"""Multiresolution service-count ladder with the historical v2 competition layer.

The underlying 60/30/15/5 s detector and structural refinement live in
mppd_r1_hz_multires_service_count. The representation-competition layer is the
same shared-AFC-event logic used by the historical v2 count-free solver. The
adapter deliberately ignores the v1 merge_time_s argument because v2 competes
on event identity rather than path-relative reference-time proximity.
"""

import scripts.mppd_r1_hz_count_free_service_v2 as competition
import scripts.mppd_r1_hz_multires_service_count as mr

_original_discover_level = mr.discover_level


def _shared_event_adapter(candidates, merge_time_s=None):
    del merge_time_s
    return competition.deduplicate_shared_event_candidates(candidates, overlap_threshold=0.60)


# Importing competition already patches these base functions; assign explicitly
# so the multires caller has a stable signature and auditable semantics.
mr.base.deduplicate_path_candidates = _shared_event_adapter
mr.base.materialize_trajectories = competition.materialize_with_ambiguity


def discover_level_with_v2_contract(*args, **kwargs):
    result = _original_discover_level(*args, **kwargs)
    sem = dict(result.get("semantics", {}))
    sem.update(
        {
            "service_trajectory_count_is_inferred": True,
            "shared_event_representation_competition": True,
            "shared_event_overlap_threshold": 0.60,
            "competition_layer_source": "mppd_r1_hz_count_free_service_v2",
            "v1_reference_time_merge_argument_ignored_by_v2_competition": True,
        }
    )
    result["semantics"] = sem
    return result


mr.discover_level = discover_level_with_v2_contract


if __name__ == "__main__":
    mr.main()
