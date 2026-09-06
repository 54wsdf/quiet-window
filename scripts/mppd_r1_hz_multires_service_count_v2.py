from __future__ import annotations

"""Multiresolution service-count ladder with the historical v2 competition layer.

The underlying 60/30/15/5 s detector and structural refinement live in
mppd_r1_hz_multires_service_count. Importing the v2 count-free module replaces
only the representation-competition/materialization layer with the same
shared-AFC-event logic that produced the historical 1841-trajectory solution.
This keeps resolution as the intended experimental difference.
"""

import scripts.mppd_r1_hz_count_free_service_v2 as competition  # noqa: F401
import scripts.mppd_r1_hz_multires_service_count as mr

_original_discover_level = mr.discover_level


def discover_level_with_v2_contract(*args, **kwargs):
    result = _original_discover_level(*args, **kwargs)
    sem = dict(result.get("semantics", {}))
    sem.update(
        {
            "service_trajectory_count_is_inferred": True,
            "shared_event_representation_competition": True,
            "shared_event_overlap_threshold": 0.60,
            "competition_layer_source": "mppd_r1_hz_count_free_service_v2",
        }
    )
    result["semantics"] = sem
    return result


mr.discover_level = discover_level_with_v2_contract


if __name__ == "__main__":
    mr.main()
