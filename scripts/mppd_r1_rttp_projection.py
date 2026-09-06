from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

from scripts.mppd_r1_rttp_bridge import joint_state_to_provider_payload


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return result
    if is_dataclass(value):
        result = asdict(value)
        if isinstance(result, Mapping):
            return result
    raise TypeError(f"{name} must be mapping-like")


def circulation_annotations_from_joint_state(state: Any) -> dict[str, dict[str, Any]]:
    """Lower inferred service chains to RTTP circulation annotations.

    The mapping intentionally carries only circulation membership and boundary stop
    semantics.  It never emits a physical vehicle identifier or planned train ID.
    """

    raw = _mapping(state, name="R1 state")
    rows = raw.get("service_circulations", ())
    if rows is None:
        rows = ()
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("service_circulations must be a sequence")

    annotations: dict[str, dict[str, Any]] = {}
    circulation_ids: set[str] = set()
    for raw_row in rows:
        row = _mapping(raw_row, name="service circulation")
        circulation_id = str(row.get("circulation_id", ""))
        if not circulation_id:
            raise ValueError("service circulation requires circulation_id")
        if circulation_id in circulation_ids:
            raise ValueError(f"duplicate circulation_id {circulation_id}")
        circulation_ids.add(circulation_id)
        if row.get("physical_vehicle_identity_claimed") is not False:
            raise ValueError(
                f"{circulation_id}: RTTP projection forbids physical vehicle identity claims"
            )
        trajectory_ids_raw = row.get("ordered_trajectory_ids")
        if isinstance(trajectory_ids_raw, (str, bytes)) or not isinstance(
            trajectory_ids_raw, Sequence
        ):
            raise ValueError(f"{circulation_id}: ordered_trajectory_ids must be a sequence")
        trajectory_ids = [str(item) for item in trajectory_ids_raw]
        if not trajectory_ids or any(not item for item in trajectory_ids):
            raise ValueError(f"{circulation_id}: invalid ordered_trajectory_ids")
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError(f"{circulation_id}: duplicate trajectory in circulation")

        chain_type = str(row.get("chain_type", "TURNBACK"))
        if chain_type not in {"TURNBACK", "THROUGH_RUNNING", "MIXED"}:
            raise ValueError(f"{circulation_id}: unsupported chain_type {chain_type}")
        start_boundary = str(row.get("start_boundary", "UNRESOLVED"))
        end_boundary = str(row.get("end_boundary", "UNRESOLVED"))
        if start_boundary not in {"UNRESOLVED", "DEPOT_EXIT", "NETWORK_BOUNDARY"}:
            raise ValueError(f"{circulation_id}: invalid start boundary {start_boundary}")
        if end_boundary not in {"UNRESOLVED", "DEPOT_ENTRY", "NETWORK_BOUNDARY"}:
            raise ValueError(f"{circulation_id}: invalid end boundary {end_boundary}")

        for index, trajectory_id in enumerate(trajectory_ids):
            if trajectory_id in annotations:
                raise ValueError(
                    f"trajectory {trajectory_id} appears in multiple service circulations"
                )
            item: dict[str, Any] = {"circulation_id": circulation_id}
            if index == 0 and start_boundary == "DEPOT_EXIT":
                item["origin_stop_type"] = "depot_exit"
            if index == len(trajectory_ids) - 1:
                if end_boundary == "DEPOT_ENTRY":
                    item["terminal_stop_type"] = "depot_entry"
            elif chain_type == "TURNBACK":
                item["terminal_stop_type"] = "turnback"
            annotations[trajectory_id] = item

    return annotations


def joint_state_to_rttp_provider_payload(state: Any) -> dict[str, Any]:
    """Canonical high-level R1→RTTP handoff including inferred circulations."""

    annotations = circulation_annotations_from_joint_state(state)
    payload = joint_state_to_provider_payload(
        state,
        service_annotations=annotations or None,
    )
    payload["semantics"] = dict(payload["semantics"])
    payload["semantics"]["service_circulation_count"] = len(
        _mapping(state, name="R1 state").get("service_circulations", ()) or ()
    )
    payload["semantics"]["service_circulation_projected"] = bool(annotations)
    payload["semantics"]["physical_vehicle_identity_claimed"] = False
    return payload
