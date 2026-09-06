from __future__ import annotations

import importlib
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

RTTP_PROVIDER_REPOSITORY = "54wsdf/rail-transit-timetable-platform"
RTTP_PROVIDER_BRANCH = "development/rttp-conditional-research-continuation-20260901"
RTTP_PROVIDER_SHA = "72f4a5900bd341bbc4d29c4c6f51604e58c70b00"
RTTP_PROVIDER_CONTRACT = "rttp.mppd-r1-provider.v0.2"
RTTP_PROVIDER_MODULE = "rttp.integrations.mppd_r1"

COUNT_FREE = "COUNT_FREE"
PLAN_ANCHORED = "PLAN_ANCHORED"
FINE_STAGE = "FINE_1S"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_COUNT_FREE_TRUE = (
    "planned_timetable_used_in_primary_inference",
    "planned_absolute_times_used",
    "planned_trip_count_used",
    "planned_trip_ids_used",
    "train_count_is_input",
)
_ALLOWED_SERVICE_ANNOTATIONS = {
    "circulation_id",
    "origin_stop_type",
    "terminal_stop_type",
}


def provider_lock() -> dict[str, str]:
    """Return the immutable RTTP provider dependency used by MPPD R1."""

    if _SHA_RE.fullmatch(RTTP_PROVIDER_SHA) is None:
        raise AssertionError("RTTP provider pin is not an exact 40-hex commit SHA")
    return {
        "repository": RTTP_PROVIDER_REPOSITORY,
        "branch_at_pin": RTTP_PROVIDER_BRANCH,
        "commit_sha": RTTP_PROVIDER_SHA,
        "provider_contract": RTTP_PROVIDER_CONTRACT,
        "module": RTTP_PROVIDER_MODULE,
    }


def verify_observed_provider_sha(observed_sha: str) -> None:
    """Fail closed when a runtime RTTP checkout is not the exact qualified pin."""

    if observed_sha != RTTP_PROVIDER_SHA:
        raise ValueError(
            f"RTTP provider SHA mismatch: expected {RTTP_PROVIDER_SHA}, got {observed_sha}"
        )


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
    raise TypeError(f"{name} must be a mapping or serializable dataclass-like object")


def _bool_flag(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean when present")
    return value


def assert_count_free_flags(mapping: Mapping[str, Any], *, scope: str) -> None:
    leaked = [key for key in _FORBIDDEN_COUNT_FREE_TRUE if _bool_flag(mapping, key)]
    if leaked:
        raise ValueError(
            f"{scope} leaks planned/count authority into COUNT_FREE R1: "
            + ", ".join(leaked)
        )


def assert_count_free_network_authority(authority: Mapping[str, Any]) -> None:
    semantics = authority.get("semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("network authority requires semantics")
    required = {
        "contains_service_event_inventory": False,
        "contains_expected_service_count": False,
        "contains_planned_timetable": False,
        "service_count_must_be_inferred": True,
    }
    bad = [
        key
        for key, expected in required.items()
        if semantics.get(key) is not expected
    ]
    if bad:
        raise ValueError(
            "network authority is not count-free structural authority: " + ", ".join(bad)
        )
    assert_count_free_flags(semantics, scope="network authority semantics")


def network_authority_to_provider_payload(
    authority: Mapping[str, Any],
    *,
    network_authority_ref: str,
) -> dict[str, Any]:
    """Project Hangzhou network authority to RTTP's zero-service R1 substrate.

    This transformation carries only physical path/direction/station order. It
    deliberately creates no ServicePattern, TripInstance, StopEvent, service count,
    train identity or absolute clock.
    """

    assert_count_free_network_authority(authority)
    if not network_authority_ref:
        raise ValueError("network_authority_ref must be non-empty")
    dataset_id = str(authority.get("dataset_id", ""))
    if not dataset_id:
        raise ValueError("network authority requires dataset_id")

    raw_routes = authority.get("line_paths")
    if isinstance(raw_routes, (str, bytes)) or not isinstance(raw_routes, Sequence):
        raise ValueError("network authority line_paths must be a sequence")

    routes: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    for raw in raw_routes:
        if not isinstance(raw, Mapping):
            raise ValueError("network authority line path entries must be mappings")
        path_id = str(raw.get("path_id", ""))
        line_id = str(raw.get("line_id", ""))
        direction_id = str(raw.get("direction_id", ""))
        station_ids_raw = raw.get("station_ids")
        if (
            not path_id
            or not line_id
            or not direction_id
            or isinstance(station_ids_raw, (str, bytes))
            or not isinstance(station_ids_raw, Sequence)
        ):
            raise ValueError("network authority contains incomplete line-path structure")
        station_ids = tuple(str(item) for item in station_ids_raw)
        if len(station_ids) < 2 or any(not item for item in station_ids):
            raise ValueError(f"route {path_id}/{direction_id} has invalid station sequence")
        key = (path_id, direction_id)
        if key in keys:
            raise ValueError(f"duplicate network route key {path_id}/{direction_id}")
        keys.add(key)
        routes.append(
            {
                "path_id": path_id,
                "line_id": line_id,
                "direction_id": direction_id,
                "ordered_station_ids": station_ids,
            }
        )

    if not routes:
        raise ValueError("network authority contains no route structures")

    return {
        "provider_contract": RTTP_PROVIDER_CONTRACT,
        "dataset_id": dataset_id,
        "network_authority_ref": network_authority_ref,
        "routes": routes,
        "metadata": {
            "planned_timetable_used_in_primary_inference": False,
            "planned_absolute_times_used": False,
            "planned_trip_count_used": False,
            "planned_trip_ids_used": False,
            "train_count_is_input": False,
        },
        "semantics": {
            "canonical_timetable_materialized": False,
            "service_pattern_materialized": False,
            "service_instance_count_materialized": False,
            "physical_route_structure_only": True,
        },
    }


def _canonical_second(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    rounded = round(float(value))
    if abs(float(value) - rounded) > 1e-9:
        raise ValueError(f"{field_name} must be on the 1-second grid")
    return int(rounded)


def _normalized_annotations(
    service_annotations: Mapping[str, Mapping[str, Any]] | None,
    trajectory_id: str,
) -> dict[str, Any]:
    if service_annotations is None:
        return {}
    raw = service_annotations.get(trajectory_id, {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"annotations for {trajectory_id} must be a mapping")
    extra = set(raw) - _ALLOWED_SERVICE_ANNOTATIONS
    if extra:
        raise ValueError(
            f"annotations for {trajectory_id} contain forbidden/noncanonical fields: "
            + ", ".join(sorted(extra))
        )
    out = dict(raw)
    if "circulation_id" in out and out["circulation_id"] is not None:
        out["circulation_id"] = str(out["circulation_id"])
    for key in ("origin_stop_type", "terminal_stop_type"):
        if key in out and out[key] is not None:
            out[key] = str(out[key])
    return out


def joint_state_to_provider_payload(
    state: Any,
    *,
    service_annotations: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Serialize an R1 fine state for the pinned RTTP provider.

    Coarse anchor_time_s is never exported as arrival/departure. Circulation
    annotations identify an inferred chain only; no physical vehicle_id is accepted.
    """

    if hasattr(state, "validate") and callable(state.validate):
        errors = state.validate()
        if errors:
            raise ValueError("R1 state validation failed: " + "; ".join(errors[:10]))
    raw = _mapping(state, name="R1 state")
    if raw.get("stage") != FINE_STAGE:
        raise ValueError(
            "RTTP canonical payload requires FINE_1S; coarse AFC ridge anchors stay noncanonical"
        )

    config = _mapping(raw.get("config", {}), name="R1 config")
    assert_count_free_flags(config, scope="R1 config")
    if config.get("planned_timetable_role") not in (
        None,
        "POST_HOC_EXTERNAL_COMPARISON_ONLY",
    ):
        raise ValueError("COUNT_FREE R1 config has a non-post-hoc planned timetable role")

    raw_services = raw.get("services")
    if isinstance(raw_services, (str, bytes)) or not isinstance(raw_services, Sequence):
        raise ValueError("R1 state services must be a sequence")

    services: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_service in raw_services:
        service = _mapping(raw_service, name="R1 service")
        trajectory_id = str(service.get("trajectory_id", ""))
        if not trajectory_id:
            raise ValueError("R1 service requires trajectory_id")
        if trajectory_id in seen:
            raise ValueError(f"duplicate R1 trajectory_id {trajectory_id}")
        seen.add(trajectory_id)

        raw_events = service.get("events")
        if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
            raise ValueError(f"R1 service {trajectory_id} events must be a sequence")
        events: list[dict[str, Any]] = []
        previous_departure_s: int | None = None
        for expected_sequence, raw_event in enumerate(raw_events):
            event = _mapping(raw_event, name="R1 event")
            sequence_index = int(event.get("sequence_index", -1))
            if sequence_index != expected_sequence:
                raise ValueError(
                    f"R1 service {trajectory_id} event sequence must be contiguous from zero"
                )
            if event.get("arrival_time_s") is None or event.get("departure_time_s") is None:
                raise ValueError(
                    f"R1 service {trajectory_id} fine event requires separate arrival/departure"
                )
            arrival_s = _canonical_second(
                event.get("arrival_time_s"), field_name="arrival_time_s"
            )
            departure_s = _canonical_second(
                event.get("departure_time_s"), field_name="departure_time_s"
            )
            if departure_s < arrival_s:
                raise ValueError(
                    f"R1 service {trajectory_id} departure precedes arrival"
                )
            if previous_departure_s is not None and arrival_s <= previous_departure_s:
                raise ValueError(
                    f"R1 service {trajectory_id} has non-positive interstation running time"
                )
            previous_departure_s = departure_s
            events.append(
                {
                    "station_id": str(event.get("station_id", "")),
                    "sequence_index": sequence_index,
                    "arrival_time_s": arrival_s,
                    "departure_time_s": departure_s,
                    "evidence_class": str(
                        event.get("evidence_class", "AFC_PASSENGER_FACING_RIDGE")
                    ),
                }
            )
        if len(events) < 2 or any(not item["station_id"] for item in events):
            raise ValueError(f"R1 service {trajectory_id} has incomplete event sequence")

        out = {
            "trajectory_id": trajectory_id,
            "line_id": str(service.get("line_id", "")),
            "direction_id": str(service.get("direction_id", "")),
            "path_id": str(service.get("path_id", "")),
            "events": events,
        }
        if not out["line_id"] or not out["direction_id"] or not out["path_id"]:
            raise ValueError(f"R1 service {trajectory_id} has incomplete route identity")
        out.update(_normalized_annotations(service_annotations, trajectory_id))
        services.append(out)

    if not services:
        raise ValueError("R1 fine state contains no services")

    return {
        "provider_contract": RTTP_PROVIDER_CONTRACT,
        "mode": COUNT_FREE,
        "stage": FINE_STAGE,
        "config": {
            "train_count_is_input": False,
            "planned_timetable_used_in_primary_inference": False,
            "planned_absolute_times_used": False,
            "planned_trip_count_used": False,
            "planned_trip_ids_used": False,
        },
        "services": services,
        "semantics": {
            "coarse_anchor_time_exported_as_canonical_event": False,
            "physical_vehicle_identity_claimed": False,
            "service_count_is_inferred": True,
        },
    }


def normalize_r1_operations_for_rttp(
    operations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Prepare R1 operations for RTTP without inventing split/merge canonicals."""

    normalized: list[dict[str, Any]] = []
    for raw in operations:
        if not isinstance(raw, Mapping):
            raise ValueError("R1 operations must be mappings")
        op_type = str(raw.get("operation_type", ""))
        if op_type in ("split", "merge"):
            lowered = raw.get("lowered_operations")
            if isinstance(lowered, (str, bytes)) or not isinstance(lowered, Sequence):
                raise ValueError(
                    f"R1 {op_type} requires explicit lowered_operations before RTTP handoff"
                )
            if any(not isinstance(item, Mapping) for item in lowered):
                raise ValueError(
                    f"R1 {op_type} lowered_operations must contain only mappings"
                )
            normalized.extend(
                normalize_r1_operations_for_rttp(list(lowered))
            )
            continue
        if op_type not in (
            "retime_event",
            "birth",
            "death",
            "assign_pattern",
            "short_turn",
            "reassign_vehicle",
        ):
            raise ValueError(f"unsupported R1 operation for RTTP handoff: {op_type}")
        normalized.append(dict(raw))
    if not normalized:
        raise ValueError("RTTP handoff contains no lowered operations")
    return normalized


def load_installed_rttp_provider() -> Any:
    """Load an installed provider and require the exact contract version.

    The source checkout still has to be verified separately with
    verify_observed_provider_sha(); a contract string is not a substitute for the
    exact Git commit lock.
    """

    module = importlib.import_module(RTTP_PROVIDER_MODULE)
    observed = getattr(module, "PROVIDER_CONTRACT_VERSION", None)
    if observed != RTTP_PROVIDER_CONTRACT:
        raise RuntimeError(
            f"RTTP provider contract mismatch: expected {RTTP_PROVIDER_CONTRACT}, got {observed}"
        )
    return module


def build_installed_runtime_substrate(
    network_payload: Mapping[str, Any],
    *,
    observed_provider_sha: str,
) -> Any:
    verify_observed_provider_sha(observed_provider_sha)
    module = load_installed_rttp_provider()
    if network_payload.get("provider_contract") != RTTP_PROVIDER_CONTRACT:
        raise ValueError("network payload provider contract mismatch")
    routes = tuple(
        module.R1RouteStructure(
            path_id=str(item["path_id"]),
            line_id=str(item["line_id"]),
            direction_id=str(item["direction_id"]),
            ordered_station_ids=tuple(str(x) for x in item["ordered_station_ids"]),
        )
        for item in network_payload["routes"]
    )
    return module.build_network_substrate(
        dataset_id=str(network_payload["dataset_id"]),
        routes=routes,
        network_authority_ref=str(network_payload["network_authority_ref"]),
        metadata=dict(network_payload.get("metadata", {})),
    )


def materialize_installed_runtime_fine_state(
    *,
    state_payload: Mapping[str, Any],
    network_substrate: Any,
    snapshot_id: str,
    service_date: str,
    timezone_id: str,
    observed_provider_sha: str,
) -> Any:
    verify_observed_provider_sha(observed_provider_sha)
    module = load_installed_rttp_provider()
    if state_payload.get("provider_contract") != RTTP_PROVIDER_CONTRACT:
        raise ValueError("R1 state payload provider contract mismatch")
    return module.materialize_r1_fine_state(
        r1_state=state_payload,
        network_substrate=network_substrate,
        snapshot_id=snapshot_id,
        service_date=service_date,
        timezone_id=timezone_id,
        mode=module.COUNT_FREE,
    )


def plan_anchored_reference_contract() -> dict[str, Any]:
    """Describe the separate route where a full RTTP plan snapshot is admissible."""

    return {
        "mode": PLAN_ANCHORED,
        "provider_contract": RTTP_PROVIDER_CONTRACT,
        "planned_timetable_may_be_input": True,
        "planned_absolute_times_may_be_input": True,
        "planned_trip_count_may_be_input": True,
        "must_not_feed_count_free_route": True,
        "count_free_comparison_role": "SEPARATE_ESTIMATOR_OR_POST_HOC_EXTERNAL_COMPARISON",
    }
