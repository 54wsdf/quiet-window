from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable, Literal

from scripts.mppd_r1_joint_model import (
    COARSE_STAGE,
    FINE_STAGE,
    EVENT_TRUST_REGION_S,
    JointState,
    ResolutionStage,
    ServiceEventState,
)

StructureOperation = Literal["birth", "death", "split", "merge", "path_reassignment"]


@dataclass(frozen=True)
class StabilityThresholds:
    """Engineering transition gate, not a scientific identifiability claim."""

    window: int = 3
    max_relative_service_count_change: float = 0.005
    min_trajectory_match_share: float = 0.98
    max_passenger_map_change_share: float = 0.02
    max_station_parameter_change_s: float = 5.0
    max_relative_objective_gain: float = 0.002


@dataclass(frozen=True)
class IterationMetrics:
    iteration: int
    inferred_service_count: int
    trajectory_match_share: float
    passenger_map_change_share: float
    station_parameter_max_change_s: float
    objective_value: float
    structure_operation_count: int = 0


@dataclass(frozen=True)
class EventUpdate:
    trajectory_id: str
    station_id: str
    sequence_index: int
    delta_s: float
    target: Literal["anchor", "arrival", "departure", "both"] = "anchor"


def allowed_event_deltas(stage: ResolutionStage) -> tuple[float, ...]:
    if stage == COARSE_STAGE:
        return (-5.0, 0.0, 5.0)
    if stage == FINE_STAGE:
        return tuple(float(x) for x in range(-5, 6))
    raise ValueError(f"unknown stage {stage}")


def validate_event_delta(delta_s: float, stage: ResolutionStage) -> None:
    if not isinstance(delta_s, (int, float)) or isinstance(delta_s, bool) or not math.isfinite(float(delta_s)):
        raise ValueError("event delta must be finite")
    delta_s = float(delta_s)
    if abs(delta_s) > EVENT_TRUST_REGION_S + 1e-9:
        raise ValueError("single event-time update exceeds the 5 s trust region")
    allowed = allowed_event_deltas(stage)
    if not any(abs(delta_s - x) <= 1e-9 for x in allowed):
        raise ValueError(f"delta {delta_s} is not admissible in {stage}")


def validate_structure_operations(stage: ResolutionStage, operations: Iterable[StructureOperation]) -> None:
    ops = list(operations)
    if stage == FINE_STAGE and ops:
        raise ValueError("fine 1 s refinement cannot silently change service structure; return to coarse stage first")
    unknown = set(ops) - {"birth", "death", "split", "merge", "path_reassignment"}
    if unknown:
        raise ValueError(f"unknown service structure operations: {sorted(unknown)}")


def _find_event(state: JointState, update: EventUpdate) -> ServiceEventState:
    for service in state.services:
        if service.trajectory_id != update.trajectory_id:
            continue
        for event in service.events:
            if event.station_id == update.station_id and event.sequence_index == update.sequence_index:
                return event
    raise KeyError(f"event not found: {update.trajectory_id}/{update.station_id}/{update.sequence_index}")


def apply_event_updates(
    state: JointState,
    updates: Iterable[EventUpdate],
    structure_operations: Iterable[StructureOperation] = (),
) -> JointState:
    """Apply one legal local update; posterior re-inference must happen after this call."""

    validate_structure_operations(state.stage, structure_operations)
    out = copy.deepcopy(state)
    seen: set[tuple[str, str, int, str]] = set()
    for update in updates:
        validate_event_delta(update.delta_s, state.stage)
        key = (update.trajectory_id, update.station_id, update.sequence_index, update.target)
        if key in seen:
            raise ValueError(f"duplicate event update in one iteration: {key}")
        seen.add(key)
        event = _find_event(out, update)
        d = float(update.delta_s)
        if update.target == "anchor":
            event.anchor_time_s += d
        elif update.target == "arrival":
            if event.arrival_time_s is None:
                raise ValueError("arrival update requires fine-stage arrival/departure initialization")
            event.arrival_time_s += d
        elif update.target == "departure":
            if event.departure_time_s is None:
                raise ValueError("departure update requires fine-stage arrival/departure initialization")
            event.departure_time_s += d
        elif update.target == "both":
            if event.arrival_time_s is None or event.departure_time_s is None:
                raise ValueError("both update requires fine-stage arrival/departure initialization")
            event.arrival_time_s += d
            event.departure_time_s += d
        else:
            raise ValueError(update.target)

    out.iteration += 1
    out.metadata = dict(out.metadata)
    out.metadata["last_update"] = {
        "stage": state.stage,
        "event_update_count": len(seen),
        "structure_operations": list(structure_operations),
        "posterior_reinference_required": True,
    }
    errors = out.validate()
    if errors:
        raise ValueError("illegal joint-state update: " + "; ".join(errors[:10]))
    return out


def initialize_fine_arrival_departure(state: JointState, initial_dwell_s: float = 0.0) -> JointState:
    if state.stage != COARSE_STAGE:
        raise ValueError("fine initialization requires a coarse state")
    if initial_dwell_s < 0 or initial_dwell_s > EVENT_TRUST_REGION_S:
        raise ValueError("initial dwell must be in [0, 5] s; larger dwell must be learned iteratively")
    out = copy.deepcopy(state)
    out.stage = FINE_STAGE
    for service in out.services:
        for event in service.events:
            event.arrival_time_s = float(event.anchor_time_s)
            event.departure_time_s = float(event.anchor_time_s) + float(initial_dwell_s)
    out.metadata = dict(out.metadata)
    out.metadata["fine_stage_initialization"] = {
        "source": "coarse_anchor_times",
        "initial_dwell_s": float(initial_dwell_s),
        "structure_locked_until_return_to_coarse": True,
    }
    errors = out.validate()
    if errors:
        raise ValueError("fine initialization failed: " + "; ".join(errors[:10]))
    return out


def _relative_change(a: float, b: float) -> float:
    return abs(b - a) / max(1.0, abs(a))


def coarse_structure_is_stable(
    history: list[IterationMetrics],
    thresholds: StabilityThresholds = StabilityThresholds(),
) -> bool:
    """Return whether the engineering conditions are sufficient to try 1 s refinement."""

    if len(history) < thresholds.window:
        return False
    rows = history[-thresholds.window :]
    for row in rows:
        if row.trajectory_match_share < thresholds.min_trajectory_match_share:
            return False
        if row.passenger_map_change_share > thresholds.max_passenger_map_change_share:
            return False
        if row.station_parameter_max_change_s > thresholds.max_station_parameter_change_s:
            return False
    for a, b in zip(rows, rows[1:]):
        if _relative_change(a.inferred_service_count, b.inferred_service_count) > thresholds.max_relative_service_count_change:
            return False
        gain = max(0.0, b.objective_value - a.objective_value)
        if gain / max(1.0, abs(a.objective_value)) > thresholds.max_relative_objective_gain:
            return False
    return True


def require_return_to_coarse(
    fine_metrics: IterationMetrics,
    *,
    structure_operation_proposed: bool,
    passenger_map_change_limit: float = 0.05,
) -> bool:
    """Fine-stage structural instability is a signal to return to 5 s inference."""

    return bool(
        structure_operation_proposed
        or fine_metrics.trajectory_match_share < 0.95
        or fine_metrics.passenger_map_change_share > passenger_map_change_limit
    )
