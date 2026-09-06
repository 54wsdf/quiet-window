from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

JOINT_STATE_SCHEMA = "mppd.r1-joint-state.v2"
JOINT_CONFIG_SCHEMA = "mppd.r1-joint-config.v2"

COARSE_STAGE = "COARSE_5S"
FINE_STAGE = "FINE_1S"
ResolutionStage = Literal["COARSE_5S", "FINE_1S"]

ACCESS_MIN_S = 15.0
EGRESS_MIN_S = 15.0
TRANSFER_MIN_S = 5.0
EVENT_TRUST_REGION_S = 5.0
COARSE_UPDATE_STEP_S = 5.0
FINE_UPDATE_STEP_S = 1.0


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


@dataclass(frozen=True)
class R1Config:
    schema: str = JOINT_CONFIG_SCHEMA
    train_count_is_input: bool = False
    planned_timetable_used_in_primary_inference: bool = False
    legacy_candidate_roots_used_as_input: bool = False
    access_min_s: float = ACCESS_MIN_S
    egress_min_s: float = EGRESS_MIN_S
    transfer_min_s: float = TRANSFER_MIN_S
    event_trust_region_s: float = EVENT_TRUST_REGION_S
    coarse_update_step_s: float = COARSE_UPDATE_STEP_S
    fine_update_step_s: float = FINE_UPDATE_STEP_S
    planned_timetable_role: str = "POST_HOC_EXTERNAL_COMPARISON_ONLY"
    legacy_1673_role: str = "HISTORICAL_BASELINE_ONLY"
    legacy_plus_60_clock_role: str = "DIAGNOSTIC_ONLY_NOT_ADMISSIBLE_STATE_UPDATE"

    def validate(self) -> None:
        if self.train_count_is_input:
            raise ValueError("R1 requires count-free service inference: train_count_is_input must be False")
        if self.planned_timetable_used_in_primary_inference:
            raise ValueError("planned timetable absolute times are excluded from primary R1 inference")
        if self.legacy_candidate_roots_used_as_input:
            raise ValueError("legacy candidate roots are baseline-only and cannot define the R1 service world")
        if self.access_min_s < 15.0 or self.egress_min_s < 15.0:
            raise ValueError("R1 access and egress physical lower bounds must be at least 15 s")
        if self.transfer_min_s < 5.0:
            raise ValueError("R1 transfer physical lower bound must be at least 5 s")
        if self.event_trust_region_s > 5.0:
            raise ValueError("a single R1 event-time update may not exceed 5 s")
        if self.coarse_update_step_s != 5.0:
            raise ValueError("coarse structural inference uses 5 s update resolution")
        if self.fine_update_step_s != 1.0:
            raise ValueError("fine event-time refinement uses 1 s update resolution")


@dataclass
class ServiceEventState:
    station_id: str
    sequence_index: int
    anchor_time_s: float
    arrival_time_s: float | None = None
    departure_time_s: float | None = None
    evidence_class: str = "AFC_PASSENGER_FACING_RIDGE"

    def validate(self, stage: ResolutionStage) -> list[str]:
        errors: list[str] = []
        if not finite(self.anchor_time_s):
            errors.append(f"station {self.station_id}: anchor_time_s is not finite")
        if self.sequence_index < 0:
            errors.append(f"station {self.station_id}: sequence_index < 0")
        if stage == FINE_STAGE:
            if self.arrival_time_s is None or self.departure_time_s is None:
                errors.append(f"station {self.station_id}: fine stage requires separate arrival/departure times")
            elif not finite(self.arrival_time_s) or not finite(self.departure_time_s):
                errors.append(f"station {self.station_id}: non-finite arrival/departure")
            elif float(self.departure_time_s) < float(self.arrival_time_s):
                errors.append(f"station {self.station_id}: departure precedes arrival")
        return errors


@dataclass
class ServiceTrajectoryState:
    trajectory_id: str
    line_id: str
    direction_id: str
    path_id: str
    events: list[ServiceEventState]
    path_ambiguous: bool = False
    direction_ambiguous: bool = False
    support_weight: float = 0.0
    evidence_score: float = 0.0
    warm_start_source: str | None = None

    def validate(self, stage: ResolutionStage) -> list[str]:
        errors: list[str] = []
        if not self.trajectory_id:
            errors.append("empty trajectory_id")
        if len(self.events) < 2:
            errors.append(f"{self.trajectory_id}: fewer than two service events")
            return errors
        ordered = sorted(self.events, key=lambda e: e.sequence_index)
        if [e.sequence_index for e in ordered] != list(range(len(ordered))):
            errors.append(f"{self.trajectory_id}: sequence indices are not contiguous from zero")
        errors.extend(f"{self.trajectory_id}: {x}" for e in ordered for x in e.validate(stage))
        for a, b in zip(ordered, ordered[1:]):
            if finite(a.anchor_time_s) and finite(b.anchor_time_s) and b.anchor_time_s <= a.anchor_time_s:
                errors.append(f"{self.trajectory_id}: non-increasing coarse anchor times")
            if stage == FINE_STAGE and a.departure_time_s is not None and b.arrival_time_s is not None:
                if float(b.arrival_time_s) <= float(a.departure_time_s):
                    errors.append(f"{self.trajectory_id}: non-positive interstation running time")
        return errors


@dataclass
class StationMovementState:
    station_id: str
    identification: Literal["DIRECTLY_IDENTIFIED", "HIERARCHICALLY_INFERRED", "UNRESOLVED"]
    access_q05_s: float | None = None
    access_q50_s: float | None = None
    access_q95_s: float | None = None
    egress_q05_s: float | None = None
    egress_q50_s: float | None = None
    egress_q95_s: float | None = None
    evidence_mass: float = 0.0

    def validate(self, cfg: R1Config) -> list[str]:
        errors: list[str] = []
        for prefix, lower in (("access", cfg.access_min_s), ("egress", cfg.egress_min_s)):
            vals = [getattr(self, f"{prefix}_q05_s"), getattr(self, f"{prefix}_q50_s"), getattr(self, f"{prefix}_q95_s")]
            if all(v is None for v in vals):
                if self.identification != "UNRESOLVED":
                    errors.append(f"station {self.station_id}: {prefix} distribution missing")
                continue
            if any(v is None or not finite(v) for v in vals):
                errors.append(f"station {self.station_id}: incomplete {prefix} quantiles")
                continue
            q05, q50, q95 = map(float, vals)
            if q05 < lower:
                errors.append(f"station {self.station_id}: {prefix} q05 below physical lower bound")
            if not q05 <= q50 <= q95:
                errors.append(f"station {self.station_id}: non-monotone {prefix} quantiles")
        return errors


@dataclass
class TransferPathState:
    path_id: str
    q05_s: float
    q50_s: float
    q95_s: float
    weight: float

    def validate(self, cfg: R1Config) -> list[str]:
        errors: list[str] = []
        if not all(finite(v) for v in (self.q05_s, self.q50_s, self.q95_s, self.weight)):
            return [f"transfer path {self.path_id}: non-finite parameter"]
        if self.q05_s < cfg.transfer_min_s:
            errors.append(f"transfer path {self.path_id}: q05 below physical lower bound")
        if not self.q05_s <= self.q50_s <= self.q95_s:
            errors.append(f"transfer path {self.path_id}: non-monotone quantiles")
        if self.weight <= 0:
            errors.append(f"transfer path {self.path_id}: non-positive mixture weight")
        return errors


@dataclass
class TransferMovementState:
    movement_id: str
    station_id: str
    from_line_id: str
    from_direction_id: str
    to_line_id: str
    to_direction_id: str
    paths: list[TransferPathState]

    def validate(self, cfg: R1Config) -> list[str]:
        errors = [f"{self.movement_id}: {x}" for p in self.paths for x in p.validate(cfg)]
        if not self.paths:
            errors.append(f"{self.movement_id}: no latent/physical transfer paths")
        if self.paths:
            total = sum(p.weight for p in self.paths)
            if abs(total - 1.0) > 1e-6:
                errors.append(f"{self.movement_id}: transfer path weights sum to {total}, not 1")
        return errors


@dataclass
class PassengerPosteriorSummary:
    passenger_mass: float
    resolved_mass: float
    no_route_support_mass: float = 0.0
    unresolved_with_route_mass: float = 0.0
    map_chain_change_share: float | None = None
    max_time_closure_error_s: float | None = None

    @property
    def resolved_share(self) -> float:
        return self.resolved_mass / self.passenger_mass if self.passenger_mass > 0 else 0.0


@dataclass
class JointState:
    schema: str
    stage: ResolutionStage
    iteration: int
    config: R1Config
    services: list[ServiceTrajectoryState]
    station_movements: list[StationMovementState] = field(default_factory=list)
    transfer_movements: list[TransferMovementState] = field(default_factory=list)
    passenger_summary: PassengerPosteriorSummary | None = None
    objective_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def inferred_service_count(self) -> int:
        return len(self.services)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.schema != JOINT_STATE_SCHEMA:
            errors.append(f"expected schema {JOINT_STATE_SCHEMA}")
        self.config.validate()
        if self.stage not in (COARSE_STAGE, FINE_STAGE):
            errors.append(f"unknown stage {self.stage}")
        if self.iteration < 0:
            errors.append("iteration < 0")
        if not self.services:
            errors.append("joint state contains no inferred service trajectories")
        ids = [s.trajectory_id for s in self.services]
        if len(ids) != len(set(ids)):
            errors.append("duplicate trajectory_id")
        errors.extend(x for s in self.services for x in s.validate(self.stage))
        errors.extend(x for s in self.station_movements for x in s.validate(self.config))
        errors.extend(x for m in self.transfer_movements for x in m.validate(self.config))
        if self.passenger_summary is not None:
            p = self.passenger_summary
            if p.passenger_mass <= 0 or p.resolved_mass < 0 or p.resolved_mass > p.passenger_mass:
                errors.append("invalid passenger mass accounting")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def bootstrap_from_count_free_discovery(doc: dict[str, Any]) -> JointState:
    """Translate the current AFC-only count-free result into a new R1 warm start.

    The count-free trajectory set is explicitly a warm start. It is not an authority,
    its size is not fixed, and coarse ridge event times are not yet claimed as
    realized arrival/departure times.
    """
    sem = doc.get("semantics", {})
    if sem.get("train_count_is_input") is not False:
        raise ValueError("count-free bootstrap requires train_count_is_input=False")
    if sem.get("planned_timetable_used") is not False:
        raise ValueError("count-free bootstrap cannot use planned absolute timetable times")
    if sem.get("legacy_candidate_roots_used_as_input") is not False:
        raise ValueError("legacy candidate roots cannot define the new R1 warm start")

    services: list[ServiceTrajectoryState] = []
    for tr in doc.get("trajectories", []):
        events = [
            ServiceEventState(
                station_id=str(e["station"]),
                sequence_index=int(e["sequence_index"]),
                anchor_time_s=float(e["time_s"]),
            )
            for e in tr.get("events", [])
        ]
        services.append(
            ServiceTrajectoryState(
                trajectory_id=str(tr["trajectory_id"]),
                line_id=str(tr["afc_line"]),
                direction_id=str(tr["direction"]),
                path_id=str(tr["path_id"]),
                events=events,
                path_ambiguous=bool(tr.get("path_ambiguous", False)),
                direction_ambiguous=bool(tr.get("direction_ambiguous", False)),
                support_weight=float(tr.get("support_weight", 0.0)),
                evidence_score=float(tr.get("evidence_score", 0.0)),
                warm_start_source=str(doc.get("schema", "UNKNOWN")),
            )
        )

    state = JointState(
        schema=JOINT_STATE_SCHEMA,
        stage=COARSE_STAGE,
        iteration=0,
        config=R1Config(),
        services=services,
        metadata={
            "warm_start_only": True,
            "service_count_is_free_after_bootstrap": True,
            "source_service_count": len(services),
            "source_schema": doc.get("schema"),
            "legacy_baseline": doc.get("legacy_baseline"),
            "planned_timetable_used": False,
            "legacy_candidate_roots_used_as_input": False,
        },
    )
    errors = state.validate()
    if errors:
        raise ValueError("invalid count-free warm start: " + "; ".join(errors[:10]))
    return state


def load_count_free_bootstrap(input_path: Path, output_path: Path) -> JointState:
    doc = json.loads(input_path.read_text(encoding="utf-8"))
    state = bootstrap_from_count_free_discovery(doc)
    state.write(output_path)
    return state
