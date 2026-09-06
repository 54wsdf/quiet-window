from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from scripts.mppd_r1_joint_model import (
    COARSE_STAGE,
    FINE_STAGE,
    JointState,
    ServiceCirculationState,
    ServiceTrajectoryState,
)

INFERENCE_METHOD = "ORDER_PRESERVING_MAX_CARDINALITY_MIN_GAP_DP"


@dataclass(frozen=True)
class TurnbackRule:
    rule_id: str
    path_id: str
    line_id: str
    from_direction_id: str
    to_direction_id: str
    station_id: str
    min_turnback_s: float
    max_turnback_s: float

    def __post_init__(self) -> None:
        if not all(
            (
                self.rule_id,
                self.path_id,
                self.line_id,
                self.from_direction_id,
                self.to_direction_id,
                self.station_id,
            )
        ):
            raise ValueError("turnback rule IDs and route fields must be non-empty")
        if self.from_direction_id == self.to_direction_id:
            raise ValueError("turnback rule must change direction")
        if not math.isfinite(float(self.min_turnback_s)) or not math.isfinite(
            float(self.max_turnback_s)
        ):
            raise ValueError("turnback bounds must be finite")
        if self.min_turnback_s < 0:
            raise ValueError("min_turnback_s must be non-negative")
        if self.max_turnback_s < self.min_turnback_s:
            raise ValueError("max_turnback_s must be >= min_turnback_s")


@dataclass(frozen=True)
class CirculationLink:
    rule_id: str
    from_trajectory_id: str
    to_trajectory_id: str
    station_id: str
    turnback_gap_s: float


@dataclass(frozen=True)
class CirculationInferenceAudit:
    service_count: int
    rule_count: int
    matched_link_count: int
    circulation_count: int
    singleton_circulation_count: int
    unmatched_chain_start_count: int
    unmatched_chain_end_count: int
    physical_vehicle_identity_claimed: bool = False
    planned_timetable_used: bool = False


def turnback_rules_from_network_authority(
    authority: Mapping[str, Any],
    *,
    min_turnback_s: float,
    max_turnback_s: float,
) -> tuple[TurnbackRule, ...]:
    """Derive terminal reversal relations from physical route geometry only.

    The time bounds are caller-supplied physical/experimental assumptions; this
    function never learns them from a planned timetable and never imports planned
    train IDs or service counts.
    """

    semantics = authority.get("semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("network authority requires semantics")
    if semantics.get("contains_planned_timetable") is not False:
        raise ValueError("turnback rules require plan-free network authority")
    if semantics.get("contains_expected_service_count") is not False:
        raise ValueError("turnback rules may not use expected service count")
    if semantics.get("contains_service_event_inventory") is not False:
        raise ValueError("turnback rules may not use service event inventory")

    raw_paths = authority.get("line_paths")
    if isinstance(raw_paths, (str, bytes)) or not isinstance(raw_paths, Sequence):
        raise ValueError("network authority line_paths must be a sequence")

    rows: list[dict[str, Any]] = []
    for raw in raw_paths:
        if not isinstance(raw, Mapping):
            raise ValueError("network authority line path must be a mapping")
        stations_raw = raw.get("station_ids")
        if isinstance(stations_raw, (str, bytes)) or not isinstance(
            stations_raw, Sequence
        ):
            raise ValueError("network authority route requires station_ids")
        stations = tuple(str(item) for item in stations_raw)
        if len(stations) < 2:
            raise ValueError("network authority route requires at least two stations")
        rows.append(
            {
                "path_id": str(raw.get("path_id", "")),
                "line_id": str(raw.get("line_id", "")),
                "direction_id": str(raw.get("direction_id", "")),
                "station_ids": stations,
            }
        )

    rules: list[TurnbackRule] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for left in rows:
        for right in rows:
            if left is right:
                continue
            if left["path_id"] != right["path_id"] or left["line_id"] != right["line_id"]:
                continue
            if left["direction_id"] == right["direction_id"]:
                continue
            station = left["station_ids"][-1]
            if station != right["station_ids"][0]:
                continue
            key = (
                left["path_id"],
                left["direction_id"],
                right["direction_id"],
                station,
                left["line_id"],
            )
            if key in seen:
                continue
            seen.add(key)
            rules.append(
                TurnbackRule(
                    rule_id=(
                        f"terminal:{left['path_id']}:{left['direction_id']}->"
                        f"{right['direction_id']}:{station}"
                    ),
                    path_id=left["path_id"],
                    line_id=left["line_id"],
                    from_direction_id=left["direction_id"],
                    to_direction_id=right["direction_id"],
                    station_id=station,
                    min_turnback_s=float(min_turnback_s),
                    max_turnback_s=float(max_turnback_s),
                )
            )
    if not rules:
        raise ValueError("network authority yields no terminal turnback relations")
    return tuple(sorted(rules, key=lambda row: row.rule_id))


def _ordered_events(service: ServiceTrajectoryState):
    return sorted(service.events, key=lambda event: event.sequence_index)


def _terminal_time(service: ServiceTrajectoryState, stage: str) -> float:
    event = _ordered_events(service)[-1]
    if stage == FINE_STAGE:
        if event.arrival_time_s is None:
            raise ValueError(f"{service.trajectory_id}: fine terminal arrival missing")
        return float(event.arrival_time_s)
    if stage == COARSE_STAGE:
        return float(event.anchor_time_s)
    raise ValueError(f"unknown R1 stage {stage}")


def _origin_time(service: ServiceTrajectoryState, stage: str) -> float:
    event = _ordered_events(service)[0]
    if stage == FINE_STAGE:
        if event.departure_time_s is None:
            raise ValueError(f"{service.trajectory_id}: fine origin departure missing")
        return float(event.departure_time_s)
    if stage == COARSE_STAGE:
        return float(event.anchor_time_s)
    raise ValueError(f"unknown R1 stage {stage}")


def _score_better(
    left: tuple[int, float], right: tuple[int, float]
) -> tuple[int, float]:
    """Lexicographic objective: maximize links, then minimize total turnback gap."""

    if right[0] > left[0]:
        return right
    if right[0] < left[0]:
        return left
    return right if right[1] < left[1] - 1e-12 else left


def _match_one_rule(
    state: JointState,
    rule: TurnbackRule,
) -> list[CirculationLink]:
    predecessors: list[tuple[float, str, ServiceTrajectoryState]] = []
    successors: list[tuple[float, str, ServiceTrajectoryState]] = []

    for service in state.services:
        events = _ordered_events(service)
        if not events:
            continue
        if (
            service.path_id == rule.path_id
            and service.line_id == rule.line_id
            and service.direction_id == rule.from_direction_id
            and events[-1].station_id == rule.station_id
        ):
            predecessors.append(
                (_terminal_time(service, state.stage), service.trajectory_id, service)
            )
        if (
            service.path_id == rule.path_id
            and service.line_id == rule.line_id
            and service.direction_id == rule.to_direction_id
            and events[0].station_id == rule.station_id
        ):
            successors.append(
                (_origin_time(service, state.stage), service.trajectory_id, service)
            )

    predecessors.sort(key=lambda row: (row[0], row[1]))
    successors.sort(key=lambda row: (row[0], row[1]))
    n, m = len(predecessors), len(successors)
    if n == 0 or m == 0:
        return []

    # dp[i][j] stores the lexicographic optimum for first i predecessors and j successors.
    # choices: 0 skip predecessor, 1 skip successor, 2 match i-1 with j-1.
    dp: list[list[tuple[int, float]]] = [
        [(0, 0.0) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    choice: list[list[int]] = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    for j in range(1, m + 1):
        choice[0][j] = 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = dp[i - 1][j]
            best_choice = 0
            candidate = dp[i][j - 1]
            selected = _score_better(best, candidate)
            if selected is candidate:
                best = candidate
                best_choice = 1

            terminal_time = predecessors[i - 1][0]
            origin_time = successors[j - 1][0]
            gap = origin_time - terminal_time
            if rule.min_turnback_s <= gap <= rule.max_turnback_s:
                prior = dp[i - 1][j - 1]
                candidate = (prior[0] + 1, prior[1] + gap)
                selected = _score_better(best, candidate)
                if selected is candidate:
                    best = candidate
                    best_choice = 2
            dp[i][j] = best
            choice[i][j] = best_choice

    links: list[CirculationLink] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            continue
        if j == 0:
            i -= 1
            continue
        c = choice[i][j]
        if c == 2:
            terminal_time, pred_id, _ = predecessors[i - 1]
            origin_time, succ_id, _ = successors[j - 1]
            links.append(
                CirculationLink(
                    rule_id=rule.rule_id,
                    from_trajectory_id=pred_id,
                    to_trajectory_id=succ_id,
                    station_id=rule.station_id,
                    turnback_gap_s=float(origin_time - terminal_time),
                )
            )
            i -= 1
            j -= 1
        elif c == 1:
            j -= 1
        else:
            i -= 1
    links.reverse()
    return links


def infer_service_circulations(
    state: JointState,
    rules: Sequence[TurnbackRule],
) -> tuple[JointState, CirculationInferenceAudit]:
    """Infer revenue-service successor chains without fixing service count or vehicle IDs."""

    errors = state.validate()
    if errors:
        raise ValueError("cannot infer circulation from invalid R1 state: " + "; ".join(errors[:10]))
    if not rules:
        raise ValueError("at least one turnback rule is required")

    links: list[CirculationLink] = []
    for rule in rules:
        links.extend(_match_one_rule(state, rule))

    successor: dict[str, str] = {}
    predecessor: dict[str, str] = {}
    for link in links:
        existing = successor.get(link.from_trajectory_id)
        if existing is not None and existing != link.to_trajectory_id:
            raise ValueError(
                f"trajectory {link.from_trajectory_id} receives conflicting circulation successors"
            )
        existing = predecessor.get(link.to_trajectory_id)
        if existing is not None and existing != link.from_trajectory_id:
            raise ValueError(
                f"trajectory {link.to_trajectory_id} receives conflicting circulation predecessors"
            )
        successor[link.from_trajectory_id] = link.to_trajectory_id
        predecessor[link.to_trajectory_id] = link.from_trajectory_id

    by_id = {service.trajectory_id: service for service in state.services}
    ordered_ids = sorted(
        by_id,
        key=lambda trajectory_id: (
            _origin_time(by_id[trajectory_id], state.stage),
            trajectory_id,
        ),
    )

    chains: list[list[str]] = []
    visited: set[str] = set()
    starts = [trajectory_id for trajectory_id in ordered_ids if trajectory_id not in predecessor]
    for start in starts:
        if start in visited:
            continue
        chain: list[str] = []
        current: str | None = start
        while current is not None:
            if current in visited:
                raise ValueError("circulation inference created a temporal cycle")
            visited.add(current)
            chain.append(current)
            current = successor.get(current)
        chains.append(chain)
    for trajectory_id in ordered_ids:
        if trajectory_id not in visited:
            raise ValueError(
                f"trajectory {trajectory_id} belongs to a circulation cycle or disconnected conflict"
            )

    chain_states = [
        ServiceCirculationState(
            circulation_id=f"r1-circ:{index:05d}",
            ordered_trajectory_ids=chain,
            chain_type="TURNBACK",
            start_boundary="UNRESOLVED",
            end_boundary="UNRESOLVED",
            linkage_method=INFERENCE_METHOD,
            confidence=None,
            evidence_score=float(len(chain) - 1),
            physical_vehicle_identity_claimed=False,
        )
        for index, chain in enumerate(chains)
    ]

    out = copy.deepcopy(state)
    out.service_circulations = chain_states
    out.metadata = dict(out.metadata)
    out.metadata["service_circulation_status"] = "INFERRED_COUNT_FREE_SUCCESSOR_CHAINS"
    out.metadata["service_circulation_method"] = INFERENCE_METHOD
    out.metadata["physical_vehicle_identity_claimed"] = False
    out.metadata["planned_timetable_used_for_service_circulation"] = False
    out.metadata["circulation_link_count"] = len(links)
    out.metadata["circulation_count"] = len(chain_states)

    errors = out.validate()
    if errors:
        raise ValueError("inferred circulation state is invalid: " + "; ".join(errors[:10]))

    audit = CirculationInferenceAudit(
        service_count=len(state.services),
        rule_count=len(rules),
        matched_link_count=len(links),
        circulation_count=len(chain_states),
        singleton_circulation_count=sum(len(chain) == 1 for chain in chains),
        unmatched_chain_start_count=len(chain_states),
        unmatched_chain_end_count=len(chain_states),
        physical_vehicle_identity_claimed=False,
        planned_timetable_used=False,
    )
    return out, audit
