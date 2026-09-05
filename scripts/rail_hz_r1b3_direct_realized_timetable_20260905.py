from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

SCHEMA = "rail.hz-r1b3-direct-realized-timetable.v1"
Z90 = 1.6448536269514722
Z95 = 1.959963984540054
MIN_PROGRESS_S = 5.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_id(prefix: str, values: Iterable[Any]) -> str:
    text = "|".join(str(x) for x in values)
    return prefix + hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()


def event_key(root_id: str, event_index: int) -> str:
    return f"{root_id}::e{event_index:03d}"


def load_root_mass(path: Path) -> dict[str, float]:
    table = pq.read_table(path, columns=["root_id", "lineage_usage_mass"])
    return {str(r["root_id"]): float(r["lineage_usage_mass"]) for r in table.to_pylist()}


def load_event_support(path: Path) -> dict[str, dict[str, Any]]:
    return {str(r["event_key"]): r for r in iter_jsonl_gz(path)}


def contexts_by_edge(day: dict[str, Any]) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    out: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in day["propagation_field"]["contexts"]:
        key = (str(row["path_id"]), str(row["direction"]), int(row["from_station"]), int(row["to_station"]))
        out[key].append(row)
    for rows in out.values():
        rows.sort(key=lambda x: float(x["context_start_s"]))
    return out


def choose_context(rows: list[dict[str, Any]], upstream_seed_s: float) -> dict[str, Any] | None:
    if not rows:
        return None
    t = upstream_seed_s % 86400.0
    inside = [r for r in rows if float(r["context_start_s"]) <= t < float(r["context_end_s"])]
    if inside:
        return inside[0]
    return min(
        rows,
        key=lambda r: min(abs(t - float(r["context_start_s"])), abs(t - float(r["context_end_s"]))),
    )


def transition_factor(
    path_id: str,
    direction: str,
    left: dict[str, Any],
    right: dict[str, Any],
    ctx: dict[tuple[str, str, int, int], list[dict[str, Any]]],
) -> tuple[float, float, str]:
    u, v = int(left["station"]), int(right["station"])
    row = choose_context(ctx.get((path_id, direction, u, v), []), float(left["time_s"]))
    if row is None:
        lag = max(MIN_PROGRESS_S, float(right["time_s"]) - float(left["time_s"]))
        return lag, 90.0, "ROOT_RELATIVE_WEAK_FALLBACK"
    lag = max(MIN_PROGRESS_S, float(row["effective_initial_lag_s"]))
    eclass = str(row.get("evidence_class", "STRUCTURAL_PRIOR_FALLBACK"))
    sigma = 20.0 if eclass == "AFC_PLUS_STRUCTURAL_PRIOR" else 45.0
    return lag, sigma, eclass


def physical_group_key(root: dict[str, Any]) -> tuple[str, str, int, float]:
    return (
        str(root["afc_line"]),
        str(root["direction"]),
        int(root["terminal_station"]),
        round(float(root["terminal_anchor_s"]), 6),
    )


def build_group_meta(roots: list[dict[str, Any]], root_mass: dict[str, float]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    groups: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for r in roots:
        groups[physical_group_key(r)].append(r)
    group_meta: dict[str, dict[str, Any]] = {}
    root_to_group: dict[str, str] = {}
    for key, members in sorted(groups.items(), key=lambda kv: kv[0]):
        gid = stable_id("phys_", key)
        masses = [max(0.0, float(root_mass.get(str(r["root_id"]), 0.0))) for r in members]
        total = sum(masses)
        if total > 0:
            probs = [m / total for m in masses]
        else:
            probs = [1.0 / len(members)] * len(members)
        entropy = -sum(p * math.log(max(p, 1e-300)) for p in probs)
        rows = []
        for r, m, p in zip(members, masses, probs):
            rid = str(r["root_id"])
            root_to_group[rid] = gid
            rows.append({
                "root_id": rid,
                "path_id": str(r["path_id"]),
                "lineage_usage_mass": m,
                "path_family_weight_within_physical_service": p,
            })
        group_meta[gid] = {
            "physical_service_id": gid,
            "group_key": {"afc_line": key[0], "direction": key[1], "terminal_station": key[2], "terminal_anchor_s": key[3]},
            "member_count": len(members),
            "members": rows,
            "path_family_entropy_nats": entropy,
            "max_path_family_weight": max(probs),
            "duplicate_path_family_hypothesis": len(members) > 1,
        }
    return group_meta, root_to_group


def solve_root(
    root: dict[str, Any],
    ctx: dict[tuple[str, str, int, int], list[dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], bool]:
    events = list(root["events"])
    n = len(events)
    Q = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    observations = 0
    transitions: list[dict[str, Any]] = []

    # Absolute-time factors come only from AFC-inferred passenger-facing pulse events.
    # Structurally propagated event centers are never used as absolute observations.
    for j, ev in enumerate(events):
        if bool(ev.get("matched_observed_pulse", False)):
            y = float(ev["time_s"])
            sd = max(1e-6, float(ev["sd_s"]))
            w = 1.0 / (sd * sd)
            Q[j, j] += w
            b[j] += w * y
            observations += 1

    path_id = str(root["path_id"])
    direction = str(root["direction"])
    for j in range(n - 1):
        lag, sd, eclass = transition_factor(path_id, direction, events[j], events[j + 1], ctx)
        w = 1.0 / (sd * sd)
        # Factor: x[j+1] - x[j] ~ N(lag, sd^2)
        Q[j, j] += w
        Q[j + 1, j + 1] += w
        Q[j, j + 1] -= w
        Q[j + 1, j] -= w
        b[j] -= w * lag
        b[j + 1] += w * lag
        transitions.append({
            "from_station": int(events[j]["station"]),
            "to_station": int(events[j + 1]["station"]),
            "lag_mean_s": lag,
            "lag_sd_s": sd,
            "evidence_class": eclass,
        })

    if observations <= 0:
        raise SystemExit(f"root has no AFC absolute-time anchor: {root['root_id']}")
    # Numerical jitter only; it is many orders weaker than any scientific factor.
    Q += np.eye(n) * 1e-12
    mean = np.linalg.solve(Q, b)
    cov = np.linalg.inv(Q)
    sd = np.sqrt(np.maximum(0.0, np.diag(cov)))
    monotone = bool(np.all(np.diff(mean) >= MIN_PROGRESS_S - 1e-7))
    return mean, sd, transitions, monotone


def weighted_quantile(values: list[float], weights: list[float], p: float) -> float | None:
    if not values:
        return None
    order = np.argsort(np.asarray(values, dtype=float))
    xs = np.asarray(values, dtype=float)[order]
    ws = np.asarray(weights, dtype=float)[order]
    total = float(ws.sum())
    if total <= 0:
        return float(np.quantile(xs, p))
    c = np.cumsum(ws) / total
    return float(xs[min(len(xs) - 1, int(np.searchsorted(c, p, side="left")))])


def solve_shard(a: argparse.Namespace) -> dict[str, Any]:
    roots_raw = load_json(a.roots)
    day = load_json(a.service_init)
    if roots_raw.get("status") != "QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION":
        raise SystemExit("service roots not qualified")
    if day.get("status") != "QUALIFIED_SINGLE_FULL_SERVICE_DAY_SERVICE_INIT":
        raise SystemExit("service initialization not qualified")
    roots = list(roots_raw["roots"])
    root_mass = load_root_mass(a.root_mass)
    support = load_event_support(a.event_support)
    ctx = contexts_by_edge(day)
    group_meta, root_to_group = build_group_meta(roots, root_mass)
    selected = [r for r in roots if str(r["path_id"]) == a.path_id and str(r["direction"]) == a.direction]
    if not selected:
        raise SystemExit("no roots in requested path-direction")

    event_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    width95: list[float] = []
    center_change: list[float] = []
    observed_count = latent_count = 0
    monotone_failures = 0
    for root in selected:
        rid = str(root["root_id"])
        gid = root_to_group[rid]
        gm = group_meta[gid]
        member = next(x for x in gm["members"] if x["root_id"] == rid)
        mean, psd, transitions, monotone = solve_root(root, ctx)
        monotone_failures += int(not monotone)
        if not monotone:
            raise SystemExit(f"direct realized-timetable posterior violates monotonicity: {rid}")
        r_events = []
        for idx, (ev, m, s) in enumerate(zip(root["events"], mean, psd)):
            ek = event_key(rid, idx)
            sup = support.get(ek, {})
            matched = bool(ev.get("matched_observed_pulse", False))
            observed_count += int(matched)
            latent_count += int(not matched)
            width95.append(2.0 * Z95 * float(s))
            center_change.append(float(m) - float(ev["time_s"]))
            row = {
                "schema": SCHEMA,
                "service_date": str(root["date"]),
                "physical_service_id": gid,
                "root_id": rid,
                "path_id": str(root["path_id"]),
                "direction": str(root["direction"]),
                "path_family_weight_within_physical_service": float(member["path_family_weight_within_physical_service"]),
                "event_key": ek,
                "event_index": idx,
                "station": int(ev["station"]),
                "realized_time_mean_s": float(m),
                "realized_time_sd_s": float(s),
                "realized_time_q05_s": float(m - Z90 * s),
                "realized_time_q50_s": float(m),
                "realized_time_q95_s": float(m + Z90 * s),
                "realized_time_lower95_s": float(m - Z95 * s),
                "realized_time_upper95_s": float(m + Z95 * s),
                "source_seed_time_s": float(ev["time_s"]),
                "source_seed_sd_s": float(ev["sd_s"]),
                "source_seed_is_absolute_factor": matched,
                "event_evidence_class": str(ev["evidence_class"]),
                "matched_afc_passenger_facing_pulse": matched,
                "pulse_score": ev.get("pulse_score"),
                "pulse_excess_mass": ev.get("pulse_excess_mass"),
                "traversal_lineage_mass": float(sup.get("traversal_lineage_mass", 0.0)),
                "boarding_lineage_mass": float(sup.get("boarding_lineage_mass", 0.0)),
                "alighting_lineage_mass": float(sup.get("alighting_lineage_mass", 0.0)),
                "transfer_arrival_lineage_mass": float(sup.get("transfer_arrival_lineage_mass", 0.0)),
                "transfer_departure_lineage_mass": float(sup.get("transfer_departure_lineage_mass", 0.0)),
                "genealogy_support_state": str(sup.get("support_state", "NO_CURRENT_GENEALOGY_TRAVERSAL_SUPPORT")),
                "absolute_planned_timetable_used": False,
            }
            event_rows.append(row)
            r_events.append(row)
        root_rows.append({
            "schema": SCHEMA,
            "service_date": str(root["date"]),
            "physical_service_id": gid,
            "root_id": rid,
            "path_id": str(root["path_id"]),
            "direction": str(root["direction"]),
            "terminal_station": int(root["terminal_station"]),
            "terminal_anchor_s": float(root["terminal_anchor_s"]),
            "path_family_weight_within_physical_service": float(member["path_family_weight_within_physical_service"]),
            "lineage_usage_mass": float(member["lineage_usage_mass"]),
            "physical_group_member_count": int(gm["member_count"]),
            "path_family_entropy_nats": float(gm["path_family_entropy_nats"]),
            "max_path_family_weight": float(gm["max_path_family_weight"]),
            "matched_afc_event_share": sum(1 for x in root["events"] if x.get("matched_observed_pulse")) / len(root["events"]),
            "event_count": len(root["events"]),
            "posterior_monotone": monotone,
            "transition_factor_count": len(transitions),
            "absolute_planned_timetable_used": False,
        })

    write_jsonl_gz(a.out_events, event_rows)
    write_jsonl_gz(a.out_roots, root_rows)
    result = {
        "schema": SCHEMA,
        "command": "solve-shard",
        "status": "QUALIFIED_R1B3_REALIZED_TIMETABLE_FACTOR_SHARD",
        "service_date": "2019-01-04",
        "scope": "FULL_SERVICE_DAY_0400_TO_NEXT_0400_FULL_NETWORK_PARTITION_OF_SAME_SCIENTIFIC_OBJECT",
        "path_id": a.path_id,
        "direction": a.direction,
        "root_count": len(selected),
        "event_count": len(event_rows),
        "afc_absolute_observation_event_count": observed_count,
        "structurally_propagated_event_count": latent_count,
        "monotone_failure_count": monotone_failures,
        "posterior_interval_width95_s": {
            "p10": weighted_quantile(width95, [1.0] * len(width95), 0.10),
            "median": weighted_quantile(width95, [1.0] * len(width95), 0.50),
            "p90": weighted_quantile(width95, [1.0] * len(width95), 0.90),
        },
        "posterior_minus_source_seed_center_s": {
            "p10": weighted_quantile(center_change, [1.0] * len(center_change), 0.10),
            "median": weighted_quantile(center_change, [1.0] * len(center_change), 0.50),
            "p90": weighted_quantile(center_change, [1.0] * len(center_change), 0.90),
        },
        "scientific_semantics": {
            "target": "REALIZED_SERVICE_TIMETABLE_NOT_PLANNED_TIMETABLE_CORRECTION",
            "absolute_time_factors": "AFC_INFERRED_PASSENGER_FACING_PULSES_ONLY",
            "relative_factors": "SAME_DAY_AFC_PLUS_WEAK_STRUCTURAL_INTERSTATION_LAGS",
            "planned_absolute_timestamp_used": False,
            "posterior_recycling_used": False,
            "numerical_solver": "ONE_FIXED_LINEAR_GAUSSIAN_FACTOR_GRAPH_PER_SERVICE_HYPOTHESIS",
        },
    }
    a.out_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result


def merge_global(a: argparse.Namespace) -> dict[str, Any]:
    shard_summaries = [load_json(p) for p in a.summary]
    if len(shard_summaries) != 8 or not all(x.get("status") == "QUALIFIED_R1B3_REALIZED_TIMETABLE_FACTOR_SHARD" for x in shard_summaries):
        raise SystemExit("expected eight qualified R1B3 path-direction shards")
    root_rows = [r for p in a.root_file for r in iter_jsonl_gz(p)]
    event_rows = [r for p in a.event_file for r in iter_jsonl_gz(p)]
    if len(root_rows) != 1673 or len(event_rows) != 43584:
        raise SystemExit(f"unexpected R1B3 inventory: roots={len(root_rows)} events={len(event_rows)}")

    root_by_id = {str(r["root_id"]): r for r in root_rows}
    events_by_phys: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roots_by_phys: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in root_rows:
        roots_by_phys[str(r["physical_service_id"])].append(r)
    for e in event_rows:
        events_by_phys[str(e["physical_service_id"])].append(e)

    if len(roots_by_phys) != 1438:
        raise SystemExit(f"physical service grouping changed unexpectedly: {len(roots_by_phys)}")
    duplicate_groups = [g for g in roots_by_phys.values() if len(g) > 1]
    if len(duplicate_groups) != 235 or any(len(g) != 2 for g in duplicate_groups):
        raise SystemExit("expected 235 two-member B Up path-family ambiguity groups")

    physical_events: list[dict[str, Any]] = []
    max_family_weights: list[float] = []
    family_entropies: list[float] = []
    expected_actual_event_count = 0.0
    for gid, member_roots in sorted(roots_by_phys.items()):
        probs = {str(r["root_id"]): float(r["path_family_weight_within_physical_service"]) for r in member_roots}
        max_family_weights.append(max(probs.values()))
        family_entropies.append(float(member_roots[0]["path_family_entropy_nats"]))
        by_station: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for e in events_by_phys[gid]:
            by_station[int(e["station"])].append(e)
        for station, comps in sorted(by_station.items()):
            existence_weight = min(1.0, sum(probs[str(e["root_id"])] for e in comps))
            expected_actual_event_count += existence_weight
            if len(comps) == len(member_roots):
                # Shared physical station across all path-family hypotheses. Mixture moments
                # preserve any residual path-conditioned timing uncertainty without double-counting a train.
                weights = np.asarray([probs[str(e["root_id"])] for e in comps], dtype=float)
                weights = weights / weights.sum()
                means = np.asarray([float(e["realized_time_mean_s"]) for e in comps])
                vars_ = np.asarray([float(e["realized_time_sd_s"]) ** 2 for e in comps])
                mean = float(np.sum(weights * means))
                var = float(np.sum(weights * (vars_ + means * means)) - mean * mean)
                sd = math.sqrt(max(0.0, var))
                path_label = "SHARED_ACROSS_PATH_HYPOTHESES" if len(member_roots) > 1 else str(comps[0]["path_id"])
            else:
                # Branch-specific station: the event exists only conditional on that path-family hypothesis.
                if len(comps) != 1:
                    raise SystemExit(f"unexpected partial station mixture: {gid} station={station}")
                e = comps[0]
                mean = float(e["realized_time_mean_s"])
                sd = float(e["realized_time_sd_s"])
                path_label = str(e["path_id"])
            physical_events.append({
                "schema": SCHEMA,
                "service_date": "2019-01-04",
                "physical_service_id": gid,
                "station": station,
                "path_state": path_label,
                "event_existence_weight_within_physical_service": existence_weight,
                "realized_time_mean_s": mean,
                "realized_time_sd_s": sd,
                "realized_time_q05_s": mean - Z90 * sd,
                "realized_time_q50_s": mean,
                "realized_time_q95_s": mean + Z90 * sd,
                "realized_time_lower95_s": mean - Z95 * sd,
                "realized_time_upper95_s": mean + Z95 * sd,
                "component_root_ids": [str(e["root_id"]) for e in comps],
                "component_path_weights": {str(e["path_id"]): probs[str(e["root_id"])] for e in comps},
                "matched_afc_component_weight": sum(probs[str(e["root_id"])] for e in comps if bool(e["matched_afc_passenger_facing_pulse"])),
                "traversal_lineage_mass": sum(float(e["traversal_lineage_mass"]) for e in comps),
                "boarding_lineage_mass": sum(float(e["boarding_lineage_mass"]) for e in comps),
                "alighting_lineage_mass": sum(float(e["alighting_lineage_mass"]) for e in comps),
                "absolute_planned_timetable_used": False,
            })

    write_jsonl_gz(a.out_physical_events, physical_events)
    write_jsonl_gz(a.out_root_hypotheses, root_rows)

    widths = [2.0 * Z95 * float(e["realized_time_sd_s"]) for e in physical_events]
    eweights = [float(e["event_existence_weight_within_physical_service"]) for e in physical_events]
    direct_weights = [float(e["matched_afc_component_weight"]) for e in physical_events]
    physical_event_hypothesis_rows = len(physical_events)
    duplicate_candidate_roots_removed = 1673 - len(roots_by_phys)
    decisive90 = sum(1 for g in duplicate_groups if max(float(r["path_family_weight_within_physical_service"]) for r in g) >= 0.90)
    decisive75 = sum(1 for g in duplicate_groups if max(float(r["path_family_weight_within_physical_service"]) for r in g) >= 0.75)
    near_tie = sum(1 for g in duplicate_groups if max(float(r["path_family_weight_within_physical_service"]) for r in g) <= 0.60)

    gates = {
        "eight_path_direction_shards_qualified": len(shard_summaries) == 8,
        "all_candidate_roots_processed_once": len(root_rows) == 1673 and len(root_by_id) == 1673,
        "all_candidate_events_processed_once": len(event_rows) == 43584 and len({str(e["event_key"]) for e in event_rows}) == 43584,
        "physical_service_hypothesis_count_after_duplicate_family_collapse": len(roots_by_phys) == 1438,
        "expected_b_up_duplicate_family_groups": len(duplicate_groups) == 235,
        "all_root_posteriors_monotone": all(bool(r["posterior_monotone"]) for r in root_rows),
        "no_absolute_planned_timetable_time_used": all(not bool(e["absolute_planned_timetable_used"]) for e in event_rows),
        "no_scientific_posterior_recycling": True,
        "physical_event_intervals_finite": all(math.isfinite(float(e["realized_time_mean_s"])) and math.isfinite(float(e["realized_time_sd_s"])) for e in physical_events),
    }
    if not all(gates.values()):
        raise SystemExit("R1B3 global qualification failed: " + json.dumps(gates))

    result = {
        "schema": SCHEMA,
        "status": "QUALIFIED_R1B3_DIRECT_REALIZED_TIMETABLE_POSTERIOR",
        "service_date": "2019-01-04",
        "scope": "FULL_SERVICE_DAY_0400_TO_NEXT_0400_FULL_NETWORK_FULL_QUALIFIED_PASSENGER_DOMAIN",
        "candidate_root_hypothesis_count": 1673,
        "physical_service_hypothesis_count": len(roots_by_phys),
        "duplicate_path_family_hypotheses_collapsed": duplicate_candidate_roots_removed,
        "ambiguous_two-family_physical_service_count": len(duplicate_groups),
        "candidate_root_event_state_count": 43584,
        "physical_event_hypothesis_row_count": physical_event_hypothesis_rows,
        "expected_actual_physical_event_count_under_path_family_posterior": expected_actual_event_count,
        "afc_absolute_observation_event_count_across_candidate_hypotheses": sum(int(x["afc_absolute_observation_event_count"]) for x in shard_summaries),
        "structurally_propagated_event_count_across_candidate_hypotheses": sum(int(x["structurally_propagated_event_count"]) for x in shard_summaries),
        "physical_event_matched_afc_expected_count": sum(direct_weights),
        "realized_time_interval_width95_s_weighted_by_event_existence": {
            "p10": weighted_quantile(widths, eweights, 0.10),
            "median": weighted_quantile(widths, eweights, 0.50),
            "p90": weighted_quantile(widths, eweights, 0.90),
        },
        "path_family_ambiguity_for_235_b_up_physical_services": {
            "max_family_weight_median": weighted_quantile(max_family_weights[-235:] if False else [max(float(r["path_family_weight_within_physical_service"]) for r in g) for g in duplicate_groups], [1.0] * len(duplicate_groups), 0.50),
            "decisive_ge_0_90_count": decisive90,
            "decisive_ge_0_75_count": decisive75,
            "near_tie_le_0_60_count": near_tie,
        },
        "qualification_gates": gates,
        "scientific_semantics": {
            "target": "PASSENGER_FACING_REALIZED_SERVICE_TIMETABLE_POSTERIOR",
            "planned_timetable_role": "WEAK_RELATIVE_STRUCTURE_ONLY",
            "planned_absolute_timestamp_role": "FORBIDDEN",
            "actual_equals_planned_plus_correction_model": False,
            "absolute_timing_evidence": "AFC_INFERRED_PASSENGER_FACING_PULSE_EVENTS",
            "relative_timing_evidence": "SAME_DAY_AFC_PLUS_WEAK_STRUCTURAL_INTERSTATION_PROPAGATION",
            "genealogy_role": "PHYSICAL_SERVICE_AND_PATH_FAMILY_SUPPORT_WITHOUT_POSTERIOR_RECYCLING",
            "scientific_self_iteration": False,
            "numerical_iteration": "NONE_REQUIRED_FOR_LINEAR_GAUSSIAN_ROOT_FACTOR_SOLVES",
            "candidate_root_identity": "LATENT_HYPOTHESIS_NOT_OBSERVED_ATS_TRAIN_ID",
        },
        "next_stage": "R1B_4_QUALIFICATION_AUDIT_ONLY_NO_NEW_SERVICE_PASSENGER_SERVICE_LOOP",
    }
    a.out_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("solve-shard")
    s.add_argument("--roots", type=Path, required=True)
    s.add_argument("--service-init", type=Path, required=True)
    s.add_argument("--root-mass", type=Path, required=True)
    s.add_argument("--event-support", type=Path, required=True)
    s.add_argument("--path-id", required=True)
    s.add_argument("--direction", required=True)
    s.add_argument("--out-events", type=Path, required=True)
    s.add_argument("--out-roots", type=Path, required=True)
    s.add_argument("--out-summary", type=Path, required=True)

    s = sub.add_parser("merge-global")
    s.add_argument("--event-file", type=Path, action="append", required=True)
    s.add_argument("--root-file", type=Path, action="append", required=True)
    s.add_argument("--summary", type=Path, action="append", required=True)
    s.add_argument("--out-physical-events", type=Path, required=True)
    s.add_argument("--out-root-hypotheses", type=Path, required=True)
    s.add_argument("--out-summary", type=Path, required=True)
    a = ap.parse_args()
    if a.command == "solve-shard":
        result = solve_shard(a)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        merge_global(a)


if __name__ == "__main__":
    main()
