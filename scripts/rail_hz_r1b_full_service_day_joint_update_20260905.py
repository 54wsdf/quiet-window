from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BEAM = 16
ROUTE_PRIOR_SCALE_S = 900.0
KERNEL_DAMPING = 0.35
SHIFT_GRID = [-180, -120, -90, -60, -30, 0, 30, 60, 90, 120, 180]
MIN_ROOT_USAGE_MASS = 50.0
TIME_SEARCH_MARGIN_S = 900.0


@dataclass(frozen=True)
class Kernel:
    median: float
    sigma: float
    evidence_class: str


@dataclass(frozen=True)
class ServiceLegEvent:
    root_id: str
    path_id: str
    direction: str
    from_station: int
    to_station: int
    from_t: float
    to_t: float
    from_sd: float
    to_sd: float


@dataclass
class ChainState:
    score: float
    roots: tuple[str, ...]
    last_to_t: float
    last_to_sd: float
    factors: tuple[dict[str, Any], ...]


def logsumexp(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    if not vals:
        return -math.inf
    m = max(vals)
    return m + math.log(sum(math.exp(x - m) for x in vals))


def lognormal_cdf(x: float, kernel: Kernel) -> float:
    if x <= 0:
        return 0.0
    mu = math.log(kernel.median)
    z = (math.log(x) - mu) / (kernel.sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def log_interval_density(center: float, sd: float, kernel: Kernel) -> float:
    if not math.isfinite(center) or not math.isfinite(sd):
        return -math.inf
    sd = max(0.0, float(sd))
    if sd < 0.5:
        if center <= 0:
            return -math.inf
        mu = math.log(kernel.median)
        lx = math.log(center)
        return -math.log(center * kernel.sigma * math.sqrt(2.0 * math.pi)) - ((lx - mu) ** 2) / (2.0 * kernel.sigma ** 2)
    lo = center - 1.96 * sd
    hi = center + 1.96 * sd
    if hi <= 0:
        return -math.inf
    lo_pos = max(1e-6, lo)
    p = max(0.0, lognormal_cdf(hi, kernel) - lognormal_cdf(lo_pos, kernel))
    width = max(1.0, hi - lo_pos)
    if p <= 1e-300:
        return -math.inf
    return math.log(p / width)


def stable_chain_hash(route_rank: int, roots: tuple[str, ...]) -> int:
    text = f"r{route_rank}|" + "|".join(roots)
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big", signed=False)


def initial_kernels() -> dict[str, Kernel]:
    return {
        "access": Kernel(180.0, 0.90, "BROAD_INITIAL_PRIOR"),
        "transfer": Kernel(180.0, 0.85, "BROAD_INITIAL_PRIOR"),
        "egress": Kernel(120.0, 0.80, "BROAD_INITIAL_PRIOR"),
    }


def kernel_payload(k: Kernel) -> dict[str, Any]:
    return {"family": "LOGNORMAL_K1", "median_sec": k.median, "sigma": k.sigma, "evidence_class": k.evidence_class}


def damp_kernel(before: Kernel, fit_median: float, fit_sigma: float, evidence_class: str) -> Kernel:
    median = math.exp((1.0 - KERNEL_DAMPING) * math.log(before.median) + KERNEL_DAMPING * math.log(max(1e-6, fit_median)))
    sigma = math.exp((1.0 - KERNEL_DAMPING) * math.log(before.sigma) + KERNEL_DAMPING * math.log(max(1e-6, fit_sigma)))
    return Kernel(float(median), float(sigma), evidence_class)


def fit_kernel_from_log_stats(before: Kernel, stats: dict[str, float], evidence_class: str) -> tuple[Kernel, dict[str, Any]]:
    w = float(stats.get("w", 0.0))
    if w <= 0:
        return before, {"fitted": False, "effective_weight": 0.0, "reason": "NO_FINITE_FACTOR_MASS"}
    mean = float(stats["sum_log"] / w)
    var = max(1e-8, float(stats["sum_log2"] / w - mean * mean))
    fit_median = math.exp(mean)
    fit_sigma = min(2.5, max(0.12, math.sqrt(var)))
    after = damp_kernel(before, fit_median, fit_sigma, evidence_class)
    return after, {
        "fitted": True,
        "effective_weight": w,
        "raw_mle_median_sec": fit_median,
        "raw_mle_sigma": fit_sigma,
        "damping": KERNEL_DAMPING,
        "damped_median_sec": after.median,
        "damped_sigma": after.sigma,
    }


def load_routes(path: Path) -> dict[tuple[str, int, str, int], list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw["status"] != "QUALIFIED_LINE_AWARE_ROUTE_SUPPORT":
        raise SystemExit(f"route support not qualified: {raw['status']}")
    out: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}
    for key, candidates in raw["route_support"].items():
        left, right = key.split("->")
        ol, os = left.split(":")
        dl, ds = right.split(":")
        out[(ol, int(os), dl, int(ds))] = candidates
    return out


def load_roots(path: Path, offsets: dict[str, int] | None = None) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw["status"] != "QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION":
        raise SystemExit(f"service roots not qualified: {raw['status']}")
    offsets = offsets or {}
    by_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    root_meta: dict[str, dict[str, Any]] = {}
    for root in raw["roots"]:
        rid = str(root["root_id"])
        shift = float(offsets.get(rid, 0))
        events = {int(e["station"]): {**e, "time_s": float(e["time_s"]) + shift} for e in root["events"]}
        copy = {k: v for k, v in root.items() if k != "events"}
        copy["events_by_station"] = events
        copy["applied_offset_sec"] = shift
        by_family[(str(root["path_id"]), str(root["direction"]))].append(copy)
        sds = [float(e["sd_s"]) for e in events.values()]
        root_meta[rid] = {
            "root_id": rid,
            "path_id": str(root["path_id"]),
            "direction": str(root["direction"]),
            "prior_sd_sec": max(30.0, statistics.median(sds) if sds else 90.0),
            "evidence_class": str(root["evidence_class"]),
            "matched_pulse_share": float(root.get("matched_pulse_share", 0.0)),
        }
    return by_family, root_meta


class LegCache:
    def __init__(self, roots_by_family: dict[tuple[str, str], list[dict[str, Any]]]):
        self.roots_by_family = roots_by_family
        self.cache: dict[tuple[int, int, tuple[tuple[str, str], ...]], tuple[list[ServiceLegEvent], list[float]]] = {}

    def get(self, leg: dict[str, Any]) -> tuple[list[ServiceLegEvent], list[float]]:
        options = tuple(sorted((str(x["path_id"]), str(x["direction"])) for x in leg["compatible_service_options"]))
        key = (int(leg["from_station"]), int(leg["to_station"]), options)
        if key in self.cache:
            return self.cache[key]
        out: list[ServiceLegEvent] = []
        seen: set[str] = set()
        for family in options:
            for root in self.roots_by_family.get(family, []):
                rid = str(root["root_id"])
                if rid in seen:
                    continue
                ev = root["events_by_station"]
                u, v = key[0], key[1]
                if u not in ev or v not in ev:
                    continue
                a, b = ev[u], ev[v]
                if float(b["time_s"]) <= float(a["time_s"]):
                    continue
                out.append(ServiceLegEvent(
                    root_id=rid,
                    path_id=family[0],
                    direction=family[1],
                    from_station=u,
                    to_station=v,
                    from_t=float(a["time_s"]),
                    to_t=float(b["time_s"]),
                    from_sd=float(a["sd_s"]),
                    to_sd=float(b["sd_s"]),
                ))
                seen.add(rid)
        out.sort(key=lambda x: (x.from_t, x.to_t, x.root_id))
        times = [x.from_t for x in out]
        self.cache[key] = (out, times)
        return out, times

    def feasible(self, leg: dict[str, Any], lower: float, upper: float) -> list[ServiceLegEvent]:
        events, times = self.get(leg)
        lo = bisect.bisect_left(times, lower - TIME_SEARCH_MARGIN_S)
        hi = bisect.bisect_right(times, upper + TIME_SEARCH_MARGIN_S)
        return events[lo:hi]


def route_prior_logs(candidates: list[dict[str, Any]]) -> list[float]:
    if not candidates:
        return []
    c0 = min(float(x["base_ranking_cost_s"]) for x in candidates)
    return [-(float(x["base_ranking_cost_s"]) - c0) / ROUTE_PRIOR_SCALE_S for x in candidates]


def station_only_score(entry: float, exit: float) -> float:
    dt = exit - entry
    proxy = Kernel(300.0, 1.20, "STATION_ONLY_PROXY")
    return log_interval_density(dt, 1.0, proxy)


def evaluate_route(
    candidate: dict[str, Any],
    route_prior_log: float,
    entry: float,
    exit: float,
    cache: LegCache,
    kernels: dict[str, Kernel],
    beam: int,
) -> tuple[float, list[ChainState], dict[str, Any]]:
    legs = candidate.get("ride_legs", [])
    if not legs:
        score = station_only_score(entry, exit)
        if not math.isfinite(score):
            return -math.inf, [], {"reason": "STATION_ONLY_INCOMPATIBLE"}
        chain = ChainState(score + route_prior_log, tuple(), exit, 0.0, ({"type": "STATION_ONLY", "center": exit - entry, "sd": 1.0},))
        return chain.score, [chain], {"local_beam_retained_fraction_product": 1.0, "station_only": True}

    first = legs[0]
    roots = cache.feasible(first, entry, exit)
    expanded: list[ChainState] = []
    for r in roots:
        center = r.from_t - entry
        ll = log_interval_density(center, r.from_sd, kernels["access"])
        if not math.isfinite(ll):
            continue
        expanded.append(ChainState(
            score=ll,
            roots=(r.root_id,),
            last_to_t=r.to_t,
            last_to_sd=r.to_sd,
            factors=({"type": "ACCESS", "root": r.root_id, "center": center, "sd": r.from_sd},),
        ))
    if not expanded:
        return -math.inf, [], {"reason": "NO_ACCESS_COMPATIBLE_SERVICE"}
    total_before = logsumexp(x.score for x in expanded)
    expanded.sort(key=lambda x: x.score, reverse=True)
    states = expanded[:beam]
    retained_log = logsumexp(x.score for x in states)
    retained_product = math.exp(min(0.0, retained_log - total_before)) if math.isfinite(total_before) else 0.0

    movements = list(candidate.get("transfer_movements", []))
    for leg_index, leg in enumerate(legs[1:], start=1):
        movement = movements[leg_index - 1] if leg_index - 1 < len(movements) else f"TRANSFER_{leg_index}"
        nxt: list[ChainState] = []
        for state in states:
            options = cache.feasible(leg, state.last_to_t, exit)
            for r in options:
                center = r.from_t - state.last_to_t
                sd = math.sqrt(max(0.0, state.last_to_sd ** 2 + r.from_sd ** 2))
                ll = log_interval_density(center, sd, kernels["transfer"])
                if not math.isfinite(ll):
                    continue
                nxt.append(ChainState(
                    score=state.score + ll,
                    roots=state.roots + (r.root_id,),
                    last_to_t=r.to_t,
                    last_to_sd=r.to_sd,
                    factors=state.factors + ({
                        "type": "TRANSFER",
                        "movement": movement,
                        "lower_root": state.roots[-1],
                        "upper_root": r.root_id,
                        "center": center,
                        "sd": sd,
                    },),
                ))
        if not nxt:
            return -math.inf, [], {"reason": f"NO_TRANSFER_COMPATIBLE_SERVICE_LEG_{leg_index}"}
        total_before = logsumexp(x.score for x in nxt)
        nxt.sort(key=lambda x: x.score, reverse=True)
        states = nxt[:beam]
        retained_log = logsumexp(x.score for x in states)
        frac = math.exp(min(0.0, retained_log - total_before)) if math.isfinite(total_before) else 0.0
        retained_product *= frac

    final: list[ChainState] = []
    for state in states:
        center = exit - state.last_to_t
        ll = log_interval_density(center, state.last_to_sd, kernels["egress"])
        if not math.isfinite(ll):
            continue
        final.append(ChainState(
            score=state.score + ll + route_prior_log,
            roots=state.roots,
            last_to_t=state.last_to_t,
            last_to_sd=state.last_to_sd,
            factors=state.factors + ({"type": "EGRESS", "root": state.roots[-1], "center": center, "sd": state.last_to_sd},),
        ))
    if not final:
        return -math.inf, [], {"reason": "NO_EGRESS_COMPATIBLE_SERVICE"}
    marginal = logsumexp(x.score for x in final)
    return marginal, final, {"local_beam_retained_fraction_product": retained_product, "station_only": False}


def evaluate_cohort(
    origin_line: str, origin_station: int, destination_line: str, destination_station: int,
    entry: float, exit: float, routes: dict[tuple[str, int, str, int], list[dict[str, Any]]],
    cache: LegCache, kernels: dict[str, Kernel], beam: int, retain_chains: bool,
) -> dict[str, Any]:
    key = (origin_line, int(origin_station), destination_line, int(destination_station))
    candidates = routes.get(key)
    if not candidates:
        return {"finite": False, "reason": "NO_ROUTE_SUPPORT"}
    priors = route_prior_logs(candidates)
    route_rows = []
    all_chains: list[tuple[int, ChainState]] = []
    beam_fracs = []
    station_only = False
    failure_reasons = Counter()
    for idx, (candidate, prior_log) in enumerate(zip(candidates, priors), start=1):
        marginal, chains, diag = evaluate_route(candidate, prior_log, entry, exit, cache, kernels, beam)
        if not math.isfinite(marginal):
            failure_reasons[diag.get("reason", "ROUTE_INCOMPATIBLE")] += 1
            continue
        route_rows.append((idx, marginal, chains))
        beam_fracs.append(float(diag.get("local_beam_retained_fraction_product", 1.0)))
        station_only = station_only or bool(diag.get("station_only", False))
        for chain in chains:
            all_chains.append((idx, chain))
    if not route_rows:
        return {"finite": False, "reason": "TIME_OR_SERVICE_INCOMPATIBLE", "failure_reasons": dict(failure_reasons)}

    route_norm = logsumexp(x[1] for x in route_rows)
    route_probs = [(rank, math.exp(marginal - route_norm)) for rank, marginal, _ in route_rows]
    entropy = -sum(p * math.log(max(p, 1e-300)) for _, p in route_probs)
    top_route_rank, top_route_prob = max(route_probs, key=lambda x: (x[1], -x[0]))

    chain_norm = logsumexp(chain.score for _, chain in all_chains)
    chain_probs = [(rank, chain, math.exp(chain.score - chain_norm)) for rank, chain in all_chains]
    top_rank, top_chain, top_chain_prob = max(chain_probs, key=lambda x: (x[2], -x[0], x[1].roots))
    result = {
        "finite": True,
        "route_probs": route_probs,
        "route_entropy": entropy,
        "top_route_rank": int(top_route_rank),
        "top_route_prob": float(top_route_prob),
        "top_chain_hash": stable_chain_hash(top_rank, top_chain.roots),
        "top_chain_prob": float(top_chain_prob),
        "beam_retained_fraction_min": min(beam_fracs) if beam_fracs else 1.0,
        "station_only_supported": station_only,
    }
    if retain_chains:
        result["chain_probs"] = chain_probs
    return result


def factor_log_stats_add(stats: dict[str, dict[str, float]], factor_type: str, center: float, weight: float) -> None:
    if center <= 0 or weight <= 0 or not math.isfinite(center):
        return
    x = math.log(center)
    s = stats.setdefault(factor_type, {"w": 0.0, "sum_log": 0.0, "sum_log2": 0.0})
    s["w"] += weight
    s["sum_log"] += weight * x
    s["sum_log2"] += weight * x * x


def parquet_batches(path: Path, batch_size: int = 25000):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def pass_e0(
    cohorts_path: Path, routes, cache, kernels0, beam: int, e0_sidecar: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, float]]:
    total_mass = finite_mass = no_route_mass = incompatible_mass = station_only_mass = 0.0
    cohort_count = finite_cohorts = 0
    weighted_entropy = weighted_top_route_prob = weighted_beam = 0.0
    factor_stats: dict[str, dict[str, float]] = {}
    root_usage: dict[str, float] = defaultdict(float)

    schema = pa.schema([
        ("finite", pa.bool_()), ("reason_code", pa.int8()), ("top_route_rank", pa.int16()),
        ("top_chain_hash", pa.uint64()), ("route_entropy", pa.float32()), ("top_route_prob", pa.float32()),
    ])
    writer = pq.ParquetWriter(e0_sidecar, schema=schema, compression="zstd")
    try:
        for df in parquet_batches(cohorts_path):
            out = {name: [] for name in schema.names}
            for row in df.itertuples(index=False):
                mass = float(row.passenger_mass)
                total_mass += mass; cohort_count += 1
                result = evaluate_cohort(str(row.origin_line), int(row.origin_station), str(row.destination_line), int(row.destination_station), float(row.entry_sec), float(row.exit_sec), routes, cache, kernels0, beam, retain_chains=True)
                if not result["finite"]:
                    reason = result["reason"]
                    if reason == "NO_ROUTE_SUPPORT": no_route_mass += mass; code = 1
                    else: incompatible_mass += mass; code = 2
                    out["finite"].append(False); out["reason_code"].append(code); out["top_route_rank"].append(-1)
                    out["top_chain_hash"].append(0); out["route_entropy"].append(float("nan")); out["top_route_prob"].append(float("nan"))
                    continue
                finite_mass += mass; finite_cohorts += 1
                if result["station_only_supported"]: station_only_mass += mass
                weighted_entropy += mass * float(result["route_entropy"])
                weighted_top_route_prob += mass * float(result["top_route_prob"])
                weighted_beam += mass * float(result["beam_retained_fraction_min"])
                for _rank, chain, p in result["chain_probs"]:
                    w = mass * float(p)
                    for f in chain.factors:
                        if f["type"] == "ACCESS":
                            factor_log_stats_add(factor_stats, "access", float(f["center"]), w)
                            root_usage[f["root"]] += w
                        elif f["type"] == "TRANSFER":
                            factor_log_stats_add(factor_stats, "transfer", float(f["center"]), w)
                            root_usage[f["lower_root"]] += w
                            root_usage[f["upper_root"]] += w
                        elif f["type"] == "EGRESS":
                            factor_log_stats_add(factor_stats, "egress", float(f["center"]), w)
                            root_usage[f["root"]] += w
                out["finite"].append(True); out["reason_code"].append(0); out["top_route_rank"].append(int(result["top_route_rank"]))
                out["top_chain_hash"].append(int(result["top_chain_hash"])); out["route_entropy"].append(float(result["route_entropy"])); out["top_route_prob"].append(float(result["top_route_prob"]))
            writer.write_table(pa.Table.from_pydict(out, schema=schema))
    finally:
        writer.close()

    if abs((finite_mass + no_route_mass + incompatible_mass) - total_mass) > 1e-6:
        raise SystemExit("E0 passenger mass conservation failed")
    metrics = {
        "cohort_count": cohort_count,
        "finite_cohort_count": finite_cohorts,
        "passenger_mass": total_mass,
        "finite_posterior_mass": finite_mass,
        "finite_posterior_share": finite_mass / total_mass if total_mass else None,
        "no_route_support_mass": no_route_mass,
        "time_or_service_incompatible_mass": incompatible_mass,
        "station_only_supported_mass": station_only_mass,
        "weighted_mean_route_entropy": weighted_entropy / finite_mass if finite_mass else None,
        "weighted_mean_top_route_probability": weighted_top_route_prob / finite_mass if finite_mass else None,
        "weighted_mean_min_local_beam_retained_fraction": weighted_beam / finite_mass if finite_mass else None,
        "mass_conservation_pass": abs((finite_mass + no_route_mass + incompatible_mass) - total_mass) <= 1e-6,
    }
    return metrics, factor_stats, root_usage


def kernels_after_e0(kernels0: dict[str, Kernel], stats: dict[str, dict[str, float]]) -> tuple[dict[str, Kernel], dict[str, Any]]:
    out = {}
    diag = {}
    for key in ("access", "transfer", "egress"):
        out[key], diag[key] = fit_kernel_from_log_stats(kernels0[key], stats.get(key, {}), "R1B_DAMPED_POSTERIOR_UPDATE")
    return out, diag


def accumulate_shift_scores(
    cohorts_path: Path, routes, cache, kernels0, kernels1, beam: int,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    score_sums: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(SHIFT_GRID))
    weights: dict[str, float] = defaultdict(float)
    for df in parquet_batches(cohorts_path):
        for row in df.itertuples(index=False):
            mass = float(row.passenger_mass)
            result = evaluate_cohort(str(row.origin_line), int(row.origin_station), str(row.destination_line), int(row.destination_station), float(row.entry_sec), float(row.exit_sec), routes, cache, kernels0, beam, retain_chains=True)
            if not result["finite"]:
                continue
            for _rank, chain, p in result["chain_probs"]:
                w = mass * float(p)
                for f in chain.factors:
                    if f["type"] == "ACCESS":
                        attachments = [(f["root"], +1, kernels1["access"])]
                    elif f["type"] == "TRANSFER":
                        attachments = [(f["lower_root"], -1, kernels1["transfer"]), (f["upper_root"], +1, kernels1["transfer"])]
                    elif f["type"] == "EGRESS":
                        attachments = [(f["root"], -1, kernels1["egress"])]
                    else:
                        continue
                    for rid, sign, kernel in attachments:
                        weights[rid] += w
                        for j, delta in enumerate(SHIFT_GRID):
                            ll = log_interval_density(float(f["center"]) + sign * delta, float(f["sd"]), kernel)
                            if math.isfinite(ll):
                                score_sums[rid][j] += w * ll
                            else:
                                score_sums[rid][j] += w * -1e6
    return score_sums, weights


def choose_offsets(score_sums, weights, root_meta) -> tuple[dict[str, int], dict[str, Any]]:
    offsets: dict[str, int] = {}
    per_root = {}
    shifted = 0
    weighted_abs = 0.0
    total_usage = 0.0
    histogram = Counter()
    for rid, meta in root_meta.items():
        usage = float(weights.get(rid, 0.0))
        total_usage += usage
        if usage < MIN_ROOT_USAGE_MASS or rid not in score_sums:
            offsets[rid] = 0
            per_root[rid] = {"usage_mass": usage, "selected_offset_sec": 0, "frozen_low_usage": True, "prior_sd_sec": meta["prior_sd_sec"]}
            histogram[0] += 1
            continue
        prior_sd = float(meta["prior_sd_sec"])
        objective = []
        for delta, raw in zip(SHIFT_GRID, score_sums[rid]):
            mean_ll = raw / usage
            prior_log = -0.5 * (float(delta) / prior_sd) ** 2
            objective.append(mean_ll + prior_log)
        best_j = max(range(len(SHIFT_GRID)), key=lambda j: (objective[j], -abs(SHIFT_GRID[j])))
        selected = int(SHIFT_GRID[best_j])
        zero_j = SHIFT_GRID.index(0)
        gain = float(objective[best_j] - objective[zero_j])
        offsets[rid] = selected
        shifted += int(selected != 0)
        weighted_abs += usage * abs(selected)
        histogram[selected] += 1
        per_root[rid] = {
            "usage_mass": usage, "selected_offset_sec": selected, "score_gain_vs_zero": gain,
            "prior_sd_sec": prior_sd, "frozen_low_usage": False, "matched_pulse_share": meta["matched_pulse_share"],
        }
    diag = {
        "shift_grid_sec": SHIFT_GRID,
        "min_root_usage_mass": MIN_ROOT_USAGE_MASS,
        "root_count": len(root_meta),
        "roots_shifted_nonzero": shifted,
        "root_shifted_share": shifted / len(root_meta) if root_meta else None,
        "usage_weighted_mean_abs_shift_sec": weighted_abs / total_usage if total_usage else None,
        "offset_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "per_root": per_root,
    }
    return offsets, diag


def pass_e1_compare(cohorts_path, e0_sidecar, routes, cache1, kernels1, beam: int) -> dict[str, Any]:
    total_mass = finite_mass = no_route_mass = incompatible_mass = 0.0
    became_finite = lost_finite = both_finite = 0.0
    top_route_changed = top_chain_changed = 0.0
    weighted_entropy = weighted_entropy_delta = weighted_top_prob = weighted_top_prob_delta = 0.0
    cohort_count = 0
    pf0 = pq.ParquetFile(e0_sidecar)
    e0_batches = pf0.iter_batches(batch_size=25000)
    for df, b0 in zip(parquet_batches(cohorts_path), e0_batches):
        e0 = b0.to_pandas()
        if len(df) != len(e0):
            raise SystemExit("E0 sidecar row alignment failed")
        for row, old in zip(df.itertuples(index=False), e0.itertuples(index=False)):
            mass = float(row.passenger_mass); total_mass += mass; cohort_count += 1
            new = evaluate_cohort(str(row.origin_line), int(row.origin_station), str(row.destination_line), int(row.destination_station), float(row.entry_sec), float(row.exit_sec), routes, cache1, kernels1, beam, retain_chains=False)
            if not new["finite"]:
                if new["reason"] == "NO_ROUTE_SUPPORT": no_route_mass += mass
                else: incompatible_mass += mass
                if bool(old.finite): lost_finite += mass
                continue
            finite_mass += mass
            weighted_entropy += mass * float(new["route_entropy"])
            weighted_top_prob += mass * float(new["top_route_prob"])
            if not bool(old.finite):
                became_finite += mass
            else:
                both_finite += mass
                top_route_changed += mass * int(int(old.top_route_rank) != int(new["top_route_rank"]))
                top_chain_changed += mass * int(int(old.top_chain_hash) != int(new["top_chain_hash"]))
                weighted_entropy_delta += mass * (float(new["route_entropy"]) - float(old.route_entropy))
                weighted_top_prob_delta += mass * (float(new["top_route_prob"]) - float(old.top_route_prob))
    if abs((finite_mass + no_route_mass + incompatible_mass) - total_mass) > 1e-6:
        raise SystemExit("E1 passenger mass conservation failed")
    return {
        "cohort_count": cohort_count,
        "passenger_mass": total_mass,
        "finite_posterior_mass": finite_mass,
        "finite_posterior_share": finite_mass / total_mass if total_mass else None,
        "no_route_support_mass": no_route_mass,
        "time_or_service_incompatible_mass": incompatible_mass,
        "became_finite_mass": became_finite,
        "lost_finite_mass": lost_finite,
        "finite_both_mass": both_finite,
        "top_route_changed_mass": top_route_changed,
        "top_route_changed_share_among_finite_both": top_route_changed / both_finite if both_finite else None,
        "top_boarding_chain_changed_mass": top_chain_changed,
        "top_boarding_chain_changed_share_among_finite_both": top_chain_changed / both_finite if both_finite else None,
        "weighted_mean_route_entropy": weighted_entropy / finite_mass if finite_mass else None,
        "weighted_mean_route_entropy_change_among_finite_both": weighted_entropy_delta / both_finite if both_finite else None,
        "weighted_mean_top_route_probability": weighted_top_prob / finite_mass if finite_mass else None,
        "weighted_mean_top_route_probability_change_among_finite_both": weighted_top_prob_delta / both_finite if both_finite else None,
        "mass_conservation_pass": True,
    }


def run_day(cohorts: Path, route_support: Path, service_roots: Path, output: Path, temp_dir: Path, beam: int) -> dict[str, Any]:
    routes = load_routes(route_support)
    roots0, root_meta = load_roots(service_roots)
    cache0 = LegCache(roots0)
    kernels0 = initial_kernels()
    e0_sidecar = temp_dir / (cohorts.stem + ".e0.parquet")
    e0, factor_stats, root_usage = pass_e0(cohorts, routes, cache0, kernels0, beam, e0_sidecar)
    kernels1, kernel_diag = kernels_after_e0(kernels0, factor_stats)
    score_sums, shift_weights = accumulate_shift_scores(cohorts, routes, cache0, kernels0, kernels1, beam)
    offsets, service_diag = choose_offsets(score_sums, shift_weights, root_meta)
    roots1, _ = load_roots(service_roots, offsets=offsets)
    cache1 = LegCache(roots1)
    e1 = pass_e1_compare(cohorts, e0_sidecar, routes, cache1, kernels1, beam)
    e0_sidecar.unlink(missing_ok=True)

    date = json.loads(service_roots.read_text(encoding="utf-8"))["source_date"]
    total = float(e0["passenger_mass"])
    result = {
        "schema": "rail.hz-r1b-full-service-day-joint-update.v1",
        "dataset_id": "CN_HZ_Tianchi_2019",
        "service_date": date,
        "status": "COMPLETED_FORMAL_R1B_FULL_SERVICE_DAY_UPDATE",
        "scope": {
            "time": "FULL_SERVICE_DAY_0400_TO_NEXT_0400",
            "network": "FULL_NETWORK",
            "passenger_subsample": False,
            "transfer_count_cap": None,
            "route_support_beam_k": 32,
            "boarding_dynamic_posterior_beam": beam,
            "boarding_skip_count_cap": None,
            "beam_boundary": "posterior beam is a computational approximation; omitted chain support is not interpreted as behavioral zero",
        },
        "kernels_before": {k: kernel_payload(v) for k, v in kernels0.items()},
        "kernel_update": kernel_diag,
        "kernels_after": {k: kernel_payload(v) for k, v in kernels1.items()},
        "E0": e0,
        "service_timing_update": service_diag,
        "E1": e1,
        "bidirectional_evidence": {
            "passenger_posterior_updates_temporal_kernels": any(kernel_diag[k].get("fitted") for k in kernel_diag),
            "passenger_posterior_updates_service_timing": service_diag["roots_shifted_nonzero"] > 0,
            "updated_state_redistributes_route_posterior": e1["top_route_changed_mass"] > 0,
            "updated_state_redistributes_boarding_posterior": e1["top_boarding_chain_changed_mass"] > 0,
        },
        "mass_conservation": {
            "E0_pass": bool(e0["mass_conservation_pass"]),
            "E1_pass": bool(e1["mass_conservation_pass"]),
            "passenger_mass": total,
        },
        "scientific_boundary": [
            "Service roots are AFC-anchored latent passenger-facing service hypotheses, not observed ATS train identities.",
            "This is formal full-service-day R1B, but hierarchical station/movement/time kernels remain the responsibility of R1C.",
            "The route K=32 support and boarding posterior beam are computational support approximations; omitted support is never claimed empirically zero.",
            "No max-skip behavioral truncation is used.",
        ],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "service_date": date,
        "status": result["status"],
        "passenger_mass": total,
        "E0_finite_share": e0["finite_posterior_share"],
        "E1_finite_share": e1["finite_posterior_share"],
        "roots_shifted_nonzero": service_diag["roots_shifted_nonzero"],
        "top_route_changed_mass": e1["top_route_changed_mass"],
        "top_boarding_chain_changed_mass": e1["top_boarding_chain_changed_mass"],
        "bidirectional_evidence": result["bidirectional_evidence"],
    }, ensure_ascii=False, indent=2))
    return result


def aggregate(input_dir: Path, output: Path) -> dict[str, Any]:
    paths = sorted(input_dir.glob("2019-01-*.r1b.json"))
    days = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    expected = [f"2019-01-{d:02d}" for d in range(1, 25)]
    actual = [x["service_date"] for x in days]
    total_mass = sum(float(x["E0"]["passenger_mass"]) for x in days)
    e0_finite = sum(float(x["E0"]["finite_posterior_mass"]) for x in days)
    e1_finite = sum(float(x["E1"]["finite_posterior_mass"]) for x in days)
    route_changed = sum(float(x["E1"]["top_route_changed_mass"]) for x in days)
    chain_changed = sum(float(x["E1"]["top_boarding_chain_changed_mass"]) for x in days)
    shifted_days = sum(int(x["service_timing_update"]["roots_shifted_nonzero"] > 0) for x in days)
    gates = {
        "exactly_24_complete_service_days": len(days) == 24 and actual == expected,
        "all_days_full_scope": all(not x["scope"]["passenger_subsample"] and x["scope"]["boarding_skip_count_cap"] is None and x["scope"]["transfer_count_cap"] is None for x in days),
        "all_days_mass_conservation": all(x["mass_conservation"]["E0_pass"] and x["mass_conservation"]["E1_pass"] for x in days),
        "passenger_updates_temporal_kernels_all_days": all(x["bidirectional_evidence"]["passenger_posterior_updates_temporal_kernels"] for x in days),
        "passenger_updates_service_timing_at_least_one_day": shifted_days > 0,
        "route_or_boarding_redistribution_present": route_changed > 0 or chain_changed > 0,
    }
    result = {
        "schema": "rail.hz24-r1b-full-service-day-multiday-summary.v1",
        "status": "QUALIFIED_HZ24_FORMAL_R1B_FULL_SERVICE_DAY_MULTIDAY" if all(gates.values()) else "HZ24_R1B_QUALIFICATION_GATE_FAILED",
        "complete_service_days": len(days),
        "dates": actual,
        "passenger_mass": total_mass,
        "E0_finite_posterior_mass": e0_finite,
        "E0_finite_posterior_share": e0_finite / total_mass if total_mass else None,
        "E1_finite_posterior_mass": e1_finite,
        "E1_finite_posterior_share": e1_finite / total_mass if total_mass else None,
        "top_route_changed_mass": route_changed,
        "top_boarding_chain_changed_mass": chain_changed,
        "days_with_nonzero_service_timing_update": shifted_days,
        "integrity_gates": gates,
        "day_summaries": [{
            "date": x["service_date"],
            "passenger_mass": x["E0"]["passenger_mass"],
            "E0_finite_share": x["E0"]["finite_posterior_share"],
            "E1_finite_share": x["E1"]["finite_posterior_share"],
            "roots_shifted_nonzero": x["service_timing_update"]["roots_shifted_nonzero"],
            "top_route_changed_mass": x["E1"]["top_route_changed_mass"],
            "top_boarding_chain_changed_mass": x["E1"]["top_boarding_chain_changed_mass"],
            "weighted_mean_route_entropy_E0": x["E0"]["weighted_mean_route_entropy"],
            "weighted_mean_route_entropy_E1": x["E1"]["weighted_mean_route_entropy"],
        } for x in days],
        "next_stage": "HZ24_R1C_FULL_SERVICE_DAY_HIERARCHICAL_TEMPORAL_KERNELS",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "complete_service_days", "passenger_mass", "E0_finite_posterior_share", "E1_finite_posterior_share", "top_route_changed_mass", "top_boarding_chain_changed_mass", "days_with_nonzero_service_timing_update", "integrity_gates"]}, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)
    return result


def main() -> None:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("day")
    s.add_argument("--cohorts", type=Path, required=True); s.add_argument("--routes", type=Path, required=True); s.add_argument("--roots", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True); s.add_argument("--temp-dir", type=Path, required=True); s.add_argument("--beam", type=int, default=BEAM)
    s = sub.add_parser("aggregate"); s.add_argument("--input-dir", type=Path, required=True); s.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.command == "day":
        a.temp_dir.mkdir(parents=True, exist_ok=True)
        run_day(a.cohorts, a.routes, a.roots, a.output, a.temp_dir, a.beam)
    else:
        aggregate(a.input_dir, a.output)


if __name__ == "__main__":
    main()
