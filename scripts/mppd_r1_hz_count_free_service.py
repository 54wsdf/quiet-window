from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "mppd.r1-hz-count-free-service-discovery.v1"
SERVICE_DAY_BOUNDARY_HOUR = 4
DAY_S = 24 * 3600

# Network structure only. No planned absolute timetable or planned train count is used.
LINE_PATHS: dict[str, dict[str, Any]] = {
    "B_main": {"afc_line": "B", "nodes": list(range(0, 28))},
    "B_branch": {"afc_line": "B", "nodes": list(range(0, 21)) + list(range(28, 34))},
    "C_main": {"afc_line": "C", "nodes": list(range(34, 67))},
    "A_main": {"afc_line": "A", "nodes": [67, 68, 69, 70, 71, 72, 73, 74, 5, 75, 76, 77, 46, 78, 79, 80, 15, 16]},
}


@dataclass(frozen=True)
class Event:
    event_id: str
    station: int
    center_s: float
    score: float
    excess_mass: float

    @property
    def weight(self) -> float:
        return max(0.05, self.score) * math.sqrt(max(1.0, self.excess_mass))


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width == 1:
        return np.asarray(x, dtype=float)
    return np.convolve(np.asarray(x, dtype=float), np.ones(width, dtype=float) / width, mode="same")


def highpass(x: np.ndarray, short_bins: int, long_bins: int) -> np.ndarray:
    return moving_average(x, short_bins) - moving_average(x, long_bins)


def service_day_seconds(t: datetime, service_date: str) -> float | None:
    start = datetime.strptime(service_date, "%Y-%m-%d") + timedelta(hours=SERVICE_DAY_BOUNDARY_HOUR)
    rel = (t - start).total_seconds()
    return rel if 0 <= rel < DAY_S else None


def load_exit_counts(paths: list[Path], service_date: str, bin_s: int) -> tuple[np.ndarray, dict[str, Any]]:
    nbins = DAY_S // bin_s
    counts = np.zeros((81, nbins), dtype=np.uint32)
    rows = exits = retained = 0
    by_station: Counter[int] = Counter()
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"time", "stationID", "status"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise SystemExit(f"missing AFC columns in {path}: {sorted(missing)}")
            for row in reader:
                rows += 1
                if str(row["status"]).strip() != "0":
                    continue
                exits += 1
                try:
                    station = int(row["stationID"])
                    t = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    continue
                if not 0 <= station < 81:
                    continue
                rel = service_day_seconds(t, service_date)
                if rel is None:
                    continue
                k = int(rel) // bin_s
                if 0 <= k < nbins:
                    counts[station, k] += 1
                    retained += 1
                    by_station[station] += 1
    return counts, {
        "raw_rows_scanned": rows,
        "exit_rows_scanned": exits,
        "retained_service_day_exit_rows": retained,
        "stations_with_exit_rows": len(by_station),
    }


def detect_station_events(
    counts: np.ndarray,
    bin_s: int,
    threshold_quantile: float = 0.95,
    minimum_score: float = 1.0,
    minimum_separation_s: int = 30,
) -> dict[int, list[Event]]:
    out: dict[int, list[Event]] = defaultdict(list)
    short_bins = max(1, round(15 / bin_s))
    background_bins = max(5, round(300 / bin_s))
    min_sep_bins = max(1, math.ceil(minimum_separation_s / bin_s))
    half = max(1, math.ceil(60 / bin_s))
    for station in range(counts.shape[0]):
        x = counts[station].astype(float)
        if x.sum() <= 0:
            continue
        short = moving_average(x, short_bins)
        background = moving_average(x, background_bins)
        residual = short - background
        score = residual / np.sqrt(np.maximum(background, 0.25))
        threshold = max(float(np.quantile(score, threshold_quantile)), minimum_score)
        left = np.r_[score[0], score[:-1]]
        right = np.r_[score[1:], score[-1]]
        candidates = np.flatnonzero((score >= threshold) & (score >= left) & (score >= right) & (residual > 0))
        ranked = candidates[np.argsort(-score[candidates], kind="stable")]
        selected: list[int] = []
        for idx in ranked:
            i = int(idx)
            if all(abs(i - j) >= min_sep_bins for j in selected):
                selected.append(i)
        selected.sort()
        for i in selected:
            lo = i
            hi = i
            while lo > 0 and i - lo < half and residual[lo - 1] > 0:
                lo -= 1
            while hi + 1 < len(x) and hi - i < half and residual[hi + 1] > 0:
                hi += 1
            raw = float(np.sum(x[lo : hi + 1]))
            bg = float(np.sum(background[lo : hi + 1]))
            center_s = float((i + 0.5) * bin_s)
            out[station].append(Event(
                event_id=f"{station}@{center_s:.1f}",
                station=station,
                center_s=center_s,
                score=float(score[i]),
                excess_mass=max(0.0, raw - bg),
            ))
    return out


def normalized_lag_score(a: np.ndarray, b: np.ndarray, lag_bins: int) -> float:
    if lag_bins <= 0 or lag_bins >= len(a) - 10:
        return -math.inf
    x = a[:-lag_bins]
    y = b[lag_bins:]
    if len(x) < 60:
        return -math.inf
    sx = float(x.std())
    sy = float(y.std())
    if sx <= 1e-9 or sy <= 1e-9:
        return -math.inf
    return float(np.dot(x - x.mean(), y - y.mean()) / (len(x) * sx * sy))


def estimate_directed_edge_lags(
    counts: np.ndarray,
    bin_s: int,
    min_lag_s: int = 30,
    max_lag_s: int = 600,
) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    hp = np.vstack([
        highpass(counts[s].astype(float), max(1, round(20 / bin_s)), max(5, round(300 / bin_s)))
        for s in range(counts.shape[0])
    ])
    out: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    finite_lags: list[float] = []
    pending: list[tuple[str, str, int, int]] = []
    lo = max(1, math.ceil(min_lag_s / bin_s))
    hi = max(lo, math.floor(max_lag_s / bin_s))
    for path_id, meta in LINE_PATHS.items():
        base = list(meta["nodes"])
        for direction in ("Down", "Up"):
            nodes = base if direction == "Down" else list(reversed(base))
            for u, v in zip(nodes, nodes[1:]):
                best_score = -math.inf
                best_lag = None
                for lag_bins in range(lo, hi + 1):
                    c = normalized_lag_score(hp[u], hp[v], lag_bins)
                    if c > best_score:
                        best_score = c
                        best_lag = lag_bins
                key = (path_id, direction, u, v)
                if best_lag is not None and math.isfinite(best_score) and best_score >= 0.02:
                    lag_s = float(best_lag * bin_s)
                    out[key] = {"lag_s": lag_s, "score": float(best_score), "source": "AFC_CROSS_CORRELATION"}
                    finite_lags.append(lag_s)
                else:
                    pending.append(key)
    fallback = float(np.median(finite_lags)) if finite_lags else 150.0
    for key in pending:
        out[key] = {"lag_s": fallback, "score": None, "source": "AFC_NETWORK_MEDIAN_FALLBACK"}
    return out


def cumulative_offsets(path_id: str, direction: str, edge_lags: dict[tuple[str, str, int, int], dict[str, Any]]) -> dict[int, float]:
    base = list(LINE_PATHS[path_id]["nodes"])
    nodes = base if direction == "Down" else list(reversed(base))
    out = {nodes[0]: 0.0}
    t = 0.0
    for u, v in zip(nodes, nodes[1:]):
        t += float(edge_lags[(path_id, direction, u, v)]["lag_s"])
        out[v] = t
    return out


def weighted_cluster(values: list[tuple[float, Event]], radius_s: float) -> list[dict[str, Any]]:
    if not values:
        return []
    vals = sorted(values, key=lambda x: x[0])
    groups: list[list[tuple[float, Event]]] = []
    current: list[tuple[float, Event]] = []
    center = None
    for x, e in vals:
        if not current:
            current = [(x, e)]
            center = x
            continue
        if abs(x - float(center)) <= radius_s:
            current.append((x, e))
            w = np.array([ev.weight for _v, ev in current], dtype=float)
            a = np.array([v for v, _ev in current], dtype=float)
            center = float(np.average(a, weights=w))
        else:
            groups.append(current)
            current = [(x, e)]
            center = x
    if current:
        groups.append(current)
    out = []
    for g in groups:
        w = np.array([e.weight for _x, e in g], dtype=float)
        x = np.array([v for v, _e in g], dtype=float)
        out.append({"center_s": float(np.average(x, weights=w)), "items": g})
    return out


def two_means_split(items: list[tuple[float, Event]]) -> tuple[list[tuple[float, Event]], list[tuple[float, Event]]] | None:
    if len(items) < 6:
        return None
    xs = np.array([x for x, _e in items], dtype=float)
    c1 = float(np.quantile(xs, 0.25))
    c2 = float(np.quantile(xs, 0.75))
    if abs(c2 - c1) < 20.0:
        return None
    for _ in range(12):
        a: list[tuple[float, Event]] = []
        b: list[tuple[float, Event]] = []
        for item in items:
            (a if abs(item[0] - c1) <= abs(item[0] - c2) else b).append(item)
        if not a or not b:
            return None
        nc1 = float(np.average([x for x, _e in a], weights=[e.weight for _x, e in a]))
        nc2 = float(np.average([x for x, _e in b], weights=[e.weight for _x, e in b]))
        if abs(nc1 - c1) + abs(nc2 - c2) < 1e-3:
            break
        c1, c2 = nc1, nc2
    return a, b


def support_summary(items: list[tuple[float, Event]], center_s: float) -> dict[str, Any]:
    stations = {e.station for _x, e in items}
    event_ids = {e.event_id for _x, e in items}
    residuals = np.array([x - center_s for x, _e in items], dtype=float)
    return {
        "station_count": len(stations),
        "event_count": len(event_ids),
        "event_ids": sorted(event_ids),
        "support_weight": float(sum(e.weight for _x, e in items)),
        "residual_median_abs_s": float(np.median(np.abs(residuals))) if len(residuals) else None,
        "residual_p90_abs_s": float(np.quantile(np.abs(residuals), 0.90)) if len(residuals) else None,
        "residual_std_s": float(np.std(residuals)) if len(residuals) else None,
    }


def candidate_evidence_score(summary: dict[str, Any], path_len: int, cluster_radius_s: float) -> float:
    coverage = float(summary["station_count"]) / max(1, path_len)
    strength = math.log1p(max(0.0, float(summary["support_weight"])))
    p90 = float(summary["residual_p90_abs_s"] or 0.0)
    fit = math.exp(-p90 / max(1.0, cluster_radius_s))
    return coverage * strength * fit


def discover_path_candidates(
    events: dict[int, list[Event]],
    path_id: str,
    direction: str,
    offsets: dict[int, float],
    station_phase: dict[int, float],
    cluster_radius_s: float,
    split_std_s: float,
    min_station_support: int,
    complexity_penalty: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    projected: list[tuple[float, Event]] = []
    for station, off in offsets.items():
        phase = float(station_phase.get(station, 0.0))
        for e in events.get(station, []):
            projected.append((e.center_s - off - phase, e))
    operations: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    queue = list(weighted_cluster(projected, cluster_radius_s))
    while queue:
        g = queue.pop(0)
        items = g["items"]
        center = float(g["center_s"])
        ss = support_summary(items, center)
        if ss["station_count"] < min_station_support:
            operations["death_low_support"] += 1
            continue
        evidence_score = candidate_evidence_score(ss, len(LINE_PATHS[path_id]["nodes"]), cluster_radius_s)
        if evidence_score <= complexity_penalty:
            operations["death_complexity_penalty"] += 1
            continue
        if ss["residual_std_s"] is not None and ss["residual_std_s"] > split_std_s:
            split = two_means_split(items)
            if split:
                left, right = split
                lc = float(np.average([x for x, _e in left], weights=[e.weight for _x, e in left]))
                rc = float(np.average([x for x, _e in right], weights=[e.weight for _x, e in right]))
                ls = support_summary(left, lc)
                rs = support_summary(right, rc)
                if ls["station_count"] >= min_station_support and rs["station_count"] >= min_station_support and abs(rc - lc) >= 25.0:
                    queue.insert(0, {"center_s": rc, "items": right})
                    queue.insert(0, {"center_s": lc, "items": left})
                    operations["split"] += 1
                    continue
        out.append({
            "path_id": path_id,
            "afc_line": LINE_PATHS[path_id]["afc_line"],
            "direction": direction,
            "reference_time_s": center,
            "evidence_score": evidence_score,
            "objective_gain": evidence_score - complexity_penalty,
            **ss,
        })
        operations["birth_retained"] += 1
    out.sort(key=lambda z: z["reference_time_s"])
    return out, operations


def event_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    aa = set(a["event_ids"])
    bb = set(b["event_ids"])
    return len(aa & bb) / max(1, min(len(aa), len(bb))) if aa and bb else 0.0


def deduplicate_path_candidates(candidates: list[dict[str, Any]], merge_time_s: float = 30.0) -> tuple[list[dict[str, Any]], Counter[str]]:
    operations: Counter[str] = Counter()
    by_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_family[(c["afc_line"], c["direction"])].append(c)
    final: list[dict[str, Any]] = []
    for (_line, _direction), vals in by_family.items():
        vals = sorted(vals, key=lambda z: (-z["support_weight"], z["reference_time_s"]))
        used: set[int] = set()
        for i, c in enumerate(vals):
            if i in used:
                continue
            family = [c]
            for j in range(i + 1, len(vals)):
                if j in used:
                    continue
                d = vals[j]
                if abs(float(d["reference_time_s"]) - float(c["reference_time_s"])) <= merge_time_s and event_overlap(c, d) >= 0.55:
                    family.append(d)
                    used.add(j)
            best = max(family, key=lambda z: (z["station_count"], z["support_weight"], -z["residual_p90_abs_s"]))
            merged = dict(best)
            merged["path_alternatives"] = sorted([
                {"path_id": x["path_id"], "support_weight": x["support_weight"], "station_count": x["station_count"]}
                for x in family
            ], key=lambda z: -z["support_weight"])
            merged["path_ambiguous"] = len({x["path_id"] for x in family}) > 1
            if len(family) > 1:
                operations["merge_cross_path_duplicate"] += len(family) - 1
                if merged["path_ambiguous"]:
                    operations["path_reassignment_competition"] += 1
            final.append(merged)
    final.sort(key=lambda z: (z["afc_line"], z["direction"], z["reference_time_s"]))
    return final, operations


def estimate_station_phase(
    trajectories: list[dict[str, Any]],
    events: dict[int, list[Event]],
    offsets_by_pd: dict[tuple[str, str], dict[int, float]],
    max_residual_s: float = 120.0,
) -> dict[int, float]:
    residuals: dict[int, list[float]] = defaultdict(list)
    for tr in trajectories:
        offmap = offsets_by_pd[(tr["path_id"], tr["direction"])]
        t0 = float(tr["reference_time_s"])
        for station, off in offmap.items():
            target = t0 + off
            evs = events.get(station, [])
            if not evs:
                continue
            best = min(evs, key=lambda e: abs(e.center_s - target))
            r = best.center_s - target
            if abs(r) <= max_residual_s:
                residuals[station].append(float(r))
    phase = {s: float(np.median(v)) for s, v in residuals.items() if len(v) >= 3}
    if phase:
        center = float(np.median(list(phase.values())))
        phase = {s: v - center for s, v in phase.items()}
    return phase


def materialize_trajectories(
    candidates: list[dict[str, Any]],
    offsets_by_pd: dict[tuple[str, str], dict[int, float]],
    station_phase: dict[int, float],
) -> list[dict[str, Any]]:
    out = []
    for idx, c in enumerate(candidates):
        offsets = offsets_by_pd[(c["path_id"], c["direction"])]
        t0 = float(c["reference_time_s"])
        evs = []
        for seq, (station, off) in enumerate(offsets.items()):
            evs.append({
                "station": station,
                "sequence_index": seq,
                "time_s": t0 + off,
                "station_phase_nuisance_s": float(station_phase.get(station, 0.0)),
            })
        out.append({
            "trajectory_id": f"cf:{c['afc_line']}:{c['direction']}:{idx}:{int(round(t0))}",
            "afc_line": c["afc_line"],
            "path_id": c["path_id"],
            "path_alternatives": c.get("path_alternatives", []),
            "path_ambiguous": bool(c.get("path_ambiguous", False)),
            "direction": c["direction"],
            "reference_time_s": t0,
            "support_station_count": c["station_count"],
            "support_event_count": c["event_count"],
            "support_weight": c["support_weight"],
            "evidence_score": c["evidence_score"],
            "objective_gain": c["objective_gain"],
            "residual_median_abs_s": c["residual_median_abs_s"],
            "residual_p90_abs_s": c["residual_p90_abs_s"],
            "events": evs,
        })
    return out


def discover_count_free(
    counts: np.ndarray,
    bin_s: int = 5,
    iterations: int = 3,
    cluster_radius_s: float = 55.0,
    split_std_s: float = 55.0,
    min_station_support: int = 4,
    complexity_penalty: float = 0.60,
) -> dict[str, Any]:
    events = detect_station_events(counts, bin_s)
    edge_lags = estimate_directed_edge_lags(counts, bin_s)
    offsets_by_pd = {(p, d): cumulative_offsets(p, d, edge_lags) for p in LINE_PATHS for d in ("Down", "Up")}
    station_phase: dict[int, float] = {}
    operation_total: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    iteration_summaries = []
    for it in range(iterations):
        allc = []
        ops: Counter[str] = Counter()
        for p in LINE_PATHS:
            for d in ("Down", "Up"):
                required_support = max(min_station_support, math.ceil(0.25 * len(LINE_PATHS[p]["nodes"])))
                cc, oo = discover_path_candidates(
                    events, p, d, offsets_by_pd[(p, d)], station_phase,
                    cluster_radius_s, split_std_s, required_support, complexity_penalty,
                )
                allc.extend(cc)
                ops.update(oo)
        candidates, dops = deduplicate_path_candidates(allc)
        ops.update(dops)
        operation_total.update(ops)
        new_phase = estimate_station_phase(candidates, events, offsets_by_pd)
        max_phase_change = max([
            abs(new_phase.get(s, 0.0) - station_phase.get(s, 0.0))
            for s in set(new_phase) | set(station_phase)
        ] or [0.0])
        station_phase = new_phase
        iteration_summaries.append({
            "iteration": it + 1,
            "candidate_trajectory_count": len(candidates),
            "station_phase_count": len(station_phase),
            "max_station_phase_change_s": float(max_phase_change),
            "operations": dict(ops),
        })
    trajectories = materialize_trajectories(candidates, offsets_by_pd, station_phase)
    edge_sources = Counter(v["source"] for v in edge_lags.values())
    by_pd = Counter(f"{x['path_id']}:{x['direction']}" for x in trajectories)
    return {
        "schema": SCHEMA,
        "status": "COUNT_FREE_SERVICE_DISCOVERY_ESTIMATED_REQUIRES_JOINT_PASSENGER_REFINEMENT",
        "semantics": {
            "train_count_is_input": False,
            "planned_timetable_used": False,
            "legacy_candidate_roots_used_as_input": False,
            "service_trajectory_count_is_inferred": True,
            "station_phase_nuisance_is_not_egress_time": True,
            "operations_supported": ["birth", "death", "split", "merge", "path_reassignment"],
            "physical_access_egress_lower_bound_for_next_joint_step_s": 15.0,
            "physical_transfer_lower_bound_for_next_joint_step_s": 5.0,
        },
        "event_detection": {
            "total_station_events": sum(len(v) for v in events.values()),
            "stations_with_events": len(events),
        },
        "edge_lag_sources": dict(edge_sources),
        "iterations": iteration_summaries,
        "operation_totals": dict(operation_total),
        "objective": {
            "complexity_penalty_per_trajectory": complexity_penalty,
            "retained_total_objective_gain": float(sum(max(0.0, float(c.get("objective_gain", 0.0))) for c in candidates)),
            "selection_semantics": "retain only trajectories whose normalized AFC evidence exceeds per-trajectory complexity penalty",
        },
        "inferred_service_trajectory_count": len(trajectories),
        "trajectory_counts": dict(by_pd),
        "station_phase_nuisance_s": {str(k): v for k, v in sorted(station_phase.items())},
        "trajectories": trajectories,
    }


def baseline_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    x = json.loads(path.read_text(encoding="utf-8"))
    roots = x.get("roots", [])
    return {
        "source_status": x.get("status"),
        "legacy_candidate_root_count": len(roots) if roots else x.get("root_count_total"),
        "legacy_root_counts": x.get("root_counts"),
        "comparison_only_not_used_by_discovery": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, action="append", required=True)
    p.add_argument("--service-date", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline-roots", type=Path)
    p.add_argument("--bin-s", type=int, default=5)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--cluster-radius-s", type=float, default=55.0)
    p.add_argument("--split-std-s", type=float, default=55.0)
    p.add_argument("--min-station-support", type=int, default=4)
    p.add_argument("--complexity-penalty", type=float, default=0.60)
    a = p.parse_args()
    counts, source = load_exit_counts(a.input, a.service_date, a.bin_s)
    result = discover_count_free(
        counts, a.bin_s, a.iterations, a.cluster_radius_s, a.split_std_s,
        a.min_station_support, a.complexity_penalty,
    )
    result["service_date"] = a.service_date
    result["source_profile"] = source
    result["legacy_baseline"] = baseline_summary(a.baseline_roots)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "service_date": a.service_date,
        "inferred_service_trajectory_count": result["inferred_service_trajectory_count"],
        "legacy_candidate_root_count": (result.get("legacy_baseline") or {}).get("legacy_candidate_root_count"),
        "event_detection": result["event_detection"],
        "edge_lag_sources": result["edge_lag_sources"],
        "operation_totals": result["operation_totals"],
        "trajectory_counts": result["trajectory_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
