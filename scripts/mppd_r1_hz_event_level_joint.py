from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import scripts.mppd_r1_hz_joint_full_day as base

SCHEMA = "mppd.r1-hz-event-level-parameters.v1"
MAX_EVENT_OFFSET_S = 180.0
MIN_EVENT_GAP_S = 1.0


def event_key(root_id: str, station_id: str | int) -> str:
    return f"{root_id}|{int(station_id)}"


def load_raw_roots(path: Path) -> dict[str, Any]:
    x = json.loads(path.read_text(encoding="utf-8"))
    if x.get("status") != "QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION":
        raise SystemExit("candidate service roots are not qualified")
    return x


def write_adjusted_roots(source: Path, params: dict[str, Any] | None, target: Path) -> None:
    x = load_raw_roots(source)
    offsets = {} if params is None else {str(k): float(v) for k, v in params.get("event_offsets_s", {}).items()}
    for r in x["roots"]:
        rid = str(r["root_id"])
        for e in r["events"]:
            k = event_key(rid, e["station"])
            e["time_s"] = float(e["time_s"]) + float(offsets.get(k, 0.0))
    target.write_text(json.dumps(x, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def run_shard(edges: Path, roots: Path, routes: Path, params_path: Path | None, out_stats: Path, shard_index: int, chains_out: Path | None):
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path else None
    with tempfile.TemporaryDirectory(prefix="mppd-r1-event-roots-") as td:
        adj = Path(td) / "roots.adjusted.json"
        write_adjusted_roots(roots, params, adj)
        # base solver sees one shared timetable.  event_offsets_s is intentionally not
        # consumed inside base.load_roots; it has already been applied to the root file.
        result = base.run_shard(edges, adj, routes, params_path, out_stats, shard_index, chains_out)
    return result


def merge_root_egress(stats: list[dict[str, Any]]):
    out = defaultdict(lambda: {"root_id": None, "station_id": None, "w": 0.0, "sum_seconds": 0.0})
    for s in stats:
        for k, r in s.get("root_egress_stats", {}).items():
            z = out[k]
            z["root_id"] = str(r["root_id"]); z["station_id"] = str(r["station_id"])
            z["w"] += float(r["w"]); z["sum_seconds"] += float(r["sum_seconds"])
    return dict(out)


def _interpolate_offsets(root: dict[str, Any], direct: dict[str, tuple[float, float]], old: dict[str, float]):
    events = root["events"]
    rid = str(root["root_id"])
    n = len(events)
    base_times = np.asarray([float(e["time_s"]) for e in events], dtype=float)
    oldv = np.asarray([float(old.get(event_key(rid, e["station"]), 0.0)) for e in events], dtype=float)
    obs_idx = []
    obs_val = []
    obs_w = []
    for i, e in enumerate(events):
        k = event_key(rid, e["station"])
        if k in direct:
            v, w = direct[k]
            obs_idx.append(i); obs_val.append(float(v)); obs_w.append(float(w))
    if not obs_idx:
        return oldv, 0

    # Piecewise-linear service trajectory through passenger-supported event anchors.
    # Outside the anchored range use the nearest anchor; this is equivalent to a
    # root-level shift only when a train has just one directly observed anchor.
    x = np.asarray(obs_idx, dtype=float); y = np.asarray(obs_val, dtype=float)
    interp = np.interp(np.arange(n, dtype=float), x, y)
    # Smooth weak anchors toward their previous event state; strong passenger mass wins.
    anchor_mass = max(obs_w) if obs_w else 0.0
    alpha = anchor_mass / (anchor_mass + 300.0)
    v = alpha * interp + (1.0 - alpha) * oldv
    v = np.clip(v, -MAX_EVENT_OFFSET_S, MAX_EVENT_OFFSET_S)

    # Preserve train-event ordering.  Candidate roots already have valid order; repairs
    # should therefore be rare and small.
    repairs = 0
    t = base_times + v
    for i in range(1, n):
        need = t[i-1] + MIN_EVENT_GAP_S
        if t[i] < need:
            t[i] = need; repairs += 1
    v = np.clip(t - base_times, -MAX_EVENT_OFFSET_S, MAX_EVENT_OFFSET_S)
    return v, repairs


def merge_stats(paths: list[Path], old_params_path: Path | None, roots_path: Path, out_params: Path, out_summary: Path, update_events: bool):
    stats = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    old = json.loads(old_params_path.read_text(encoding="utf-8")) if old_params_path else None

    # Reuse the same joint M-step for station access/egress distributions, transfer-path
    # mixtures and boarding hazards, but explicitly disable the obsolete root-level shift.
    with tempfile.TemporaryDirectory(prefix="mppd-r1-event-merge-") as td:
        tmp_params = Path(td) / "base.params.json"; tmp_summary = Path(td) / "base.summary.json"
        base.merge_stats(paths, old_params_path, roots_path, tmp_params, tmp_summary, update_offsets=False)
        new = json.loads(tmp_params.read_text(encoding="utf-8"))
        bsum = json.loads(tmp_summary.read_text(encoding="utf-8"))

    new["schema"] = SCHEMA
    new["root_offsets_s"] = {}
    old_event = {} if old is None else {str(k): float(v) for k, v in old.get("event_offsets_s", {}).items()}
    raw = load_raw_roots(roots_path)

    if update_events:
        agg = merge_root_egress(stats)
        direct: dict[str, tuple[float, float]] = {}
        for k, r in agg.items():
            w = float(r["w"])
            if w <= 0:
                continue
            station = str(r["station_id"])
            mean_egress = float(r["sum_seconds"]) / w
            em = new["egress"].get(station)
            if em is None:
                continue
            target_egress = base.dist_median(float(em["mu"]), float(em["sigma"]))
            prev = float(old_event.get(k, 0.0))
            residual = mean_egress - target_egress
            alpha = w / (w + 150.0)
            direct[k] = (max(-MAX_EVENT_OFFSET_S, min(MAX_EVENT_OFFSET_S, prev + alpha * residual)), w)

        offsets = {}; repairs = 0; roots_with_anchor = 0
        for r in raw["roots"]:
            rid = str(r["root_id"])
            if any(event_key(rid, e["station"]) in direct for e in r["events"]):
                roots_with_anchor += 1
            vals, rep = _interpolate_offsets(r, direct, old_event); repairs += rep
            for e, v in zip(r["events"], vals.tolist()):
                if abs(float(v)) > 1e-9:
                    offsets[event_key(rid, e["station"])] = float(v)
        new["event_offsets_s"] = offsets
        event_anchor_count = len(direct)
    else:
        new["event_offsets_s"] = old_event
        event_anchor_count = 0; roots_with_anchor = 0; repairs = 0

    new["semantics"] = {
        **new.get("semantics", {}),
        "actual_service_state_is_train_station_event_level": True,
        "event_times_jointly_updated_from_passenger_egress_factors": True,
        "unanchored_events_interpolated_within_service_root": True,
        "single_root_level_time_shift_is_not_the_service_model": True,
    }
    out_params.write_text(json.dumps(new, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    vals = np.asarray(list(new.get("event_offsets_s", {}).values()), dtype=float)
    av = np.abs(vals) if len(vals) else np.asarray([], dtype=float)
    summary = {
        **bsum,
        "schema": "mppd.r1-hz-event-level-merge-summary.v1",
        "event_direct_anchor_count": int(event_anchor_count),
        "service_roots_with_direct_anchor": int(roots_with_anchor),
        "service_event_offsets_nonzero": int(len(vals)),
        "service_event_abs_offset_median_s": float(np.median(av)) if len(av) else 0.0,
        "service_event_abs_offset_p90_s": float(np.quantile(av, .9)) if len(av) else 0.0,
        "service_event_abs_offset_max_s": float(np.max(av)) if len(av) else 0.0,
        "service_event_offset_bound_hits": int(np.sum(np.isclose(av, MAX_EVENT_OFFSET_S, atol=1e-8))) if len(av) else 0,
        "service_monotonicity_repairs": int(repairs),
    }
    summary["roots_with_nonzero_joint_offset"] = 0
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return new, summary


def finalize(params_path: Path, roots_path: Path, merge_summary_path: Path, out: Path):
    params = json.loads(params_path.read_text(encoding="utf-8"))
    merge = json.loads(merge_summary_path.read_text(encoding="utf-8"))
    raw = load_raw_roots(roots_path)
    original = {event_key(str(r["root_id"]), e["station"]): float(e["time_s"]) for r in raw["roots"] for e in r["events"]}
    with tempfile.TemporaryDirectory(prefix="mppd-r1-event-final-") as td:
        adj = Path(td) / "roots.adjusted.json"
        write_adjusted_roots(roots_path, params, adj)
        base.final_result(params_path, adj, merge_summary_path, out)
    x = json.loads(out.read_text(encoding="utf-8"))
    offsets = {str(k): float(v) for k, v in params.get("event_offsets_s", {}).items()}
    for r in x["realized_service_timetable"]:
        k = event_key(str(r["service_id"]), r["station_id"])
        r["candidate_event_time_s"] = original[k]
        r["joint_event_offset_s"] = float(offsets.get(k, 0.0))
        r.pop("joint_root_offset_s", None)
    vals = np.asarray(list(offsets.values()), dtype=float); av=np.abs(vals) if len(vals) else np.asarray([],dtype=float)
    x["schema"] = "mppd.r1-hz-event-level-full-day-result.v1"
    x["semantics"] = {
        **x.get("semantics", {}),
        "actual_service_state_is_train_station_event_level": True,
        "single_root_level_shift_rejected": True,
    }
    x["joint_parameters"] = {
        "access_boarding_hazard": float(params["access_hazard"]),
        "transfer_boarding_hazard": float(params["transfer_hazard"]),
        "service_event_offsets_nonzero": int(len(vals)),
        "service_event_abs_offset_median_s": float(np.median(av)) if len(av) else 0.0,
        "service_event_abs_offset_p90_s": float(np.quantile(av,.9)) if len(av) else 0.0,
        "service_event_abs_offset_max_s": float(np.max(av)) if len(av) else 0.0,
        "service_event_offset_bound_hits": int(np.sum(np.isclose(av,MAX_EVENT_OFFSET_S,atol=1e-8))) if len(av) else 0,
    }
    out.write_text(json.dumps(x, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"status":x["status"],"coverage":x["coverage"],"joint_parameters":x["joint_parameters"],"diagnostics":x["diagnostics"]},ensure_ascii=False,indent=2))
    return x


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    s=sub.add_parser("shard"); s.add_argument("--edges",type=Path,required=True); s.add_argument("--roots",type=Path,required=True); s.add_argument("--routes",type=Path,required=True)
    s.add_argument("--params",type=Path); s.add_argument("--out-stats",type=Path,required=True); s.add_argument("--shard-index",type=int,required=True); s.add_argument("--chains-out",type=Path)
    s=sub.add_parser("merge"); s.add_argument("--stats",type=Path,action="append",required=True); s.add_argument("--old-params",type=Path); s.add_argument("--roots",type=Path,required=True)
    s.add_argument("--out-params",type=Path,required=True); s.add_argument("--out-summary",type=Path,required=True); s.add_argument("--update-events",action="store_true")
    s=sub.add_parser("finalize"); s.add_argument("--params",type=Path,required=True); s.add_argument("--roots",type=Path,required=True); s.add_argument("--merge-summary",type=Path,required=True); s.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.cmd=="shard": run_shard(a.edges,a.roots,a.routes,a.params,a.out_stats,a.shard_index,a.chains_out)
    elif a.cmd=="merge": merge_stats(a.stats,a.old_params,a.roots,a.out_params,a.out_summary,a.update_events)
    else: finalize(a.params,a.roots,a.merge_summary,a.out)


if __name__=="__main__":
    main()
