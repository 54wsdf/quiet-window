from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_STATS = "mppd.r1-hz-joint-shard-stats.v1"
SCHEMA_PARAMS = "mppd.r1-hz-joint-parameters.v1"
SCHEMA_RESULT = "mppd.r1-hz-joint-full-day-result.v1"
MIN_MOVE_S = 5.0
SUPPORT_TOP_K = 32
EPS = 1e-300
N = NormalDist()


def logsumexp(xs: Iterable[float]) -> float:
    a = [float(x) for x in xs if math.isfinite(float(x))]
    if not a:
        return -math.inf
    m = max(a)
    return m + math.log(sum(math.exp(x - m) for x in a))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def normal_pdf(z: float) -> float:
    if math.isinf(z):
        return 0.0
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def interval_normal_moments(mu: float, sigma: float, lo: float, hi: float) -> tuple[float, float, float]:
    """Return untruncated normal mass and conditional E[Y], E[Y^2] on log-time interval."""
    if hi <= lo or hi <= 0:
        return 0.0, 0.0, 0.0
    lo = max(MIN_MOVE_S, lo)
    if hi <= lo:
        return 0.0, 0.0, 0.0
    a = (math.log(lo) - mu) / sigma
    b = (math.log(hi) - mu) / sigma
    p = max(0.0, N.cdf(b) - N.cdf(a))
    if p <= EPS:
        return 0.0, 0.0, 0.0
    ez = (normal_pdf(a) - normal_pdf(b)) / p
    ez2 = 1.0 + (a * normal_pdf(a) - b * normal_pdf(b)) / p
    ey = mu + sigma * ez
    ey2 = mu * mu + 2.0 * mu * sigma * ez + sigma * sigma * ez2
    return p, ey, ey2


def truncated_survival(mu: float, sigma: float) -> float:
    z = (math.log(MIN_MOVE_S) - mu) / sigma
    return max(EPS, 1.0 - N.cdf(z))


def truncated_logpdf(x: float, mu: float, sigma: float) -> float:
    if x < MIN_MOVE_S or not math.isfinite(x):
        return -math.inf
    y = math.log(x)
    z = (y - mu) / sigma
    return -math.log(x * sigma * math.sqrt(2.0 * math.pi)) - 0.5 * z * z - math.log(truncated_survival(mu, sigma))


def truncated_quantile(mu: float, sigma: float, q: float) -> float:
    f0 = N.cdf((math.log(MIN_MOVE_S) - mu) / sigma)
    u = clamp(f0 + q * (1.0 - f0), 1e-12, 1.0 - 1e-12)
    return max(MIN_MOVE_S, math.exp(mu + sigma * N.inv_cdf(u)))


def dist_median(mu: float, sigma: float) -> float:
    return truncated_quantile(mu, sigma, 0.5)


def boarding_component(
    base_time: float,
    selected_time: float,
    previous_times: list[float],
    mu: float,
    sigma: float,
    hazard: float,
) -> dict[str, float] | None:
    """Integrate a physical movement time over service-opportunity intervals.

    base_time is gate entry for access or previous-train arrival for transfer.
    Once movement is complete, every earlier eligible service can be skipped with
    probability (1-hazard); the selected service is boarded with hazard.
    """
    upper = selected_time - base_time
    if upper < MIN_MOVE_S:
        return None
    prev_gaps = sorted(
        t - base_time for t in previous_times
        if base_time + MIN_MOVE_S < t < selected_time - 1e-9
    )
    thresholds = [MIN_MOVE_S] + prev_gaps + [upper]
    surv = truncated_survival(mu, sigma)
    terms: list[tuple[float, float, float, float, int]] = []
    nprev = len(prev_gaps)
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        p0, ey, ey2 = interval_normal_moments(mu, sigma, lo, hi)
        if p0 <= 0:
            continue
        p = p0 / surv
        skipped = nprev - i
        weight = p * hazard * ((1.0 - hazard) ** skipped)
        if weight > 0:
            # log-time moments are exact within the interval; seconds use exp(E log X)
            terms.append((weight, ey, ey2, math.exp(ey), skipped))
    total = sum(x[0] for x in terms)
    if total <= EPS:
        return None
    return {
        "loglik": math.log(total),
        "elog": sum(w * ey for w, ey, _ey2, _ex, _sk in terms) / total,
        "elog2": sum(w * ey2 for w, _ey, ey2, _ex, _sk in terms) / total,
        "etime": sum(w * ex for w, _ey, _ey2, ex, _sk in terms) / total,
        "eskip": sum(w * sk for w, _ey, _ey2, _ex, sk in terms) / total,
        "upper": upper,
    }


def transfer_mixture(
    base_time: float,
    selected_time: float,
    previous_times: list[float],
    model: dict[str, Any],
    hazard: float,
) -> dict[str, Any] | None:
    comps = model["components"]
    rows = []
    for idx, c in enumerate(comps):
        r = boarding_component(base_time, selected_time, previous_times, float(c["mu"]), float(c["sigma"]), hazard)
        if r is None:
            continue
        lw = math.log(max(EPS, float(c["weight"]))) + float(r["loglik"])
        rows.append((idx, lw, r))
    z = logsumexp(x[1] for x in rows)
    if not math.isfinite(z):
        return None
    post = []
    for idx, lw, r in rows:
        q = math.exp(lw - z)
        post.append({"component": idx, "prob": q, **r})
    return {
        "loglik": z,
        "components": post,
        "etime": sum(float(x["prob"]) * float(x["etime"]) for x in post),
        "eskip": sum(float(x["prob"]) * float(x["eskip"]) for x in post),
    }


def load_roots(path: Path, offsets: dict[str, float] | None = None):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status") != "QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION":
        raise SystemExit("candidate service roots are not qualified")
    offsets = offsets or {}
    roots: dict[str, dict[str, Any]] = {}
    stations: set[str] = set()
    for r in raw["roots"]:
        rid = str(r["root_id"])
        off = float(offsets.get(rid, 0.0))
        events = {}
        for seq, e in enumerate(r["events"]):
            s = str(int(e["station"]))
            stations.add(s)
            events[s] = {
                "station": s,
                "sequence_index": seq,
                "base_time_s": float(e["time_s"]),
                "time_s": float(e["time_s"]) + off,
                "sd_s": float(e.get("sd_s", 90.0)),
            }
        roots[rid] = {
            "root_id": rid,
            "path_id": str(r["path_id"]),
            "direction": str(r["direction"]),
            "events": events,
            "offset_s": off,
            "evidence_class": str(r.get("evidence_class", "UNKNOWN")),
            "matched_pulse_share": float(r.get("matched_pulse_share", 0.0)),
        }
    return raw, roots, stations


def build_sequences(roots: dict[str, dict[str, Any]]):
    seq: dict[tuple[str, str, str], list[tuple[float, str]]] = defaultdict(list)
    for rid, r in roots.items():
        for s, e in r["events"].items():
            seq[(r["path_id"], r["direction"], s)].append((float(e["time_s"]), rid))
    for k in seq:
        seq[k].sort(key=lambda x: (x[0], x[1]))
    return seq


def load_routes(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status") != "QUALIFIED_LINE_AWARE_ROUTE_SUPPORT":
        raise SystemExit("route support is not qualified")
    out = {}
    for key, vals in raw["route_support"].items():
        out[key] = {int(v["rank"]): v for v in vals}
    return raw, out


def default_params(stations: set[str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA_PARAMS,
        "iteration": 0,
        "minimum_physical_move_s": MIN_MOVE_S,
        "access_hazard": 0.85,
        "transfer_hazard": 0.75,
        "access": {s: {"mu": math.log(60.0), "sigma": 0.75, "evidence_mass": 0.0} for s in sorted(stations)},
        "egress": {s: {"mu": math.log(60.0), "sigma": 0.75, "evidence_mass": 0.0} for s in sorted(stations)},
        "transfers": {},
        "root_offsets_s": {},
        "semantics": {
            "joint_model": True,
            "candidate_root_times_are_latent_initialization_not_realized_truth": True,
            "planned_absolute_timetable_not_used_as_realized_truth": True,
            "transfer_paths_are_latent_when_station_topology_is_unavailable": True,
        },
    }


def transfer_default() -> dict[str, Any]:
    return {
        "components": [
            {"path_id": "latent_path_1", "weight": 0.5, "mu": math.log(35.0), "sigma": 0.55, "evidence_mass": 0.0},
            {"path_id": "latent_path_2", "weight": 0.5, "mu": math.log(120.0), "sigma": 0.55, "evidence_mass": 0.0},
        ]
    }


def family_previous_times(seq, root: dict[str, Any], station: str, selected_time: float, base_time: float) -> list[float]:
    return [t for t, _rid in seq.get((root["path_id"], root["direction"], station), []) if base_time + MIN_MOVE_S < t < selected_time - 1e-9]


def route_key(row: dict[str, Any]) -> str:
    return f"{row['origin_line']}:{int(row['origin_station'])}->{row['destination_line']}:{int(row['destination_station'])}"


def movement_key(route: dict[str, Any], j: int) -> str:
    metas = route.get("inter_leg_movement_meta", [])
    if j < len(metas):
        m = metas[j]
        if m.get("movement"):
            return str(m["movement"])
    moves = route.get("transfer_movements", [])
    return str(moves[j]) if j < len(moves) else f"UNKNOWN_TRANSFER_{j}"


def evaluate_candidate(row: dict[str, Any], route: dict[str, Any], roots: dict[str, Any], seq, params: dict[str, Any]):
    chain = [x for x in str(row.get("root_chain", "")).split(">") if x]
    legs = route.get("ride_legs", [])
    if not chain or len(chain) != len(legs):
        return None
    if any(rid not in roots for rid in chain):
        return None
    entry = float(row["entry_sec"]); exit_t = float(row["exit_sec"])
    origin = str(int(row["origin_station"])); dest = str(int(row["destination_station"]))
    route_cost = float(route.get("base_ranking_cost_s", 0.0))

    rr = []
    for rid, leg in zip(chain, legs):
        r = roots[rid]
        b = str(int(leg["from_station"])); a = str(int(leg["to_station"]))
        if b not in r["events"] or a not in r["events"]:
            return None
        dep = float(r["events"][b]["time_s"]); arr = float(r["events"][a]["time_s"])
        if arr <= dep:
            return None
        rr.append((r, b, a, dep, arr))

    am = params["access"].get(origin, {"mu": math.log(60.0), "sigma": 0.75})
    prev = family_previous_times(seq, rr[0][0], rr[0][1], rr[0][3], entry)
    access = boarding_component(entry, rr[0][3], prev, float(am["mu"]), float(am["sigma"]), float(params["access_hazard"]))
    if access is None:
        return None

    egress_t = exit_t - rr[-1][4]
    em = params["egress"].get(dest, {"mu": math.log(60.0), "sigma": 0.75})
    ell = truncated_logpdf(egress_t, float(em["mu"]), float(em["sigma"]))
    if not math.isfinite(ell):
        return None

    transfers = []
    tll = 0.0
    for j in range(len(rr) - 1):
        if rr[j][2] != rr[j + 1][1]:
            return None
        mk = movement_key(route, j)
        model = params["transfers"].get(mk, transfer_default())
        prev_times = family_previous_times(seq, rr[j + 1][0], rr[j + 1][1], rr[j + 1][3], rr[j][4])
        tr = transfer_mixture(rr[j][4], rr[j + 1][3], prev_times, model, float(params["transfer_hazard"]))
        if tr is None:
            return None
        tll += float(tr["loglik"])
        transfers.append((mk, tr))

    # Weak route ranking prior only; old E0 probabilities define support, not authority.
    logprior = -route_cost / 1800.0
    loglik = float(access["loglik"]) + ell + tll + logprior
    return {
        "loglik": loglik,
        "chain": chain,
        "route_rank": int(row["route_rank"]),
        "rides": rr,
        "access": access,
        "egress_time": egress_t,
        "transfers": transfers,
    }


def iter_cohort_groups(path: Path):
    pf = pq.ParquetFile(path)
    current = None; rows = []
    for batch in pf.iter_batches(batch_size=50000):
        for row in batch.to_pylist():
            cid = str(row["cohort_id"])
            if current is None:
                current = cid
            if cid != current:
                yield rows
                rows = []; current = cid
            rows.append(row)
    if rows:
        yield rows


def add_moment(d: dict[str, dict[str, float]], key: str, w: float, elog: float, elog2: float):
    z = d.setdefault(key, {"w": 0.0, "sum_log": 0.0, "sum_log2": 0.0})
    z["w"] += w; z["sum_log"] += w * elog; z["sum_log2"] += w * elog2


def add_root_egress(d, rid: str, station: str, w: float, seconds: float):
    key = f"{rid}|{station}"
    z = d.setdefault(key, {"root_id": rid, "station_id": station, "w": 0.0, "sum_seconds": 0.0})
    z["w"] += w; z["sum_seconds"] += w * seconds


def chain_schema() -> pa.Schema:
    return pa.schema([
        ("cohort_id", pa.string()), ("origin_station", pa.int32()), ("destination_station", pa.int32()),
        ("entry_sec", pa.float64()), ("exit_sec", pa.float64()), ("passenger_mass", pa.float64()),
        ("route_rank", pa.int16()), ("root_chain", pa.string()), ("posterior_probability", pa.float64()),
        ("access_time_s", pa.float64()), ("initial_wait_s", pa.float64()), ("egress_time_s", pa.float64()),
        ("transfer_assignments_json", pa.string()), ("ride_assignments_json", pa.string()),
        ("travel_time_closure_error_s", pa.float64()),
    ])


def run_shard(edges_path: Path, roots_path: Path, routes_path: Path, params_path: Path | None, out_stats: Path, shard_index: int, chains_out: Path | None):
    root_raw, roots0, stations = load_roots(roots_path)
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path else default_params(stations)
    _raw2, roots, _ = load_roots(roots_path, {k: float(v) for k, v in params.get("root_offsets_s", {}).items()})
    seq = build_sequences(roots)
    _route_raw, routes = load_routes(routes_path)

    access_stats = {}; egress_stats = {}; transfer_stats = {}; root_egress = {}
    boards_access = skips_access = boards_transfer = skips_transfer = 0.0
    passenger_mass = resolved_mass = unresolved_mass = 0.0
    cohort_count = resolved_cohorts = 0
    candidate_evaluations = 0; support_truncated_cohorts = 0
    closure_max = 0.0

    writer = None; chain_buffer = []
    if chains_out:
        chains_out.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(chains_out, chain_schema(), compression="zstd")

    def flush():
        nonlocal chain_buffer
        if writer is not None and chain_buffer:
            writer.write_table(pa.Table.from_pylist(chain_buffer, schema=chain_schema()))
            chain_buffer = []

    try:
        for group in iter_cohort_groups(edges_path):
            cohort_count += 1
            first = group[0]; mass = float(first["passenger_mass"]); passenger_mass += mass
            finite_rows = [r for r in group if str(r.get("descendant_state_type")) == "SERVICE_CHAIN_STATE" and str(r.get("root_chain", ""))]
            if not finite_rows:
                unresolved_mass += mass; continue
            finite_rows.sort(key=lambda r: float(r.get("posterior_probability", 0.0)), reverse=True)
            if len(finite_rows) > SUPPORT_TOP_K:
                support_truncated_cohorts += 1
                finite_rows = finite_rows[:SUPPORT_TOP_K]
            evals = []
            rk = route_key(first); route_map = routes.get(rk, {})
            mincost = min((float(x.get("base_ranking_cost_s", 0.0)) for x in route_map.values()), default=0.0)
            for r in finite_rows:
                route = route_map.get(int(r["route_rank"]))
                if route is None:
                    continue
                # center route prior at the best route for this OD
                route = dict(route); route["base_ranking_cost_s"] = float(route.get("base_ranking_cost_s", 0.0)) - mincost
                ev = evaluate_candidate(r, route, roots, seq, params)
                candidate_evaluations += 1
                if ev is not None:
                    evals.append((r, ev))
            if not evals:
                unresolved_mass += mass; continue
            z = logsumexp(ev["loglik"] for _r, ev in evals)
            if not math.isfinite(z):
                unresolved_mass += mass; continue
            resolved_mass += mass; resolved_cohorts += 1
            best = None
            for r, ev in evals:
                q = math.exp(float(ev["loglik"]) - z); w = mass * q
                a = ev["access"]
                add_moment(access_stats, str(int(first["origin_station"])), w, float(a["elog"]), float(a["elog2"]))
                boards_access += w; skips_access += w * float(a["eskip"])
                e = float(ev["egress_time"])
                le = math.log(e)
                add_moment(egress_stats, str(int(first["destination_station"])), w, le, le * le)
                last_root = str(ev["chain"][-1]); last_station = str(ev["rides"][-1][2])
                add_root_egress(root_egress, last_root, last_station, w, e)
                for mk, tr in ev["transfers"]:
                    for c in tr["components"]:
                        key = f"{mk}|{int(c['component'])}"
                        add_moment(transfer_stats, key, w * float(c["prob"]), float(c["elog"]), float(c["elog2"]))
                    boards_transfer += w; skips_transfer += w * float(tr["eskip"])
                score = q
                if best is None or score > best[0]:
                    best = (score, r, ev)
            if writer is not None and best is not None:
                q, r, ev = best
                acc = float(ev["access"]["etime"])
                first_dep = float(ev["rides"][0][3]); initial_wait = first_dep - float(first["entry_sec"]) - acc
                trs = []; tsum = wsum = 0.0
                for j, (mk, tr) in enumerate(ev["transfers"]):
                    comp = max(tr["components"], key=lambda x: float(x["prob"]))
                    kt = max(MIN_MOVE_S, float(comp["etime"]))
                    prev_arr = float(ev["rides"][j][4]); next_dep = float(ev["rides"][j+1][3])
                    wait = next_dep - prev_arr - kt
                    tsum += kt; wsum += wait
                    model = params["transfers"].get(mk, transfer_default())
                    path_id = str(model["components"][int(comp["component"])]["path_id"])
                    trs.append({"movement": mk, "path_id": path_id, "transfer_time_s": kt, "wait_time_s": wait})
                rides_json = []
                ride_time = 0.0
                for rr in ev["rides"]:
                    root, b, a, dep, arr = rr
                    rides_json.append({"service_id": root["root_id"], "board_station": int(b), "alight_station": int(a), "board_time_s": dep, "alight_time_s": arr})
                    ride_time += arr - dep
                egr = float(ev["egress_time"])
                total = acc + initial_wait + ride_time + tsum + wsum + egr
                observed = float(first["exit_sec"]) - float(first["entry_sec"])
                cerr = total - observed; closure_max = max(closure_max, abs(cerr))
                chain_buffer.append({
                    "cohort_id": str(first["cohort_id"]), "origin_station": int(first["origin_station"]),
                    "destination_station": int(first["destination_station"]), "entry_sec": float(first["entry_sec"]),
                    "exit_sec": float(first["exit_sec"]), "passenger_mass": mass, "route_rank": int(ev["route_rank"]),
                    "root_chain": ">".join(ev["chain"]), "posterior_probability": float(q),
                    "access_time_s": acc, "initial_wait_s": initial_wait, "egress_time_s": egr,
                    "transfer_assignments_json": json.dumps(trs, ensure_ascii=False, separators=(",", ":")),
                    "ride_assignments_json": json.dumps(rides_json, ensure_ascii=False, separators=(",", ":")),
                    "travel_time_closure_error_s": cerr,
                })
                if len(chain_buffer) >= 20000:
                    flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    result = {
        "schema": SCHEMA_STATS, "service_date": root_raw.get("source_date", "2019-01-04"), "shard_index": int(shard_index),
        "parameter_iteration": int(params.get("iteration", 0)), "passenger_mass": passenger_mass,
        "resolved_mass": resolved_mass, "unresolved_mass": unresolved_mass, "cohort_count": cohort_count,
        "resolved_cohort_count": resolved_cohorts, "candidate_evaluations": candidate_evaluations,
        "support_top_k": SUPPORT_TOP_K, "support_truncated_cohort_count": support_truncated_cohorts,
        "access_stats": access_stats, "egress_stats": egress_stats, "transfer_stats": transfer_stats,
        "root_egress_stats": root_egress,
        "hazard_stats": {"access_boards": boards_access, "access_skips": skips_access,
                         "transfer_boards": boards_transfer, "transfer_skips": skips_transfer},
        "diagnostics": {"max_abs_map_chain_time_closure_error_s": closure_max},
        "semantics": {"old_e0_posterior_used_only_to_define_finite_candidate_support": True,
                      "old_e0_posterior_not_used_as_new_posterior_weight": True,
                      "all_cohorts_in_shard_processed": True},
    }
    out_stats.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["shard_index","passenger_mass","resolved_mass","unresolved_mass","cohort_count","resolved_cohort_count","candidate_evaluations"]}, ensure_ascii=False, indent=2))
    return result


def merge_moments(stats: list[dict[str, Any]], field: str):
    out = defaultdict(lambda: {"w": 0.0, "sum_log": 0.0, "sum_log2": 0.0})
    for s in stats:
        for k, v in s[field].items():
            for f in ("w", "sum_log", "sum_log2"):
                out[k][f] += float(v[f])
    return dict(out)


def update_dist(old: dict[str, Any], st: dict[str, float], prior_mass: float = 25.0):
    w = float(st.get("w", 0.0))
    omu = float(old["mu"]); osig = float(old["sigma"])
    if w <= 0:
        return {**old, "evidence_mass": 0.0}
    m = float(st["sum_log"] / w); v = max(0.02**2, float(st["sum_log2"] / w - m * m))
    mu = (w * m + prior_mass * omu) / (w + prior_mass)
    # blend second moments to avoid zero-variance collapse
    sig = math.sqrt((w * v + prior_mass * osig * osig) / (w + prior_mass))
    sig = clamp(sig, 0.12, 1.6)
    return {"mu": mu, "sigma": sig, "evidence_mass": w}


def merge_stats(paths: list[Path], old_params_path: Path | None, roots_path: Path, out_params: Path, out_summary: Path, update_offsets: bool):
    stats = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    _root_raw, roots0, stations = load_roots(roots_path)
    old = json.loads(old_params_path.read_text(encoding="utf-8")) if old_params_path else default_params(stations)
    acc = merge_moments(stats, "access_stats"); egr = merge_moments(stats, "egress_stats"); trs = merge_moments(stats, "transfer_stats")
    new = default_params(stations)
    new["iteration"] = int(old.get("iteration", 0)) + 1
    new["access"] = {s: update_dist(old["access"].get(s, {"mu":math.log(60.0),"sigma":0.75}), acc.get(s, {})) for s in sorted(stations)}
    new["egress"] = {s: update_dist(old["egress"].get(s, {"mu":math.log(60.0),"sigma":0.75}), egr.get(s, {})) for s in sorted(stations)}

    h = defaultdict(float)
    for s in stats:
        for k,v in s["hazard_stats"].items(): h[k] += float(v)
    new["access_hazard"] = clamp(h["access_boards"] / max(EPS, h["access_boards"] + h["access_skips"]), 0.05, 0.995)
    new["transfer_hazard"] = clamp(h["transfer_boards"] / max(EPS, h["transfer_boards"] + h["transfer_skips"]), 0.05, 0.995)

    movement_names = sorted({k.rsplit("|",1)[0] for k in trs} | set(old.get("transfers",{})))
    new_transfers = {}
    for mk in movement_names:
        oldm = old.get("transfers", {}).get(mk, transfer_default())
        comps = []
        for i in range(2):
            oc = oldm["components"][i]
            uc = update_dist(oc, trs.get(f"{mk}|{i}", {}), prior_mass=50.0)
            comps.append({"path_id": str(oc.get("path_id",f"latent_path_{i+1}")), "weight": float(oc.get("weight",0.5)), **uc})
        # Mixture weights from posterior component evidence with a small symmetric Dirichlet prior.
        ws = [max(0.0, float(c.get("evidence_mass",0.0))) + 10.0 for c in comps]; sw=sum(ws)
        for c,w in zip(comps,ws): c["weight"] = w/sw
        comps.sort(key=lambda c: float(c["mu"]))
        for i,c in enumerate(comps): c["path_id"] = f"latent_path_{i+1}"
        new_transfers[mk] = {"components": comps}
    new["transfers"] = new_transfers

    # Root-time update is anchored by the AFC-derived candidate-root field.  Only the
    # first merge updates offsets; the final merge freezes them so final chains and
    # timetable refer to exactly the same latent service state.
    old_offsets = {k: float(v) for k,v in old.get("root_offsets_s",{}).items()}
    if update_offsets:
        re = defaultdict(lambda: {"w":0.0,"sum_resid":0.0})
        for s in stats:
            for _k, r in s["root_egress_stats"].items():
                rid=str(r["root_id"]); station=str(r["station_id"]); w=float(r["w"])
                if w<=0: continue
                mean_e=float(r["sum_seconds"])/w
                target_e=dist_median(float(new["egress"][station]["mu"]), float(new["egress"][station]["sigma"]))
                re[rid]["w"] += w; re[rid]["sum_resid"] += w*(mean_e-target_e)
        offsets={}
        for rid in roots0:
            prev=float(old_offsets.get(rid,0.0)); r=re.get(rid); off=prev
            if r and r["w"]>0:
                resid=r["sum_resid"]/r["w"]
                alpha=r["w"]/(r["w"]+500.0)
                off=clamp(prev+alpha*resid,-180.0,180.0)
            if abs(off)>1e-9: offsets[rid]=off
        new["root_offsets_s"] = offsets
    else:
        new["root_offsets_s"] = old_offsets
    new["semantics"] = old.get("semantics", new["semantics"])

    out_params.write_text(json.dumps(new,ensure_ascii=False,separators=(",", ":")),encoding="utf-8")
    passenger=sum(float(x["passenger_mass"]) for x in stats); resolved=sum(float(x["resolved_mass"]) for x in stats)
    summary={
        "schema":"mppd.r1-hz-joint-merge-summary.v1","iteration":new["iteration"],"shard_count":len(stats),
        "passenger_mass":passenger,"resolved_mass":resolved,"resolved_share":resolved/passenger if passenger else None,
        "cohort_count":sum(int(x["cohort_count"]) for x in stats),
        "resolved_cohort_count":sum(int(x["resolved_cohort_count"]) for x in stats),
        "access_hazard":new["access_hazard"],"transfer_hazard":new["transfer_hazard"],
        "station_access_models_with_evidence":sum(float(x.get("evidence_mass",0))>0 for x in new["access"].values()),
        "station_egress_models_with_evidence":sum(float(x.get("evidence_mass",0))>0 for x in new["egress"].values()),
        "transfer_movement_model_count":len(new["transfers"]),
        "roots_with_nonzero_joint_offset":sum(abs(float(v))>1e-9 for v in new["root_offsets_s"].values()),
        "max_abs_map_chain_time_closure_error_s":max(float(x["diagnostics"].get("max_abs_map_chain_time_closure_error_s",0)) for x in stats),
    }
    out_summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return new,summary


def final_result(params_path: Path, roots_path: Path, merge_summary_path: Path, out: Path):
    params=json.loads(params_path.read_text(encoding="utf-8")); merge=json.loads(merge_summary_path.read_text(encoding="utf-8"))
    root_raw, roots, stations=load_roots(roots_path,{k:float(v) for k,v in params.get("root_offsets_s",{}).items()})
    timetable=[]
    for rid,r in sorted(roots.items()):
        for s,e in sorted(r["events"].items(),key=lambda kv:int(kv[1]["sequence_index"])):
            timetable.append({"service_id":rid,"path_id":r["path_id"],"direction":r["direction"],"station_id":int(s),
                              "sequence_index":int(e["sequence_index"]),"realized_event_time_s":float(e["time_s"]),
                              "event_time_sd_s":float(e["sd_s"]),"joint_root_offset_s":float(r["offset_s"]),
                              "candidate_event_time_s":float(e["base_time_s"])})
    station_intervals=[]
    for s in sorted(stations,key=int):
        a=params["access"][s]; e=params["egress"][s]
        station_intervals.append({"station_id":int(s),
          "access_interval_s":[truncated_quantile(float(a["mu"]),float(a["sigma"]),.05),truncated_quantile(float(a["mu"]),float(a["sigma"]),.95)],
          "access_median_s":dist_median(float(a["mu"]),float(a["sigma"])),"access_evidence_mass":float(a.get("evidence_mass",0)),
          "egress_interval_s":[truncated_quantile(float(e["mu"]),float(e["sigma"]),.05),truncated_quantile(float(e["mu"]),float(e["sigma"]),.95)],
          "egress_median_s":dist_median(float(e["mu"]),float(e["sigma"])),"egress_evidence_mass":float(e.get("evidence_mass",0))})
    transfer_intervals=[]
    for mk,m in sorted(params["transfers"].items()):
        paths=[]
        for c in m["components"]:
            paths.append({"path_id":c["path_id"],"weight":float(c["weight"]),
                          "transfer_interval_s":[truncated_quantile(float(c["mu"]),float(c["sigma"]),.05),truncated_quantile(float(c["mu"]),float(c["sigma"]),.95)],
                          "median_s":dist_median(float(c["mu"]),float(c["sigma"])),"evidence_mass":float(c.get("evidence_mass",0))})
        transfer_intervals.append({"movement":mk,"paths":paths})
    result={
      "schema":SCHEMA_RESULT,"dataset_id":"CN_HZ_Tianchi_2019","service_date":root_raw.get("source_date","2019-01-04"),
      "scope":"FULL_SERVICE_DAY_0400_TO_NEXT_0400_FULL_NETWORK_NO_PASSENGER_SUBSAMPLE",
      "status":"R1_JOINT_FULL_DAY_ESTIMATED_REQUIRES_SCIENTIFIC_AUDIT",
      "semantics":{"r1_is_single_joint_inversion":True,"planned_timetable_is_soft_structural_prior_only":True,
                   "realized_service_event_times_are_latent":True,"station_access_egress_and_transfer_paths_jointly_estimated":True,
                   "old_r1b_r1c_results_not_frozen_as_truth":True},
      "hard_constraints":{"minimum_access_s":MIN_MOVE_S,"minimum_egress_s":MIN_MOVE_S,"minimum_transfer_s":MIN_MOVE_S,"minimum_wait_s":0.0},
      "coverage":{"service_root_count":len(roots),"service_event_count":len(timetable),"station_count":len(station_intervals),
                  "transfer_movement_count":len(transfer_intervals),"transfer_path_count":sum(len(x["paths"]) for x in transfer_intervals),
                  "passenger_mass":merge["passenger_mass"],"resolved_passenger_mass":merge["resolved_mass"],"resolved_share":merge["resolved_share"]},
      "joint_parameters":{"access_boarding_hazard":float(params["access_hazard"]),"transfer_boarding_hazard":float(params["transfer_hazard"]),
                          "roots_with_nonzero_joint_offset":sum(abs(float(v))>1e-9 for v in params.get("root_offsets_s",{}).values())},
      "diagnostics":{"max_abs_map_chain_time_closure_error_s":merge["max_abs_map_chain_time_closure_error_s"],
                     "support_top_k_per_cohort":SUPPORT_TOP_K,
                     "candidate_support_source":"historical E0 genealogy used only as finite candidate-chain proposal support; new posterior recomputed from unified model"},
      "realized_service_timetable":timetable,"station_movement_intervals":station_intervals,"transfer_path_intervals":transfer_intervals,
    }
    out.write_text(json.dumps(result,ensure_ascii=False,separators=(",", ":")),encoding="utf-8")
    print(json.dumps({"status":result["status"],"coverage":result["coverage"],"joint_parameters":result["joint_parameters"],"diagnostics":result["diagnostics"]},ensure_ascii=False,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    s=sub.add_parser("shard"); s.add_argument("--edges",type=Path,required=True); s.add_argument("--roots",type=Path,required=True); s.add_argument("--routes",type=Path,required=True)
    s.add_argument("--params",type=Path); s.add_argument("--out-stats",type=Path,required=True); s.add_argument("--shard-index",type=int,required=True); s.add_argument("--chains-out",type=Path)
    s=sub.add_parser("merge"); s.add_argument("--stats",type=Path,action="append",required=True); s.add_argument("--old-params",type=Path); s.add_argument("--roots",type=Path,required=True)
    s.add_argument("--out-params",type=Path,required=True); s.add_argument("--out-summary",type=Path,required=True); s.add_argument("--update-offsets",action="store_true")
    s=sub.add_parser("finalize"); s.add_argument("--params",type=Path,required=True); s.add_argument("--roots",type=Path,required=True); s.add_argument("--merge-summary",type=Path,required=True); s.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.cmd=="shard": run_shard(a.edges,a.roots,a.routes,a.params,a.out_stats,a.shard_index,a.chains_out)
    elif a.cmd=="merge": merge_stats(a.stats,a.old_params,a.roots,a.out_params,a.out_summary,a.update_offsets)
    else: final_result(a.params,a.roots,a.merge_summary,a.out)


if __name__=="__main__": main()
