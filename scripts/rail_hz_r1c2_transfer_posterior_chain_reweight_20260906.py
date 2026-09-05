from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import ndtr

SCHEMA = "rail.hz-r1c2-transfer-posterior-chain-reweight.v1"
TIME_BIN_S = 1800.0
EPS = 1e-300
GH_X, GH_W = np.polynomial.hermite.hermgauss(5)
GH_Z = math.sqrt(2.0) * GH_X
GH_WN = GH_W / math.sqrt(math.pi)


def dump(path: Path, x: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def logsumexp(xs: list[float]) -> float:
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(x - m) for x in finite))


def lognormal_cdf(x: float, median: float, sigma: float) -> float:
    if x <= 0:
        return 0.0
    z = (math.log(x) - math.log(median)) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def old_log_interval_density(center: float, sd: float, median: float, sigma: float) -> float:
    if not math.isfinite(center) or not math.isfinite(sd):
        return -math.inf
    sd = max(0.0, float(sd))
    if sd < 0.5:
        if center <= 0:
            return -math.inf
        lx = math.log(center)
        mu = math.log(median)
        return -math.log(center * sigma * math.sqrt(2.0 * math.pi)) - ((lx - mu) ** 2) / (2.0 * sigma ** 2)
    lo = center - 1.96 * sd
    hi = center + 1.96 * sd
    if hi <= 0:
        return -math.inf
    lo_pos = max(1e-6, lo)
    p = max(0.0, lognormal_cdf(hi, median, sigma) - lognormal_cdf(lo_pos, median, sigma))
    width = max(1.0, hi - lo_pos)
    if p <= EPS:
        return -math.inf
    return math.log(p / width)


def load_old_roots(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status") != "QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION":
        raise SystemExit(f"old root authority not qualified: {raw.get('status')}")
    out = {}
    for r in raw["roots"]:
        rid = str(r["root_id"])
        for e in r["events"]:
            out[(rid, int(e["station"]))] = (float(e["time_s"]), float(e["sd_s"]))
    return out


def load_old_transfer_kernel(path: Path):
    x = json.loads(path.read_text(encoding="utf-8"))
    k = x["kernels0"]["transfer"]
    return float(k["median_sec"]), float(k["sigma"]), str(k.get("evidence_class", ""))


def load_realized_schedule(path: Path):
    events = list(iter_jsonl_gz(path))
    if len(events) != 43584:
        raise SystemExit(f"expected 43584 frozen R1B root-event states, found {len(events)}")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    root_meta: dict[str, tuple[str, str, str]] = {}
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        rid = str(e["root_id"])
        st = int(e["station"])
        by_key[(rid, st)] = e
        root_meta[rid] = (str(e["path_id"]), str(e["direction"]), str(e["physical_service_id"]))
        groups[(str(e["path_id"]), str(e["direction"]), st)].append(e)
    service_seq = {}
    for key, rows in groups.items():
        uniq = {}
        for e in rows:
            pid = str(e["physical_service_id"])
            prev = uniq.get(pid)
            if prev is None or str(e["root_id"]) < str(prev["root_id"]):
                uniq[pid] = e
        service_seq[key] = sorted(
            uniq.values(), key=lambda e: (float(e["realized_time_mean_s"]), str(e["physical_service_id"]))
        )
    return by_key, root_meta, service_seq


def load_k1_fits(paths: list[Path]):
    fits = {}
    for p in paths:
        x = json.loads(p.read_text(encoding="utf-8"))
        if x.get("status") != "QUALIFIED_R1C_MOVEMENT_GENERATIVE_K1_FIT":
            raise SystemExit(f"R1C1 movement result not qualified: {p}")
        movement = str(x["movement"])
        if movement in fits:
            raise SystemExit(f"duplicate movement fit: {movement}")
        contexts = {int(c["time_bin_index_30m"]): c["strict_first_boardable_generative_estimate"] for c in x["time_contexts"]}
        fits[movement] = {
            "global": x["movement_level_fit"]["parameters"],
            "contexts": contexts,
        }
    if len(fits) != 9:
        raise SystemExit(f"expected 9 R1C1 movement fits, found {len(fits)}")
    return fits


def movement_station(movement: str) -> int:
    parts = movement.split(":")
    if len(parts) < 2:
        raise ValueError(f"cannot parse movement station: {movement}")
    return int(parts[1])


def effective_cdf(bounds: np.ndarray, sds: np.ndarray, median: float, sigma: float) -> np.ndarray:
    x = bounds[:, None] + sds[:, None] * GH_Z[None, :]
    z = np.full_like(x, -np.inf, dtype=float)
    pos = x > 0
    z[pos] = (np.log(x[pos]) - math.log(median)) / sigma
    return np.sum(ndtr(z) * GH_WN[None, :], axis=1)


def k1_boarded_probability(
    movement: str,
    lower_root: str,
    upper_root: str,
    by_key: dict[tuple[str, int], dict[str, Any]],
    root_meta: dict[str, tuple[str, str, str]],
    service_seq: dict[tuple[str, str, int], list[dict[str, Any]]],
    fits: dict[str, Any],
) -> tuple[float | None, str]:
    try:
        station = movement_station(movement)
    except Exception:
        return None, "MOVEMENT_PARSE_FAILURE"
    lo = by_key.get((lower_root, station))
    up = by_key.get((upper_root, station))
    if lo is None or up is None:
        return None, "R1B_REALIZED_EVENT_MAPPING_FAILURE"
    arr = float(lo["realized_time_mean_s"])
    arr_sd = float(lo["realized_time_sd_laplace_s"])
    boarded = float(up["realized_time_mean_s"])
    boarded_pid = str(up["physical_service_id"])
    if boarded <= arr:
        return None, "NONPOSITIVE_REALIZED_CONNECTION"
    if upper_root not in root_meta:
        return None, "UPPER_ROOT_META_FAILURE"
    path, direction, _ = root_meta[upper_root]
    seq = service_seq.get((path, direction, station))
    if not seq:
        return None, "DOWNSTREAM_SERVICE_SEQUENCE_MISSING"
    gaps, sds = [], []
    found = False
    for e in seq:
        dep = float(e["realized_time_mean_s"])
        if dep <= arr:
            continue
        if dep > boarded + 1e-6:
            break
        gaps.append(dep - arr)
        sds.append(math.sqrt(arr_sd * arr_sd + float(e["realized_time_sd_laplace_s"]) ** 2))
        if str(e["physical_service_id"]) == boarded_pid:
            found = True
            break
    if not found or not gaps:
        return None, "OBSERVED_BOARDED_SERVICE_NOT_IN_SEQUENCE"
    fit = fits.get(movement)
    if fit is None:
        return None, "MOVEMENT_K1_FIT_MISSING"
    b = int(math.floor(arr / TIME_BIN_S))
    pars = fit["contexts"].get(b, fit["global"])
    median = float(pars["median_s"])
    sigma = float(pars["log_sigma"])
    skip = float(pars["skip_or_left_behind_probability"])
    cdf = effective_cdf(np.asarray(gaps, float), np.asarray(sds, float), median, sigma)
    prev = np.concatenate(([0.0], cdf[:-1]))
    delta = np.maximum(0.0, cdf - prev)
    m = len(gaps)
    powers = skip ** np.arange(m - 1, -1, -1, dtype=float)
    terms = delta * powers * (1.0 - skip)
    prob = float(np.sum(terms))
    if not math.isfinite(prob) or prob <= EPS:
        return None, "K1_ZERO_OR_NONFINITE_PROBABILITY"
    return prob, "OK"


def old_transfer_log_factor(
    movement: str,
    lower_root: str,
    upper_root: str,
    old_events: dict[tuple[str, int], tuple[float, float]],
    old_median: float,
    old_sigma: float,
) -> tuple[float | None, str]:
    try:
        station = movement_station(movement)
    except Exception:
        return None, "MOVEMENT_PARSE_FAILURE"
    lo = old_events.get((lower_root, station))
    up = old_events.get((upper_root, station))
    if lo is None or up is None:
        return None, "OLD_EVENT_MAPPING_FAILURE"
    center = float(up[0] - lo[0])
    sd = math.sqrt(float(lo[1]) ** 2 + float(up[1]) ** 2)
    ll = old_log_interval_density(center, sd, old_median, old_sigma)
    if not math.isfinite(ll):
        return None, "OLD_TRANSFER_FACTOR_NONFINITE"
    return ll, "OK"


def edge_log_ratio(
    row: dict[str, Any], old_events, old_median, old_sigma, by_key, root_meta, service_seq, fits
) -> tuple[float, bool, str]:
    if str(row["descendant_state_type"]) == "UNRESOLVED" or int(row["transfer_count"]) <= 0:
        return 0.0, True, "NO_TRANSFER_FACTOR"
    roots = [x for x in str(row["root_chain"]).split(">") if x]
    movements = [x for x in str(row["transfer_chain"]).split(">") if x]
    if len(roots) != len(movements) + 1:
        return 0.0, False, "ROOT_TRANSFER_CHAIN_ARITY_MISMATCH"
    total = 0.0
    for i, movement in enumerate(movements):
        lower_root, upper_root = roots[i], roots[i + 1]
        new_p, new_reason = k1_boarded_probability(movement, lower_root, upper_root, by_key, root_meta, service_seq, fits)
        if new_p is None:
            return 0.0, False, "NEW:" + new_reason
        old_ll, old_reason = old_transfer_log_factor(movement, lower_root, upper_root, old_events, old_median, old_sigma)
        if old_ll is None:
            return 0.0, False, "OLD:" + old_reason
        total += math.log(max(new_p, EPS)) - old_ll
    return total, True, "OK"


def entropy(ps: list[float]) -> float:
    return -sum(p * math.log(max(p, EPS)) for p in ps)


def process_shard(a) -> dict[str, Any]:
    old_events = load_old_roots(a.old_roots)
    old_median, old_sigma, old_evidence = load_old_transfer_kernel(a.e0_global)
    by_key, root_meta, service_seq = load_realized_schedule(a.schedule)
    fits = load_k1_fits(a.k1_fit)
    schedule_hash = sha256_file(a.schedule)

    pf = pq.ParquetFile(a.edges)
    in_schema = pf.schema_arrow
    extra_fields = [
        pa.field("posterior_probability_r1b1", pa.float64()),
        pa.field("lineage_mass_r1b1", pa.float64()),
        pa.field("transfer_log_likelihood_ratio", pa.float64()),
        pa.field("r1c2_transfer_update_supported", pa.bool_()),
        pa.field("r1c2_transfer_update_reason", pa.string()),
    ]
    out_schema = in_schema
    for f in extra_fields:
        if out_schema.get_field_index(f.name) < 0:
            out_schema = out_schema.append(f)
    a.out_edges.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(a.out_edges, out_schema, compression="zstd")
    out_buf: list[dict[str, Any]] = []

    passenger_mass = resolved_mass = unresolved_mass = 0.0
    transfer_sensitive_mass = 0.0
    tv_mass = 0.0
    top_chain_changed_mass = 0.0
    top_route_changed_mass = 0.0
    entropy_before_num = entropy_after_num = 0.0
    normalization_error_max = 0.0
    cohort_count = resolved_cohorts = 0
    unsupported_candidate_mass = 0.0
    unsupported_reasons: dict[str, float] = defaultdict(float)
    transfer_before: dict[str, float] = defaultdict(float)
    transfer_after: dict[str, float] = defaultdict(float)
    root_before: dict[str, float] = defaultdict(float)
    root_after: dict[str, float] = defaultdict(float)

    def flush():
        nonlocal out_buf
        if out_buf:
            writer.write_table(pa.Table.from_pylist(out_buf, schema=out_schema))
            out_buf = []

    def finish_cohort(rows: list[dict[str, Any]]):
        nonlocal passenger_mass, resolved_mass, unresolved_mass, transfer_sensitive_mass
        nonlocal tv_mass, top_chain_changed_mass, top_route_changed_mass
        nonlocal entropy_before_num, entropy_after_num, normalization_error_max
        nonlocal cohort_count, resolved_cohorts, unsupported_candidate_mass, out_buf
        if not rows:
            return
        cohort_count += 1
        mass = float(rows[0]["passenger_mass"])
        passenger_mass += mass
        if any(str(r["descendant_state_type"]) == "UNRESOLVED" for r in rows):
            if len(rows) != 1:
                raise SystemExit(f"unresolved cohort has {len(rows)} edges: {rows[0]['cohort_id']}")
            r = dict(rows[0])
            unresolved_mass += mass
            r["posterior_probability_r1b1"] = float(r["posterior_probability"])
            r["lineage_mass_r1b1"] = float(r["lineage_mass"])
            r["transfer_log_likelihood_ratio"] = 0.0
            r["r1c2_transfer_update_supported"] = False
            r["r1c2_transfer_update_reason"] = "UNRESOLVED_COHORT_PRESERVED"
            out_buf.append(r)
            if len(out_buf) >= 50000:
                flush()
            return

        resolved_cohorts += 1
        resolved_mass += mass
        old_ps = [float(r["posterior_probability"]) for r in rows]
        s0 = sum(old_ps)
        if s0 <= 0:
            raise SystemExit(f"resolved cohort has zero posterior mass: {rows[0]['cohort_id']}")
        old_ps = [p / s0 for p in old_ps]
        logw: list[float] = []
        ratios: list[float] = []
        supported: list[bool] = []
        reasons: list[str] = []
        transfer_sensitive = any(int(r["transfer_count"]) > 0 for r in rows)
        if transfer_sensitive:
            transfer_sensitive_mass += mass
        for r, p in zip(rows, old_ps):
            ratio, ok, reason = edge_log_ratio(r, old_events, old_median, old_sigma, by_key, root_meta, service_seq, fits)
            ratios.append(ratio)
            supported.append(ok)
            reasons.append(reason)
            if int(r["transfer_count"]) > 0 and not ok:
                unsupported_candidate_mass += mass * p
                unsupported_reasons[reason] += mass * p
            logw.append(math.log(max(p, EPS)) + (ratio if ok else 0.0))
        z = logsumexp(logw)
        if not math.isfinite(z):
            raise SystemExit(f"R1C2 normalization failed: {rows[0]['cohort_id']}")
        new_ps = [math.exp(x - z) for x in logw]
        normalization_error_max = max(normalization_error_max, abs(sum(new_ps) - 1.0))
        tv = 0.5 * sum(abs(a - b) for a, b in zip(old_ps, new_ps))
        tv_mass += mass * tv
        entropy_before_num += mass * entropy(old_ps)
        entropy_after_num += mass * entropy(new_ps)
        old_top = max(range(len(rows)), key=lambda i: old_ps[i])
        new_top = max(range(len(rows)), key=lambda i: new_ps[i])
        if str(rows[old_top]["descendant_state_id"]) != str(rows[new_top]["descendant_state_id"]):
            top_chain_changed_mass += mass
        if int(rows[old_top]["route_rank"]) != int(rows[new_top]["route_rank"]):
            top_route_changed_mass += mass

        for r, p0, p1, ratio, ok, reason in zip(rows, old_ps, new_ps, ratios, supported, reasons):
            old_lm, new_lm = mass * p0, mass * p1
            roots = [x for x in str(r["root_chain"]).split(">") if x]
            movements = [x for x in str(r["transfer_chain"]).split(">") if x]
            for rid in roots:
                root_before[rid] += old_lm
                root_after[rid] += new_lm
            for movement in movements:
                transfer_before[movement] += old_lm
                transfer_after[movement] += new_lm
            rr = dict(r)
            rr["posterior_probability_r1b1"] = float(r["posterior_probability"])
            rr["lineage_mass_r1b1"] = float(r["lineage_mass"])
            rr["posterior_probability"] = p1
            rr["lineage_mass"] = new_lm
            rr["transfer_log_likelihood_ratio"] = ratio if ok else 0.0
            rr["r1c2_transfer_update_supported"] = bool(ok and int(r["transfer_count"]) > 0)
            rr["r1c2_transfer_update_reason"] = reason
            out_buf.append(rr)
            if len(out_buf) >= 50000:
                flush()

    pending_id = None
    pending: list[dict[str, Any]] = []
    try:
        for batch in pf.iter_batches(batch_size=50000):
            for row in batch.to_pylist():
                cid = str(row["cohort_id"])
                if pending_id is None:
                    pending_id = cid
                if cid != pending_id:
                    finish_cohort(pending)
                    pending = []
                    pending_id = cid
                pending.append(row)
        finish_cohort(pending)
        flush()
    finally:
        writer.close()

    conservation_error = abs((resolved_mass + unresolved_mass) - passenger_mass)
    shift_share_resolved = tv_mass / resolved_mass if resolved_mass else 0.0
    shift_share_transfer_sensitive = tv_mass / transfer_sensitive_mass if transfer_sensitive_mass else 0.0
    result = {
        "schema": SCHEMA,
        "status": "QUALIFIED_R1C2_CHAIN_GENEALOGY_REWEIGHT_SHARD" if conservation_error <= 1e-6 and normalization_error_max <= 1e-9 else "FAILED_R1C2_SHARD",
        "service_date": "2019-01-04",
        "shard_index": int(a.shard_index),
        "cohort_count": int(cohort_count),
        "resolved_cohort_count": int(resolved_cohorts),
        "passenger_mass": passenger_mass,
        "resolved_genealogy_mass": resolved_mass,
        "unresolved_genealogy_mass": unresolved_mass,
        "transfer_sensitive_passenger_mass": transfer_sensitive_mass,
        "posterior_total_variation_mass": tv_mass,
        "posterior_total_variation_share_of_resolved": shift_share_resolved,
        "posterior_total_variation_share_of_transfer_sensitive": shift_share_transfer_sensitive,
        "top_descendant_changed_passenger_mass": top_chain_changed_mass,
        "top_route_changed_passenger_mass": top_route_changed_mass,
        "weighted_mean_lineage_entropy_before_nats": entropy_before_num / resolved_mass if resolved_mass else None,
        "weighted_mean_lineage_entropy_after_nats": entropy_after_num / resolved_mass if resolved_mass else None,
        "unsupported_transfer_candidate_lineage_mass": unsupported_candidate_mass,
        "unsupported_transfer_candidate_reasons": dict(sorted(unsupported_reasons.items())),
        "posterior_normalization_max_abs_error": normalization_error_max,
        "mass_conservation_abs_error": conservation_error,
        "frozen_r1b_realized_schedule_sha256": schedule_hash,
        "r1b_timetable_modified": False,
        "old_transfer_factor": {"median_s": old_median, "log_sigma": old_sigma, "evidence_class": old_evidence},
        "movement_lineage_mass_before": dict(sorted(transfer_before.items())),
        "movement_lineage_mass_after": dict(sorted(transfer_after.items())),
        "root_lineage_mass_before": dict(sorted(root_before.items())),
        "root_lineage_mass_after": dict(sorted(root_after.items())),
        "scientific_semantics": {
            "r1c1_transfer_posterior_enters_passenger_chain_likelihood": True,
            "replaces_old_connection_slack_transfer_factor_by_first_boardable_generative_factor": True,
            "resolved_unresolved_mass_boundary_preserved": True,
            "posterior_reweight_only_within_existing_resolved_candidate_chains": True,
            "route_boarding_genealogy_probabilities_allowed_to_redistribute": True,
            "r1b_realized_timetable_frozen": True,
            "planned_absolute_timetable_used": False,
            "scientific_self_iteration": False,
        },
    }
    if result["status"].startswith("FAILED"):
        raise SystemExit(json.dumps(result, ensure_ascii=False))
    dump(a.out_summary, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def merge_global(a) -> dict[str, Any]:
    ss = [json.loads(p.read_text(encoding="utf-8")) for p in a.summary]
    if len(ss) != 8:
        raise SystemExit(f"expected 8 R1C2 shard summaries, found {len(ss)}")
    if not all(x.get("status") == "QUALIFIED_R1C2_CHAIN_GENEALOGY_REWEIGHT_SHARD" for x in ss):
        raise SystemExit("not all R1C2 shards qualified")
    hashes = {x["frozen_r1b_realized_schedule_sha256"] for x in ss}
    if len(hashes) != 1:
        raise SystemExit("R1B realized schedule hash differs across shards")
    passenger = sum(float(x["passenger_mass"]) for x in ss)
    resolved = sum(float(x["resolved_genealogy_mass"]) for x in ss)
    unresolved = sum(float(x["unresolved_genealogy_mass"]) for x in ss)
    sensitive = sum(float(x["transfer_sensitive_passenger_mass"]) for x in ss)
    tv = sum(float(x["posterior_total_variation_mass"]) for x in ss)
    top_chain = sum(float(x["top_descendant_changed_passenger_mass"]) for x in ss)
    top_route = sum(float(x["top_route_changed_passenger_mass"]) for x in ss)
    unsupported = sum(float(x["unsupported_transfer_candidate_lineage_mass"]) for x in ss)
    e0 = sum(float(x["weighted_mean_lineage_entropy_before_nats"] or 0.0) * float(x["resolved_genealogy_mass"]) for x in ss)
    e1 = sum(float(x["weighted_mean_lineage_entropy_after_nats"] or 0.0) * float(x["resolved_genealogy_mass"]) for x in ss)
    normerr = max(float(x["posterior_normalization_max_abs_error"]) for x in ss)
    conserr = abs((resolved + unresolved) - passenger)
    before: dict[str, float] = defaultdict(float)
    after: dict[str, float] = defaultdict(float)
    reasons: dict[str, float] = defaultdict(float)
    for x in ss:
        for k, v in x["movement_lineage_mass_before"].items(): before[k] += float(v)
        for k, v in x["movement_lineage_mass_after"].items(): after[k] += float(v)
        for k, v in x["unsupported_transfer_candidate_reasons"].items(): reasons[k] += float(v)
    movement_shift = {}
    for k in sorted(set(before) | set(after)):
        b, c = before.get(k, 0.0), after.get(k, 0.0)
        movement_shift[k] = {"before_lineage_mass": b, "after_lineage_mass": c, "delta_lineage_mass": c - b, "relative_delta": (c-b)/b if b else None}
    gates = {
        "all_eight_exact_partitions_qualified": len(ss) == 8,
        "full_day_passenger_mass_conserved": conserr <= 1e-6,
        "cohort_posteriors_normalized": normerr <= 1e-9,
        "r1b_realized_timetable_identical_and_frozen": len(hashes) == 1 and all(x["r1b_timetable_modified"] is False for x in ss),
        "transfer_posterior_changes_chain_genealogy_probability_mass": tv > 1e-6,
        "transfer_sensitive_passenger_mass_positive": sensitive > 0,
        "unresolved_mass_remains_explicit": unresolved > 0,
    }
    result = {
        "schema": SCHEMA,
        "status": "QUALIFIED_R1C2_FULL_DAY_CHAIN_GENEALOGY_REFINEMENT" if all(gates.values()) else "FAILED_R1C2_FULL_DAY_QUALIFICATION",
        "service_date": "2019-01-04",
        "scope": "FULL_SERVICE_DAY_FULL_NETWORK_ALL_QUALIFIED_PASSENGER_DOMAIN_EIGHT_EXACT_PARTITIONS",
        "passenger_mass": passenger,
        "resolved_genealogy_mass": resolved,
        "unresolved_genealogy_mass": unresolved,
        "transfer_sensitive_passenger_mass": sensitive,
        "posterior_total_variation_mass": tv,
        "posterior_total_variation_share_of_resolved": tv / resolved if resolved else None,
        "posterior_total_variation_share_of_transfer_sensitive": tv / sensitive if sensitive else None,
        "top_descendant_changed_passenger_mass": top_chain,
        "top_route_changed_passenger_mass": top_route,
        "weighted_mean_lineage_entropy_before_nats": e0 / resolved if resolved else None,
        "weighted_mean_lineage_entropy_after_nats": e1 / resolved if resolved else None,
        "unsupported_transfer_candidate_lineage_mass": unsupported,
        "unsupported_transfer_candidate_share_of_resolved": unsupported / resolved if resolved else None,
        "unsupported_transfer_candidate_reasons": dict(sorted(reasons.items())),
        "mass_conservation_abs_error": conserr,
        "posterior_normalization_max_abs_error": normerr,
        "frozen_r1b_realized_schedule_sha256": next(iter(hashes)),
        "movement_lineage_mass_shift": movement_shift,
        "qualification_gates": gates,
        "scientific_semantics": {
            "r1c1_is_stage_1_transfer_time_inference": True,
            "r1c2_is_stage_2_passenger_chain_and_genealogy_refinement": True,
            "r1c_scientifically_closed_only_if_this_product_qualifies": True,
            "r1b_realized_timetable_frozen": True,
            "no_feedback_to_r1b": True,
            "no_scientific_self_iteration": True,
        },
    }
    dump(a.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"].startswith("FAILED"):
        raise SystemExit("R1C2 full-day qualification failed")
    return result


def parser():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("process-shard")
    s.add_argument("--edges", type=Path, required=True)
    s.add_argument("--old-roots", type=Path, required=True)
    s.add_argument("--e0-global", type=Path, required=True)
    s.add_argument("--schedule", type=Path, required=True)
    s.add_argument("--k1-fit", type=Path, action="append", required=True)
    s.add_argument("--shard-index", type=int, required=True)
    s.add_argument("--out-edges", type=Path, required=True)
    s.add_argument("--out-summary", type=Path, required=True)
    m = sp.add_parser("merge-global")
    m.add_argument("--summary", type=Path, action="append", required=True)
    m.add_argument("--out", type=Path, required=True)
    return p


def main():
    a = parser().parse_args()
    if a.cmd == "process-shard":
        process_shard(a)
    elif a.cmd == "merge-global":
        merge_global(a)


if __name__ == "__main__":
    main()
