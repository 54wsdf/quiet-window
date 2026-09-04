import argparse
import csv
import gzip
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r0_cached_full_network_support_rescan_20260904 as support
import scripts.mppd_r0_full_network_factor_engine_20260904 as r0
import scripts.mppd_g2v2_uncertain_service_full_network_posterior_20260904 as g2
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch

EPS = 1e-14


def logsumexp(vals):
    vals = [float(x) for x in vals if math.isfinite(float(x))]
    if not vals:
        return -math.inf
    m = max(vals)
    return m + math.log(sum(math.exp(x - m) for x in vals))


def kernel_from_median_sigma(median, sigma, evidence_class):
    return {
        "family": "LOGNORMAL_MIXTURE",
        "components": [{
            "weight": 1.0,
            "median_sec": float(median),
            "sigma": float(sigma),
            "evidence_class": evidence_class,
        }],
        "mixture_capable": True,
        "component_count": 1,
    }


def kernel_cdf(x, kernel):
    if x <= 0:
        return 0.0
    return sum(
        float(c["weight"]) * g2.lognorm_cdf(x, float(c["median_sec"]), float(c["sigma"]))
        for c in kernel["components"]
    )


def kernel_pdf(x, kernel):
    if x <= 0:
        return 0.0
    return sum(
        float(c["weight"]) * g2.lognorm_pdf(x, float(c["median_sec"]), float(c["sigma"]))
        for c in kernel["components"]
    )


def expected_kernel_cdf_normal_difference(mean, sd, kernel):
    if sd <= 1e-9:
        return kernel_cdf(mean, kernel)
    return sum(w * kernel_cdf(mean + z * sd, kernel) for z, w in g2.QUAD)


def expected_kernel_pdf_normal_difference(mean, sd, kernel):
    if sd <= 1e-9:
        return kernel_pdf(mean, kernel)
    return sum(w * kernel_pdf(mean + z * sd, kernel) for z, w in g2.QUAD)


def interval_logprob_kernel(lower_dep, lower_sd, upper_dep, upper_sd, ready, ready_sd, kernel):
    um = (upper_dep - ready).total_seconds()
    usd = math.sqrt(float(upper_sd) ** 2 + float(ready_sd) ** 2)
    Fu = expected_kernel_cdf_normal_difference(um, usd, kernel)
    if lower_dep is None:
        Fl = 0.0
    else:
        lm = (lower_dep - ready).total_seconds()
        lsd = math.sqrt(float(lower_sd) ** 2 + float(ready_sd) ** 2)
        Fl = expected_kernel_cdf_normal_difference(lm, lsd, kernel)
    return math.log(max(EPS, Fu - Fl))


def egress_logdensity_kernel(exit_time, arr, arr_sd, kernel):
    mean = (exit_time - arr).total_seconds()
    val = expected_kernel_pdf_normal_difference(mean, float(arr_sd), kernel)
    return math.log(max(EPS, val))


def root_key(line, root):
    return f"{line}||{root}"


def build_joint_rides(roots):
    cache = {}

    def rides(line, origin, destination):
        key = (line, origin, destination)
        if key in cache:
            return cache[key]
        out = []
        for (ln, root), variants in roots.items():
            if ln != line:
                continue
            choices = []
            for st in variants:
                if origin not in st["events"] or destination not in st["events"]:
                    continue
                eo = st["events"][origin]
                ed = st["events"][destination]
                if eo["departure"] >= ed["arrival"]:
                    continue
                choices.append((
                    g2.evidence_rank(st["evidence_class"]),
                    eo["sd"] + ed["sd"],
                    st["id"],
                    eo,
                    ed,
                    st["evidence_class"],
                ))
            if not choices:
                continue
            _, _, sid, eo, ed, evc = min(choices, key=lambda x: (x[0], x[1], x[2]))
            out.append({
                "line": line,
                "dep": eo["departure"],
                "arr": ed["arrival"],
                "dep_sd": float(eo["sd"]),
                "arr_sd": float(ed["sd"]),
                "root": root,
                "root_key": root_key(line, root),
                "variant_id": sid,
                "evidence_class": evc,
            })
        out.sort(key=lambda x: (x["dep"], x["root"]))
        cache[key] = out
        return out

    return rides, cache


def kernel_for(kind, movement, kernels):
    if kind == "ACCESS":
        return kernels["access"]
    if kind == "EGRESS":
        return kernels["egress"]
    return kernels["transfer_by_movement"].get(movement, kernels["transfer_global"])


def skip_intervals(rides, selected_index, max_skip):
    opts = []
    for n_skip in range(max_skip + 1):
        upper_i = selected_index - n_skip
        if upper_i < 0:
            break
        lower_i = upper_i - 1
        upper = rides[upper_i]
        lower = rides[lower_i] if lower_i >= 0 else None
        opts.append((n_skip, lower, upper))
    return opts


def path_is_non_simple(cand):
    path = cand.get("path") or []
    return len(path) != len(set(path))


def route_beam_joint(cand, meta, rides_fn, tin, tout, beam, kernels, max_skip):
    legs = r0.path_legs(cand["path"], meta)
    states = [{
        "ready": tin,
        "ready_sd": 0.0,
        "ready_root": None,
        "logp": 0.0,
        "chain": [],
        "factors": [],
    }]
    for leg_index, (line, origin, destination) in enumerate(legs):
        if origin == destination:
            continue
        rides = rides_fn(line, origin, destination)
        if not rides:
            return [], {
                "reason": "MISSING_SERVICE_SEGMENT",
                "line": line,
                "origin": origin,
                "destination": destination,
            }
        nxt = []
        for st in states:
            ready = st["ready"]
            ready_sd = st["ready_sd"]
            ready_root = st["ready_root"]
            prev_leg = st["chain"][-1] if st["chain"] else None
            movement = None if prev_leg is None else f"{prev_leg['line']}:{prev_leg['destination']}->{line}:{origin}"
            kind = "ACCESS" if prev_leg is None else "TRANSFER"
            kern = kernel_for(kind, movement, kernels)
            for i, rd in enumerate(rides):
                if rd["arr"] > tout and (rd["arr"] - tout).total_seconds() > 3 * max(1.0, rd["arr_sd"]):
                    continue
                opts = skip_intervals(rides, i, max_skip)
                if not opts:
                    continue
                log_uniform = -math.log(len(opts))
                for n_skip, lower, upper in opts:
                    lp = interval_logprob_kernel(
                        lower["dep"] if lower else None,
                        lower["dep_sd"] if lower else 0.0,
                        upper["dep"],
                        upper["dep_sd"],
                        ready,
                        ready_sd,
                        kern,
                    ) + log_uniform
                    if not math.isfinite(lp):
                        continue
                    factor = {
                        "type": "INTERVAL",
                        "kind": kind,
                        "movement": movement,
                        "ready": ready,
                        "ready_sd": ready_sd,
                        "ready_root": ready_root,
                        "lower": lower["dep"] if lower else None,
                        "lower_sd": lower["dep_sd"] if lower else 0.0,
                        "lower_root": lower["root_key"] if lower else None,
                        "upper": upper["dep"],
                        "upper_sd": upper["dep_sd"],
                        "upper_root": upper["root_key"],
                        "selected_root": rd["root_key"],
                        "n_skip": n_skip,
                    }
                    leg = {
                        "line": line,
                        "origin": origin,
                        "destination": destination,
                        "root": rd["root"],
                        "root_key": rd["root_key"],
                        "variant_id": rd["variant_id"],
                        "dep": rd["dep"],
                        "arr": rd["arr"],
                        "dep_sd": rd["dep_sd"],
                        "arr_sd": rd["arr_sd"],
                        "evidence_class": rd["evidence_class"],
                        "n_skip": n_skip,
                        "movement": movement,
                    }
                    nxt.append({
                        "ready": rd["arr"],
                        "ready_sd": rd["arr_sd"],
                        "ready_root": rd["root_key"],
                        "logp": st["logp"] + lp,
                        "chain": st["chain"] + [leg],
                        "factors": st["factors"] + [factor],
                    })
        if not nxt:
            return [], {"reason": "TIME_INCOMPATIBLE_CHAIN"}
        nxt.sort(key=lambda x: (x["logp"], -x["ready"].timestamp()), reverse=True)
        states = nxt[:beam]

    final = []
    egress_kernel = kernels["egress"]
    for st in states:
        if not st["chain"]:
            continue
        last = st["chain"][-1]
        le = egress_logdensity_kernel(tout, last["arr"], last["arr_sd"], egress_kernel)
        if not math.isfinite(le):
            continue
        ef = {
            "type": "EGRESS",
            "exit_time": tout,
            "arr": last["arr"],
            "arr_sd": last["arr_sd"],
            "arr_root": last["root_key"],
        }
        final.append({
            "logp": st["logp"] + le,
            "chain": st["chain"],
            "factors": st["factors"] + [ef],
        })
    final.sort(key=lambda x: x["logp"], reverse=True)
    return final[:beam], None if final else {"reason": "EGRESS_INCOMPATIBLE"}


def deterministic_keep(origin, destination, tin, tout, sample_mod):
    if sample_mod <= 1:
        return True
    key = f"{origin}|{destination}|{tin.isoformat()}|{tout.isoformat()}".encode()
    return int(hashlib.sha1(key).hexdigest()[:16], 16) % sample_mod == 0


def load_sampled_cohorts(path, sample_mod, max_cohorts):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            tin = datetime.fromisoformat(r["entry_time"])
            tout = datetime.fromisoformat(r["exit_time"])
            oc = r["origin_code"]
            dc = r["destination_code"]
            if not deterministic_keep(oc, dc, tin, tout, sample_mod):
                continue
            rows.append((oc, dc, tin, tout, int(r["passenger_mass"])))
            if max_cohorts and len(rows) >= max_cohorts:
                break
    return rows


def structural_pressure_weights(cands):
    if not cands:
        return []
    min_cost = min(float(c.get("base_cost", 0.0)) for c in cands)
    raw = [math.exp(-0.30 * max(0.0, float(c.get("base_cost", 0.0)) - min_cost)) for c in cands]
    z = sum(raw) or 1.0
    return [x / z for x in raw]


def posterior_pass(cohorts, routes, meta, roots, kernels, beam, max_skip, retain_details=True):
    rides_fn, _ = build_joint_rides(roots)
    result = {
        "processed_mass": 0.0,
        "finite_mass": 0.0,
        "status_mass": Counter(),
        "failure_mass": Counter(),
        "route_entropy_weighted": 0.0,
        "skip_mass": Counter(),
        "non_simple_mass": 0.0,
        "first_boarding_entropy_contraction_weighted": 0.0,
        "access_tail_mass_gt_15min": 0.0,
        "egress_tail_mass_gt_15min": 0.0,
        "missing_service_pressure": Counter(),
        "cohorts": {},
        "factors": [],
        "root_usage": Counter(),
    }
    for oc, dc, tin, tout, mass in cohorts:
        result["processed_mass"] += mass
        cands = routes.get((oc, dc), [])
        cohort_id = f"{oc}|{dc}|{tin.isoformat()}|{tout.isoformat()}"
        if not cands:
            result["status_mass"]["NO_STRUCTURAL_ROUTE"] += mass
            if retain_details:
                result["cohorts"][cohort_id] = {"status": "NO_STRUCTURAL_ROUTE", "mass": mass}
            continue

        min_cost = min(float(c.get("base_cost", 0.0)) for c in cands)
        hypotheses = []
        fail_details = []
        route_scores = []
        for ci, cand in enumerate(cands):
            chains, failure = route_beam_joint(cand, meta, rides_fn, tin, tout, beam, kernels, max_skip)
            fail_details.append(failure)
            if not chains:
                route_scores.append(-math.inf)
                continue
            route_lp = logsumexp([x["logp"] for x in chains]) - 0.30 * max(
                0.0, float(cand.get("base_cost", 0.0)) - min_cost
            )
            route_scores.append(route_lp)
            for ch in chains:
                hypotheses.append({
                    "candidate_index": ci,
                    "line_sequence": cand.get("line_sequence", ""),
                    "transfer_count": int(cand.get("transfer_count", 0)),
                    "non_simple": path_is_non_simple(cand),
                    "logp": ch["logp"] - 0.30 * max(0.0, float(cand.get("base_cost", 0.0)) - min_cost),
                    "chain": ch["chain"],
                    "factors": ch["factors"],
                })

        finite_routes = [(i, lp) for i, lp in enumerate(route_scores) if math.isfinite(lp)]
        if not hypotheses or not finite_routes:
            reason_counts = Counter((f or {}).get("reason", "UNKNOWN") for f in fail_details)
            reason = reason_counts.most_common(1)[0][0]
            result["failure_mass"][reason] += mass
            result["status_mass"]["NO_FINITE_POSTERIOR"] += mass
            if reason == "MISSING_SERVICE_SEGMENT":
                sw = structural_pressure_weights(cands)
                for ci, f in enumerate(fail_details):
                    if f and f.get("reason") == "MISSING_SERVICE_SEGMENT":
                        key = f"{f.get('line')}|{f.get('origin')}|{f.get('destination')}"
                        result["missing_service_pressure"][key] += mass * sw[ci]
            if retain_details:
                result["cohorts"][cohort_id] = {"status": reason, "mass": mass}
            continue

        z_h = logsumexp([h["logp"] for h in hypotheses])
        for h in hypotheses:
            h["probability"] = math.exp(h["logp"] - z_h)

        route_prob = defaultdict(float)
        first_root_prob = defaultdict(float)
        route_first = defaultdict(lambda: defaultdict(float))
        for h in hypotheses:
            p = h["probability"]
            route_prob[h["line_sequence"]] += p
            first_root = h["chain"][0]["root_key"] if h["chain"] else "NONE"
            first_root_prob[first_root] += p
            route_first[first_root][h["line_sequence"]] += p
            result["non_simple_mass"] += mass * p * (1.0 if h["non_simple"] else 0.0)
            for leg in h["chain"]:
                result["skip_mass"][int(leg["n_skip"])] += mass * p
                result["root_usage"][leg["root_key"]] += mass * p
            if h["factors"]:
                af = next((f for f in h["factors"] if f["type"] == "INTERVAL" and f["kind"] == "ACCESS"), None)
                if af:
                    access_upper = (af["upper"] - af["ready"]).total_seconds()
                    if access_upper > 900:
                        result["access_tail_mass_gt_15min"] += mass * p
                ef = next((f for f in reversed(h["factors"]) if f["type"] == "EGRESS"), None)
                if ef and (ef["exit_time"] - ef["arr"]).total_seconds() > 900:
                    result["egress_tail_mass_gt_15min"] += mass * p
            if retain_details:
                for factor in h["factors"]:
                    ff = dict(factor)
                    ff["weight"] = mass * p
                    result["factors"].append(ff)

        probs = list(route_prob.values())
        H_before = -sum(p * math.log(max(p, EPS)) for p in probs if p > 0)
        H_after = 0.0
        for fr, frp in first_root_prob.items():
            if frp <= 0:
                continue
            conditional = [v / frp for v in route_first[fr].values()]
            H_after += frp * (-sum(p * math.log(max(p, EPS)) for p in conditional if p > 0))
        contraction = max(0.0, H_before - H_after)

        result["finite_mass"] += mass
        result["status_mass"]["FINITE_POSTERIOR"] += mass
        result["route_entropy_weighted"] += mass * H_before
        result["first_boarding_entropy_contraction_weighted"] += mass * contraction

        if retain_details:
            top_route = max(route_prob.items(), key=lambda x: x[1])
            top_first = max(first_root_prob.items(), key=lambda x: x[1])
            result["cohorts"][cohort_id] = {
                "status": "FINITE_POSTERIOR",
                "mass": mass,
                "route_probs": dict(route_prob),
                "route_entropy": H_before,
                "top_route": top_route[0],
                "top_route_prob": top_route[1],
                "first_root_probs": dict(first_root_prob),
                "top_first_root": top_first[0],
                "top_first_prob": top_first[1],
            }

    return result


def weighted_interval_fit(factors, initial_kernel, min_weight=20.0):
    clean = []
    total_weight = 0.0
    for f in factors:
        w = float(f.get("weight", 0.0))
        if w <= 0 or f.get("type") != "INTERVAL":
            continue
        lower = 0.0 if f["lower"] is None else (f["lower"] - f["ready"]).total_seconds()
        upper = (f["upper"] - f["ready"]).total_seconds()
        if upper <= max(0.0, lower) + 1e-6:
            continue
        clean.append((max(0.0, lower), upper, w))
        total_weight += w
    if total_weight < min_weight:
        return initial_kernel, {"fitted": False, "effective_weight": total_weight}

    c0 = initial_kernel["components"][0]
    x0 = np.array([math.log(max(1.0, float(c0["median_sec"]))), math.log(max(0.15, float(c0["sigma"])))])
    bounds = [(math.log(10.0), math.log(1800.0)), (math.log(0.12), math.log(2.5))]

    def obj(x):
        med = math.exp(float(x[0]))
        sig = math.exp(float(x[1]))
        val = 0.0
        for lower, upper, w in clean:
            Fu = g2.lognorm_cdf(upper, med, sig)
            Fl = g2.lognorm_cdf(lower, med, sig) if lower > 0 else 0.0
            val -= w * math.log(max(EPS, Fu - Fl))
        return val / total_weight

    fit = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
    med = math.exp(float(fit.x[0]))
    sig = math.exp(float(fit.x[1]))
    out = kernel_from_median_sigma(med, sig, "R1B_POSTERIOR_WEIGHTED_INTERVAL_FIT")
    return out, {
        "fitted": bool(fit.success),
        "effective_weight": total_weight,
        "median_sec": med,
        "sigma": sig,
        "objective_per_mass": float(fit.fun),
    }


def weighted_egress_fit(factors, initial_kernel, min_weight=20.0):
    vals = []
    total_weight = 0.0
    for f in factors:
        if f.get("type") != "EGRESS":
            continue
        x = (f["exit_time"] - f["arr"]).total_seconds()
        w = float(f.get("weight", 0.0))
        if 0 < x <= 3600 and w > 0:
            vals.append((x, w))
            total_weight += w
    if total_weight < min_weight:
        return initial_kernel, {"fitted": False, "effective_weight": total_weight}
    logs = np.array([math.log(x) for x, _ in vals], dtype=float)
    weights = np.array([w for _, w in vals], dtype=float)
    mu = float(np.average(logs, weights=weights))
    var = float(np.average((logs - mu) ** 2, weights=weights))
    sig = min(2.5, max(0.12, math.sqrt(max(1e-8, var))))
    med = math.exp(mu)
    out = kernel_from_median_sigma(med, sig, "R1B_POSTERIOR_WEIGHTED_POINT_FIT")
    return out, {
        "fitted": True,
        "effective_weight": total_weight,
        "median_sec": med,
        "sigma": sig,
    }


def update_kernels(factors, kernels):
    access_factors = [f for f in factors if f.get("type") == "INTERVAL" and f.get("kind") == "ACCESS"]
    transfer_factors = [f for f in factors if f.get("type") == "INTERVAL" and f.get("kind") == "TRANSFER"]
    egress_factors = [f for f in factors if f.get("type") == "EGRESS"]

    new_access, access_diag = weighted_interval_fit(access_factors, kernels["access"])
    new_egress, egress_diag = weighted_egress_fit(egress_factors, kernels["egress"])
    new_global_k, global_k_diag = weighted_interval_fit(transfer_factors, kernels["transfer_global"])

    by_movement = defaultdict(list)
    movement_weight = Counter()
    for f in transfer_factors:
        movement = f.get("movement") or "UNKNOWN"
        by_movement[movement].append(f)
        movement_weight[movement] += float(f.get("weight", 0.0))

    specific = {}
    specific_diag = {}
    for movement, rows in by_movement.items():
        if movement_weight[movement] < 50.0:
            continue
        kfit, diag = weighted_interval_fit(rows, new_global_k, min_weight=50.0)
        specific[movement] = kfit
        specific_diag[movement] = diag

    updated = {
        "access": new_access,
        "egress": new_egress,
        "transfer_global": new_global_k,
        "transfer_by_movement": specific,
        "schema": "mppd.r1b-kernels.v1-mixture-capable-k1-first-pass",
    }
    diagnostics = {
        "access": access_diag,
        "egress": egress_diag,
        "transfer_global": global_k_diag,
        "transfer_specific_count": len(specific),
        "top_transfer_movements": [
            {
                "movement": k,
                "effective_weight": movement_weight[k],
                "median_sec": specific[k]["components"][0]["median_sec"],
                "sigma": specific[k]["components"][0]["sigma"],
            }
            for k in sorted(specific, key=lambda x: movement_weight[x], reverse=True)[:50]
        ],
        "transfer_specific_diagnostics": specific_diag,
    }
    return updated, diagnostics


def shift_dt(value, delta_sec):
    return None if value is None else value + timedelta(seconds=delta_sec)


def factor_loglik_with_root_shift(factor, target_root, delta_sec, kernels):
    if factor["type"] == "EGRESS":
        arr = shift_dt(factor["arr"], delta_sec if factor.get("arr_root") == target_root else 0.0)
        return egress_logdensity_kernel(
            factor["exit_time"],
            arr,
            factor["arr_sd"],
            kernels["egress"],
        )

    ready = shift_dt(factor["ready"], delta_sec if factor.get("ready_root") == target_root else 0.0)
    lower = shift_dt(factor["lower"], delta_sec if factor.get("lower_root") == target_root else 0.0)
    upper = shift_dt(factor["upper"], delta_sec if factor.get("upper_root") == target_root else 0.0)
    kern = kernel_for(factor["kind"], factor.get("movement"), kernels)
    return interval_logprob_kernel(
        lower,
        factor["lower_sd"],
        upper,
        factor["upper_sd"],
        ready,
        factor["ready_sd"],
        kern,
    )


def root_metadata(roots):
    meta = {}
    for (line, root), variants in roots.items():
        key = root_key(line, root)
        evc = min(
            (v.get("evidence_class", "") for v in variants),
            key=lambda x: g2.evidence_rank(x),
            default="UNKNOWN",
        )
        event_sds = [
            float(e.get("sd", 0.0))
            for v in variants
            for e in v.get("events", {}).values()
            if e.get("sd") is not None
        ]
        meta[key] = {
            "line": line,
            "root": root,
            "evidence_class": evc,
            "median_event_sd": float(np.median(event_sds)) if event_sds else 60.0,
        }
    return meta


def update_service_offsets(roots, factors, kernels, root_usage, grid, min_usage):
    rmeta = root_metadata(roots)
    by_root_factors = defaultdict(list)
    for f in factors:
        touched = set()
        if f["type"] == "EGRESS":
            if f.get("arr_root"):
                touched.add(f["arr_root"])
        else:
            for key in ("ready_root", "lower_root", "upper_root"):
                if f.get(key):
                    touched.add(f[key])
        for root in touched:
            by_root_factors[root].append(f)

    offsets = {}
    diagnostics = {}
    for root, usage in root_usage.items():
        md = rmeta.get(root)
        if not md or usage < min_usage:
            continue
        if md["evidence_class"] == "PARTIAL_DIRECT_SERVICE_ANCHOR":
            offsets[root] = 0
            diagnostics[root] = {
                "usage_mass": usage,
                "evidence_class": md["evidence_class"],
                "selected_offset_sec": 0,
                "frozen": True,
            }
            continue
        rows = by_root_factors.get(root, [])
        if not rows:
            continue
        total_w = sum(float(f.get("weight", 0.0)) for f in rows)
        if total_w <= 0:
            continue
        prior_sd = max(30.0, md["median_event_sd"], 90.0 if "WEAK" in md["evidence_class"] else 60.0)
        scored = []
        for delta in grid:
            ll = 0.0
            used_w = 0.0
            for f in rows:
                w = float(f.get("weight", 0.0))
                lp = factor_loglik_with_root_shift(f, root, delta, kernels)
                if math.isfinite(lp) and w > 0:
                    ll += w * lp
                    used_w += w
            if used_w <= 0:
                continue
            mean_ll = ll / used_w
            prior = -0.5 * (float(delta) / prior_sd) ** 2
            scored.append((mean_ll + prior, delta, mean_ll, prior, used_w))
        if not scored:
            continue
        scored.sort(reverse=True)
        best = scored[0]
        base = next((x for x in scored if x[1] == 0), None)
        gain = best[0] - base[0] if base else None
        selected = int(best[1])
        if gain is not None and gain < 1e-4:
            selected = 0
        offsets[root] = selected
        diagnostics[root] = {
            "usage_mass": usage,
            "factor_weight": best[4],
            "evidence_class": md["evidence_class"],
            "prior_sd_sec": prior_sd,
            "selected_offset_sec": selected,
            "score_gain_vs_zero": gain,
            "best_mean_loglik": best[2],
            "best_prior_log": best[3],
            "frozen": False,
        }
    return offsets, diagnostics


def apply_root_offsets(roots, offsets):
    moved = 0
    moved_roots = 0
    for (line, root), variants in roots.items():
        key = root_key(line, root)
        delta = int(offsets.get(key, 0))
        if delta == 0:
            continue
        moved_roots += 1
        for v in variants:
            for e in v.get("events", {}).values():
                e["arrival"] = e["arrival"] + timedelta(seconds=delta)
                e["departure"] = e["departure"] + timedelta(seconds=delta)
                moved += 1
    return moved_roots, moved


def summarize_pass(pass_result):
    processed = float(pass_result["processed_mass"])
    finite = float(pass_result["finite_mass"])
    leg_mass = sum(float(v) for v in pass_result["skip_mass"].values())
    return {
        "processed_passenger_mass": processed,
        "finite_posterior_mass": finite,
        "finite_posterior_share": finite / processed if processed else 0.0,
        "status_mass": dict(pass_result["status_mass"]),
        "failure_mass": dict(pass_result["failure_mass"]),
        "weighted_mean_route_entropy_nats": (
            pass_result["route_entropy_weighted"] / finite if finite else None
        ),
        "weighted_mean_first_boarding_route_entropy_contraction_nats": (
            pass_result["first_boarding_entropy_contraction_weighted"] / finite if finite else None
        ),
        "skip_mass_by_count": dict(sorted(pass_result["skip_mass"].items())),
        "skip_positive_leg_share": (
            sum(v for k, v in pass_result["skip_mass"].items() if int(k) > 0) / leg_mass
            if leg_mass else None
        ),
        "non_simple_route_posterior_mass": pass_result["non_simple_mass"],
        "non_simple_route_share_of_finite_mass": (
            pass_result["non_simple_mass"] / finite if finite else None
        ),
        "access_tail_gt_15min_posterior_mass": pass_result["access_tail_mass_gt_15min"],
        "egress_tail_gt_15min_posterior_mass": pass_result["egress_tail_mass_gt_15min"],
        "top_missing_service_pressure": [
            {"segment": k, "pressure_mass": v}
            for k, v in pass_result["missing_service_pressure"].most_common(100)
        ],
    }


def compare_passes(e0, e1):
    common = set(e0["cohorts"]).intersection(e1["cohorts"])
    mass_common = 0.0
    route_changed = 0.0
    first_changed = 0.0
    finite_both = 0.0
    became_finite = 0.0
    lost_finite = 0.0
    route_tv_weighted = 0.0
    for cid in common:
        a = e0["cohorts"][cid]
        b = e1["cohorts"][cid]
        mass = float(a.get("mass", b.get("mass", 0.0)))
        mass_common += mass
        af = a.get("status") == "FINITE_POSTERIOR"
        bf = b.get("status") == "FINITE_POSTERIOR"
        if not af and bf:
            became_finite += mass
        if af and not bf:
            lost_finite += mass
        if not (af and bf):
            continue
        finite_both += mass
        if a.get("top_route") != b.get("top_route"):
            route_changed += mass
        if a.get("top_first_root") != b.get("top_first_root"):
            first_changed += mass
        keys = set(a.get("route_probs", {})).union(b.get("route_probs", {}))
        tv = 0.5 * sum(
            abs(float(a.get("route_probs", {}).get(k, 0.0)) - float(b.get("route_probs", {}).get(k, 0.0)))
            for k in keys
        )
        route_tv_weighted += mass * tv
    return {
        "common_passenger_mass": mass_common,
        "finite_in_both_mass": finite_both,
        "became_finite_mass": became_finite,
        "lost_finite_mass": lost_finite,
        "top_route_changed_mass": route_changed,
        "top_route_changed_share_among_finite_both": route_changed / finite_both if finite_both else None,
        "top_first_service_changed_mass": first_changed,
        "top_first_service_changed_share_among_finite_both": first_changed / finite_both if finite_both else None,
        "mean_route_total_variation_among_finite_both": route_tv_weighted / finite_both if finite_both else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--cohorts", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--service-init", required=True)
    ap.add_argument("--topology-patch")
    ap.add_argument("--gtxa-overlay")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--max-skip", type=int, default=2)
    ap.add_argument("--sample-mod", type=int, default=50)
    ap.add_argument("--max-cohorts", type=int, default=0)
    ap.add_argument("--min-service-usage", type=float, default=25.0)
    ap.add_argument("--shift-grid", default="-90,-60,-30,0,30,60,90")
    args = ap.parse_args()

    wall0 = time.perf_counter()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    if args.topology_patch:
        apply_topology_patch(G, meta, load_patch(args.topology_patch))
    if args.gtxa_overlay:
        apply_gtxa_overlay(G, meta, code_to_nodes, load_overlay(args.gtxa_overlay))

    routes = support.load_routes(args.routes)
    roots, service_manifest, service_payload = g2.load_uncertain_service(args.service_init)
    cohorts = load_sampled_cohorts(args.cohorts, args.sample_mod, args.max_cohorts)
    sample_mass = sum(x[4] for x in cohorts)

    kernels0 = {
        "schema": "mppd.r1b-kernels.v1-mixture-capable-k1-first-pass",
        "access": kernel_from_median_sigma(180.0, 0.90, "G2V2_BROAD_INITIAL_PRIOR"),
        "transfer_global": kernel_from_median_sigma(180.0, 0.85, "G2V2_BROAD_INITIAL_PRIOR"),
        "transfer_by_movement": {},
        "egress": kernel_from_median_sigma(120.0, 0.80, "G2V2_BROAD_INITIAL_PRIOR"),
    }

    e0 = posterior_pass(
        cohorts, routes, meta, roots, kernels0, args.beam, args.max_skip, retain_details=True
    )
    kernels1, kernel_diag = update_kernels(e0["factors"], kernels0)

    grid = sorted(set(int(x.strip()) for x in args.shift_grid.split(",") if x.strip()))
    if 0 not in grid:
        grid.append(0)
        grid.sort()
    offsets, service_diag = update_service_offsets(
        roots,
        e0["factors"],
        kernels1,
        e0["root_usage"],
        grid,
        args.min_service_usage,
    )
    moved_roots, moved_events = apply_root_offsets(roots, offsets)

    e1 = posterior_pass(
        cohorts, routes, meta, roots, kernels1, args.beam, args.max_skip, retain_details=True
    )
    movement = compare_passes(e0, e1)

    e0_summary = summarize_pass(e0)
    e1_summary = summarize_pass(e1)

    result = {
        "schema": "mppd.r1b-first-bidirectional-joint-update-smoke.v1",
        "date": "2026-09-04",
        "status": "R1B_FIRST_BIDIRECTIONAL_JOINT_UPDATE_SMOKE_COMPLETED",
        "scientific_unit": "SEOUL_2026-08-29_0700_1000_FULL_NETWORK_HASH_SAMPLE",
        "scope": {
            "full_network_candidate_domain": True,
            "hash_sampled_cohorts": True,
            "sample_mod": args.sample_mod,
            "sample_cohort_count": len(cohorts),
            "sample_passenger_mass": sample_mass,
            "line_filter": False,
            "segment_filter": False,
            "transfer_count_cap": False,
            "behavioral_regularities_hard_coded": False,
            "max_skip_latent_support": args.max_skip,
            "capacity_state_active": False,
        },
        "inputs": {
            "service_schema": service_payload.get("schema"),
            "service_status": service_payload.get("status"),
            "service_manifest": service_manifest,
            "route_od_count": len(routes),
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
        },
        "iteration": {
            "E0_passenger_posterior": e0_summary,
            "M_kernel_update": {
                "kernels_before": kernels0,
                "kernels_after": kernels1,
                "diagnostics": kernel_diag,
            },
            "M_service_timing_update": {
                "candidate_grid_sec": grid,
                "root_count_with_offset_decision": len(offsets),
                "moved_root_count": moved_roots,
                "moved_event_count": moved_events,
                "nonzero_offsets": {k: v for k, v in offsets.items() if int(v) != 0},
                "top_diagnostics": [
                    {"root": k, **v}
                    for k, v in sorted(
                        service_diag.items(),
                        key=lambda kv: float(kv[1].get("usage_mass", 0.0)),
                        reverse=True,
                    )[:100]
                ],
            },
            "E1_passenger_posterior": e1_summary,
            "posterior_redistribution": movement,
        },
        "r1b_minimum_gate": {
            "passenger_evidence_updates_kernel_posterior": any(
                abs(
                    kernels1[k]["components"][0]["median_sec"]
                    - kernels0[k]["components"][0]["median_sec"]
                ) > 1e-6
                for k in ("access", "egress", "transfer_global")
            ),
            "passenger_evidence_updates_service_timing": moved_roots > 0,
            "updated_state_redistributes_route_or_boarding_posterior": (
                (movement["mean_route_total_variation_among_finite_both"] or 0.0) > 1e-9
                or (movement["top_first_service_changed_mass"] or 0.0) > 0
            ),
            "not_sequential_fixed_service_pipeline": moved_roots > 0,
        },
        "behavioral_diagnostics_boundary": {
            "skip_count_is_latent_with_uniform_nuisance_prior": True,
            "first_feasible_preference_not_imposed": True,
            "non_simple_route_mass_is_diagnostic_not_prior": True,
            "access_egress_first_pass_default_component_count": 1,
            "access_egress_interface_mixture_capable": True,
            "transfer_interface_movement_conditioned_and_mixture_capable": True,
        },
        "service_support_boundary": {
            "missing_service_pressure_is_reported_not_auto_inserted": True,
            "reason": "A smoke update must not turn each residual passenger into a fabricated service event; support activation requires an evidence-qualified proposal gate.",
        },
        "scientific_boundary": [
            "This is a deterministic full-network-domain hash-sample R1B smoke, not the qualified 632143-passenger result.",
            "One approximate EM/message-passing iteration is executed: passenger route/boarding responsibilities update shared kernels and inferred service-root timing offsets, then the passenger posterior is recomputed.",
            "Partial direct service anchors are frozen in the service-timing M-step.",
            "Service timing updates are trajectory-level root offsets and remain AFC-inferred latent-state updates, never observed ATS.",
            "Theta_A and Theta_E use a single main component in this smoke but the serialized distribution interface is mixture-capable.",
            "Theta_K is movement-conditioned when posterior mass supports a specific movement; K=1 is used in this smoke while the interface remains mixture-capable.",
            "Skipped feasible services are allowed as a latent nuisance state and are not hard-zeroed; no behavioral preference is promoted from this smoke.",
            "Capacity/left-behind is not activated in this smoke, so positive skip mass cannot be interpreted as voluntary skipping.",
            "Missing-service residuals are converted only into pressure diagnostics; no service event is auto-created from residual passenger mass.",
        ],
        "performance": {
            "total_wall_sec": time.perf_counter() - wall0,
            "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "next_gate": "If all R1B minimum-gate booleans pass and redistribution is nontrivial, qualify stability on a larger/full sample, then add evidence-qualified missing-service support proposals and R1C hierarchical kernels.",
        "no_email_notification_logic": True,
    }

    (outdir / "r1b_first_bidirectional_joint_update_smoke_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (outdir / "r1b_service_root_offset_diagnostics.json").write_text(
        json.dumps(service_diag, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
