import argparse
import csv
import gzip
import json
import math
import resource
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r0_full_network_factor_engine_20260904 as r0

START = g0.START
END = g0.END
RAIL = g0.RAIL


def logsumexp(vals):
    vals = [x for x in vals if math.isfinite(x)]
    if not vals:
        return -math.inf
    m = max(vals)
    return m + math.log(sum(math.exp(x - m) for x in vals))


def lognorm_cdf(x, median, sigma):
    if x <= 0:
        return 0.0
    mu = math.log(max(1e-6, median))
    z = (math.log(x) - mu) / (max(1e-6, sigma) * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def log_interval_prob(lower, upper, median, sigma):
    if upper <= max(0.0, lower):
        return -math.inf
    fu = lognorm_cdf(upper, median, sigma)
    fl = lognorm_cdf(max(0.0, lower), median, sigma) if lower > 0 else 0.0
    return math.log(max(1e-14, fu - fl))


def log_point_density(x, median, sigma):
    if x <= 0:
        return -math.inf
    mu = math.log(max(1e-6, median))
    s = max(1e-6, sigma)
    z = (math.log(x) - mu) / s
    return -math.log(x * s * math.sqrt(2.0 * math.pi)) - 0.5 * z * z


def load_route_cache(path):
    cache = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cache[(obj["origin_code"], obj["destination_code"])] = obj["candidates"]
    return cache


def load_cohorts(taims_path, code_to_nodes):
    cohorts = Counter()
    stats = Counter()
    with zipfile.ZipFile(taims_path) as z:
        f = g0.zr(z, "VW_KSCC_DX_CARD.csv")
        for row in csv.DictReader(f):
            if str(row.get("TRNS_MNS_CD") or "").strip() not in RAIL:
                continue
            t0 = g0.dt(row.get("RIDE_DTM"))
            t1 = g0.dt(row.get("ALGH_DTM"))
            if not t0 or not t1 or not (START <= t0 < END) or t1 <= t0 or t1 - t0 > timedelta(hours=3):
                continue
            stats["eligible"] += 1
            oc = g0.z4(row.get("RIDE_BSST_ID"))
            dc = g0.z4(row.get("ALGH_BSST_ID"))
            if oc not in code_to_nodes or dc not in code_to_nodes:
                stats["unmapped"] += 1
                continue
            stats["mapped"] += 1
            # Seoul public AFC is minute-quantized. Exact t0/t1 grouping therefore
            # preserves the observed temporal information while removing duplicate
            # computation. No card identifier is retained.
            cohorts[(oc, dc, t0, t1)] += 1
        f.close()
    return cohorts, stats


def previous_departure(rides, idx, ready):
    for j in range(idx - 1, -1, -1):
        d = rides[j][0]
        if d >= ready:
            return d
        break
    return None


def route_chain_beam(route, meta, rides_fn, t_entry, t_exit, beam, priors):
    legs = r0.path_legs(route["path"], meta)
    states = [(t_entry, 0.0, [])]
    pruned_logmass = -math.inf

    for leg_index, (line, origin, destination) in enumerate(legs):
        if origin == destination:
            states = [
                (arr, lp, chain + [(line, origin, destination, None, None, None)])
                for arr, lp, chain in states
            ]
            continue
        rides = rides_fn(line, origin, destination)
        if not rides:
            return [], pruned_logmass, "MISSING_SERVICE_SEGMENT"
        candidates = []
        for ready, base_lp, chain in states:
            for i, (dep, arr, sid, evc) in enumerate(rides):
                if dep < ready or arr > t_exit:
                    continue
                upper = (dep - ready).total_seconds()
                if upper <= 0:
                    continue
                pd = previous_departure(rides, i, ready)
                lower = max(0.0, (pd - ready).total_seconds()) if pd else 0.0
                if leg_index == 0:
                    lp = log_interval_prob(lower, upper, priors["access_median"], priors["access_sigma"])
                else:
                    lp = log_interval_prob(lower, upper, priors["transfer_median"], priors["transfer_sigma"])
                if not math.isfinite(lp):
                    continue
                candidates.append((
                    arr,
                    base_lp + lp,
                    chain + [(line, origin, destination, sid, dep, arr)],
                ))
        if not candidates:
            return [], pruned_logmass, "TIME_INCOMPATIBLE_CHAIN"
        candidates.sort(key=lambda x: (x[1], -x[0].timestamp()), reverse=True)
        if len(candidates) > beam:
            removed = [x[1] for x in candidates[beam:]]
            pruned_logmass = logsumexp([pruned_logmass, logsumexp(removed)])
            candidates = candidates[:beam]
        states = candidates

    final = []
    for arr, lp, chain in states:
        eg = (t_exit - arr).total_seconds()
        le = log_point_density(eg, priors["egress_median"], priors["egress_sigma"])
        if math.isfinite(le):
            final.append((lp + le, chain))
    final.sort(key=lambda x: x[0], reverse=True)
    if not final:
        return [], pruned_logmass, "EGRESS_INCOMPATIBLE"
    return final[:beam], pruned_logmass, None


def normalized_route_posterior(cands, route_scores):
    finite = [(i, x) for i, x in enumerate(route_scores) if math.isfinite(x)]
    if not finite:
        return []
    z = logsumexp([x for _, x in finite])
    return [(i, math.exp(x - z)) for i, x in finite]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taims", required=True)
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--service", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--checkpoint-every", type=int, default=50000)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()
    phase = {}

    priors = {
        "access_median": 180.0,
        "access_sigma": 0.90,
        "transfer_median": 180.0,
        "transfer_sigma": 0.85,
        "egress_median": 120.0,
        "egress_sigma": 0.80,
        "evidence_class": "PROVISIONAL_BROAD_INITIAL_PRIOR_FOR_G2_ONLY",
    }

    t = time.perf_counter()
    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    phase["network_build_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    cohorts, afc_stats = load_cohorts(args.taims, code_to_nodes)
    phase["cohort_load_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    route_cache = load_route_cache(args.routes)
    phase["route_cache_load_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    trajectories, service_stats = r0.service_trajectories(
        args.taims, args.p1c, args.service, meta, code_to_nodes
    )
    rides_fn, ride_cache = r0.build_segment_ride_cache(trajectories)
    phase["service_bootstrap_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    route_posterior_mass = Counter()
    transfer_posterior_mass = Counter()
    best_route_mass = Counter()
    failure_mass = Counter()
    cohort_status_mass = Counter()
    pruned_mass_upper = 0.0
    processed_cohorts = 0
    processed_mass = 0
    finite_mass = 0
    high_order_finite_mass = 0

    posterior_path = outdir / "g2_route_chain_posterior_cohorts.jsonl.gz"
    with gzip.open(posterior_path, "wt", encoding="utf-8") as fout:
        for (oc, dc, t0, t1), mass in cohorts.items():
            processed_cohorts += 1
            processed_mass += mass
            cands = route_cache.get((oc, dc), [])
            if not cands:
                failure_mass["NO_STRUCTURAL_ROUTE_CANDIDATE"] += mass
                continue

            min_cost = min(float(c.get("base_cost", 0.0)) for c in cands)
            route_scores = []
            route_chain_scores = []
            route_failures = []
            route_pruned = []
            for c in cands:
                chains, pruned_logmass, fail = route_chain_beam(
                    c, meta, rides_fn, t0, t1, args.beam, priors
                )
                route_failures.append(fail)
                route_pruned.append(pruned_logmass)
                if not chains:
                    route_scores.append(-math.inf)
                    route_chain_scores.append([])
                    continue
                chain_logs = [x[0] for x in chains]
                # Structural prior is deliberately weak and has no explicit
                # transfer-count penalty; high-order paths are not disfavoured
                # merely because they contain more transfers.
                cost_penalty = -0.30 * max(0.0, float(c["base_cost"]) - min_cost)
                policy_bonus = math.log(1.0 + len(c.get("policies", [])))
                score = logsumexp(chain_logs) + cost_penalty + 0.20 * policy_bonus
                route_scores.append(score)
                route_chain_scores.append(chains)

            post = normalized_route_posterior(cands, route_scores)
            if not post:
                reasons = Counter(x or "UNKNOWN" for x in route_failures)
                failure_mass[reasons.most_common(1)[0][0]] += mass
                cohort_status_mass["NO_FINITE_CHAIN"] += mass
                continue

            finite_mass += mass
            cohort_status_mass["FINITE_POSTERIOR"] += mass
            out_routes = []
            best = max(post, key=lambda x: x[1])
            if int(cands[best[0]]["transfer_count"]) >= 3:
                high_order_finite_mass += mass
            best_route_mass[cands[best[0]]["line_sequence"]] += mass

            for idx, prob in post:
                c = cands[idx]
                weighted = mass * prob
                route_posterior_mass[c["line_sequence"]] += weighted
                transfer_posterior_mass[int(c["transfer_count"])] += weighted
                chains = route_chain_scores[idx]
                chain_z = logsumexp([x[0] for x in chains])
                top_chains = []
                for lp, chain in chains[: min(5, len(chains))]:
                    cp = math.exp(lp - chain_z) if math.isfinite(chain_z) else 0.0
                    top_chains.append({
                        "conditional_probability": cp,
                        "services": [
                            {
                                "line": x[0],
                                "origin": x[1],
                                "destination": x[2],
                                "service_id": x[3],
                                "departure": x[4].isoformat() if x[4] else None,
                                "arrival": x[5].isoformat() if x[5] else None,
                            }
                            for x in chain
                        ],
                    })
                out_routes.append({
                    "line_sequence": c["line_sequence"],
                    "transfer_count": c["transfer_count"],
                    "route_probability": prob,
                    "base_cost": c["base_cost"],
                    "policies": c.get("policies", []),
                    "top_chains": top_chains,
                })

            fout.write(json.dumps({
                "origin_code": oc,
                "destination_code": dc,
                "entry_time": t0.isoformat(),
                "exit_time": t1.isoformat(),
                "cohort_mass": mass,
                "routes": out_routes,
            }, ensure_ascii=False) + "\n")

            if args.checkpoint_every > 0 and processed_cohorts % args.checkpoint_every == 0:
                cp = {
                    "processed_cohorts": processed_cohorts,
                    "processed_mass": processed_mass,
                    "finite_mass": finite_mass,
                    "elapsed_sec": time.perf_counter() - wall0,
                }
                (outdir / "g2_checkpoint.json").write_text(
                    json.dumps(cp, ensure_ascii=False, indent=2), encoding="utf-8"
                )

    phase["posterior_scan_sec"] = time.perf_counter() - t

    route_rows = [
        {"line_sequence": rf, "posterior_passenger_mass": mass}
        for rf, mass in route_posterior_mass.most_common()
    ]
    with (outdir / "g2_route_family_posterior_mass.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["line_sequence", "posterior_passenger_mass"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(route_rows)

    transfer_rows = [
        {"transfer_count": tc, "posterior_passenger_mass": mass}
        for tc, mass in sorted(transfer_posterior_mass.items())
    ]
    with (outdir / "g2_transfer_count_posterior_mass.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["transfer_count", "posterior_passenger_mass"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(transfer_rows)

    total_wall = time.perf_counter() - wall0
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "schema": "mppd.g2-full-network-route-chain-posterior.v1",
        "date": "2026-09-04",
        "status": "G2_PROVISIONAL_FULL_NETWORK_ROUTE_CHAIN_POSTERIOR_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "time_window": "2026-08-29 07:00-10:00",
        "scope_assertions": {
            "full_network": True,
            "line_filter_applied": False,
            "segment_filter_applied": False,
            "transfer_count_cap_applied": False,
            "all_mapped_passenger_mass_enters_accounting": True,
            "minute_cohort_aggregation_is_lossless_for_qualified_seoul_afc": True,
            "beam_pruning_is_chain_probability_pruning_not_spatial_pruning": True,
        },
        "network": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "lines": len({m["line"] for m in meta.values()}),
            "transfer_groups": len(transfer_groups),
        },
        "afc": {
            "eligible": afc_stats["eligible"],
            "mapped": afc_stats["mapped"],
            "unmapped": afc_stats["unmapped"],
            "unique_time_cohorts": len(cohorts),
            "processed_mass": processed_mass,
            "finite_posterior_mass": finite_mass,
            "finite_posterior_share_of_mapped": finite_mass / processed_mass if processed_mass else 0.0,
            "high_order_best_route_mass_transfer_ge_3": high_order_finite_mass,
        },
        "initial_factor_priors": priors,
        "posterior_mass_by_transfer_count": dict(sorted(transfer_posterior_mass.items())),
        "cohort_status_mass": dict(cohort_status_mass),
        "failure_mass": dict(failure_mass),
        "top_route_family_posterior_mass": route_rows[:100],
        "service_bootstrap": service_stats,
        "performance": {
            "beam": args.beam,
            "phase_sec": phase,
            "total_wall_sec": total_wall,
            "max_rss_kb": max_rss_kb,
            "cohorts_per_sec": processed_cohorts / max(total_wall, 1e-9),
            "passenger_mass_per_sec": processed_mass / max(total_wall, 1e-9),
        },
        "scientific_boundary": [
            "G2 constructs a provisional route/boarding posterior over the complete network; it is not the final joint inversion.",
            "Theta_A, Theta_K and Theta_E are broad initialization priors in G2 and must be jointly updated with S, R and B in G3.",
            "No explicit transfer-count penalty is used; high-order routes are retained when structurally and temporally supported.",
            "Beam pruning acts only on low-scoring service-chain states within a full-network route candidate; it does not remove lines, segments, OD classes or transfer-count classes.",
            "Route candidates come from R0 full-network structural policies and are not observed passenger route truth.",
            "Partial direct service anchors and AFC_INFERRED_SERVICE_FIELD remain distinct evidence classes.",
            "No raw card identifier is retained.",
        ],
        "next_gate": "Fit/update shared station access, physical-transfer-direction and egress factors from the full-network posterior, then jointly update the complete service field and route/boarding posterior in G3.",
        "no_email_notification_logic": True,
    }
    (outdir / "g2_full_network_route_chain_posterior_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
