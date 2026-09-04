import argparse
import csv
import gzip
import json
import math
import resource
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_g2_full_network_route_chain_posterior_20260904 as g2
import scripts.mppd_r0_cached_full_network_support_rescan_20260904 as support
import scripts.mppd_r0_full_network_factor_engine_20260904 as r0


def load_cached_cohorts(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yield {
                "oc": row["origin_code"],
                "dc": row["destination_code"],
                "t0": datetime.fromisoformat(row["entry_time"]),
                "t1": datetime.fromisoformat(row["exit_time"]),
                "mass": int(row["passenger_mass"]),
            }


def entropy(probs):
    return -sum(p * math.log(max(p, 1e-15)) for p in probs if p > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--cohorts", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--service-init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--checkpoint-every", type=int, default=50000)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()
    phase = {}

    # These are intentionally broad initial factors. They provide a proper
    # non-zero likelihood substrate only; G3 must jointly update them with S/R/B.
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
    routes = support.load_routes(args.routes)
    phase["route_cache_load_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    trajectories, evidence_counts, service_manifest = support.load_service_trajectories(args.service_init)
    rides_fn, ride_cache = r0.build_segment_ride_cache(trajectories)
    phase["service_cache_load_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    route_mass = Counter()
    transfer_mass = Counter()
    best_transfer_mass = Counter()
    status_mass = Counter()
    failure_mass = Counter()
    entropy_mass_sum = 0.0
    posterior_mass_total = 0
    processed_cohorts = 0
    processed_mass = 0
    max_transfer_seen = 0

    with gzip.open(outdir / "g2_cached_route_chain_posterior_cohorts.jsonl.gz", "wt", encoding="utf-8") as fout:
        for row in load_cached_cohorts(args.cohorts):
            processed_cohorts += 1
            mass = row["mass"]
            processed_mass += mass
            cands = routes.get((row["oc"], row["dc"]), [])
            if not cands:
                status_mass["NO_STRUCTURAL_ROUTE"] += mass
                continue

            min_cost = min(float(c.get("base_cost", 0.0)) for c in cands)
            route_scores = []
            route_chains = []
            route_fail = []
            for cand in cands:
                tc = int(cand.get("transfer_count", 0))
                max_transfer_seen = max(max_transfer_seen, tc)
                chains, pruned_logmass, fail = g2.route_chain_beam(
                    cand, meta, rides_fn, row["t0"], row["t1"], args.beam, priors
                )
                route_fail.append(fail)
                route_chains.append(chains)
                if not chains:
                    route_scores.append(-math.inf)
                    continue
                chain_logs = [x[0] for x in chains]
                cost_penalty = -0.30 * max(0.0, float(cand.get("base_cost", 0.0)) - min_cost)
                policy_bonus = math.log(1.0 + len(cand.get("policies", [])))
                route_scores.append(g2.logsumexp(chain_logs) + cost_penalty + 0.20 * policy_bonus)

            post = g2.normalized_route_posterior(cands, route_scores)
            if not post:
                reason_counts = Counter(x or "UNKNOWN" for x in route_fail)
                failure_mass[reason_counts.most_common(1)[0][0]] += mass
                status_mass["NO_FINITE_POSTERIOR"] += mass
                continue

            status_mass["FINITE_POSTERIOR"] += mass
            posterior_mass_total += mass
            probs = [p for _, p in post]
            entropy_mass_sum += mass * entropy(probs)
            best_idx, best_prob = max(post, key=lambda x: x[1])
            best_transfer_mass[int(cands[best_idx].get("transfer_count", 0))] += mass

            out_routes = []
            for idx, prob in post:
                cand = cands[idx]
                tc = int(cand.get("transfer_count", 0))
                weighted = mass * prob
                transfer_mass[tc] += weighted
                route_mass[cand.get("line_sequence", "")] += weighted
                chains = route_chains[idx]
                z = g2.logsumexp([x[0] for x in chains])
                top_chains = []
                for lp, chain in chains[: min(3, len(chains))]:
                    cp = math.exp(lp - z) if math.isfinite(z) else 0.0
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
                    "line_sequence": cand.get("line_sequence"),
                    "transfer_count": tc,
                    "route_probability": prob,
                    "base_cost": cand.get("base_cost"),
                    "policies": cand.get("policies", []),
                    "top_chains": top_chains,
                })

            fout.write(json.dumps({
                "origin_code": row["oc"],
                "destination_code": row["dc"],
                "entry_time": row["t0"].isoformat(),
                "exit_time": row["t1"].isoformat(),
                "cohort_mass": mass,
                "route_entropy_nats": entropy(probs),
                "routes": out_routes,
            }, ensure_ascii=False) + "\n")

            if args.checkpoint_every > 0 and processed_cohorts % args.checkpoint_every == 0:
                (outdir / "g2_cached_checkpoint.json").write_text(json.dumps({
                    "processed_cohorts": processed_cohorts,
                    "processed_mass": processed_mass,
                    "finite_posterior_mass": posterior_mass_total,
                    "elapsed_sec": time.perf_counter() - wall0,
                }, ensure_ascii=False, indent=2), encoding="utf-8")

    phase["posterior_scan_sec"] = time.perf_counter() - t

    route_rows = [
        {"line_sequence": k, "posterior_passenger_mass": v}
        for k, v in route_mass.most_common()
    ]
    with (outdir / "g2_cached_route_family_posterior_mass.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["line_sequence", "posterior_passenger_mass"])
        w.writeheader(); w.writerows(route_rows)

    transfer_rows = [
        {"transfer_count": k, "posterior_passenger_mass": v, "best_route_passenger_mass": best_transfer_mass[k]}
        for k, v in sorted(transfer_mass.items())
    ]
    with (outdir / "g2_cached_transfer_count_posterior_mass.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["transfer_count", "posterior_passenger_mass", "best_route_passenger_mass"])
        w.writeheader(); w.writerows(transfer_rows)

    total_wall = time.perf_counter() - wall0
    result = {
        "schema": "mppd.g2-cached-full-network-route-chain-posterior.v1",
        "date": "2026-09-04",
        "status": "G2_CACHED_PROVISIONAL_FULL_NETWORK_POSTERIOR_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "time_window": "2026-08-29 07:00-10:00",
        "scope_assertions": {
            "full_network": True,
            "line_filter_applied": False,
            "segment_filter_applied": False,
            "transfer_count_cap_applied": False,
            "raw_taims_rescan": False,
            "all_cached_mapped_passenger_mass_accounted": True,
            "high_order_routes_not_penalized_by_transfer_count": True,
        },
        "network": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "lines": len({m["line"] for m in meta.values()}),
            "transfer_groups": len(transfer_groups),
        },
        "afc": {
            "processed_cohort_count": processed_cohorts,
            "processed_passenger_mass": processed_mass,
            "finite_posterior_mass": posterior_mass_total,
            "finite_posterior_share": posterior_mass_total / processed_mass if processed_mass else 0.0,
        },
        "service_initialization": {
            "trajectory_count_by_evidence_class": dict(evidence_counts),
            "manifest": service_manifest,
        },
        "initial_station_time_priors": priors,
        "posterior": {
            "mass_by_transfer_count": dict(sorted(transfer_mass.items())),
            "best_route_mass_by_transfer_count": dict(sorted(best_transfer_mass.items())),
            "weighted_mean_route_entropy_nats": entropy_mass_sum / posterior_mass_total if posterior_mass_total else None,
            "top_route_family_posterior_mass": route_rows[:100],
            "status_mass": dict(status_mass),
            "failure_mass": dict(failure_mass),
            "max_transfer_count_seen": max_transfer_seen,
        },
        "performance": {
            "beam": args.beam,
            "phase_sec": phase,
            "total_wall_sec": total_wall,
            "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "passenger_mass_per_sec": processed_mass / max(total_wall, 1e-9),
        },
        "scientific_boundary": [
            "This is a provisional complete-network route/boarding posterior initialization, not the final city-day reconstruction.",
            "Access, transfer and egress parameters are deliberately broad initial priors and must be jointly updated in G3.",
            "Weak latent service lattices are initialization states and may move/disappear/reweight in G3.",
            "No explicit transfer-count penalty is used; high-order paths survive or disappear only through structural and temporal likelihood support plus weak path-cost regularization.",
            "Route candidates are structural hypotheses, not observed passenger route truth.",
            "No raw card identifier is retained."
        ],
        "next_gate": "Use the complete-network posterior sufficient statistics to update Theta_A/Theta_K/Theta_E and service-event posterior support jointly; then iterate G2/G3 until objective and held-out diagnostics stabilize.",
        "no_email_notification_logic": True,
    }
    (outdir / "g2_cached_full_network_route_chain_posterior_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
