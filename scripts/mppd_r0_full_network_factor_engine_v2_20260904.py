import argparse
import csv
import gzip
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r0_full_network_factor_engine_20260904 as v1


def build_route_candidates_streaming(G, meta, code_to_nodes, od_mass):
    """Generate full-network route candidates with bounded memory.

    V1 retained all source->destination path maps for all routing policies at once.
    V2 instead runs one policy and one source at a time, immediately reducing each
    OD to its best path under that policy. Only final OD candidates remain resident.
    Spatial scope and transfer-count scope are unchanged.
    """
    source_to_ods = defaultdict(list)
    for oc, dc in od_mass:
        for src in code_to_nodes.get(oc, []):
            if src in G:
                source_to_ods[src].append((oc, dc))

    by_od = defaultdict(dict)
    timings = {}
    for policy in v1.ROUTE_POLICIES:
        pname = policy[0]
        weight = v1.edge_weight(policy)
        t0 = time.perf_counter()
        best = {}
        for src, ods in source_to_ods.items():
            pmap = nx.single_source_dijkstra_path(G, src, weight=weight)
            for oc, dc in ods:
                local = best.get((oc, dc))
                for dst in code_to_nodes.get(dc, []):
                    path = pmap.get(dst)
                    if not path:
                        continue
                    base_cost = v1.path_cost(G, path)
                    key = (base_cost, len(path), src, dst)
                    if local is None or key < local[0]:
                        local = (key, path)
                if local is not None:
                    best[(oc, dc)] = local

        for od, (_, path) in best.items():
            sig = tuple(path)
            rf, tc = v1.route_signature(path, meta)
            obj = by_od[od].setdefault(sig, {
                "path": path,
                "line_sequence": rf,
                "transfer_count": tc,
                "base_cost": v1.path_cost(G, path),
                "policies": [],
            })
            obj["policies"].append(pname)
        timings[pname] = time.perf_counter() - t0
        del best

    route_cache = {}
    for od, candmap in by_od.items():
        cands = list(candmap.values())
        cands.sort(key=lambda x: (x["base_cost"], x["transfer_count"], len(x["path"]), x["line_sequence"]))
        route_cache[od] = cands
    return route_cache, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taims", required=True)
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--service", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-every", type=int, default=100000)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()
    phase = {}

    t = time.perf_counter()
    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    phase["network_build_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    passengers, od_mass, afc_stats = v1.load_passengers(args.taims, code_to_nodes)
    phase["passenger_load_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    route_cache, policy_timings = build_route_candidates_streaming(G, meta, code_to_nodes, od_mass)
    phase["route_candidate_build_sec"] = time.perf_counter() - t

    unrouted_od = 0
    route_family_mass = Counter()
    transfer_count_mass = Counter()
    alternative_count = Counter()
    with gzip.open(outdir / "r0_route_candidates_by_od.jsonl.gz", "wt", encoding="utf-8") as f:
        for (oc, dc), mass in od_mass.items():
            cands = route_cache.get((oc, dc), [])
            if not cands:
                unrouted_od += 1
            else:
                alternative_count[len(cands)] += mass
                for c in cands:
                    route_family_mass[c["line_sequence"]] += mass
                    transfer_count_mass[c["transfer_count"]] += mass
            f.write(json.dumps({
                "origin_code": oc,
                "destination_code": dc,
                "passenger_mass": mass,
                "candidates": cands,
            }, ensure_ascii=False) + "\n")

    t = time.perf_counter()
    trajectories, service_stats = v1.service_trajectories(
        args.taims, args.p1c, args.service, meta, code_to_nodes
    )
    rides_fn, ride_cache = v1.build_segment_ride_cache(trajectories)
    phase["service_bootstrap_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    support_by_transfer = defaultdict(Counter)
    failure_by_transfer = defaultdict(Counter)
    processed = 0
    mapped_with_route = 0
    feasible_any = 0
    max_transfer_seen = 0
    high_order_feasible = 0
    checkpoints = []

    for oc, dc, t0, t1 in passengers:
        processed += 1
        cands = route_cache.get((oc, dc), [])
        if not cands:
            continue
        mapped_with_route += 1
        any_ok = False
        for c in cands:
            tc = int(c["transfer_count"])
            max_transfer_seen = max(max_transfer_seen, tc)
            support_by_transfer[tc]["candidate_evaluations"] += 1
            chain, fail = v1.first_feasible_chain(c, meta, rides_fn, t0, t1)
            if chain is not None:
                support_by_transfer[tc]["finite_chain"] += 1
                any_ok = True
                if tc >= 3:
                    high_order_feasible += 1
            else:
                support_by_transfer[tc]["no_chain"] += 1
                if fail and fail.get("available_rides", 0) == 0:
                    failure_by_transfer[tc]["missing_segment_service_coverage"] += 1
                else:
                    failure_by_transfer[tc]["time_incompatible_service_chain"] += 1
        if any_ok:
            feasible_any += 1

        if args.checkpoint_every > 0 and processed % args.checkpoint_every == 0:
            cp = {
                "processed_passengers": processed,
                "feasible_any": feasible_any,
                "elapsed_sec": time.perf_counter() - wall0,
            }
            checkpoints.append(cp)
            (outdir / "r0_checkpoint.json").write_text(json.dumps(cp, ensure_ascii=False, indent=2), encoding="utf-8")

    phase["chain_support_scan_sec"] = time.perf_counter() - t

    transfer_rows = []
    for tc in sorted(set(support_by_transfer) | set(failure_by_transfer)):
        s = support_by_transfer[tc]
        f = failure_by_transfer[tc]
        n = s["candidate_evaluations"]
        transfer_rows.append({
            "transfer_count": tc,
            "candidate_evaluations": n,
            "finite_chain": s["finite_chain"],
            "finite_chain_share": s["finite_chain"] / n if n else 0.0,
            "no_chain": s["no_chain"],
            "missing_segment_service_coverage": f["missing_segment_service_coverage"],
            "time_incompatible_service_chain": f["time_incompatible_service_chain"],
        })

    with (outdir / "r0_chain_support_by_transfer_count.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(transfer_rows[0]) if transfer_rows else ["transfer_count"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(transfer_rows)

    fam_rows = [
        {
            "line_sequence": rf,
            "transfer_count": max(0, len(rf.split(">")) - 1),
            "candidate_support_mass": mass,
        }
        for rf, mass in route_family_mass.most_common()
    ]
    with (outdir / "r0_route_family_candidate_support.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["line_sequence", "transfer_count", "candidate_support_mass"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(fam_rows)

    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    total_wall = time.perf_counter() - wall0
    result = {
        "schema": "mppd.r0-full-network-factor-engine-qualification.v2-streaming",
        "date": "2026-09-04",
        "status": "R0_FULL_NETWORK_FACTOR_ENGINE_V2_QUALIFICATION_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "time_window": "2026-08-29 07:00-10:00",
        "scope_assertions": {
            "full_network": True,
            "line_filter_applied": False,
            "segment_filter_applied": False,
            "transfer_count_cap_applied": False,
            "high_order_routes_retained": True,
            "route_alternatives_generated_on_complete_network": True,
            "missing_service_lines_retained_as_latent": True,
            "all_pairs_path_matrix_retained": False,
            "streaming_policy_source_routing": True,
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
            "unique_mapped_od": len(od_mass),
            "mapped_with_at_least_one_structural_route": mapped_with_route,
            "passengers_with_at_least_one_finite_service_chain": feasible_any,
            "finite_service_chain_share_of_mapped": feasible_any / len(passengers) if passengers else 0.0,
        },
        "route_engine": {
            "policies": [x[0] for x in v1.ROUTE_POLICIES],
            "policy_sec": policy_timings,
            "od_without_route_candidates": unrouted_od,
            "passenger_mass_by_number_of_distinct_candidates": dict(sorted(alternative_count.items())),
            "max_transfer_count_seen": max_transfer_seen,
            "candidate_support_transfer_count_mass": dict(sorted(transfer_count_mass.items())),
        },
        "service_bootstrap": service_stats,
        "chain_support_by_transfer_count": transfer_rows,
        "high_order_finite_chain_evaluations_transfer_ge_3": high_order_feasible,
        "performance": {
            "phase_sec": phase,
            "total_wall_sec": total_wall,
            "max_rss_kb": max_rss_kb,
            "passengers_per_sec": processed / max(total_wall, 1e-9),
            "checkpoint_count": len(checkpoints),
        },
        "scientific_boundary": [
            "R0 V2 changes only computational memory strategy; scientific data scope is identical to V1.",
            "Route alternatives are full-network structural candidates, not observed passenger routes or a final route posterior.",
            "The first-feasible chain scan uses zero lower-bound transfer movement only as a support diagnostic; Theta_K is not estimated here.",
            "Finite-chain share measures current service bootstrap support coverage, not passenger assignment accuracy.",
            "No raw card identifier is retained.",
        ],
        "next_gate": "Use the qualified streaming engine and R0 route cache for G2 arbitrary-transfer posterior, then G3 full-network joint update.",
        "no_email_notification_logic": True,
    }
    (outdir / "r0_full_network_factor_engine_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
