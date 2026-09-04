import argparse
import csv
import gzip
import io
import json
import math
import resource
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_g1_full_network_service_bootstrap_20260904 as g1

START = g0.START
END = g0.END
RAIL = g0.RAIL

ROUTE_POLICIES = (
    ("BASE", 1.0, 1.0),
    ("TRANSFER_EXPENSIVE", 1.0, 1.45),
    ("TRANSFER_CHEAP", 1.0, 0.70),
    ("UNCERTAIN_EXPENSIVE", 1.65, 1.0),
)


def edge_weight(policy):
    _, uncertain_mult, transfer_mult = policy

    def fn(u, v, d):
        w = float(d.get("weight", 1.0))
        kind = d.get("kind")
        if kind == "transfer":
            return w * transfer_mult
        if kind == "inline_uncertain":
            return w * uncertain_mult
        return w

    return fn


def path_cost(G, path):
    out = 0.0
    for u, v in zip(path, path[1:]):
        out += float((G.get_edge_data(u, v) or {}).get("weight", 1.0))
    return out


def path_legs(path, meta):
    if not path:
        return []
    legs = []
    start = 0
    current = meta[path[0]]["line"]
    for i in range(1, len(path)):
        line = meta[path[i]]["line"]
        if line != current:
            # path[i-1] is the interchange node on the old line; path[i] is the
            # physical interchange counterpart on the new line.
            if i - 1 >= start:
                legs.append((current, path[start], path[i - 1]))
            start = i
            current = line
    legs.append((current, path[start], path[-1]))
    return legs


def route_signature(path, meta):
    legs = path_legs(path, meta)
    return ">".join(x[0] for x in legs), max(0, len(legs) - 1)


def build_policy_paths(G, code_to_nodes):
    # Full-network alternative generation without per-OD repeated Yen/Dijkstra.
    # Each policy runs single-source Dijkstra once per line-aware origin node.
    # No line, segment, or transfer-count filter is applied.
    source_nodes = sorted({n for vals in code_to_nodes.values() for n in vals if n in G})
    all_maps = {}
    timings = {}
    for policy in ROUTE_POLICIES:
        name = policy[0]
        t0 = time.perf_counter()
        weight = edge_weight(policy)
        maps = {}
        for src in source_nodes:
            maps[src] = nx.single_source_dijkstra_path(G, src, weight=weight)
        all_maps[name] = maps
        timings[name] = time.perf_counter() - t0
    return all_maps, timings


def resolve_route_candidates(G, meta, code_to_nodes, policy_maps, oc, dc):
    origins = code_to_nodes.get(oc, [])
    dests = code_to_nodes.get(dc, [])
    by_nodes = {}
    for policy in ROUTE_POLICIES:
        pname = policy[0]
        maps = policy_maps[pname]
        best = None
        for o in origins:
            pmap = maps.get(o, {})
            for d in dests:
                path = pmap.get(d)
                if not path:
                    continue
                base_cost = path_cost(G, path)
                key = (base_cost, len(path), o, d)
                if best is None or key < best[0]:
                    best = (key, path)
        if best is None:
            continue
        path = best[1]
        sig = tuple(path)
        rf, tc = route_signature(path, meta)
        obj = by_nodes.setdefault(sig, {
            "path": path,
            "line_sequence": rf,
            "transfer_count": tc,
            "base_cost": path_cost(G, path),
            "policies": [],
        })
        obj["policies"].append(pname)
    out = list(by_nodes.values())
    out.sort(key=lambda x: (x["base_cost"], x["transfer_count"], len(x["path"]), x["line_sequence"]))
    return out


def load_passengers(taims_path, code_to_nodes):
    rows = []
    stats = Counter()
    od_mass = Counter()
    with zipfile.ZipFile(taims_path) as z:
        f = g0.zr(z, "VW_KSCC_DX_CARD.csv")
        for r in csv.DictReader(f):
            if str(r.get("TRNS_MNS_CD") or "").strip() not in RAIL:
                continue
            t0 = g0.dt(r.get("RIDE_DTM"))
            t1 = g0.dt(r.get("ALGH_DTM"))
            if not t0 or not t1 or not (START <= t0 < END) or t1 <= t0 or t1 - t0 > timedelta(hours=3):
                continue
            stats["eligible"] += 1
            oc = g0.z4(r.get("RIDE_BSST_ID"))
            dc = g0.z4(r.get("ALGH_BSST_ID"))
            if oc not in code_to_nodes or dc not in code_to_nodes:
                stats["unmapped"] += 1
                continue
            stats["mapped"] += 1
            rows.append((oc, dc, t0, t1))
            od_mass[(oc, dc)] += 1
        f.close()
    return rows, od_mass, stats


def service_trajectories(taims_path, p1c_path, service_path, meta, code_to_nodes):
    trains = g1.load_service_events(service_path, meta)
    entry_hist, exit_hist, entry_mass, exit_mass, afc_stats = g1.load_afc_hist(
        taims_path, code_to_nodes, meta
    )
    models, by_line_train = g1.build_line_models(trains, meta, exit_hist, exit_mass)
    inferred, line_summary = g1.infer_candidates(models, exit_hist, exit_mass)

    trajectories = defaultdict(list)
    observed = 0
    for (line, tr), sts in trains.items():
        direction = by_line_train[line][tr]["direction"]
        events = {}
        for n, ev in sts.items():
            events[n] = {
                "arrival": ev["arrival"],
                "departure": ev["departure"],
            }
        trajectories[line].append({
            "id": f"OBS::{line}::{tr}",
            "line": line,
            "direction": direction,
            "evidence_class": "PARTIAL_DIRECT_SERVICE_ANCHOR",
            "events": events,
        })
        observed += 1

    for obj in inferred:
        events = {}
        for e in obj.get("station_events", []):
            t = datetime.fromisoformat(e["time"])
            events[e["node"]] = {"arrival": t, "departure": t}
        trajectories[obj["line"]].append({
            "id": obj["candidate_id"],
            "line": obj["line"],
            "direction": obj.get("direction"),
            "evidence_class": "AFC_INFERRED_SERVICE_FIELD",
            "events": events,
        })

    for line in trajectories:
        trajectories[line].sort(
            key=lambda x: min((z["departure"] for z in x["events"].values()), default=END)
        )
    return trajectories, {
        "observed_trajectory_count": observed,
        "inferred_trajectory_count": len(inferred),
        "line_direction_summary": line_summary,
        "afc_hist": afc_stats,
    }


def build_segment_ride_cache(trajectories):
    cache = {}

    def rides(line, origin, destination):
        key = (line, origin, destination)
        if key in cache:
            return cache[key]
        out = []
        for tr in trajectories.get(line, []):
            ev = tr["events"]
            if origin not in ev or destination not in ev:
                continue
            dep = ev[origin]["departure"]
            arr = ev[destination]["arrival"]
            if dep < arr:
                out.append((dep, arr, tr["id"], tr["evidence_class"]))
        out.sort(key=lambda x: (x[0], x[1], x[2]))
        cache[key] = out
        return out

    return rides, cache


def first_feasible_chain(route, meta, rides_fn, t_entry, t_exit):
    legs = path_legs(route["path"], meta)
    chain = []
    ready = t_entry
    for line, o, d in legs:
        if o == d:
            # A zero-distance same-line leg can occur when the structural route
            # enters and leaves a line at one physical interchange. It carries no
            # service ride and is retained as a route-structure state.
            chain.append({
                "line": line,
                "origin": o,
                "destination": d,
                "service_id": None,
                "departure": None,
                "arrival": None,
                "evidence_class": "ZERO_DISTANCE_STRUCTURAL_LEG",
            })
            continue
        candidates = rides_fn(line, o, d)
        chosen = None
        for dep, arr, sid, evc in candidates:
            if dep >= ready and arr <= t_exit:
                chosen = (dep, arr, sid, evc)
                break
        if chosen is None:
            return None, {
                "failed_line": line,
                "failed_origin": o,
                "failed_destination": d,
                "available_rides": len(candidates),
            }
        dep, arr, sid, evc = chosen
        chain.append({
            "line": line,
            "origin": o,
            "destination": d,
            "service_id": sid,
            "departure": dep,
            "arrival": arr,
            "evidence_class": evc,
        })
        ready = arr
    if ready >= t_exit:
        return None, {"failed_line": "EGRESS", "available_rides": 0}
    return chain, None


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
    passengers, od_mass, afc_stats = load_passengers(args.taims, code_to_nodes)
    phase["passenger_load_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    policy_maps, policy_timings = build_policy_paths(G, code_to_nodes)
    phase["route_policy_precompute_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    route_cache = {}
    unrouted_od = 0
    route_family_mass = Counter()
    transfer_count_mass = Counter()
    alternative_count = Counter()
    with gzip.open(outdir / "r0_route_candidates_by_od.jsonl.gz", "wt", encoding="utf-8") as f:
        for (oc, dc), mass in od_mass.items():
            cands = resolve_route_candidates(G, meta, code_to_nodes, policy_maps, oc, dc)
            route_cache[(oc, dc)] = cands
            if not cands:
                unrouted_od += 1
            else:
                alternative_count[len(cands)] += mass
                # Each OD contributes its mass to every candidate family only in
                # this candidate-support census; this is not route-assigned mass.
                for c in cands:
                    route_family_mass[c["line_sequence"]] += mass
                    transfer_count_mass[c["transfer_count"]] += mass
            f.write(json.dumps({
                "origin_code": oc,
                "destination_code": dc,
                "passenger_mass": mass,
                "candidates": [
                    {
                        "line_sequence": c["line_sequence"],
                        "transfer_count": c["transfer_count"],
                        "base_cost": c["base_cost"],
                        "policies": c["policies"],
                        "path": c["path"],
                    }
                    for c in cands
                ],
            }, ensure_ascii=False) + "\n")
    phase["route_candidate_build_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    trajectories, service_stats = service_trajectories(
        args.taims, args.p1c, args.service, meta, code_to_nodes
    )
    rides_fn, ride_cache = build_segment_ride_cache(trajectories)
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
            tc = c["transfer_count"]
            max_transfer_seen = max(max_transfer_seen, tc)
            support_by_transfer[tc]["candidate_evaluations"] += 1
            chain, fail = first_feasible_chain(c, meta, rides_fn, t0, t1)
            if chain is not None:
                support_by_transfer[tc]["finite_chain"] += 1
                any_ok = True
                if tc >= 3:
                    high_order_feasible += 1
            else:
                support_by_transfer[tc]["no_chain"] += 1
                if fail:
                    if fail.get("available_rides", 0) == 0:
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
            (outdir / "r0_checkpoint.json").write_text(
                json.dumps(cp, ensure_ascii=False, indent=2), encoding="utf-8"
            )

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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(transfer_rows)

    fam_rows = []
    for rf, mass in route_family_mass.most_common():
        fam_rows.append({
            "line_sequence": rf,
            "transfer_count": max(0, len(rf.split(">")) - 1),
            "candidate_support_mass": mass,
        })
    with (outdir / "r0_route_family_candidate_support.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(fam_rows[0]) if fam_rows else ["line_sequence", "transfer_count", "candidate_support_mass"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(fam_rows)

    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    total_wall = time.perf_counter() - wall0
    result = {
        "schema": "mppd.r0-full-network-factor-engine-qualification.v1",
        "date": "2026-09-04",
        "status": "R0_FULL_NETWORK_FACTOR_ENGINE_QUALIFICATION_COMPLETED",
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
            "policies": [x[0] for x in ROUTE_POLICIES],
            "policy_precompute_sec": policy_timings,
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
            "R0 qualifies the complete-network computational substrate; it is not the final city-day inversion.",
            "Route alternatives are deterministic full-network structural candidates generated under multiple global edge-weight policies; they are not observed passenger routes and not yet a learned route posterior.",
            "The first-feasible chain scan uses zero lower-bound transfer movement only as a computational support test. Theta_K is not estimated here.",
            "Finite-chain share measures current service-field support coverage, not passenger assignment accuracy.",
            "Partial direct service anchors remain partial operational evidence; AFC candidates remain AFC_INFERRED_SERVICE_FIELD.",
            "No raw card identifier is retained.",
        ],
        "next_gate": "Use the R0 route/chain substrate to implement G2 probabilistic route-family and arbitrary-transfer chain posterior with shared Theta_A/Theta_K/Theta_E factors, then run the first complete-network joint update.",
        "no_email_notification_logic": True,
    }

    (outdir / "r0_full_network_factor_engine_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
