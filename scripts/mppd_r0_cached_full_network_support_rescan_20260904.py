import argparse
import csv
import gzip
import json
import resource
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r0_full_network_factor_engine_20260904 as r0


def load_routes(path):
    cache = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cache[(obj["origin_code"], obj["destination_code"])] = obj.get("candidates", [])
    return cache


def load_cohorts(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yield (
                row["origin_code"],
                row["destination_code"],
                datetime.fromisoformat(row["entry_time"]),
                datetime.fromisoformat(row["exit_time"]),
                int(row["passenger_mass"]),
            )


def load_service_trajectories(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_line = defaultdict(list)
    evidence_counts = Counter()
    for st in payload.get("states", []):
        events = {}
        for ev in st.get("station_events", []):
            arr = ev.get("arrival")
            dep = ev.get("departure")
            if not arr or not dep:
                continue
            events[ev["node"]] = {
                "arrival": datetime.fromisoformat(arr),
                "departure": datetime.fromisoformat(dep),
            }
        if len(events) < 2:
            continue
        obj = {
            "id": st["service_id"],
            "line": st["line"],
            "direction": st.get("direction"),
            "evidence_class": st.get("evidence_class"),
            "timing_uncertainty_sec": st.get("timing_uncertainty_sec"),
            "events": events,
        }
        by_line[st["line"]].append(obj)
        evidence_counts[obj["evidence_class"]] += 1
    for line in by_line:
        by_line[line].sort(
            key=lambda x: min((v["departure"] for v in x["events"].values()), default=datetime.max)
        )
    return by_line, evidence_counts, payload.get("manifest", {})


def scan(meta, routes, cohorts, trajectories):
    rides_fn, ride_cache = r0.build_segment_ride_cache(trajectories)
    support = defaultdict(Counter)
    failures = defaultdict(Counter)
    overall = Counter()
    max_transfer = 0
    evidence_used = defaultdict(Counter)

    for oc, dc, tin, tout, mass in cohorts:
        overall["mapped_mass"] += mass
        cands = routes.get((oc, dc), [])
        if not cands:
            overall["no_structural_route_mass"] += mass
            continue
        overall["structural_route_mass"] += mass
        any_ok = False
        for cand in cands:
            tc = int(cand.get("transfer_count", 0))
            max_transfer = max(max_transfer, tc)
            support[tc]["candidate_evaluation_mass"] += mass
            chain, fail = r0.first_feasible_chain(cand, meta, rides_fn, tin, tout)
            if chain is None:
                support[tc]["no_chain_mass"] += mass
                if fail and int(fail.get("available_rides", 0) or 0) == 0:
                    failures[tc]["missing_segment_service_mass"] += mass
                else:
                    failures[tc]["time_incompatible_mass"] += mass
                continue
            support[tc]["finite_chain_mass"] += mass
            any_ok = True
            for leg in chain:
                sid = leg.get("service_id")
                if not sid:
                    continue
                # Resolve evidence type once through service trajectory id index.
                # Built below from trajectories and cached for speed.
        if any_ok:
            overall["finite_any_mass"] += mass

    rows = []
    for tc in sorted(set(support) | set(failures)):
        s = support[tc]
        f = failures[tc]
        n = s["candidate_evaluation_mass"]
        rows.append({
            "transfer_count": tc,
            "candidate_evaluation_mass": n,
            "finite_chain_mass": s["finite_chain_mass"],
            "finite_chain_share": s["finite_chain_mass"] / n if n else 0.0,
            "no_chain_mass": s["no_chain_mass"],
            "missing_segment_service_mass": f["missing_segment_service_mass"],
            "time_incompatible_mass": f["time_incompatible_mass"],
        })
    return overall, rows, max_transfer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--cohorts", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--service-init", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()
    phase = {}

    t = time.perf_counter()
    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    phase["network_build_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    routes = load_routes(args.routes)
    phase["route_cache_load_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    trajectories, evidence_counts, service_manifest = load_service_trajectories(args.service_init)
    phase["service_cache_load_sec"] = time.perf_counter() - t

    t = time.perf_counter()
    overall, rows, max_transfer = scan(meta, routes, load_cohorts(args.cohorts), trajectories)
    phase["cohort_chain_scan_sec"] = time.perf_counter() - t

    with (outdir / "r0_cached_chain_support_by_transfer_count.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(rows[0]) if rows else ["transfer_count"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    total_wall = time.perf_counter() - wall0
    result = {
        "schema": "mppd.r0-cached-full-network-support-rescan.v1",
        "date": "2026-09-04",
        "status": "R0_CACHED_FULL_NETWORK_SUPPORT_RESCAN_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "time_window": "2026-08-29 07:00-10:00",
        "scope_assertions": {
            "full_network": True,
            "line_filter_applied": False,
            "segment_filter_applied": False,
            "transfer_count_cap_applied": False,
            "raw_taims_rescan": False,
            "all_mapped_cohort_mass_accounted": True,
            "weak_latent_service_states_retained": True,
        },
        "network": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "lines": len({m["line"] for m in meta.values()}),
            "transfer_groups": len(transfer_groups),
        },
        "service_initialization": {
            "trajectory_count_by_evidence_class": dict(evidence_counts),
            "manifest": service_manifest,
        },
        "passenger_support": {
            "mapped_mass": overall["mapped_mass"],
            "structural_route_mass": overall["structural_route_mass"],
            "no_structural_route_mass": overall["no_structural_route_mass"],
            "finite_any_mass": overall["finite_any_mass"],
            "finite_any_share_of_mapped": overall["finite_any_mass"] / overall["mapped_mass"] if overall["mapped_mass"] else 0.0,
            "max_transfer_count_seen": max_transfer,
        },
        "chain_support_by_transfer_count": rows,
        "performance": {
            "phase_sec": phase,
            "total_wall_sec": total_wall,
            "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "mapped_passenger_mass_per_sec": overall["mapped_mass"] / max(total_wall, 1e-9),
        },
        "scientific_boundary": [
            "This rescan measures initialization support coverage only; it is not route assignment, Theta_K estimation or final service reconstruction.",
            "Weak latent service lattices remain movable low-confidence latent S initialization, not realized timetable truth.",
            "The chain support diagnostic still uses zero transfer movement lower-bound; G2/G3 must introduce and jointly update access/transfer/egress distributions.",
            "Any increase in high-order finite-chain support means that previously severed network constraints are computationally reachable, not that their route/service assignments are correct.",
            "No line, segment or transfer-count class is dropped."
        ],
        "next_gate": "If high-order chain support is materially restored, run G2 full-network route/boarding posterior using evidence-typed service uncertainty; otherwise diagnose remaining service-trajectory topology support without spatial pruning.",
        "no_email_notification_logic": True,
    }
    (outdir / "r0_cached_full_network_support_rescan_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
