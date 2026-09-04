import argparse
import csv
import gzip
import json
import zipfile
from collections import Counter
from datetime import timedelta
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch


def in_gtxa_family(code):
    try:
        x = int(code)
    except Exception:
        return False
    return 9000 <= x <= 9099


def component_map(G):
    comps = list(nx.connected_components(G))
    out = {}
    for i, comp in enumerate(comps):
        for n in comp:
            out[n] = i
    return comps, out


def structurally_reachable(code_to_nodes, component_of, oc, dc):
    os = {component_of[n] for n in code_to_nodes.get(oc, []) if n in component_of}
    ds = {component_of[n] for n in code_to_nodes.get(dc, []) if n in component_of}
    return bool(os & ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taims", required=True)
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--topology-patch", required=True)
    ap.add_argument("--gtxa-overlay", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    base = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "connected_components": nx.number_connected_components(G),
        "strict_code_count": len(code_to_nodes),
    }
    topology_patch = load_patch(args.topology_patch)
    topology_result = apply_topology_patch(G, meta, topology_patch)
    overlay = load_overlay(args.gtxa_overlay)
    overlay_result = apply_gtxa_overlay(G, meta, code_to_nodes, overlay)
    comps, component_of = component_map(G)

    stats = Counter()
    od_mass = Counter()
    cohort_mass = Counter()
    unmapped_pair_mass = Counter()
    unrouted_pair_mass = Counter()
    gt_mapped = Counter()
    gt_pairs = Counter()

    with zipfile.ZipFile(args.taims) as z:
        f = g0.zr(z, "VW_KSCC_DX_CARD.csv")
        for r in csv.DictReader(f):
            if str(r.get("TRNS_MNS_CD") or "").strip() not in g0.RAIL:
                continue
            t0 = g0.dt(r.get("RIDE_DTM"))
            t1 = g0.dt(r.get("ALGH_DTM"))
            if not t0 or not t1 or not (g0.START <= t0 < g0.END) or t1 <= t0 or t1 - t0 > timedelta(hours=3):
                continue
            stats["eligible"] += 1
            oc = g0.z4(r.get("RIDE_BSST_ID"))
            dc = g0.z4(r.get("ALGH_BSST_ID"))
            is_gt = in_gtxa_family(oc) or in_gtxa_family(dc)
            if is_gt:
                stats["gtxa_family_touch"] += 1
                if in_gtxa_family(oc) and in_gtxa_family(dc):
                    stats["gtxa_family_both"] += 1
                    gt_pairs[(oc, dc)] += 1
                else:
                    stats["gtxa_family_cross_family"] += 1
            if oc not in code_to_nodes or dc not in code_to_nodes:
                stats["unmapped"] += 1
                unmapped_pair_mass[(oc, dc)] += 1
                if is_gt:
                    stats["gtxa_unmapped_after_overlay"] += 1
                continue
            stats["mapped"] += 1
            if is_gt:
                stats["gtxa_mapped_after_overlay"] += 1
                gt_mapped[oc] += 1
                gt_mapped[dc] += 1
            od_mass[(oc, dc)] += 1
            cohort_mass[(oc, dc, t0.isoformat(), t1.isoformat())] += 1
        f.close()

    routed_mass = 0
    unrouted_mass = 0
    routed_od = 0
    unrouted_od = 0
    for (oc, dc), mass in od_mass.items():
        if structurally_reachable(code_to_nodes, component_of, oc, dc):
            routed_mass += mass
            routed_od += 1
        else:
            unrouted_mass += mass
            unrouted_od += 1
            unrouted_pair_mass[(oc, dc)] += mass

    with gzip.open(outdir / "seoul_20260829_0700_1000_afc_time_cohorts_g0h.csv.gz", "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["origin_code", "destination_code", "entry_time", "exit_time", "passenger_mass"])
        w.writeheader()
        for (oc, dc, tin, tout), mass in sorted(cohort_mass.items()):
            w.writerow({"origin_code": oc, "destination_code": dc, "entry_time": tin, "exit_time": tout, "passenger_mass": mass})

    with (outdir / "g0h_corrected_od_mass.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["origin_code", "destination_code", "passenger_mass", "structurally_routed"])
        w.writeheader()
        for (oc, dc), mass in sorted(od_mass.items()):
            w.writerow({"origin_code": oc, "destination_code": dc, "passenger_mass": mass, "structurally_routed": structurally_reachable(code_to_nodes, component_of, oc, dc)})

    with (outdir / "g0h_remaining_unmapped_codepairs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["origin_code", "destination_code", "passenger_mass"])
        w.writeheader()
        for (oc, dc), mass in unmapped_pair_mass.most_common():
            w.writerow({"origin_code": oc, "destination_code": dc, "passenger_mass": mass})

    with (outdir / "g0h_structurally_unrouted_codepairs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["origin_code", "destination_code", "passenger_mass"])
        w.writeheader()
        for (oc, dc), mass in unrouted_pair_mass.most_common():
            w.writerow({"origin_code": oc, "destination_code": dc, "passenger_mass": mass})

    old_strict_mapped = 620195
    mapped = int(stats["mapped"])
    result = {
        "schema": "mppd.g0h-gtxa-corrected-cohort-rebuild.v1",
        "date": "2026-09-04",
        "status": "G0H_GTXA_AWARE_RAW_AFC_COHORT_REBUILD_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "window": {"start": g0.START.isoformat(), "end": g0.END.isoformat()},
        "network": {
            "base": base,
            "after_g0e_patch": {
                "patch_schema": topology_result.get("schema"),
                "inserted_edges": topology_result.get("edge_count_inserted")
            },
            "gtxa_overlay": overlay_result,
            "final_nodes": G.number_of_nodes(),
            "final_edges": G.number_of_edges(),
            "final_connected_components": len(comps),
            "component_sizes": sorted((len(c) for c in comps), reverse=True)
        },
        "afc": {
            "eligible_rows": int(stats["eligible"]),
            "mapped_rows_after_g0h": mapped,
            "unmapped_rows_after_g0h": int(stats["unmapped"]),
            "mapped_share_of_eligible": mapped / stats["eligible"] if stats["eligible"] else None,
            "old_strict_p1c_mapped_rows": old_strict_mapped,
            "mapped_denominator_gain": mapped - old_strict_mapped,
            "cohort_count": len(cohort_mass),
            "od_count": len(od_mass)
        },
        "gtxa": {
            "family_touch_rows": int(stats["gtxa_family_touch"]),
            "family_both_endpoint_rows": int(stats["gtxa_family_both"]),
            "family_cross_family_rows": int(stats["gtxa_family_cross_family"]),
            "mapped_after_overlay_rows": int(stats["gtxa_mapped_after_overlay"]),
            "unmapped_after_overlay_rows": int(stats["gtxa_unmapped_after_overlay"]),
            "endpoint_mapping_mass": dict(sorted(gt_mapped.items())),
            "top_od_pairs": [{"origin_code": oc, "destination_code": dc, "passenger_mass": mass} for (oc, dc), mass in gt_pairs.most_common(100)]
        },
        "structural_routing": {
            "mapped_passenger_mass": mapped,
            "routed_passenger_mass": routed_mass,
            "unrouted_passenger_mass": unrouted_mass,
            "routed_share_of_mapped": routed_mass / mapped if mapped else None,
            "routed_od_count": routed_od,
            "unrouted_od_count": unrouted_od
        },
        "qualification_checks": {
            "eligible_mass_conservation": int(stats["eligible"]) == int(stats["mapped"] + stats["unmapped"]),
            "mapped_routing_mass_conservation": mapped == routed_mass + unrouted_mass,
            "all_gtxa_raw_rows_have_both_gtxa_endpoints": stats["gtxa_family_touch"] == stats["gtxa_family_both"],
            "all_gtxa_rows_mapped_after_overlay": stats["gtxa_unmapped_after_overlay"] == 0,
            "north_south_direct_gtxa_edge_forbidden": overlay_result.get("north_south_direct_edge_forbidden") is True
        },
        "scientific_boundary": [
            "G0H regenerates the denominator from raw TAIMS after an explicit line-aware GTX-A crosswalk overlay; it supersedes old strict-P1C cohort denominators for downstream experiments.",
            "The remaining unmapped AFC rows are retained for a generalized mapping-scope audit and are not classified as passenger anomalies.",
            "GTX-A north and south service components are date-aware and receive no direct central through-service edge on 2026-08-29.",
            "Structural routability is graph reachability only and does not assert passenger route truth or service-chain feasibility.",
            "The old 620195-passenger cohort remains an immutable historical baseline for comparison."
        ],
        "next_gate": "Build the 26-policy route ensemble on the exact G0H graph and corrected cohort cache; replace the invalid line-1032 weak service lattice with date-aware north/south initialization before corrected-denominator G2v2.",
        "no_email_notification_logic": True
    }
    (outdir / "g0h_gtxa_corrected_cohort_rebuild_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "g0h_gtxa_overlay_application.json").write_text(json.dumps(overlay_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
