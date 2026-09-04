import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch


def read_unrouted(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "origin_code": str(r.get("origin_code") or "").strip(),
                    "destination_code": str(r.get("destination_code") or "").strip(),
                    "passenger_mass": int(r.get("passenger_mass") or 0),
                }
            )
    return rows


def components(G):
    comps = list(nx.connected_components(G))
    component_of = {}
    for i, comp in enumerate(comps):
        for n in comp:
            component_of[n] = i
    return comps, component_of


def best_path(G, origins, dests, cache):
    best = None
    for o in origins:
        if o not in G:
            continue
        if o not in cache:
            cache[o] = nx.single_source_dijkstra(G, o, weight="weight")
        lengths, paths = cache[o]
        for d in dests:
            if d not in paths:
                continue
            path = paths[d]
            key = (float(lengths[d]), len(path), o, d)
            if best is None or key < best[0]:
                best = (key, path)
    return best


def patch_edges_used(G, path):
    ids = []
    for u, v in zip(path, path[1:]):
        data = G.get_edge_data(u, v) or {}
        edge_id = data.get("topology_patch_edge_id")
        if edge_id:
            ids.append(str(edge_id))
    return ids


def write_csv(path, rows, fallback):
    fields = list(rows[0]) if rows else fallback
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        if rows:
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--unrouted", required=True)
    ap.add_argument("--g0-summary", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    g0_summary = json.loads(Path(args.g0_summary).read_text(encoding="utf-8"))
    patch = load_patch(args.patch)
    rows = read_unrouted(args.unrouted)

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    comps_before, comp_before = components(G)
    base_nodes = G.number_of_nodes()
    base_edges = G.number_of_edges()

    patch_result = apply_topology_patch(G, meta, patch)
    comps_after, comp_after = components(G)

    if G.number_of_nodes() != base_nodes:
        raise RuntimeError("topology patch unexpectedly changed node count")
    if G.number_of_edges() != base_edges + patch_result["edge_count_inserted"]:
        raise RuntimeError("topology patch edge accounting mismatch")

    direct_patch_component_pairs = {
        tuple(sorted(map(int, edge.get("prior_component_pair") or [])))
        for edge in patch.get("edges") or []
        if len(edge.get("prior_component_pair") or []) == 2
    }

    path_cache = {}
    recovered_rows = []
    remaining_rows = []
    recovered_mass = 0
    remaining_mass = 0
    recovered_od = 0
    patch_edge_mass = Counter()
    prior_component_pair_mass = Counter()
    recovery_mode_mass = Counter()
    transfer_count_mass = Counter()
    route_family_mass = Counter()

    for r in rows:
        oc = r["origin_code"]
        dc = r["destination_code"]
        mass = r["passenger_mass"]
        origins = code_to_nodes.get(oc, [])
        dests = code_to_nodes.get(dc, [])
        before_pairs = sorted(
            {
                tuple(sorted((comp_before[o], comp_before[d])))
                for o in origins
                for d in dests
                if o in comp_before and d in comp_before and comp_before[o] != comp_before[d]
            }
        )

        best = best_path(G, origins, dests, path_cache)
        if best is None:
            remaining_mass += mass
            remaining_rows.append(
                {
                    "origin_code": oc,
                    "destination_code": dc,
                    "passenger_mass": mass,
                    "prior_component_pairs": "|".join(f"{a}-{b}" for a, b in before_pairs),
                    "status": "STILL_UNROUTED_AFTER_G0C_PATCH",
                }
            )
            continue

        (_, _, chosen_o, chosen_d), path = best
        used = patch_edges_used(G, path)
        if not used:
            raise RuntimeError(f"previously-unrouted OD recovered without using a patch edge: {oc}->{dc}")

        recovered_mass += mass
        recovered_od += 1
        for edge_id in set(used):
            patch_edge_mass[edge_id] += mass

        chosen_pair = tuple(sorted((comp_before[chosen_o], comp_before[chosen_d])))
        prior_component_pair_mass[chosen_pair] += mass
        recovery_mode = (
            "DIRECT_COMPONENT_PAIR_REPAIR"
            if chosen_pair in direct_patch_component_pairs
            else "TRANSITIVE_COMPONENT_MERGE_RECOVERY"
        )
        recovery_mode_mass[recovery_mode] += mass

        seq = g0.path_line_sequence(path, meta)
        route_family = ">".join(seq)
        transfer_count = max(0, len(seq) - 1)
        route_family_mass[route_family] += mass
        transfer_count_mass[transfer_count] += mass

        recovered_rows.append(
            {
                "origin_code": oc,
                "destination_code": dc,
                "passenger_mass": mass,
                "chosen_origin_node": chosen_o,
                "chosen_destination_node": chosen_d,
                "prior_component_pair": f"{chosen_pair[0]}-{chosen_pair[1]}",
                "recovery_mode": recovery_mode,
                "patch_edges_used": "|".join(used),
                "line_sequence": route_family,
                "transfer_count": transfer_count,
                "path_nodes": "|".join(path),
            }
        )

    input_unrouted_mass = sum(r["passenger_mass"] for r in rows)
    if recovered_mass + remaining_mass != input_unrouted_mass:
        raise RuntimeError("unrouted mass conservation failed")

    afc = g0_summary.get("afc") or {}
    mapped_mass = int(afc.get("mapped_endpoint_rows") or 0)
    base_routed_mass = int(afc.get("routed_passenger_mass") or 0)
    declared_base_unrouted = int(afc.get("unrouted_passenger_mass") or 0)
    if declared_base_unrouted and declared_base_unrouted != input_unrouted_mass:
        raise RuntimeError(
            f"G0 summary/unrouted CSV mismatch: {declared_base_unrouted} != {input_unrouted_mass}"
        )
    new_routed_mass = base_routed_mass + recovered_mass
    new_unrouted_mass = mapped_mass - new_routed_mass
    if new_unrouted_mass != remaining_mass:
        raise RuntimeError(
            f"patched routing mass accounting mismatch: summary implies {new_unrouted_mass}, reroute has {remaining_mass}"
        )

    component_sizes_before = sorted((len(x) for x in comps_before), reverse=True)
    component_sizes_after = sorted((len(x) for x in comps_after), reverse=True)

    result = {
        "schema": "mppd.g0c-topology-patch-qualification.v1",
        "date": "2026-09-04",
        "status": "G0C_TOPOLOGY_PATCH_REROUTE_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "baseline": {
            "mapped_passenger_mass": mapped_mass,
            "routed_passenger_mass": base_routed_mass,
            "unrouted_passenger_mass": input_unrouted_mass,
            "routed_share_of_mapped_mass": (base_routed_mass / mapped_mass) if mapped_mass else None,
            "nodes": base_nodes,
            "edges": base_edges,
            "connected_components": len(comps_before),
            "largest_component_nodes": component_sizes_before[0] if component_sizes_before else 0,
        },
        "patch": {
            "schema": patch_result["schema"],
            "status": patch_result["status"],
            "edge_count_requested": patch_result["edge_count_requested"],
            "edge_count_inserted": patch_result["edge_count_inserted"],
            "edge_count_already_present": patch_result["edge_count_already_present"],
            "records": patch_result["records"],
        },
        "patched_graph": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "connected_components": len(comps_after),
            "largest_component_nodes": component_sizes_after[0] if component_sizes_after else 0,
            "component_sizes_before": component_sizes_before,
            "component_sizes_after": component_sizes_after,
        },
        "reroute": {
            "input_unrouted_od_count": len(rows),
            "recovered_od_count": recovered_od,
            "remaining_unrouted_od_count": len(remaining_rows),
            "input_unrouted_passenger_mass": input_unrouted_mass,
            "recovered_passenger_mass": recovered_mass,
            "remaining_unrouted_passenger_mass": remaining_mass,
            "fraction_of_prior_unrouted_mass_recovered": (recovered_mass / input_unrouted_mass) if input_unrouted_mass else None,
            "new_total_routed_passenger_mass": new_routed_mass,
            "new_routed_share_of_mapped_mass": (new_routed_mass / mapped_mass) if mapped_mass else None,
            "remaining_unrouted_share_of_mapped_mass": (remaining_mass / mapped_mass) if mapped_mass else None,
            "recovery_mode_mass": dict(recovery_mode_mass),
            "prior_component_pair_recovered_mass": {
                f"{a}-{b}": mass for (a, b), mass in sorted(prior_component_pair_mass.items())
            },
            "patch_edge_path_usage_mass": dict(patch_edge_mass),
            "recovered_transfer_count_mass": {str(k): v for k, v in sorted(transfer_count_mass.items())},
            "top_recovered_route_families": [
                {"line_sequence": rf, "passenger_mass": mass}
                for rf, mass in route_family_mass.most_common(50)
            ],
        },
        "qualification_checks": {
            "mass_conservation": True,
            "node_count_preserved": G.number_of_nodes() == base_nodes,
            "patch_edge_accounting": G.number_of_edges() == base_edges + patch_result["edge_count_inserted"],
            "all_recovered_paths_use_patch_edge": True,
            "baseline_history_mutated": False,
        },
        "scientific_boundary": [
            "G0C tests whether evidence-qualified missing transfer links explain the structural AFC residual; it does not assert passenger route ground truth.",
            "Recovered OD paths are routing hypotheses on the repaired topology, not observed passenger paths.",
            "Patch edges remain provenance-typed topology hypotheses and are not observed ATS service events.",
            "Any transitive recovery is reported separately from OD mass whose original component pair has direct patch evidence.",
            "The repaired topology can enter route-ensemble and posterior experiments only after this reroute accounting closes cleanly."
        ],
        "next_gate": "If G0C recovers substantial structural mass without accounting violations, rebuild the 26-policy route ensemble on the explicit patch, then rerun uncertainty-aware G2v2 before G3.",
        "no_email_notification_logic": True,
    }

    write_csv(
        outdir / "g0c_recovered_unrouted_codepairs.csv",
        recovered_rows,
        [
            "origin_code", "destination_code", "passenger_mass", "chosen_origin_node",
            "chosen_destination_node", "prior_component_pair", "recovery_mode",
            "patch_edges_used", "line_sequence", "transfer_count", "path_nodes"
        ],
    )
    write_csv(
        outdir / "g0c_remaining_unrouted_codepairs.csv",
        remaining_rows,
        ["origin_code", "destination_code", "passenger_mass", "prior_component_pairs", "status"],
    )
    (outdir / "g0c_topology_patch_application.json").write_text(
        json.dumps(patch_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "g0c_topology_patch_qualification_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
