import argparse
import json
from collections import Counter
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
from scripts.mppd_g0c_topology_patch_qualification_20260904 import (
    best_path,
    components,
    patch_edges_used,
    read_unrouted,
    write_csv,
)
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch


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

    G, meta, code_to_nodes, _, _, _, _ = g0.build_network(args.p1c)
    comps_before, comp_before = components(G)
    base_nodes, base_edges = G.number_of_nodes(), G.number_of_edges()
    patch_result = apply_topology_patch(G, meta, patch)
    comps_after, _ = components(G)

    if G.number_of_nodes() != base_nodes:
        raise RuntimeError("G0E changed node count; this qualification is edge-only reconciliation")
    if G.number_of_edges() != base_edges + patch_result["edge_count_inserted"]:
        raise RuntimeError("G0E patch edge accounting mismatch")

    path_cache = {}
    recovered = []
    remaining = []
    recovered_mass = 0
    remaining_mass = 0
    edge_usage_mass = Counter()
    edge_usage_od = Counter()
    direct_vs_transitive = Counter()
    transfer_count_mass = Counter()

    declared_pairs = {
        tuple(sorted(map(int, e.get("prior_component_pair") or [])))
        for e in patch.get("edges") or []
        if len(e.get("prior_component_pair") or []) == 2
    }

    for r in rows:
        oc, dc, mass = r["origin_code"], r["destination_code"], r["passenger_mass"]
        origins = code_to_nodes.get(oc, [])
        dests = code_to_nodes.get(dc, [])
        best = best_path(G, origins, dests, path_cache)
        prior_pairs = sorted(
            {
                tuple(sorted((comp_before[o], comp_before[d])))
                for o in origins
                for d in dests
                if o in comp_before and d in comp_before and comp_before[o] != comp_before[d]
            }
        )
        if best is None:
            remaining_mass += mass
            remaining.append(
                {
                    "origin_code": oc,
                    "destination_code": dc,
                    "passenger_mass": mass,
                    "prior_component_pairs": "|".join(f"{a}-{b}" for a, b in prior_pairs),
                    "status": "STILL_UNROUTED_AFTER_G0E_RECONCILIATION",
                }
            )
            continue

        (_, _, chosen_o, chosen_d), path = best
        used = patch_edges_used(G, path)
        if not used:
            raise RuntimeError(f"previously-unrouted OD recovered without G0C/G0E edge: {oc}->{dc}")
        recovered_mass += mass
        chosen_pair = tuple(sorted((comp_before[chosen_o], comp_before[chosen_d])))
        mode = "DECLARED_COMPONENT_REPAIR" if chosen_pair in declared_pairs else "TRANSITIVE_COMPONENT_MERGE_RECOVERY"
        direct_vs_transitive[mode] += mass
        for edge_id in set(used):
            edge_usage_mass[edge_id] += mass
            edge_usage_od[edge_id] += 1
        seq = g0.path_line_sequence(path, meta)
        transfer_count = max(0, len(seq) - 1)
        transfer_count_mass[transfer_count] += mass
        recovered.append(
            {
                "origin_code": oc,
                "destination_code": dc,
                "passenger_mass": mass,
                "chosen_origin_node": chosen_o,
                "chosen_destination_node": chosen_d,
                "prior_component_pair": f"{chosen_pair[0]}-{chosen_pair[1]}",
                "recovery_mode": mode,
                "patch_edges_used": "|".join(used),
                "line_sequence": ">".join(seq),
                "transfer_count": transfer_count,
                "path_nodes": "|".join(path),
            }
        )

    baseline_unrouted = sum(r["passenger_mass"] for r in rows)
    if recovered_mass + remaining_mass != baseline_unrouted:
        raise RuntimeError("G0E mass conservation failed")

    afc = g0_summary.get("afc") or {}
    mapped_mass = int(afc.get("mapped_endpoint_rows") or 0)
    base_routed_mass = int(afc.get("routed_passenger_mass") or 0)
    declared_unrouted = int(afc.get("unrouted_passenger_mass") or 0)
    if declared_unrouted != baseline_unrouted:
        raise RuntimeError(f"G0 summary/unrouted mismatch: {declared_unrouted} != {baseline_unrouted}")
    new_routed = base_routed_mass + recovered_mass
    if mapped_mass - new_routed != remaining_mass:
        raise RuntimeError("G0E total routed/unrouted accounting mismatch")

    g0e_ids = {e["edge_id"] for e in patch.get("edges") or [] if str(e.get("edge_id", "")).startswith("G0E_")}
    g0c_ids = {e["edge_id"] for e in patch.get("edges") or [] if str(e.get("edge_id", "")).startswith("G0C_")}
    recovered_using_g0e = sum(
        r["passenger_mass"]
        for r in recovered
        if set(str(r["patch_edges_used"]).split("|")) & g0e_ids
    )
    recovered_using_g0c = sum(
        r["passenger_mass"]
        for r in recovered
        if set(str(r["patch_edges_used"]).split("|")) & g0c_ids
    )

    before_sizes = sorted((len(c) for c in comps_before), reverse=True)
    after_sizes = sorted((len(c) for c in comps_after), reverse=True)
    result = {
        "schema": "mppd.g0e-topology-reconciliation-qualification.v1",
        "date": "2026-09-04",
        "status": "G0E_CROSS_LAYER_TOPOLOGY_RECONCILIATION_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "baseline": {
            "mapped_passenger_mass": mapped_mass,
            "routed_passenger_mass": base_routed_mass,
            "unrouted_passenger_mass": baseline_unrouted,
            "routed_share": base_routed_mass / mapped_mass if mapped_mass else None,
            "nodes": base_nodes,
            "edges": base_edges,
            "connected_components": len(comps_before),
            "largest_component_nodes": before_sizes[0] if before_sizes else 0,
        },
        "patch": {
            "schema": patch_result["schema"],
            "inherits_from": patch_result.get("inherits_from"),
            "inherited_schema": patch_result.get("inherited_schema"),
            "inherited_edge_count": patch_result.get("inherited_edge_count"),
            "own_edge_count": patch_result.get("own_edge_count"),
            "expanded_edge_count": patch_result["edge_count_requested"],
            "edge_count_inserted": patch_result["edge_count_inserted"],
            "edge_count_already_present": patch_result["edge_count_already_present"],
            "records": patch_result["records"],
        },
        "reconciled_graph": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "connected_components": len(comps_after),
            "largest_component_nodes": after_sizes[0] if after_sizes else 0,
            "component_sizes_before": before_sizes,
            "component_sizes_after": after_sizes,
        },
        "reroute": {
            "input_unrouted_od_count": len(rows),
            "recovered_od_count": len(recovered),
            "remaining_unrouted_od_count": len(remaining),
            "input_unrouted_passenger_mass": baseline_unrouted,
            "recovered_passenger_mass": recovered_mass,
            "remaining_unrouted_passenger_mass": remaining_mass,
            "fraction_of_original_unrouted_mass_recovered": recovered_mass / baseline_unrouted if baseline_unrouted else None,
            "new_total_routed_passenger_mass": new_routed,
            "new_routed_share_of_mapped_mass": new_routed / mapped_mass if mapped_mass else None,
            "remaining_unrouted_share_of_mapped_mass": remaining_mass / mapped_mass if mapped_mass else None,
            "recovered_mass_using_any_g0e_edge": recovered_using_g0e,
            "recovered_mass_using_any_g0c_edge": recovered_using_g0c,
            "recovery_mode_mass": dict(direct_vs_transitive),
            "patch_edge_path_usage_mass": dict(edge_usage_mass),
            "patch_edge_path_usage_od": dict(edge_usage_od),
            "recovered_transfer_count_mass": {str(k): v for k, v in sorted(transfer_count_mass.items())},
        },
        "qualification_checks": {
            "mass_conservation": True,
            "node_count_preserved": G.number_of_nodes() == base_nodes,
            "patch_edge_accounting": G.number_of_edges() == base_edges + patch_result["edge_count_inserted"],
            "all_recovered_paths_use_reconciliation_edge": True,
            "baseline_history_mutated": False,
        },
        "scientific_boundary": [
            "G0E reconciles route topology with G1C service-state continuity and current physical network order; it does not create observed ATS evidence.",
            "Same-line gap edges are coarse continuity substrates over omitted intermediate line-specific nodes, not claims that intermediate stations do not exist.",
            "Recovered paths remain structural hypotheses rather than passenger ground truth.",
            "Any residual after G0E remains explicitly unrouted and must be diagnosed rather than silently dropped."
        ],
        "next_gate": "If G0E closes the structural residual with clean accounting, rebuild the route ensemble on G0E and run G2v2 on that expanded reconciled route set.",
        "no_email_notification_logic": True,
    }

    write_csv(
        outdir / "g0e_recovered_unrouted_codepairs.csv",
        recovered,
        ["origin_code", "destination_code", "passenger_mass", "chosen_origin_node", "chosen_destination_node", "prior_component_pair", "recovery_mode", "patch_edges_used", "line_sequence", "transfer_count", "path_nodes"],
    )
    write_csv(
        outdir / "g0e_remaining_unrouted_codepairs.csv",
        remaining,
        ["origin_code", "destination_code", "passenger_mass", "prior_component_pairs", "status"],
    )
    (outdir / "g0e_topology_patch_application.json").write_text(json.dumps(patch_result, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "g0e_topology_reconciliation_qualification_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
