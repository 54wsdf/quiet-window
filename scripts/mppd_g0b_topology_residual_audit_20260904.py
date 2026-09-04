import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0


def norm_text(value):
    s = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[\(\)\[\]\{\}·ㆍ\-\_.,/\\]", "", s)
    return s


def z4(value):
    digits = "".join(c for c in str(value or "") if c.isdigit())
    return digits.zfill(4) if digits and len(digits) <= 4 else (digits[-4:] if digits else "")


def parse_pipe_ints(value):
    out = []
    for x in str(value or "").split("|"):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except ValueError:
            pass
    return sorted(set(out))


def read_unrouted(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "origin_code": str(r.get("origin_code") or "").strip(),
                    "destination_code": str(r.get("destination_code") or "").strip(),
                    "passenger_mass": int(r.get("passenger_mass") or 0),
                    "origin_candidates": str(r.get("origin_candidates") or ""),
                    "destination_candidates": str(r.get("destination_candidates") or ""),
                    "origin_components": parse_pipe_ints(r.get("origin_components")),
                    "destination_components": parse_pipe_ints(r.get("destination_components")),
                }
            )
    return rows


def current_node_entries(p1c, G):
    by_node = defaultdict(list)
    for x in p1c.get("canonical_entries", []):
        line = str(x.get("service_subway_id") or "").strip()
        st = str(x.get("service_statn_id") or "").strip()
        if not line or not st:
            continue
        n = g0.node(line, st)
        if n in G:
            by_node[n].append(x)
    return by_node


def evidence_keys(entry):
    keys = []
    dv = norm_text(entry.get("dv_name"))
    sn = norm_text(entry.get("service_name"))
    code = z4(entry.get("out_stn_num"))
    if dv:
        keys.append(("dv_name", dv))
    if sn:
        keys.append(("service_name", sn))
    if code:
        keys.append(("external_station_code", code))
    return keys


def bridge_evidence(by_node, meta, component_of):
    key_nodes = defaultdict(set)
    node_key_display = defaultdict(dict)
    for n, entries in by_node.items():
        for x in entries:
            for kind, key in evidence_keys(x):
                key_nodes[(kind, key)].add(n)
                node_key_display[n][(kind, key)] = str(
                    x.get("dv_name") if kind == "dv_name"
                    else x.get("service_name") if kind == "service_name"
                    else x.get("out_stn_num")
                    or ""
                ).strip()

    pair_evidence = defaultdict(set)
    for (kind, key), nodes in key_nodes.items():
        nodes = sorted(nodes)
        if len(nodes) < 2:
            continue
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                if meta[u]["line"] == meta[v]["line"]:
                    continue
                cu = component_of[u]
                cv = component_of[v]
                if cu == cv:
                    continue
                a, b = sorted((u, v))
                pair_evidence[(a, b)].add((kind, key))

    candidates = []
    for (u, v), evidence in pair_evidence.items():
        cu, cv = component_of[u], component_of[v]
        kinds = {kind for kind, _ in evidence}
        strength = (
            (4 if "dv_name" in kinds else 0)
            + (3 if "external_station_code" in kinds else 0)
            + (2 if "service_name" in kinds else 0)
            + max(0, len(evidence) - len(kinds))
        )
        candidates.append(
            {
                "node_u": u,
                "node_v": v,
                "line_u": meta[u]["line"],
                "line_v": meta[v]["line"],
                "station_u": meta[u]["station"],
                "station_v": meta[v]["station"],
                "component_u": cu,
                "component_v": cv,
                "evidence_strength": strength,
                "evidence": sorted(
                    {
                        f"{kind}:{node_key_display[u].get((kind, key), key)}"
                        for kind, key in evidence
                    }
                    | {
                        f"{kind}:{node_key_display[v].get((kind, key), key)}"
                        for kind, key in evidence
                    }
                ),
            }
        )
    return candidates


def od_recovered_by_component_bridge(row, a, b):
    os = set(row["origin_components"])
    ds = set(row["destination_components"])
    if not os or not ds:
        return False
    return bool((a in os and b in ds) or (b in os and a in ds))


def component_pair(row):
    pairs = []
    for a in row["origin_components"]:
        for b in row["destination_components"]:
            if a != b:
                pairs.append(tuple(sorted((a, b))))
    return sorted(set(pairs))


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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    p1c = json.loads(Path(args.p1c).read_text(encoding="utf-8"))
    rows = read_unrouted(args.unrouted)

    components = list(nx.connected_components(G))
    component_of = {}
    for i, comp in enumerate(components):
        for n in comp:
            component_of[n] = i

    by_node = current_node_entries(p1c, G)
    candidates = bridge_evidence(by_node, meta, component_of)

    component_bridge_to_candidates = defaultdict(list)
    for c in candidates:
        cp = tuple(sorted((c["component_u"], c["component_v"])))
        component_bridge_to_candidates[cp].append(c)

    pair_mass = Counter()
    pair_od = Counter()
    for r in rows:
        for cp in component_pair(r):
            pair_mass[cp] += r["passenger_mass"]
            pair_od[cp] += 1

    for c in candidates:
        a, b = sorted((c["component_u"], c["component_v"]))
        affected = [r for r in rows if od_recovered_by_component_bridge(r, a, b)]
        c["single_bridge_recoverable_mass"] = sum(r["passenger_mass"] for r in affected)
        c["single_bridge_recoverable_od"] = len(affected)
        c["evidence"] = "|".join(c["evidence"])

    candidates.sort(
        key=lambda c: (
            -c["single_bridge_recoverable_mass"],
            -c["evidence_strength"],
            c["component_u"],
            c["component_v"],
            c["node_u"],
            c["node_v"],
        )
    )

    attributed = []
    directly_bridgeable_mass = 0
    same_component_bug_mass = 0
    no_evidence_bridge_mass = 0
    missing_component_mass = 0

    for r in rows:
        cps = component_pair(r)
        bridge_cps = [cp for cp in cps if cp in component_bridge_to_candidates]
        same = bool(set(r["origin_components"]) & set(r["destination_components"]))
        if same:
            reason = "SAME_COMPONENT_ROUTING_INCONSISTENCY"
            same_component_bug_mass += r["passenger_mass"]
        elif not r["origin_components"] or not r["destination_components"]:
            reason = "ENDPOINT_COMPONENT_PARSE_OR_MAPPING_GAP"
            missing_component_mass += r["passenger_mass"]
        elif bridge_cps:
            reason = "CROSS_COMPONENT_WITH_INTERNAL_BRIDGE_EVIDENCE"
            directly_bridgeable_mass += r["passenger_mass"]
        else:
            reason = "CROSS_COMPONENT_NO_INTERNAL_BRIDGE_EVIDENCE"
            no_evidence_bridge_mass += r["passenger_mass"]
        attributed.append(
            {
                **r,
                "origin_components": "|".join(map(str, r["origin_components"])),
                "destination_components": "|".join(map(str, r["destination_components"])),
                "attribution": reason,
                "candidate_component_bridges": "|".join(f"{a}-{b}" for a, b in bridge_cps),
            }
        )

    component_rows = []
    for i, comp in enumerate(components):
        lines = sorted({meta[n]["line"] for n in comp})
        component_rows.append(
            {
                "component": i,
                "node_count": len(comp),
                "line_count": len(lines),
                "lines": "|".join(lines),
                "nodes": "|".join(sorted(comp)),
            }
        )
    component_rows.sort(key=lambda x: (-x["node_count"], x["component"]))

    pair_rows = []
    for cp, mass in pair_mass.most_common():
        cands = component_bridge_to_candidates.get(cp, [])
        pair_rows.append(
            {
                "component_a": cp[0],
                "component_b": cp[1],
                "unrouted_passenger_mass": mass,
                "unrouted_od_count": pair_od[cp],
                "internal_bridge_candidate_count": len(cands),
                "best_evidence_strength": max((c["evidence_strength"] for c in cands), default=0),
            }
        )

    patch_hypotheses = []
    seen_component_pair = set()
    for c in candidates:
        cp = tuple(sorted((c["component_u"], c["component_v"])))
        if cp in seen_component_pair:
            continue
        seen_component_pair.add(cp)
        patch_hypotheses.append(
            {
                "component_pair": list(cp),
                "u": c["node_u"],
                "v": c["node_v"],
                "line_u": c["line_u"],
                "line_v": c["line_v"],
                "evidence": c["evidence"].split("|") if c["evidence"] else [],
                "evidence_strength": c["evidence_strength"],
                "single_bridge_recoverable_mass": c["single_bridge_recoverable_mass"],
                "single_bridge_recoverable_od": c["single_bridge_recoverable_od"],
                "evidence_class": "AFC_SUPPORTED_INTERNAL_TOPOLOGY_HYPOTHESIS",
                "automatic_apply": False,
            }
        )

    total_unrouted_mass = sum(r["passenger_mass"] for r in rows)
    result = {
        "schema": "mppd.g0b-topology-residual-audit.v1",
        "date": "2026-09-04",
        "status": "G0B_TOPOLOGY_RESIDUAL_AUDIT_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "input": {
            "unrouted_od_count": len(rows),
            "unrouted_passenger_mass": total_unrouted_mass,
        },
        "graph": {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "component_count": len(components),
            "largest_component_nodes": max((len(c) for c in components), default=0),
            "transfer_group_count": len(transfer_groups),
            "ambiguous_sequence_group_count": len(ambiguous_seq),
            "ambiguous_external_code_count": len(ambiguous_codes),
            **graph_build,
        },
        "attribution_mass": {
            "cross_component_with_internal_bridge_evidence": directly_bridgeable_mass,
            "cross_component_no_internal_bridge_evidence": no_evidence_bridge_mass,
            "same_component_routing_inconsistency": same_component_bug_mass,
            "endpoint_component_parse_or_mapping_gap": missing_component_mass,
        },
        "bridge_hypotheses": {
            "node_pair_candidate_count": len(candidates),
            "component_pair_candidate_count": len(seen_component_pair),
            "top_component_pair_single_bridge_recoverable_mass": [
                {
                    "component_pair": h["component_pair"],
                    "single_bridge_recoverable_mass": h["single_bridge_recoverable_mass"],
                    "single_bridge_recoverable_od": h["single_bridge_recoverable_od"],
                    "evidence": h["evidence"],
                }
                for h in patch_hypotheses[:20]
            ],
        },
        "scientific_boundary": [
            "Unrouted AFC rows are treated as topology residual evidence, not passenger anomalies.",
            "Candidate bridges are hypotheses derived only from cross-component identity evidence already present in P1C canonical entries.",
            "No candidate bridge is automatically inserted into the network.",
            "Any topology repair must be provenance-typed and requalified against route coverage before downstream posterior inference.",
        ],
        "next_gate": "Review high-mass bridge hypotheses, apply only evidence-qualified topology patches, rebuild the route ensemble, then rerun G2v2.",
        "no_email_notification_logic": True,
    }

    write_csv(
        outdir / "g0b_topology_bridge_candidates.csv",
        candidates,
        [
            "node_u",
            "node_v",
            "line_u",
            "line_v",
            "station_u",
            "station_v",
            "component_u",
            "component_v",
            "evidence_strength",
            "evidence",
            "single_bridge_recoverable_mass",
            "single_bridge_recoverable_od",
        ],
    )
    write_csv(
        outdir / "g0b_unrouted_attribution.csv",
        attributed,
        [
            "origin_code",
            "destination_code",
            "passenger_mass",
            "origin_candidates",
            "destination_candidates",
            "origin_components",
            "destination_components",
            "attribution",
            "candidate_component_bridges",
        ],
    )
    write_csv(
        outdir / "g0b_component_summary.csv",
        component_rows,
        ["component", "node_count", "line_count", "lines", "nodes"],
    )
    write_csv(
        outdir / "g0b_component_pair_residual_mass.csv",
        pair_rows,
        [
            "component_a",
            "component_b",
            "unrouted_passenger_mass",
            "unrouted_od_count",
            "internal_bridge_candidate_count",
            "best_evidence_strength",
        ],
    )
    (outdir / "g0b_topology_patch_hypotheses.json").write_text(
        json.dumps(patch_hypotheses, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "g0b_topology_residual_audit_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
