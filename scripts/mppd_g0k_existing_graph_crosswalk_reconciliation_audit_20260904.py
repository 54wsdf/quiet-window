import argparse
import csv
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch


def norm_name(v):
    s = unicodedata.normalize("NFKC", str(v or "")).strip().casefold()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[·ㆍ\-_.]", "", s)
    return s


def base_name(v):
    s = str(v or "").strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"（[^）]*）", "", s)
    return norm_name(s)


def z4(v):
    return g0.z4(v)


def load_station_dictionary(taims):
    out = {}
    with zipfile.ZipFile(taims) as z:
        f = g0.zr(z, "VW_KSCC_DV_CARD.csv")
        for r in csv.DictReader(f):
            code = z4(r.get("OUT_STN_NUM"))
            if not code:
                continue
            out[code] = {
                "code": code,
                "name_ko": str(r.get("STN_KR_NM") or "").strip(),
                "name_en": str(r.get("STN_ENG_NM") or "").strip(),
                "opening_date": str(r.get("OPENING_DE") or "").strip(),
                "closing_date": str(r.get("CLOSURE_DE") or "").strip(),
                "use_yn": str(r.get("USE_YN") or "").strip(),
                "reference_date": str(r.get("STDR_DE") or "").strip(),
            }
        f.close()
    return out


def load_p1c_entries(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_code = defaultdict(list)
    for e in payload.get("canonical_entries", []):
        code = z4(e.get("out_stn_num"))
        if not code:
            continue
        line = str(e.get("service_subway_id") or "").strip()
        st = str(e.get("service_statn_id") or "").strip()
        by_code[code].append({
            "tier": str(e.get("tier") or "").strip(),
            "line": line,
            "station": st,
            "node": g0.node(line, st) if line and st else "",
            "dv_name": str(e.get("dv_name") or "").strip(),
            "service_name": str(e.get("service_name") or "").strip(),
            "mapping_status": str(e.get("mapping_status") or "").strip(),
        })
    return by_code


def graph_name_indexes(G, meta):
    full = defaultdict(set)
    base = defaultdict(set)
    for n in G.nodes:
        m = meta.get(n, {})
        for raw in (m.get("dv_name"), m.get("service_name")):
            if not raw:
                continue
            full[norm_name(raw)].add(n)
            base[base_name(raw)].add(n)
    return full, base


def node_record(n, meta):
    m = meta.get(n, {})
    return {
        "node": n,
        "line": m.get("line"),
        "station": m.get("station"),
        "seq": m.get("seq"),
        "dv_name": m.get("dv_name"),
        "service_name": m.get("service_name"),
        "tier": m.get("tier"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taims", required=True)
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--unmapped-pairs", required=True)
    ap.add_argument("--topology-patch", required=True)
    ap.add_argument("--gtxa-overlay", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    station_dict = load_station_dictionary(args.taims)
    p1c_by_code = load_p1c_entries(args.p1c)
    G, meta, code_to_nodes, _, _, _, _ = g0.build_network(args.p1c)
    apply_topology_patch(G, meta, load_patch(args.topology_patch))
    apply_gtxa_overlay(G, meta, code_to_nodes, load_overlay(args.gtxa_overlay))
    full_idx, base_idx = graph_name_indexes(G, meta)

    target_rows = []
    with open(args.codes, encoding="utf-8", newline="") as f:
        target_rows = list(csv.DictReader(f))

    code_results = {}
    category_touch = Counter()
    candidate_count_touch = Counter()
    unique_line_touch = Counter()

    for r in target_rows:
        code = str(r.get("code") or "").strip()
        touch_mass = int(r.get("unmapped_endpoint_touch_mass") or 0)
        old_diag = str(r.get("diagnosis") or "").strip()
        dv = station_dict.get(code, {})
        p1c_entries = p1c_by_code.get(code, [])
        d_entries = [e for e in p1c_entries if e.get("tier") == "D_ALIAS_SINGLETON_LINE"]
        d_nodes_in_graph = sorted({e["node"] for e in d_entries if e.get("node") in G})

        exact_nodes = sorted(full_idx.get(norm_name(dv.get("name_ko")), set())) if dv.get("name_ko") else []
        base_nodes = sorted(base_idx.get(base_name(dv.get("name_ko")), set())) if dv.get("name_ko") else []

        if d_nodes_in_graph:
            category = "P1C_D_TIER_LINE_AWARE_NODE_ALREADY_IN_GRAPH"
            qualified_nodes = d_nodes_in_graph
            evidence = ["P1C_D_TIER_LINE_AWARE_NODE", "TAIMS_DV_CARD_STATION_IDENTITY"]
        elif exact_nodes:
            category = "TAIMS_EXACT_NAME_MATCH_EXISTING_GRAPH"
            qualified_nodes = exact_nodes
            evidence = ["TAIMS_DV_CARD_STATION_IDENTITY", "EXACT_KOREAN_STATION_NAME_MATCH_TO_CORRECTED_GRAPH"]
        elif base_nodes:
            category = "TAIMS_BASE_NAME_MATCH_EXISTING_GRAPH_REVIEW_REQUIRED"
            qualified_nodes = []
            evidence = ["TAIMS_DV_CARD_STATION_IDENTITY", "PARENTHETICAL_STRIPPED_NAME_MATCH_ONLY"]
        else:
            category = "TAIMS_IDENTIFIED_STATION_NO_EXISTING_GRAPH_NODE"
            qualified_nodes = []
            evidence = ["TAIMS_DV_CARD_STATION_IDENTITY"]

        qualified_lines = sorted({str(meta[n].get("line")) for n in qualified_nodes})
        category_touch[category] += touch_mass
        candidate_count_touch[len(qualified_nodes)] += touch_mass
        if len(qualified_lines) == 1:
            unique_line_touch[qualified_lines[0]] += touch_mass

        result = {
            "code": code,
            "touch_mass": touch_mass,
            "g0i_diagnosis": old_diag,
            "taims_name_ko": dv.get("name_ko"),
            "taims_name_en": dv.get("name_en"),
            "taims_use_yn": dv.get("use_yn"),
            "taims_reference_date": dv.get("reference_date"),
            "category": category,
            "evidence": evidence,
            "qualified_candidate_nodes": qualified_nodes,
            "qualified_candidate_node_records": [node_record(n, meta) for n in qualified_nodes],
            "qualified_candidate_lines": qualified_lines,
            "exact_name_candidate_nodes": exact_nodes,
            "base_name_review_candidate_nodes": base_nodes,
            "p1c_entries": p1c_entries,
        }
        code_results[code] = result

    pair_class = Counter()
    pair_count = Counter()
    potential_mass = 0
    blocked_mass = 0
    pair_rows = []
    with open(args.unmapped_pairs, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            oc = str(r.get("origin_code") or "").strip()
            dc = str(r.get("destination_code") or "").strip()
            mass = int(r.get("passenger_mass") or 0)
            if mass <= 0:
                continue
            o_now = oc in code_to_nodes
            d_now = dc in code_to_nodes
            o_new = bool(code_results.get(oc, {}).get("qualified_candidate_nodes"))
            d_new = bool(code_results.get(dc, {}).get("qualified_candidate_nodes"))
            o_resolved = o_now or o_new
            d_resolved = d_now or d_new
            if o_resolved and d_resolved:
                cls = "FULLY_RECOVERABLE_BY_EXISTING_GRAPH_CROSSWALK_RECONCILIATION"
                potential_mass += mass
            else:
                missing = []
                if not o_resolved:
                    missing.append(code_results.get(oc, {}).get("category", "UNKNOWN_ORIGIN"))
                if not d_resolved:
                    missing.append(code_results.get(dc, {}).get("category", "UNKNOWN_DESTINATION"))
                cls = "BLOCKED_BY_" + "+".join(sorted(missing))
                blocked_mass += mass
            pair_class[cls] += mass
            pair_count[cls] += 1
            pair_rows.append({
                "origin_code": oc,
                "destination_code": dc,
                "passenger_mass": mass,
                "origin_currently_mapped": o_now,
                "destination_currently_mapped": d_now,
                "origin_new_qualified": o_new,
                "destination_new_qualified": d_new,
                "reconciliation_class": cls,
            })

    total_pair_mass = sum(pair_class.values())
    result = {
        "schema": "mppd.g0k-existing-graph-crosswalk-reconciliation-audit.v1",
        "date": "2026-09-04",
        "status": "G0K_EXISTING_GRAPH_CROSSWALK_RECONCILIATION_AUDIT_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "input": {
            "target_code_count": len(code_results),
            "residual_passenger_mass": total_pair_mass,
            "corrected_graph_nodes": G.number_of_nodes(),
            "corrected_graph_edges": G.number_of_edges(),
        },
        "endpoint_touch_mass_by_category_nonconserved": dict(category_touch),
        "endpoint_touch_mass_by_qualified_candidate_count_nonconserved": {str(k): v for k, v in sorted(candidate_count_touch.items())},
        "endpoint_touch_mass_with_unique_candidate_line_nonconserved": dict(unique_line_touch),
        "passenger_level_reconciliation_mass": dict(pair_class),
        "passenger_level_reconciliation_od_count": dict(pair_count),
        "fully_recoverable_passenger_mass": potential_mass,
        "blocked_passenger_mass": blocked_mass,
        "fully_recoverable_share_of_residual": potential_mass / total_pair_mass if total_pair_mass else None,
        "code_results": list(sorted(code_results.values(), key=lambda x: (-x["touch_mass"], x["code"]))),
        "scientific_boundary": [
            "A D-tier line-aware candidate is promoted only for audit when its exact P1C candidate node already exists in the corrected graph and TAIMS independently confirms the station code identity.",
            "Exact station-name reconciliation to existing graph nodes is a crosswalk hypothesis, not passenger route truth; multiple same-name line nodes remain an explicit candidate set.",
            "Parenthetical-stripped name matches are review-only and are not counted as qualified recoverability.",
            "Stations with no existing corrected-graph name match require a missing-line/operator layer and are not force-mapped to a nearby or same-name node.",
            "This audit quantifies recoverability but does not mutate the raw-AFC denominator."
        ],
        "next_gate": "Apply only the qualified existing-graph code reconciliation as an explicit G0K overlay, rebuild the raw-AFC denominator, then classify the still-blocked station families into missing operator/line layers.",
        "no_email_notification_logic": True,
    }

    (outdir / "g0k_existing_graph_crosswalk_reconciliation_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    code_rows = []
    for x in result["code_results"]:
        code_rows.append({
            "code": x["code"], "touch_mass": x["touch_mass"], "g0i_diagnosis": x["g0i_diagnosis"],
            "taims_name_ko": x["taims_name_ko"], "taims_name_en": x["taims_name_en"], "category": x["category"],
            "qualified_candidate_nodes": "|".join(x["qualified_candidate_nodes"]),
            "qualified_candidate_lines": "|".join(x["qualified_candidate_lines"]),
            "exact_name_candidate_nodes": "|".join(x["exact_name_candidate_nodes"]),
            "base_name_review_candidate_nodes": "|".join(x["base_name_review_candidate_nodes"]),
        })
    with (outdir / "g0k_code_reconciliation_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(code_rows[0]) if code_rows else ["code"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(code_rows)
    with (outdir / "g0k_residual_pair_reconciliation.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(pair_rows[0]) if pair_rows else ["origin_code", "destination_code", "passenger_mass"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(pair_rows)
    print(json.dumps({
        "status": result["status"],
        "input": result["input"],
        "endpoint_touch_mass_by_category_nonconserved": result["endpoint_touch_mass_by_category_nonconserved"],
        "passenger_level_reconciliation_mass": result["passenger_level_reconciliation_mass"],
        "fully_recoverable_passenger_mass": potential_mass,
        "blocked_passenger_mass": blocked_mass,
        "fully_recoverable_share_of_residual": result["fully_recoverable_share_of_residual"],
        "top_codes": [{"code": x["code"], "mass": x["touch_mass"], "name": x["taims_name_ko"], "category": x["category"], "nodes": x["qualified_candidate_nodes"]} for x in result["code_results"][:30]],
        "no_email_notification_logic": True
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
