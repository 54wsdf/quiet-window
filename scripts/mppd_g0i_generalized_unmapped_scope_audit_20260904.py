import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch


def norm(v):
    return str(v or "").strip()


def bucket(code):
    try:
        x = int(code)
    except Exception:
        return "NON_NUMERIC"
    lo = (x // 1000) * 1000
    return f"{lo:04d}-{lo+999:04d}"


def entry_record(e):
    return {
        "tier": norm(e.get("tier")),
        "service_subway_id": norm(e.get("service_subway_id")),
        "service_statn_id": norm(e.get("service_statn_id")),
        "service_statn_sn": e.get("service_statn_sn"),
        "dv_name": norm(e.get("dv_name")),
        "service_name": norm(e.get("service_name")),
        "mapping_status": norm(e.get("mapping_status")),
        "confidence": e.get("confidence"),
        "source": e.get("source"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--unmapped-pairs", required=True)
    ap.add_argument("--topology-patch", required=True)
    ap.add_argument("--gtxa-overlay", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.p1c).read_text(encoding="utf-8"))
    by_code = defaultdict(list)
    for e in payload.get("canonical_entries", []):
        c = g0.z4(e.get("out_stn_num"))
        if c:
            by_code[c].append(entry_record(e))

    G, meta, code_to_nodes, _, _, _, _ = g0.build_network(args.p1c)
    apply_topology_patch(G, meta, load_patch(args.topology_patch))
    overlay_result = apply_gtxa_overlay(G, meta, code_to_nodes, load_overlay(args.gtxa_overlay))

    pair_rows = []
    endpoint_touch = Counter()
    endpoint_origin = Counter()
    endpoint_destination = Counter()
    pair_class_mass = Counter()
    pair_class_od = Counter()
    bucket_touch = Counter()
    total_mass = 0

    def diagnose(code):
        if code in code_to_nodes:
            return "MAPPED_AFTER_G0H"
        entries = by_code.get(code, [])
        if not entries:
            return "ABSENT_FROM_P1C"
        tiers = {x["tier"] for x in entries if x["tier"]}
        if tiers and tiers <= {"D_ALIAS_SINGLETON_LINE"}:
            return "P1C_D_TIER_ONLY"
        if any(t in g0.STRICT_TIERS for t in tiers):
            return "STRICT_ENTRY_BUT_NOT_IN_CORRECTED_GRAPH"
        return "P1C_NONSTRICT_OTHER"

    with open(args.unmapped_pairs, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            oc = str(r.get("origin_code") or "").strip()
            dc = str(r.get("destination_code") or "").strip()
            mass = int(r.get("passenger_mass") or 0)
            if mass <= 0:
                continue
            total_mass += mass
            od = diagnose(oc)
            dd = diagnose(dc)
            endpoint_origin[oc] += mass
            endpoint_destination[dc] += mass
            if od != "MAPPED_AFTER_G0H":
                endpoint_touch[oc] += mass
                bucket_touch[bucket(oc)] += mass
            if dd != "MAPPED_AFTER_G0H":
                endpoint_touch[dc] += mass
                bucket_touch[bucket(dc)] += mass
            if od == "MAPPED_AFTER_G0H" and dd != "MAPPED_AFTER_G0H":
                pc = f"ONE_UNMAPPED:{dd}"
            elif dd == "MAPPED_AFTER_G0H" and od != "MAPPED_AFTER_G0H":
                pc = f"ONE_UNMAPPED:{od}"
            else:
                pc = "BOTH_UNMAPPED:" + "+".join(sorted((od, dd)))
            pair_class_mass[pc] += mass
            pair_class_od[pc] += 1
            pair_rows.append({
                "origin_code": oc,
                "destination_code": dc,
                "passenger_mass": mass,
                "origin_diagnosis": od,
                "destination_diagnosis": dd,
                "pair_class": pc,
            })

    code_rows = []
    classification_touch = Counter()
    for code, touch_mass in endpoint_touch.most_common():
        diag = diagnose(code)
        classification_touch[diag] += touch_mass
        entries = by_code.get(code, [])
        tiers = sorted({x["tier"] for x in entries if x["tier"]})
        lines = sorted({x["service_subway_id"] for x in entries if x["service_subway_id"]})
        names = sorted({x["dv_name"] or x["service_name"] for x in entries if x["dv_name"] or x["service_name"]})
        candidate_nodes = sorted({
            g0.node(x["service_subway_id"], x["service_statn_id"])
            for x in entries if x["service_subway_id"] and x["service_statn_id"]
        })
        graph_candidate_nodes = [n for n in candidate_nodes if n in G]
        code_rows.append({
            "code": code,
            "unmapped_endpoint_touch_mass": touch_mass,
            "origin_mass": endpoint_origin[code],
            "destination_mass": endpoint_destination[code],
            "diagnosis": diag,
            "code_bucket": bucket(code),
            "p1c_entry_count": len(entries),
            "p1c_tiers": "|".join(tiers),
            "p1c_lines": "|".join(lines),
            "p1c_names": "|".join(names),
            "candidate_nodes": "|".join(candidate_nodes),
            "candidate_nodes_already_in_corrected_graph": "|".join(graph_candidate_nodes),
        })

    pair_rows.sort(key=lambda x: (-x["passenger_mass"], x["origin_code"], x["destination_code"]))
    result = {
        "schema": "mppd.g0i-generalized-unmapped-scope-audit.v1",
        "date": "2026-09-04",
        "status": "G0I_GENERALIZED_UNMAPPED_AFC_SCOPE_AUDIT_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "input": {
            "unmapped_passenger_mass": total_mass,
            "unmapped_od_count": len(pair_rows),
            "unique_unmapped_endpoint_codes": len(endpoint_touch),
        },
        "pair_class_mass": dict(pair_class_mass),
        "pair_class_od_count": dict(pair_class_od),
        "endpoint_touch_mass_by_diagnosis": dict(classification_touch),
        "endpoint_touch_mass_by_code_bucket": dict(bucket_touch),
        "top_unmapped_endpoint_codes": code_rows[:100],
        "top_unmapped_od_pairs": pair_rows[:100],
        "overlay_context": {
            "schema": overlay_result.get("schema"),
            "gtxa_code_override_count": overlay_result.get("code_override_count"),
            "all_active_components_have_base_network_attachment": overlay_result.get("all_active_components_have_base_network_attachment")
        },
        "scientific_boundary": [
            "Endpoint touch mass can double-count a passenger when both endpoints are unmapped; pair-class mass is the conserved passenger-level residual.",
            "A D-tier P1C candidate is diagnostic evidence only and is not automatically promoted into the corrected denominator.",
            "ABSENT_FROM_P1C does not imply anomalous passenger behavior; it may indicate an operator/station scope boundary or a missing crosswalk source.",
            "Any generalized denominator expansion requires a separate provenance-qualified mapping repair and raw-AFC rebuild."
        ],
        "next_gate": "Use pair-class and high-mass code diagnostics to distinguish recoverable crosswalk defects from true data-scope boundaries; only then freeze the final eligible-AFC denominator.",
        "no_email_notification_logic": True
    }

    with (outdir / "g0i_unmapped_endpoint_code_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(code_rows[0]) if code_rows else ["code"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(code_rows)
    with (outdir / "g0i_unmapped_od_pair_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(pair_rows[0]) if pair_rows else ["origin_code", "destination_code", "passenger_mass"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(pair_rows)
    (outdir / "g0i_generalized_unmapped_scope_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
