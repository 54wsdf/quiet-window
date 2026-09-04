import argparse
import json
from collections import defaultdict
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0


def norm(v):
    return str(v or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--codes", nargs="+", default=["9000", "9001"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = json.loads(Path(args.p1c).read_text(encoding="utf-8"))
    G, meta, code_to_nodes, _, _, _, _ = g0.build_network(args.p1c)
    codes = {str(x).strip() for x in args.codes}

    canonical = []
    all_by_name = defaultdict(list)
    all_by_service_station = defaultdict(list)
    for e in payload.get("canonical_entries", []):
        rec = {
            "out_stn_num": norm(e.get("out_stn_num")),
            "tier": norm(e.get("tier")),
            "service_subway_id": norm(e.get("service_subway_id")),
            "service_statn_id": norm(e.get("service_statn_id")),
            "service_statn_sn": e.get("service_statn_sn"),
            "dv_name": norm(e.get("dv_name")),
            "service_name": norm(e.get("service_name")),
            "confidence": e.get("confidence"),
            "mapping_status": norm(e.get("mapping_status")),
            "source": e.get("source"),
        }
        if rec["dv_name"]:
            all_by_name[rec["dv_name"]].append(rec)
        if rec["service_name"]:
            all_by_name[rec["service_name"]].append(rec)
        if rec["service_subway_id"] and rec["service_statn_id"]:
            all_by_service_station[(rec["service_subway_id"], rec["service_statn_id"])].append(rec)
        if rec["out_stn_num"] in codes:
            canonical.append(rec)

    code_results = {}
    for code in sorted(codes):
        entries = [x for x in canonical if x["out_stn_num"] == code]
        graph_nodes = list(code_to_nodes.get(code, []))
        graph_node_meta = []
        for n in graph_nodes:
            graph_node_meta.append({"node": n, **meta.get(n, {})})
        peer_records = []
        seen = set()
        for e in entries:
            for name in (e["dv_name"], e["service_name"]):
                if not name:
                    continue
                for peer in all_by_name.get(name, []):
                    key = json.dumps(peer, ensure_ascii=False, sort_keys=True)
                    if key not in seen:
                        seen.add(key)
                        peer_records.append(peer)
        code_results[code] = {
            "canonical_entry_count": len(entries),
            "canonical_entries": entries,
            "graph_nodes": graph_nodes,
            "graph_node_meta": graph_node_meta,
            "same_name_peer_entries": peer_records,
        }

    result = {
        "schema": "mppd.g0f-residual-station-code-audit.v1",
        "date": "2026-09-04",
        "status": "G0F_FINAL_STRUCTURAL_RESIDUAL_STATION_CODE_AUDIT_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "codes": code_results,
        "scientific_boundary": [
            "This audit diagnoses station-code/crosswalk provenance only and does not alter topology.",
            "A code mapped to a different line than expected must not be silently repaired without evidence from the crosswalk and date-specific network state.",
            "The final 145-passenger residual remains explicit until this mapping audit is interpreted."
        ],
        "no_email_notification_logic": True,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g0f_residual_station_code_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
