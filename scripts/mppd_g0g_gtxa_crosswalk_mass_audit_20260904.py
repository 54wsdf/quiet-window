import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0


def norm(v):
    return str(v or "").strip()


def in_gtxa_code_family(code):
    try:
        x = int(code)
    except Exception:
        return False
    return 9000 <= x <= 9099


def canonical_record(e):
    return {
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taims", required=True)
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(args.p1c).read_text(encoding="utf-8"))
    G, meta, code_to_nodes, _, _, _, _ = g0.build_network(args.p1c)

    canonical_by_code = defaultdict(list)
    line1032_entries = []
    for e in payload.get("canonical_entries", []):
        rec = canonical_record(e)
        if rec["out_stn_num"]:
            canonical_by_code[rec["out_stn_num"]].append(rec)
        if rec["service_subway_id"] == "1032":
            line1032_entries.append(rec)

    endpoint_mass = Counter()
    ride_mass = Counter()
    alight_mass = Counter()
    pair_mass = Counter()
    family_pair_mass = Counter()
    eligible = 0
    family_touch_rows = 0
    family_both_rows = 0
    family_touch_duration_sec = []

    with zipfile.ZipFile(args.taims) as z:
        f = g0.zr(z, "VW_KSCC_DX_CARD.csv")
        for r in csv.DictReader(f):
            if norm(r.get("TRNS_MNS_CD")) not in g0.RAIL:
                continue
            t0 = g0.dt(r.get("RIDE_DTM"))
            t1 = g0.dt(r.get("ALGH_DTM"))
            if not t0 or not t1 or not (g0.START <= t0 < g0.END) or t1 <= t0 or t1 - t0 > timedelta(hours=3):
                continue
            eligible += 1
            oc = g0.z4(r.get("RIDE_BSST_ID"))
            dc = g0.z4(r.get("ALGH_BSST_ID"))
            oh = in_gtxa_code_family(oc)
            dh = in_gtxa_code_family(dc)
            if not (oh or dh):
                continue
            family_touch_rows += 1
            if oh and dh:
                family_both_rows += 1
            if oh:
                endpoint_mass[oc] += 1
                ride_mass[oc] += 1
            if dh:
                endpoint_mass[dc] += 1
                alight_mass[dc] += 1
            pair_mass[(oc, dc)] += 1
            family_pair_mass[(oc if oh else "NON_FAMILY", dc if dh else "NON_FAMILY")] += 1
            family_touch_duration_sec.append(int((t1 - t0).total_seconds()))
        f.close()

    seen_codes = sorted(set(endpoint_mass) | {c for c in canonical_by_code if in_gtxa_code_family(c)})
    code_rows = []
    code_detail = {}
    for code in seen_codes:
        entries = canonical_by_code.get(code, [])
        graph_nodes = list(code_to_nodes.get(code, []))
        tiers = sorted({x["tier"] for x in entries if x["tier"]})
        mapped_lines = sorted({x["service_subway_id"] for x in entries if x["service_subway_id"]})
        names = sorted({x["dv_name"] or x["service_name"] for x in entries if x["dv_name"] or x["service_name"]})
        row = {
            "code": code,
            "endpoint_mass": endpoint_mass[code],
            "ride_mass": ride_mass[code],
            "alight_mass": alight_mass[code],
            "strict_graph_mapped": bool(graph_nodes),
            "strict_graph_nodes": "|".join(graph_nodes),
            "tiers": "|".join(tiers),
            "mapped_lines": "|".join(mapped_lines),
            "names": "|".join(names),
            "canonical_entry_count": len(entries),
        }
        code_rows.append(row)
        code_detail[code] = {**row, "canonical_entries": entries}

    pair_rows = [
        {"origin_code": oc, "destination_code": dc, "passenger_mass": mass,
         "origin_family": in_gtxa_code_family(oc), "destination_family": in_gtxa_code_family(dc),
         "origin_strict_mapped": oc in code_to_nodes, "destination_strict_mapped": dc in code_to_nodes}
        for (oc, dc), mass in pair_mass.most_common()
    ]

    family_unmapped_endpoint_touch_rows = sum(
        mass for (oc, dc), mass in pair_mass.items()
        if (in_gtxa_code_family(oc) and oc not in code_to_nodes) or (in_gtxa_code_family(dc) and dc not in code_to_nodes)
    )
    family_wrong_line_suspect_touch_rows = sum(
        mass for (oc, dc), mass in pair_mass.items()
        if any(
            in_gtxa_code_family(c)
            and c in code_to_nodes
            and not any(str(n).startswith("1032|") for n in code_to_nodes.get(c, []))
            for c in (oc, dc)
        )
    )

    result = {
        "schema": "mppd.g0g-gtxa-crosswalk-mass-audit.v1",
        "date": "2026-09-04",
        "status": "G0G_GTXA_STATION_CODE_FAMILY_RAW_AFC_AUDIT_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "window": {"start": g0.START.isoformat(), "end": g0.END.isoformat()},
        "raw_afc": {
            "eligible_rows": eligible,
            "gtxa_family_touch_rows": family_touch_rows,
            "gtxa_family_both_endpoint_rows": family_both_rows,
            "gtxa_family_unmapped_endpoint_touch_rows_under_strict_p1c": family_unmapped_endpoint_touch_rows,
            "gtxa_family_wrong_line_suspect_touch_rows_under_strict_p1c": family_wrong_line_suspect_touch_rows,
            "gtxa_family_seen_codes": seen_codes,
            "gtxa_family_endpoint_mass": dict(sorted(endpoint_mass.items())),
            "duration_sec_median": (sorted(family_touch_duration_sec)[len(family_touch_duration_sec)//2] if family_touch_duration_sec else None)
        },
        "code_detail": code_detail,
        "line_1032_canonical_entries": line1032_entries,
        "top_family_pairs": [
            {"origin": a, "destination": b, "passenger_mass": m}
            for (a, b), m in family_pair_mass.most_common(100)
        ],
        "scientific_boundary": [
            "The 9000-9099 scan is a station-code family audit against raw AFC, not an assumption that every code in the numerical range is GTX-A.",
            "A GTX-A classification requires convergence of contiguous-code behavior, station names, line-1032 evidence and official route order.",
            "D-tier or wrong-line same-name mappings are counted as crosswalk diagnostics and are not silently promoted into the graph.",
            "Any repaired denominator must be regenerated from raw TAIMS before downstream coverage or posterior percentages are declared final."
        ],
        "next_gate": "Use raw family mass and line-1032 canonical evidence to define a provenance-qualified GTX-A crosswalk repair; regenerate the AFC cohort cache and rerun topology/route/posterior qualification on the corrected denominator.",
        "no_email_notification_logic": True
    }

    with open(outdir / "g0g_gtxa_code_mass.csv", "w", encoding="utf-8", newline="") as f:
        fields = list(code_rows[0]) if code_rows else ["code"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(code_rows)
    with open(outdir / "g0g_gtxa_touch_od_pairs.csv", "w", encoding="utf-8", newline="") as f:
        fields = list(pair_rows[0]) if pair_rows else ["origin_code", "destination_code", "passenger_mass"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(pair_rows)
    (outdir / "g0g_gtxa_crosswalk_mass_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
