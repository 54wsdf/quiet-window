import argparse
import csv
import gzip
import json
import resource
import time
import zipfile
from collections import Counter
from datetime import timedelta
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0

START = g0.START
END = g0.END
RAIL = g0.RAIL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taims", required=True)
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    cohorts = Counter()
    stats = Counter()

    t0 = time.perf_counter()
    with zipfile.ZipFile(args.taims) as z:
        f = g0.zr(z, "VW_KSCC_DX_CARD.csv")
        for row in csv.DictReader(f):
            if str(row.get("TRNS_MNS_CD") or "").strip() not in RAIL:
                continue
            tin = g0.dt(row.get("RIDE_DTM"))
            tout = g0.dt(row.get("ALGH_DTM"))
            if not tin or not tout or not (START <= tin < END) or tout <= tin or tout - tin > timedelta(hours=3):
                continue
            stats["eligible"] += 1
            oc = g0.z4(row.get("RIDE_BSST_ID"))
            dc = g0.z4(row.get("ALGH_BSST_ID"))
            if oc not in code_to_nodes or dc not in code_to_nodes:
                stats["unmapped"] += 1
                continue
            stats["mapped"] += 1
            cohorts[(oc, dc, tin.isoformat(), tout.isoformat())] += 1
        f.close()
    scan_sec = time.perf_counter() - t0

    out_path = outdir / "seoul_20260829_0700_1000_afc_time_cohorts.csv.gz"
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["origin_code", "destination_code", "entry_time", "exit_time", "passenger_mass"])
        for (oc, dc, tin, tout), mass in sorted(cohorts.items()):
            w.writerow([oc, dc, tin, tout, mass])

    total_mass = sum(cohorts.values())
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "schema": "mppd.r0-city-day-observation-cache.v1",
        "date": "2026-09-04",
        "status": "R0_CITY_DAY_CACHE_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "time_window": "2026-08-29 07:00-10:00",
        "scope_assertions": {
            "full_network": True,
            "line_filter_applied": False,
            "segment_filter_applied": False,
            "all_mapped_passenger_mass_preserved": True,
            "raw_card_identifier_retained": False,
            "time_aggregation_lossless_for_qualified_seoul_public_afc": True,
        },
        "network": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "lines": len({m["line"] for m in meta.values()}),
            "transfer_groups": len(transfer_groups),
        },
        "afc": {
            "eligible_rows": stats["eligible"],
            "mapped_rows": stats["mapped"],
            "unmapped_rows": stats["unmapped"],
            "cohort_rows": len(cohorts),
            "cohort_passenger_mass": total_mass,
        },
        "performance": {
            "raw_zip_scan_sec": scan_sec,
            "total_wall_sec": time.perf_counter() - wall0,
            "max_rss_kb": max_rss_kb,
        },
        "cache_contract": {
            "key": ["origin_code", "destination_code", "entry_time", "exit_time"],
            "value": "passenger_mass",
            "intended_use": "Reusable full-network passenger observation input for G2/G3 iterations without rescanning raw TAIMS.",
        },
        "scientific_boundary": [
            "This cache is an execution optimization only and does not alter the V6 scientific evidence domain.",
            "The Seoul source timestamps are already minute-quantized; grouping identical observed OD/entry/exit tuples is lossless with respect to the qualified passenger observation field.",
            "No raw card identifier is stored.",
        ],
        "no_email_notification_logic": True,
    }
    (outdir / "r0_city_day_cache_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
