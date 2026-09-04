import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0


def norm(v):
    return str(v or "").strip()


def compact_row(row, max_fields=40):
    out = {}
    for k, v in row.items():
        if len(out) >= max_fields:
            break
        s = norm(v)
        if s:
            out[str(k)] = s[:300]
    return out


def code_variants(v):
    s = norm(v)
    out = set()
    if s:
        out.add(s)
    digits = "".join(c for c in s if c.isdigit())
    if digits:
        if len(digits) <= 4:
            out.add(digits.zfill(4))
        if len(digits) == 4:
            out.add(digits)
    return out


def likely_identifier_columns(fieldnames):
    out = []
    pats = ("STN", "STATN", "STTN", "BSST", "STA", "NODE", "STOP", "ID", "CD", "CODE")
    for f in fieldnames or []:
        u = str(f).upper()
        if any(p in u for p in pats):
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taims", required=True)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-matches-per-code-file", type=int, default=8)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    targets = {}
    with open(args.codes, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            code = norm(r.get("code"))
            if code:
                targets[code] = {
                    "touch_mass": int(r.get("unmapped_endpoint_touch_mass") or 0),
                    "g0i_diagnosis": norm(r.get("diagnosis")),
                    "g0i_p1c_tiers": norm(r.get("p1c_tiers")),
                    "g0i_p1c_lines": norm(r.get("p1c_lines")),
                    "g0i_p1c_names": norm(r.get("p1c_names")),
                }
    target_set = set(targets)

    file_inventory = []
    matches = defaultdict(list)
    match_mass_by_file = Counter()
    matched_codes_by_file = defaultdict(set)
    skipped_files = []

    with zipfile.ZipFile(args.taims) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        for info in infos:
            name = info.filename
            rec = {
                "file": name,
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
                "scanned": False,
                "fieldnames": [],
                "identifier_columns": [],
                "rows_scanned": 0,
                "target_match_count": 0,
                "matched_target_code_count": 0,
            }
            if not name.lower().endswith(".csv"):
                skipped_files.append({"file": name, "reason": "NOT_CSV", "size": info.file_size})
                file_inventory.append(rec)
                continue
            if name.endswith("VW_KSCC_DX_CARD.csv"):
                skipped_files.append({"file": name, "reason": "TRANSACTION_TABLE_ALREADY_USED_AS_AFC_SOURCE", "size": info.file_size})
                file_inventory.append(rec)
                continue
            try:
                fh = g0.zr(z, name)
                reader = csv.DictReader(fh)
                fields = list(reader.fieldnames or [])
                id_cols = likely_identifier_columns(fields)
                rec["fieldnames"] = fields
                rec["identifier_columns"] = id_cols
                if not fields:
                    fh.close()
                    skipped_files.append({"file": name, "reason": "NO_HEADER", "size": info.file_size})
                    file_inventory.append(rec)
                    continue

                per_code_count = Counter()
                rows = 0
                for row_idx, row in enumerate(reader, start=2):
                    rows += 1
                    candidate_cols = id_cols if id_cols else fields
                    row_codes = defaultdict(list)
                    for col in candidate_cols:
                        v = row.get(col)
                        for variant in code_variants(v):
                            if variant in target_set:
                                row_codes[variant].append(col)
                    for code, cols in row_codes.items():
                        if per_code_count[code] >= args.max_matches_per_code_file:
                            continue
                        per_code_count[code] += 1
                        payload = {
                            "code": code,
                            "file": name,
                            "row_number": row_idx,
                            "matched_columns": sorted(set(cols)),
                            "row": compact_row(row),
                        }
                        matches[code].append(payload)
                        matched_codes_by_file[name].add(code)
                        match_mass_by_file[name] += targets[code]["touch_mass"]
                fh.close()
                rec["scanned"] = True
                rec["rows_scanned"] = rows
                rec["target_match_count"] = sum(per_code_count.values())
                rec["matched_target_code_count"] = len(per_code_count)
            except Exception as exc:
                rec["scan_error"] = f"{type(exc).__name__}: {exc}"
                skipped_files.append({"file": name, "reason": "SCAN_ERROR", "detail": rec["scan_error"], "size": info.file_size})
            file_inventory.append(rec)

    code_results = []
    matched_touch_mass = 0
    for code, base in sorted(targets.items(), key=lambda kv: (-kv[1]["touch_mass"], kv[0])):
        ms = matches.get(code, [])
        if ms:
            matched_touch_mass += base["touch_mass"]
        code_results.append({
            "code": code,
            **base,
            "source_match_count": len(ms),
            "matched_files": sorted({m["file"] for m in ms}),
            "source_matches": ms,
        })

    file_results = []
    for name, codes in sorted(matched_codes_by_file.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        file_results.append({
            "file": name,
            "matched_code_count": len(codes),
            "matched_codes": sorted(codes),
            "sum_target_touch_mass_nonconserved": match_mass_by_file[name],
        })

    result = {
        "schema": "mppd.g0j-unmapped-code-source-discovery.v1",
        "date": "2026-09-04",
        "status": "G0J_RAW_SOURCE_DISCOVERY_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "target": {
            "unique_code_count": len(targets),
            "endpoint_touch_mass_nonconserved": sum(x["touch_mass"] for x in targets.values()),
            "codes_with_any_nontransaction_source_match": sum(1 for x in code_results if x["source_match_count"] > 0),
            "touch_mass_with_any_source_match_nonconserved": matched_touch_mass,
        },
        "archive": {
            "member_count": len(file_inventory),
            "csv_scanned_count": sum(1 for x in file_inventory if x.get("scanned")),
            "files_with_target_matches": len(file_results),
        },
        "files_with_matches": file_results,
        "code_results": code_results,
        "file_inventory": file_inventory,
        "skipped_files": skipped_files,
        "scientific_boundary": [
            "This is source discovery, not mapping qualification. A numeric code occurrence in another TAIMS table is evidence of provenance only.",
            "The AFC transaction table is deliberately excluded from source discovery because it cannot independently identify an endpoint code.",
            "Touch mass is non-conserved when a passenger has two unmapped endpoints and must not be used as a passenger-level denominator.",
            "Any G0J crosswalk promotion requires semantically interpretable station/line metadata or other independent corroboration, not code equality alone."
        ],
        "next_gate": "Inspect the metadata tables that explain the largest residual code mass, define evidence-qualified mapping families, and rebuild the raw-AFC denominator only after their line/station identity is resolved.",
        "no_email_notification_logic": True
    }

    (outdir / "g0j_unmapped_code_source_discovery.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (outdir / "g0j_code_source_summary.csv").open("w", encoding="utf-8", newline="") as f:
        rows = [{k: v for k, v in x.items() if k != "source_matches"} for x in code_results]
        fields = list(rows[0]) if rows else ["code"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps({
        "status": result["status"],
        "target": result["target"],
        "archive": result["archive"],
        "files_with_matches": result["files_with_matches"],
        "top_codes": [{"code": x["code"], "touch_mass": x["touch_mass"], "source_match_count": x["source_match_count"], "matched_files": x["matched_files"]} for x in code_results[:30]],
        "no_email_notification_logic": True
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
