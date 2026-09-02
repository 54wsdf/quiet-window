#!/usr/bin/env python3
"""Consolidate the fragmented 2026 MTA OD deterministic acquisition in Google Drive.

The acquisition itself is public_data registry / official public source evidence.
This script never treats Drive folder names as unique identifiers: it inventories
known source folder IDs, forms a chunk-index union, validates duplicate bytes,
server-side copies existing Drive objects into one pre-created canonical folder,
and fetches only genuinely missing Socrata chunks when the provider version still
matches the frozen acquisition snapshot.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests

DATASET = "28vm-gjqr"
CHUNK = 50_000
FROZEN_TOTAL = 72_639_113
FROZEN_ROWS_UPDATED_AT = 1_787_243_523
EXPECTED = math.ceil(FROZEN_TOTAL / CHUNK)
PART_RE = re.compile(r"^28vm-gjqr\.part-(\d{5})\.csv\.gz$")
BASE_CSV = f"https://data.ny.gov/resource/{DATASET}.csv"
BASE_JSON = f"https://data.ny.gov/resource/{DATASET}.json"
META_URL = f"https://data.ny.gov/api/views/{DATASET}"
UA = "OpenCity-MTA-OD-Drive-union-salvage/20260902"


def sh(cmd: list[str], capture: bool = False) -> str:
    if capture:
        return subprocess.check_output(cmd, text=True)
    subprocess.run(cmd, check=True)
    return ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def clone_rclone_config(src: Path, base_remote: str, source_ids: list[str], dest_id: str, out: Path):
    cp = configparser.RawConfigParser(interpolation=None)
    cp.optionxform = str
    with src.open("r", encoding="utf-8") as f:
        cp.read_file(f)
    if base_remote not in cp:
        raise RuntimeError(f"remote [{base_remote}] not found in rclone config")
    base = dict(cp[base_remote].items())
    for i, folder_id in enumerate(source_ids):
        name = f"src{i}"
        if name in cp:
            cp.remove_section(name)
        cp.add_section(name)
        for k, v in base.items():
            cp[name][k] = v
        cp[name]["root_folder_id"] = folder_id
    if "dst" in cp:
        cp.remove_section("dst")
    cp.add_section("dst")
    for k, v in base.items():
        cp["dst"][k] = v
    cp["dst"]["root_folder_id"] = dest_id
    with out.open("w", encoding="utf-8") as f:
        cp.write(f)
    os.chmod(out, 0o600)


def lsjson(remote: str, cfg: Path) -> list[dict]:
    txt = sh([
        "rclone", "lsjson", f"{remote}:", "--files-only", "--hash",
        "--config", str(cfg), "--no-mimetype"
    ], capture=True)
    return json.loads(txt)


def get_md5(entry: dict) -> str | None:
    h = entry.get("Hashes") or {}
    for k in ("MD5", "md5"):
        if h.get(k):
            return str(h[k]).lower()
    return None


def stream_sha(remote: str, name: str, cfg: Path) -> tuple[int, str]:
    p = subprocess.Popen(["rclone", "cat", f"{remote}:{name}", "--config", str(cfg)], stdout=subprocess.PIPE)
    h = hashlib.sha256(); n = 0
    assert p.stdout is not None
    for b in iter(lambda: p.stdout.read(8 * 1024 * 1024), b""):
        h.update(b); n += len(b)
    code = p.wait()
    if code != 0:
        raise RuntimeError(f"rclone cat failed: {remote}:{name}, code={code}")
    return n, h.hexdigest()


def read_remote_json(remote: str, name: str, cfg: Path) -> dict:
    return json.loads(sh(["rclone", "cat", f"{remote}:{name}", "--config", str(cfg)], capture=True))


def provider_state(session: requests.Session) -> tuple[int, dict]:
    c = session.get(BASE_JSON, params={"$select": "count(*)"}, timeout=180)
    c.raise_for_status()
    total = int(c.json()[0]["count"])
    m = session.get(META_URL, timeout=180)
    m.raise_for_status()
    x = m.json()
    meta = {k: x.get(k) for k in ["id", "name", "rowsUpdatedAt", "dataUpdatedAt", "metadataUpdatedAt", "publicationDate"]}
    return total, meta


def fetch_missing(session: requests.Session, idx: int, work: Path) -> dict:
    offset = idx * CHUNK
    limit = min(CHUNK, FROZEN_TOTAL - offset)
    params = {"$limit": limit, "$offset": offset, "$order": ":id"}
    r = session.get(BASE_CSV, params=params, timeout=(30, 900))
    r.raise_for_status()
    text = r.text
    rows = list(csv.reader(io.StringIO(text)))
    n = max(0, len(rows) - 1)
    if n != limit:
        raise RuntimeError(f"missing chunk {idx}: provider rows {n} != expected {limit}")
    p = work / f"{DATASET}.part-{idx:05d}.csv.gz"
    # Deterministic gzip for any genuinely missing chunk acquired by salvage.
    with p.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
            gz.write(text.encode("utf-8"))
    return {
        "index": idx,
        "offset": offset,
        "rows": n,
        "file": p.name,
        "bytes": p.stat().st_size,
        "sha256": sha256_file(p),
        "origin": "official_source_missing_chunk_fill",
        "local_path": str(p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--remote", required=True)
    ap.add_argument("--source-folder-id", action="append", required=True)
    ap.add_argument("--dest-folder-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="opencity-mta-od-union-") as td0:
        td = Path(td0)
        cfg = td / "rclone-union.conf"
        clone_rclone_config(args.config, args.remote, args.source_folder_id, args.dest_folder_id, cfg)

        inventories: dict[str, list[dict]] = {}
        union: dict[int, list[dict]] = {}
        manifest_records: dict[int, list[dict]] = {}
        source_manifests: list[dict] = []

        for i, folder_id in enumerate(args.source_folder_id):
            remote = f"src{i}"
            entries = lsjson(remote, cfg)
            inventories[folder_id] = entries
            for e in entries:
                name = e.get("Name") or e.get("Path") or ""
                m = PART_RE.match(name)
                if m:
                    idx = int(m.group(1))
                    union.setdefault(idx, []).append({
                        "source_remote": remote,
                        "source_folder_id": folder_id,
                        "file": name,
                        "bytes": int(e.get("Size", 0)),
                        "md5": get_md5(e),
                    })
                elif name.startswith("manifest-shard-") and name.endswith(".json"):
                    try:
                        mj = read_remote_json(remote, name, cfg)
                        source_manifests.append({"source_folder_id": folder_id, "file": name, "manifest": mj})
                        for rec in mj.get("files", []):
                            idx = int(rec["index"])
                            manifest_records.setdefault(idx, []).append({**rec, "manifest_file": name, "source_folder_id": folder_id})
                    except Exception as exc:
                        source_manifests.append({"source_folder_id": folder_id, "file": name, "error": repr(exc)})

        observed = sorted(union)
        duplicate_conflicts = []
        duplicate_agreements = []
        chosen: dict[int, dict] = {}
        sha_by_idx: dict[int, str] = {}
        rows_by_idx: dict[int, int] = {}

        for idx, copies in sorted(union.items()):
            sizes = {c["bytes"] for c in copies}
            md5s = {c["md5"] for c in copies if c.get("md5")}
            if len(sizes) > 1 or len(md5s) > 1:
                duplicate_conflicts.append({"index": idx, "copies": copies})
                continue
            if len(copies) > 1:
                duplicate_agreements.append({"index": idx, "copies": copies})
            chosen[idx] = copies[0]
            recs = manifest_records.get(idx, [])
            rsha = {str(r.get("sha256", "")) for r in recs if r.get("sha256")}
            rbytes = {int(r.get("bytes", -1)) for r in recs if r.get("bytes") is not None}
            rrows = {int(r.get("rows", -1)) for r in recs if r.get("rows") is not None}
            if len(rsha) > 1 or (rbytes and rbytes != {copies[0]["bytes"]}) or len(rrows) > 1:
                duplicate_conflicts.append({"index": idx, "copies": copies, "manifest_records": recs})
                continue
            if rsha:
                sha_by_idx[idx] = next(iter(rsha))
            if rrows:
                rows_by_idx[idx] = next(iter(rrows))

        if duplicate_conflicts:
            (args.output / "DUPLICATE_CONFLICTS.json").write_text(json.dumps(duplicate_conflicts, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"{len(duplicate_conflicts)} duplicate/hash conflicts in Drive union")

        expected_indices = set(range(EXPECTED))
        missing_before = sorted(expected_indices - set(observed))
        extra = sorted(set(observed) - expected_indices)
        if extra:
            raise RuntimeError(f"unexpected chunk indices: {extra[:20]}")

        sess = requests.Session(); sess.headers["User-Agent"] = UA
        current_total, current_meta = provider_state(sess)
        version_match = current_total == FROZEN_TOTAL and int(current_meta.get("rowsUpdatedAt") or -1) == FROZEN_ROWS_UPDATED_AT

        # Inventory canonical destination before copying.
        dest_before_entries = lsjson("dst", cfg)
        dest_before = {int(m.group(1)): e for e in dest_before_entries if (m := PART_RE.match(e.get("Name") or e.get("Path") or ""))}

        copied = []
        # Consolidate all already-acquired bytes first, regardless of whether the provider later changed.
        for idx in sorted(chosen):
            src = chosen[idx]
            name = src["file"]
            de = dest_before.get(idx)
            if de:
                if int(de.get("Size", -1)) != src["bytes"]:
                    raise RuntimeError(f"canonical destination size conflict for {name}")
                dmd5 = get_md5(de); smd5 = src.get("md5")
                if dmd5 and smd5 and dmd5 != smd5:
                    raise RuntimeError(f"canonical destination MD5 conflict for {name}")
                continue
            cmd = [
                "rclone", "copyto", f"{src['source_remote']}:{name}", f"dst:{name}",
                "--config", str(cfg), "--drive-server-side-across-configs",
                "--retries", "8", "--low-level-retries", "20"
            ]
            try:
                sh(cmd)
            except subprocess.CalledProcessError:
                # Fall back to a Drive->runner->Drive transfer; never re-request the public source.
                local = td / name
                sh(["rclone", "copyto", f"{src['source_remote']}:{name}", str(local), "--config", str(cfg), "--retries", "5"])
                sh(["rclone", "copyto", str(local), f"dst:{name}", "--config", str(cfg), "--retries", "8", "--low-level-retries", "20"])
                local.unlink(missing_ok=True)
            copied.append(idx)

        filled = []
        if missing_before:
            if not version_match:
                state = {
                    "state": "PARTIAL_PROVIDER_VERSION_CHANGED",
                    "evidence_class": "public_data registry / official public source",
                    "dataset": DATASET,
                    "frozen_total_rows": FROZEN_TOTAL,
                    "frozen_rowsUpdatedAt": FROZEN_ROWS_UPDATED_AT,
                    "current_total_rows": current_total,
                    "current_metadata": current_meta,
                    "union_chunks": len(observed),
                    "missing_indices": missing_before,
                    "source_folder_ids": args.source_folder_id,
                    "canonical_folder_id": args.dest_folder_id,
                    "rule": "Existing Drive bytes were consolidated, but missing indices were not fetched because the provider version no longer matches the frozen snapshot."
                }
                p = args.output / "SALVAGE_STATE.json"
                p.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
                sh(["rclone", "copyto", str(p), "dst:SALVAGE_STATE.json", "--config", str(cfg), "--retries", "5"])
                raise RuntimeError("provider version changed; refusing to mix versions while filling missing chunks")
            for idx in missing_before:
                rec = fetch_missing(sess, idx, td)
                local = Path(rec.pop("local_path"))
                sh(["rclone", "copyto", str(local), f"dst:{rec['file']}", "--config", str(cfg), "--retries", "8", "--low-level-retries", "20"])
                sha_by_idx[idx] = rec["sha256"]
                rows_by_idx[idx] = rec["rows"]
                filled.append(rec)
                local.unlink(missing_ok=True)

        # Final destination inventory must contain exactly one logical chunk name for every index.
        final_entries = lsjson("dst", cfg)
        final_parts: dict[int, list[dict]] = {}
        for e in final_entries:
            name = e.get("Name") or e.get("Path") or ""
            m = PART_RE.match(name)
            if m:
                final_parts.setdefault(int(m.group(1)), []).append(e)
        missing_after = sorted(expected_indices - set(final_parts))
        extra_after = sorted(set(final_parts) - expected_indices)
        dup_after = sorted(i for i, es in final_parts.items() if len(es) != 1)
        if missing_after or extra_after or dup_after:
            raise RuntimeError(f"canonical closure failed missing={missing_after[:20]} extra={extra_after[:20]} duplicate_names={dup_after[:20]}")

        # Recover SHA-256 and row counts from shard manifests where possible; stream only the remainder.
        part_records = []
        for idx in range(EXPECTED):
            name = f"{DATASET}.part-{idx:05d}.csv.gz"
            e = final_parts[idx][0]
            size = int(e.get("Size", 0))
            expected_rows = min(CHUNK, FROZEN_TOTAL - idx * CHUNK)
            recs = manifest_records.get(idx, [])
            if idx not in rows_by_idx and recs:
                rows_by_idx[idx] = int(recs[0].get("rows", -1))
            if rows_by_idx.get(idx, expected_rows) != expected_rows:
                raise RuntimeError(f"row-count contract mismatch in manifest for {name}: {rows_by_idx.get(idx)} vs {expected_rows}")
            if idx not in sha_by_idx:
                n, dig = stream_sha("dst", name, cfg)
                if n != size:
                    raise RuntimeError(f"streamed bytes differ from Drive size for {name}: {n} vs {size}")
                sha_by_idx[idx] = dig
            part_records.append({
                "index": idx,
                "offset": idx * CHUNK,
                "rows": expected_rows,
                "file": name,
                "bytes": size,
                "sha256": sha_by_idx[idx],
                "drive_md5": get_md5(e),
                "provenance": "existing_shard_union" if idx not in {x["index"] for x in filled} else "official_source_missing_chunk_fill",
            })

        total_rows_manifest = sum(x["rows"] for x in part_records)
        if total_rows_manifest != FROZEN_TOTAL:
            raise RuntimeError(f"manifest rows {total_rows_manifest} != source rows {FROZEN_TOTAL}")

        closed_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema": "opencity-mta-od-drive-union-v1",
            "state": "ACQUIRED",
            "evidence_class": "public_data registry / official public source",
            "dataset": DATASET,
            "name": "MTA Subway Origin-Destination Ridership Estimate: Beginning 2026",
            "source_resource_url": BASE_CSV,
            "source_metadata_url": META_URL,
            "source_row_count": FROZEN_TOTAL,
            "source_rowsUpdatedAt": FROZEN_ROWS_UPDATED_AT,
            "source_metadata_at_close": current_meta,
            "provider_version_match_at_close": version_match,
            "ordering_contract": "$order=:id",
            "chunk_rows": CHUNK,
            "expected_chunks": EXPECTED,
            "actual_chunks": len(part_records),
            "source_folder_ids_fragmented": args.source_folder_id,
            "canonical_folder_id": args.dest_folder_id,
            "union_chunks_before_fill": len(observed),
            "missing_indices_before_fill": missing_before,
            "filled_from_official_source_indices": [x["index"] for x in filled],
            "copied_existing_indices_count": len(copied),
            "duplicate_indices_with_agreeing_drive_hashes": [x["index"] for x in duplicate_agreements],
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "executor_commit": os.environ.get("GITHUB_SHA"),
            "closed_at_utc": closed_at,
            "parts": part_records,
        }
        mp = args.output / "ACQUIRED_MANIFEST.json"
        sp = args.output / "SHA256SUMS.txt"
        up = args.output / "SOURCE_FOLDER_UNION.json"
        mp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        sp.write_text("".join(f"{x['sha256']}  {x['file']}\n" for x in part_records), encoding="utf-8")
        up.write_text(json.dumps({
            "source_folder_ids": args.source_folder_id,
            "observed_union_indices": observed,
            "union_count": len(observed),
            "duplicate_agreements": duplicate_agreements,
            "missing_before_fill": missing_before,
            "canonical_folder_id": args.dest_folder_id,
        }, indent=2) + "\n", encoding="utf-8")
        for p in (mp, sp, up):
            sh(["rclone", "copyto", str(p), f"dst:{p.name}", "--config", str(cfg), "--retries", "8", "--low-level-retries", "20"])
        print(json.dumps({
            "state": "ACQUIRED", "dataset": DATASET, "rows": FROZEN_TOTAL,
            "chunks": EXPECTED, "union_before_fill": len(observed),
            "filled_missing": len(filled), "canonical_folder_id": args.dest_folder_id,
        }))


if __name__ == "__main__":
    main()
