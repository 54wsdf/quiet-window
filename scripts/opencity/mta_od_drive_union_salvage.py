#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlencode

import requests

EXPECTED_CHUNKS = 1453
EXPECTED_ROWS = 72_639_113
CHUNK_SIZE = 50_000
EXPECTED_ROWS_UPDATED_AT = 1787243523
EXPECTED_VIEW_LAST_MODIFIED = 1787243537
DATA_URL = "https://data.ny.gov/resource/28vm-gjqr.json"
META_URL = "https://data.ny.gov/api/views/28vm-gjqr"
SOURCE_FOLDER_NAME = "mta_od_2026_quiet_window_reacquire"
KNOWN_SOURCE_FOLDER_IDS = [
    "1KLXVIEyyGOZqQoAUQpanmt3KuFixqZUS",
    "1JFL0T9FcY4vJz8-m9lnUikdYNoDYDGLO",
    "1CfWNC_sWdQsTwe_I_ZOU1g-0mz73MHG2",
    "1Y6WAhuGHqyi6qoQNCNuUHVsUWlngEjFu",
    "19QZJ8tnOpwonvcZH4IvS2h-93XxkxE4j",
]
CANONICAL_FOLDER_ID = "1KhB316vUHPTEtwdWPN4138TttTW67jGO"
ACQ_ROOT_FOLDER_ID = "11-QAIWNF54-xtRl7TeCsQaPwQMPh6Io2"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def run(cmd, *, capture=True, check=True, input_bytes=None):
    p = subprocess.run(
        cmd,
        input=input_bytes,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if check and p.returncode:
        err = p.stderr.decode("utf-8", "replace") if p.stderr else ""
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{err[-4000:]}")
    return p


def rclone_lsjson(remote, folder_id):
    p = run([
        "rclone", "lsjson", f"{remote}:", "--drive-root-folder-id", folder_id,
        "--max-depth", "1",
    ])
    return json.loads(p.stdout or b"[]")


def discover_source_ids(remote):
    ids = set(KNOWN_SOURCE_FOLDER_IDS)
    for x in rclone_lsjson(remote, ACQ_ROOT_FOLDER_ID):
        if x.get("IsDir") and x.get("Name") == SOURCE_FOLDER_NAME and x.get("ID"):
            ids.add(x["ID"])
    # Verify every ID is readable; a disappeared/mistyped known race folder is a hard evidence failure.
    verified = []
    for fid in sorted(ids):
        try:
            rclone_lsjson(remote, fid)
            verified.append(fid)
        except Exception as e:
            raise RuntimeError(f"source folder id is not readable: {fid}: {e}")
    if not verified:
        raise RuntimeError("no MTA OD source folders discovered")
    return verified


def file_index(remote, folder_id):
    out = {}
    for x in rclone_lsjson(remote, folder_id):
        if x.get("IsDir"):
            continue
        name = x.get("Name") or x.get("Path")
        if name:
            out[name] = x
    return out


def rclone_download(remote, folder_id, name, local):
    run([
        "rclone", "copyto", f"{remote}:{name}", str(local),
        "--drive-root-folder-id", folder_id,
        "--retries", "8", "--low-level-retries", "20",
    ], capture=True)


def rclone_upload(remote, folder_id, local, name):
    run([
        "rclone", "copyto", str(local), f"{remote}:{name}",
        "--drive-root-folder-id", folder_id,
        "--retries", "8", "--low-level-retries", "20",
    ], capture=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def expected_rows_for_index(idx):
    if idx < EXPECTED_CHUNKS - 1:
        return CHUNK_SIZE
    return EXPECTED_ROWS - CHUNK_SIZE * (EXPECTED_CHUNKS - 1)


def inspect_chunk(path, idx):
    sha = sha256_file(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise RuntimeError(f"chunk {idx}: top-level JSON is not a list")
    exp = expected_rows_for_index(idx)
    if len(rows) != exp:
        raise RuntimeError(f"chunk {idx}: row count {len(rows)} != expected {exp}")
    first_id = str(rows[0].get(":id", "")) if rows else ""
    last_id = str(rows[-1].get(":id", "")) if rows else ""
    return {"sha256": sha, "size_bytes": size, "rows": len(rows), "first_id": first_id, "last_id": last_id}


def provider_snapshot(session):
    m = session.get(META_URL, timeout=90)
    m.raise_for_status()
    meta = m.json()
    c = session.get(DATA_URL, params={"$select": "count(*)"}, timeout=90)
    c.raise_for_status()
    cj = c.json()
    count = int(cj[0].get("count") or cj[0].get("count_1") or next(iter(cj[0].values())))
    rows_updated = int(meta.get("rowsUpdatedAt") or 0)
    view_modified = int(meta.get("viewLastModified") or 0)
    return {
        "rowsUpdatedAt": rows_updated,
        "viewLastModified": view_modified,
        "row_count": count,
        "matches_required_version": rows_updated == EXPECTED_ROWS_UPDATED_AT and count == EXPECTED_ROWS,
    }


def refill_chunk(session, idx, target):
    snap = provider_snapshot(session)
    if not snap["matches_required_version"]:
        raise RuntimeError(
            f"chunk {idx} absent from Drive union and provider version no longer matches: {snap}"
        )
    params = {"$order": ":id", "$limit": CHUNK_SIZE, "$offset": idx * CHUNK_SIZE}
    last = None
    for attempt in range(8):
        try:
            r = session.get(DATA_URL, params=params, timeout=180)
            r.raise_for_status()
            target.write_bytes(r.content)
            info = inspect_chunk(target, idx)
            return info, snap, DATA_URL + "?" + urlencode(params)
        except Exception as e:
            last = e
    raise RuntimeError(f"chunk {idx}: provider refill failed after retries: {last}")


def shard_mode(remote, shard):
    if shard < 0 or shard > 7:
        raise ValueError("shard must be 0..7")
    source_ids = discover_source_ids(remote)
    source_maps = {fid: file_index(remote, fid) for fid in source_ids}
    canonical_map = file_index(remote, CANONICAL_FOLDER_ID)
    session = requests.Session()
    session.headers["User-Agent"] = "quiet-window-mta-od-drive-union-salvage/20260902"
    work = Path(tempfile.mkdtemp(prefix=f"mta_od_salvage_{shard}_"))
    done = 0
    refilled = 0
    copied = 0
    canonical_kept = 0
    duplicates = 0
    try:
        for idx in range(shard, EXPECTED_CHUNKS, 8):
            name = f"chunk_{idx:06d}.json"
            manifest_name = f"manifest_{idx:06d}.json"
            candidates = []
            for fid in source_ids:
                if name in source_maps[fid]:
                    candidates.append(("source", fid))
            if name in canonical_map:
                candidates.append(("canonical", CANONICAL_FOLDER_ID))

            chosen = work / name
            observed = []
            base_info = None
            for j, (kind, fid) in enumerate(candidates):
                tmp = work / f"candidate_{idx:06d}_{j}.json"
                rclone_download(remote, fid, name, tmp)
                info = inspect_chunk(tmp, idx)
                observed.append({"kind": kind, "folder_id": fid, **info})
                if base_info is None:
                    base_info = info
                    shutil.copyfile(tmp, chosen)
                else:
                    if info["sha256"] != base_info["sha256"]:
                        raise RuntimeError(
                            f"chunk {idx}: duplicate index hash conflict: "
                            f"{base_info['sha256']} vs {info['sha256']} from {fid}"
                        )
                tmp.unlink(missing_ok=True)

            snapshot = None
            source_query = None
            if base_info is None:
                base_info, snapshot, source_query = refill_chunk(session, idx, chosen)
                provenance = "provider_refill_after_union_missing"
                refilled += 1
                rclone_upload(remote, CANONICAL_FOLDER_ID, chosen, name)
            else:
                duplicates += max(0, len(observed) - 1)
                if any(x["kind"] == "canonical" for x in observed):
                    provenance = "existing_canonical_verified"
                    canonical_kept += 1
                else:
                    provenance = "drive_union_copy"
                    copied += 1
                    rclone_upload(remote, CANONICAL_FOLDER_ID, chosen, name)

            record = {
                "index": idx,
                "filename": name,
                "sha256": base_info["sha256"],
                "size_bytes": base_info["size_bytes"],
                "rows": base_info["rows"],
                "first_id": base_info["first_id"],
                "last_id": base_info["last_id"],
                "order": ":id",
                "offset": idx * CHUNK_SIZE,
                "limit": CHUNK_SIZE,
                "provenance": provenance,
                "identical_drive_copies": observed,
                "provider_snapshot_at_refill": snapshot,
                "provider_query_at_refill": source_query,
                "canonical_folder_id": CANONICAL_FOLDER_ID,
                "source_folder_ids_considered": source_ids,
            }
            mp = work / manifest_name
            mp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rclone_upload(remote, CANONICAL_FOLDER_ID, mp, manifest_name)
            chosen.unlink(missing_ok=True)
            mp.unlink(missing_ok=True)
            done += 1
            print(json.dumps({"idx": idx, "provenance": provenance, "sha256": base_info["sha256"]}), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({
        "shard": shard,
        "indices_verified": done,
        "drive_union_copies": copied,
        "existing_canonical": canonical_kept,
        "provider_refills": refilled,
        "duplicate_copies_verified": duplicates,
        "source_folder_ids": source_ids,
    }, indent=2))


def finalize_mode(remote):
    source_ids = discover_source_ids(remote)
    cmap = file_index(remote, CANONICAL_FOLDER_ID)
    work = Path(tempfile.mkdtemp(prefix="mta_od_finalize_"))
    records = []
    missing = []
    try:
        for idx in range(EXPECTED_CHUNKS):
            cn = f"chunk_{idx:06d}.json"
            mn = f"manifest_{idx:06d}.json"
            if cn not in cmap or mn not in cmap:
                missing.append(idx)
                continue
            mp = work / mn
            rclone_download(remote, CANONICAL_FOLDER_ID, mn, mp)
            r = json.loads(mp.read_text(encoding="utf-8"))
            if int(r.get("index", -1)) != idx:
                raise RuntimeError(f"manifest index mismatch for {idx}")
            if r.get("filename") != cn:
                raise RuntimeError(f"manifest filename mismatch for {idx}")
            sha = r.get("sha256", "")
            if not SHA_RE.match(sha):
                raise RuntimeError(f"invalid sha256 for chunk {idx}")
            exp_rows = expected_rows_for_index(idx)
            if int(r.get("rows", -1)) != exp_rows:
                raise RuntimeError(f"manifest rows mismatch for chunk {idx}")
            remote_size = int(cmap[cn].get("Size", -1))
            if remote_size != int(r.get("size_bytes", -2)):
                raise RuntimeError(f"remote size mismatch for chunk {idx}: {remote_size} vs {r.get('size_bytes')}")
            records.append(r)
            mp.unlink(missing_ok=True)
        if missing:
            raise RuntimeError(f"canonical missing chunk/manifest indices after union salvage: {missing[:80]} (n={len(missing)})")
        if len(records) != EXPECTED_CHUNKS:
            raise RuntimeError(f"manifest count {len(records)} != {EXPECTED_CHUNKS}")
        total_rows = sum(int(r["rows"]) for r in records)
        if total_rows != EXPECTED_ROWS:
            raise RuntimeError(f"row total {total_rows} != {EXPECTED_ROWS}")
        if [r["index"] for r in records] != list(range(EXPECTED_CHUNKS)):
            raise RuntimeError("index continuity 0..1452 failed")

        session = requests.Session()
        session.headers["User-Agent"] = "quiet-window-mta-od-drive-union-salvage/20260902"
        try:
            current_snapshot = provider_snapshot(session)
        except Exception as e:
            current_snapshot = {"check_error": str(e)}

        aggregate_lines = "".join(f"{r['index']:06d}:{r['sha256']}\n" for r in records)
        aggregate_digest = hashlib.sha256(aggregate_lines.encode("ascii")).hexdigest()
        refill_indices = [r["index"] for r in records if r["provenance"] == "provider_refill_after_union_missing"]
        duplicate_indices = [r["index"] for r in records if len(r.get("identical_drive_copies", [])) > 1]
        aggregate = {
            "status": "ACQUIRED",
            "dataset": "MTA Origin-Destination Ridership Estimate: 2025",
            "official_data_url": DATA_URL,
            "official_metadata_url": META_URL,
            "query_order": ":id",
            "chunk_size": CHUNK_SIZE,
            "expected_provider_version": {
                "rowsUpdatedAt": EXPECTED_ROWS_UPDATED_AT,
                "viewLastModified": EXPECTED_VIEW_LAST_MODIFIED,
                "row_count": EXPECTED_ROWS,
            },
            "provider_snapshot_at_finalize": current_snapshot,
            "producer_run_id": "33615833576",
            "source_folder_name": SOURCE_FOLDER_NAME,
            "source_folder_ids": source_ids,
            "canonical_folder_id": CANONICAL_FOLDER_ID,
            "index_range": [0, EXPECTED_CHUNKS - 1],
            "chunk_count": len(records),
            "row_count": total_rows,
            "refilled_indices": refill_indices,
            "duplicate_indices_verified_hash_identical": duplicate_indices,
            "chunk_sha256_aggregate": aggregate_digest,
            "chunks": records,
        }
        ap = work / "MTA_OD_AGGREGATE_MANIFEST.json"
        ap.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        aggregate_file_sha = sha256_file(ap)
        (work / "MTA_OD_AGGREGATE_MANIFEST.sha256").write_text(
            f"{aggregate_file_sha}  MTA_OD_AGGREGATE_MANIFEST.json\n", encoding="utf-8"
        )
        (work / "SHA256SUMS.txt").write_text(
            "".join(f"{r['sha256']}  {r['filename']}\n" for r in records), encoding="utf-8"
        )
        acquisition = {
            "status": "ACQUIRED",
            "closure": "1453 indices complete; 0..1452 continuous; total rows 72639113; per-chunk SHA-256 aggregate manifest closed",
            "canonical_folder_id": CANONICAL_FOLDER_ID,
            "source_folder_ids": source_ids,
            "source": DATA_URL,
            "version": {
                "rowsUpdatedAt": EXPECTED_ROWS_UPDATED_AT,
                "viewLastModified": EXPECTED_VIEW_LAST_MODIFIED,
                "row_count": EXPECTED_ROWS,
            },
            "order": ":id",
            "chunk_size": CHUNK_SIZE,
            "chunk_count": EXPECTED_CHUNKS,
            "chunk_sha256_aggregate": aggregate_digest,
            "aggregate_manifest_sha256": aggregate_file_sha,
            "refilled_indices": refill_indices,
        }
        (work / "ACQUISITION_MANIFEST.json").write_text(json.dumps(acquisition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (work / "ACQUISITION_CLOSED.json").write_text(json.dumps(acquisition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for name in [
            "MTA_OD_AGGREGATE_MANIFEST.json",
            "MTA_OD_AGGREGATE_MANIFEST.sha256",
            "SHA256SUMS.txt",
            "ACQUISITION_MANIFEST.json",
            "ACQUISITION_CLOSED.json",
        ]:
            rclone_upload(remote, CANONICAL_FOLDER_ID, work / name, name)
        print(json.dumps(acquisition, indent=2, sort_keys=True))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["shard", "finalize"], required=True)
    p.add_argument("--shard", type=int)
    p.add_argument("--remote", default=os.environ.get("REMOTE_NAME", ""))
    a = p.parse_args()
    if not a.remote:
        raise SystemExit("--remote/REMOTE_NAME required")
    if a.mode == "shard":
        if a.shard is None:
            raise SystemExit("--shard required for shard mode")
        shard_mode(a.remote, a.shard)
    else:
        finalize_mode(a.remote)


if __name__ == "__main__":
    main()
