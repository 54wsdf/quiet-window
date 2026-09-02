#!/usr/bin/env python3
import importlib.util
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE = HERE / "mta_od_drive_union_salvage.py"
spec = importlib.util.spec_from_file_location("mta_od_salvage_core", CORE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Exact-name Drive inventory taken after producer run 33615833576 completed.
# These five folder objects were created during the producer time window and are
# the current authoritative source set for the union.  Do not resolve the
# ambiguous pathname public_data_registry_acquisition_20260902: that parent name
# itself exists as multiple real Drive objects and hides sibling race trees.
CURRENT_SOURCE_FOLDER_IDS = [
    "1DEamJPCebt6rIhhW3Qpivxh_DG4cTMut",
    "1E359BirG1whXGaRgCYEGhoSslaReMEe5",
    "12jzAigIsi5q5OUUuz0MlMuhUFpimtYzu",
    "1mFq1mCUDB-RRpG2VoSFt8syxWGGDoOjI",
    "130kSsP6Zfe0naibmgYloRUv5BP0Z8849",
]


def discover_source_ids(remote):
    ids = list(CURRENT_SOURCE_FOLDER_IDS)
    extras = os.environ.get("MTA_OD_EXTRA_SOURCE_FOLDER_IDS", "").strip()
    if extras:
        ids.extend(x.strip() for x in extras.split(",") if x.strip())
    ids = sorted(set(ids))

    verified = []
    unreadable = []
    for fid in ids:
        try:
            mod.rclone_lsjson(remote, fid)
            verified.append(fid)
        except Exception as e:
            unreadable.append({"folder_id": fid, "error": str(e)})

    if unreadable:
        raise RuntimeError(
            "one or more current producer source folder IDs are not readable "
            "through the policy-mandated rclone credential; refusing an "
            "incomplete union: " + json.dumps(unreadable, sort_keys=True)
        )
    if len(verified) < len(CURRENT_SOURCE_FOLDER_IDS):
        raise RuntimeError(
            f"verified current source folder count {len(verified)} < "
            f"required {len(CURRENT_SOURCE_FOLDER_IDS)}"
        )

    print(json.dumps({
        "discovery_authority": "post-producer exact-name Drive object inventory",
        "producer_run_id": os.environ.get("PRODUCER_RUN_ID", "33615833576"),
        "source_folder_name": mod.SOURCE_FOLDER_NAME,
        "source_folder_ids": verified,
        "source_folder_count": len(verified),
        "ambiguous_parent_path_resolution_used": False,
    }, indent=2), flush=True)
    return verified


_original_provider_snapshot = mod.provider_snapshot


def provider_snapshot_with_retry(session):
    last = None
    for attempt in range(1, 9):
        try:
            snap = _original_provider_snapshot(session)
            print(json.dumps({
                "provider_snapshot_attempt": attempt,
                "provider_snapshot": snap,
            }, sort_keys=True), flush=True)
            return snap
        except Exception as e:
            last = e
            if attempt == 8:
                break
            delay = min(30, 2 ** (attempt - 1))
            print(json.dumps({
                "provider_snapshot_attempt": attempt,
                "transient_error": str(e),
                "retry_in_seconds": delay,
            }, sort_keys=True), flush=True)
            time.sleep(delay)
    raise RuntimeError(f"provider snapshot failed after 8 attempts: {last}")


mod.discover_source_ids = discover_source_ids
mod.provider_snapshot = provider_snapshot_with_retry
mod.main()
