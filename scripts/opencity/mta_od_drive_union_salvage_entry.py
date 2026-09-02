#!/usr/bin/env python3
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE = HERE / "mta_od_drive_union_salvage.py"
spec = importlib.util.spec_from_file_location("mta_od_salvage_core", CORE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DISCOVERY_PARENT = (
    "ResearchData/transport_challenges/OpenCity_Traffic_Challenge_2026/"
    "01_public_reference_data/public_data_registry_acquisition_20260902"
)


def discover_source_ids(remote):
    p = mod.run([
        "rclone", "lsjson", f"{remote}:{DISCOVERY_PARENT}",
        "--dirs-only", "--max-depth", "1",
    ])
    entries = json.loads(p.stdout or b"[]")
    matched = []
    for x in entries:
        if x.get("IsDir") and x.get("Name") == mod.SOURCE_FOLDER_NAME and x.get("ID"):
            matched.append(x["ID"])
    extras = os.environ.get("MTA_OD_EXTRA_SOURCE_FOLDER_IDS", "").strip()
    if extras:
        matched.extend(x.strip() for x in extras.split(",") if x.strip())
    ids = sorted(set(matched))
    if not ids:
        raise RuntimeError(
            f"no readable {mod.SOURCE_FOLDER_NAME!r} folders discovered under {DISCOVERY_PARENT}"
        )
    verified = []
    for fid in ids:
        mod.rclone_lsjson(remote, fid)
        verified.append(fid)
    print(json.dumps({
        "drive_discovery_parent": DISCOVERY_PARENT,
        "source_folder_name": mod.SOURCE_FOLDER_NAME,
        "source_folder_ids": verified,
        "source_folder_count": len(verified),
    }, indent=2), flush=True)
    return verified


mod.discover_source_ids = discover_source_ids
mod.main()
