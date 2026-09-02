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

DISCOVERY_PARENT = (
    "ResearchData/transport_challenges/OpenCity_Traffic_Challenge_2026/"
    "01_public_reference_data/public_data_registry_acquisition_20260902"
)

# Folder objects observed from the original eight-runner rclone destination race.
# These are transport locations only; the canonical consolidated object remains
# mod.CANONICAL_FOLDER_ID. Keep all readable objects in the union so downloaded
# source bytes are reused before any provider refill is attempted.
OBSERVED_RACE_FOLDER_IDS = [
    "1DEamJPCebt6rIhhW3Qpivxh_DG4cTMut",
    "1E359BirG1whXGaRgCYEGhoSslaReMEe5",
    "12jzAigIsi5q5OUUuz0MlMuhUFpimtYzu",
    "1mFq1mCUDB-RRpG2VoSFt8syxWGGDoOjI",
    "130kSsP6Zfe0naibmgYloRUv5BP0Z8849",
]


def discover_source_ids(remote):
    candidates = set(OBSERVED_RACE_FOLDER_IDS)
    candidates.update(getattr(mod, "KNOWN_SOURCE_FOLDER_IDS", []))

    try:
        p = mod.run([
            "rclone", "lsjson", f"{remote}:{DISCOVERY_PARENT}",
            "--dirs-only", "--max-depth", "1",
        ])
        entries = json.loads(p.stdout or b"[]")
        for x in entries:
            if x.get("IsDir") and x.get("Name") == mod.SOURCE_FOLDER_NAME and x.get("ID"):
                candidates.add(x["ID"])
    except Exception as exc:
        print(json.dumps({"path_discovery_warning": str(exc)}, indent=2), flush=True)

    extras = os.environ.get("MTA_OD_EXTRA_SOURCE_FOLDER_IDS", "").strip()
    if extras:
        candidates.update(x.strip() for x in extras.split(",") if x.strip())

    verified = []
    unreadable = []
    for fid in sorted(candidates):
        try:
            mod.rclone_lsjson(remote, fid)
            verified.append(fid)
        except Exception as exc:
            unreadable.append({"folder_id": fid, "error": str(exc)[-500:]})

    if not verified:
        raise RuntimeError("no readable MTA OD race/source folder objects remain")

    print(json.dumps({
        "drive_discovery_parent": DISCOVERY_PARENT,
        "source_folder_name": mod.SOURCE_FOLDER_NAME,
        "source_folder_ids": verified,
        "source_folder_count": len(verified),
        "unreadable_candidate_folder_ids": unreadable,
    }, indent=2), flush=True)
    return verified


def provider_snapshot_without_count_query(session):
    """Validate the frozen data revision without repeated count(*) queries.

    `rowsUpdatedAt` is the Socrata data-row revision marker. `viewLastModified`
    can change when metadata/view configuration changes without altering rows,
    so it is recorded but is deliberately not a data-integrity gate. The total
    row count and 50k/:id chunk contract were already frozen by producer run
    33615833576. A missing chunk is refetched only while rowsUpdatedAt remains
    exactly equal to that producer revision.
    """
    last = None
    for attempt in range(8):
        try:
            r = session.get(mod.META_URL, timeout=120)
            r.raise_for_status()
            meta = r.json()
            rows_updated = int(meta.get("rowsUpdatedAt") or 0)
            view_modified = int(meta.get("viewLastModified") or 0)
            return {
                "rowsUpdatedAt": rows_updated,
                "viewLastModified": view_modified,
                "expected_viewLastModified_at_original_pin": mod.EXPECTED_VIEW_LAST_MODIFIED,
                "viewLastModified_is_nonbinding_metadata": True,
                "row_count": mod.EXPECTED_ROWS,
                "row_count_evidence": "frozen producer contract; live count(*) intentionally skipped",
                "matches_required_version": rows_updated == mod.EXPECTED_ROWS_UPDATED_AT,
            }
        except Exception as exc:
            last = exc
            if attempt < 7:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"metadata row-revision probe failed after retries: {last}")


mod.discover_source_ids = discover_source_ids
mod.provider_snapshot = provider_snapshot_without_count_query
mod.main()
