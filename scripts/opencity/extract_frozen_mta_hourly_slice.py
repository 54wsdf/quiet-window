#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

DATASET_ID = "5wq4-mkjj"
EXPECTED_ROW_COUNT = 43_835_841
EXPECTED_PARTS = 439
EXPECTED_CHUNK_ROWS = 100_000
EXPECTED_ROWS_UPDATED_AT = 1787757107
EXPECTED_ACQUISITION_RUN = "33586650747"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def rclone_copyto(remote: str, root_folder_id: str, remote_name: str, local_path: Path) -> None:
    run([
        "rclone", "copyto", f"{remote}:{remote_name}", str(local_path),
        "--drive-root-folder-id", root_folder_id,
        "--retries", "8", "--low-level-retries", "20",
        "--transfers", "1", "--checkers", "4",
    ])


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument("--manifest-folder-id", required=True)
    ap.add_argument("--parts-folder-id", required=True)
    ap.add_argument("--manifest-name", default="MANIFEST_5wq4-mkjj_33586650747.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest-output", type=Path, required=True)
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end-exclusive", default="2026-07-01")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    start_ts = f"{args.start}T00:00:00.000"
    end_ts = f"{args.end_exclusive}T00:00:00.000"

    with tempfile.TemporaryDirectory(prefix="t8-hourly-frozen-") as td_raw:
        td = Path(td_raw)
        source_manifest_path = td / args.manifest_name
        rclone_copyto(args.remote, args.manifest_folder_id, args.manifest_name, source_manifest_path)
        source_manifest_sha = sha256(source_manifest_path)
        source = json.loads(source_manifest_path.read_text(encoding="utf-8"))

        if source.get("state") != "ACQUIRED":
            raise RuntimeError(f"hourly source manifest is not ACQUIRED: {source.get('state')}")
        if source.get("dataset_id") != DATASET_ID:
            raise RuntimeError(f"unexpected hourly dataset id: {source.get('dataset_id')}")
        if int(source.get("row_count") or -1) != EXPECTED_ROW_COUNT:
            raise RuntimeError(f"unexpected hourly archive row count: {source.get('row_count')}")
        if int(source.get("chunk_rows") or -1) != EXPECTED_CHUNK_ROWS:
            raise RuntimeError(f"unexpected hourly chunk contract: {source.get('chunk_rows')}")
        if str(source.get("github_run_id")) != EXPECTED_ACQUISITION_RUN:
            raise RuntimeError(f"unexpected hourly acquisition run: {source.get('github_run_id')}")

        parts = source.get("parts") or []
        if len(parts) != EXPECTED_PARTS:
            raise RuntimeError(f"unexpected hourly archive part count: {len(parts)}")
        indices = [int(p["part"]) for p in parts]
        if indices != list(range(EXPECTED_PARTS)):
            raise RuntimeError("hourly archive part indices are not exactly 0..438")

        required = {
            "transit_timestamp", "station_complex_id", "station_complex",
            "ridership", "latitude", "longitude",
        }
        aggregate: dict[tuple[str, str, str, str, str], Decimal] = defaultdict(Decimal)
        verified_rows = 0
        selected_raw_rows = 0
        min_timestamp: str | None = None
        max_timestamp: str | None = None
        verified_hash_lines: list[str] = []

        for ordinal, part in enumerate(parts, start=1):
            fn = str(part["file"])
            local = td / "current.csv"
            local.unlink(missing_ok=True)
            rclone_copyto(args.remote, args.parts_folder_id, fn, local)

            expected_bytes = int(part["bytes"])
            if local.stat().st_size != expected_bytes:
                raise RuntimeError(f"size mismatch for {fn}: {local.stat().st_size} != {expected_bytes}")
            digest = sha256(local)
            if digest != str(part["sha256"]):
                raise RuntimeError(f"SHA-256 mismatch for {fn}: {digest} != {part['sha256']}")
            verified_hash_lines.append(f"{int(part['part']):05d} {digest}")

            part_rows = 0
            with local.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                fields = set(reader.fieldnames or [])
                missing = sorted(required - fields)
                if missing:
                    raise RuntimeError(f"missing hourly columns in {fn}: {missing}")
                for row in reader:
                    part_rows += 1
                    ts = row["transit_timestamp"]
                    if min_timestamp is None or ts < min_timestamp:
                        min_timestamp = ts
                    if max_timestamp is None or ts > max_timestamp:
                        max_timestamp = ts
                    if not (start_ts <= ts < end_ts):
                        continue
                    try:
                        ridership = Decimal(row["ridership"] or "0")
                    except InvalidOperation as exc:
                        raise RuntimeError(f"invalid ridership in {fn}: {row['ridership']!r}") from exc
                    key = (
                        ts,
                        row["station_complex_id"],
                        row["station_complex"],
                        row["latitude"],
                        row["longitude"],
                    )
                    aggregate[key] += ridership
                    selected_raw_rows += 1

            expected_rows = int(part.get("rows_expected") or -1)
            if part_rows != expected_rows:
                raise RuntimeError(f"row-count mismatch for {fn}: {part_rows} != {expected_rows}")
            verified_rows += part_rows
            local.unlink(missing_ok=True)
            if ordinal % 20 == 0 or ordinal == len(parts):
                print(json.dumps({
                    "verified_parts": ordinal,
                    "total_parts": len(parts),
                    "verified_rows": verified_rows,
                    "selected_raw_rows": selected_raw_rows,
                    "aggregated_rows": len(aggregate),
                }))

        if verified_rows != EXPECTED_ROW_COUNT:
            raise RuntimeError(f"archive verified row total mismatch: {verified_rows} != {EXPECTED_ROW_COUNT}")
        if len(aggregate) < 1000:
            raise RuntimeError(f"unexpectedly small frozen June aggregate: {len(aggregate)} rows")

        with args.output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "transit_timestamp", "station_complex_id", "station_complex",
                "latitude", "longitude", "ridership",
            ])
            for key in sorted(aggregate):
                writer.writerow([*key, decimal_text(aggregate[key])])

        ordered_hash_aggregate = hashlib.sha256(
            ("\n".join(verified_hash_lines) + "\n").encode("utf-8")
        ).hexdigest()
        output_sha = sha256(args.output)
        result = {
            "state": "DERIVED_FROM_FROZEN_ARCHIVE",
            "evidence_class": "public_data registry / official public source frozen-archive derivation",
            "dataset_id": DATASET_ID,
            "source_archive": {
                "manifest_name": args.manifest_name,
                "manifest_sha256": source_manifest_sha,
                "manifest_folder_id": args.manifest_folder_id,
                "parts_folder_id": args.parts_folder_id,
                "acquisition_run": int(EXPECTED_ACQUISITION_RUN),
                "row_count": EXPECTED_ROW_COUNT,
                "part_count": EXPECTED_PARTS,
                "chunk_rows": EXPECTED_CHUNK_ROWS,
                "rowsUpdatedAt": EXPECTED_ROWS_UPDATED_AT,
                "verified_part_count": len(parts),
                "verified_row_count": verified_rows,
                "ordered_part_sha256_aggregate": ordered_hash_aggregate,
            },
            "period": {"start": args.start, "end_exclusive": args.end_exclusive},
            "archive_timestamp_span": {"min": min_timestamp, "max": max_timestamp},
            "selected_raw_rows": selected_raw_rows,
            "aggregated_station_hour_rows": len(aggregate),
            "output": {
                "file": args.output.name,
                "bytes": args.output.stat().st_size,
                "sha256": output_sha,
            },
            "derivation": "All 439 frozen Drive parts are byte-verified against the closed acquisition manifest before filtering. June 2026 rows are grouped by transit_timestamp, station_complex_id, station_complex, latitude, longitude and ridership is summed exactly with Decimal arithmetic. No live hourly provider bytes are used.",
        }
        args.manifest_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
