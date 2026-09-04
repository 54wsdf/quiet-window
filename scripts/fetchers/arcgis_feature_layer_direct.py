#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def run(cmd):
    subprocess.run(cmd, check=True)


def out(cmd):
    return subprocess.check_output(cmd, text=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rclone_put(path: Path, dest: str, cfg: str, remote: str) -> None:
    run([
        'rclone', 'copyto', str(path), f'{remote}:{dest}', '--config', cfg,
        '--retries', '8', '--low-level-retries', '20', '--transfers', '1', '--checkers', '4',
        '--drive-chunk-size', '128M', '--stats', '60s', '--stats-one-line',
    ])
    rows = json.loads(out(['rclone', 'lsjson', f'{remote}:{dest}', '--config', cfg, '--files-only']))
    size = int(rows[0]['Size']) if rows else -1
    if size != path.stat().st_size:
        raise RuntimeError(f'remote size mismatch: {size} != {path.stat().st_size}')


def get_json(session: requests.Session, url: str, params: dict) -> dict:
    for attempt in range(8):
        try:
            r = session.get(url, params=params, timeout=(30, 300))
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get('error'):
                raise RuntimeError(data['error'])
            return data
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(120, 2 ** attempt))
    raise RuntimeError('unreachable')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer-url', required=True)
    ap.add_argument('--dataset-id', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--filename', required=True)
    ap.add_argument('--drive-path', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--remote', required=True)
    ap.add_argument('--chunk-size', type=int, default=1000)
    args = ap.parse_args()

    session = requests.Session()
    session.headers['User-Agent'] = 'quiet-window-public-data-worker/3.0'
    query_url = args.layer_url.rstrip('/') + '/query'

    ids_payload = get_json(session, query_url, {
        'where': '1=1', 'returnIdsOnly': 'true', 'f': 'json'
    })
    object_ids = sorted(int(x) for x in (ids_payload.get('objectIds') or []))
    if not object_ids:
        raise RuntimeError('ArcGIS layer exposes no object IDs')

    with tempfile.TemporaryDirectory(prefix='qw-arcgis-') as td:
        td = Path(td)
        csv_path = td / args.filename
        fieldnames = None
        written = 0
        with csv_path.open('w', encoding='utf-8', newline='') as f:
            writer = None
            for offset in range(0, len(object_ids), args.chunk_size):
                block = object_ids[offset: offset + args.chunk_size]
                data = get_json(session, query_url, {
                    'objectIds': ','.join(str(x) for x in block),
                    'outFields': '*', 'returnGeometry': 'false', 'f': 'json'
                })
                rows = [x.get('attributes') or {} for x in data.get('features', [])]
                if not rows:
                    continue
                if fieldnames is None:
                    fieldnames = list(rows[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                assert writer is not None
                writer.writerows(rows)
                written += len(rows)
                print(json.dumps({'dataset_id': args.dataset_id, 'written': written, 'total_ids': len(object_ids)}), flush=True)

        if written <= 0 or csv_path.stat().st_size <= 0:
            raise RuntimeError('ArcGIS layer returned no rows')

        digest = sha256(csv_path)
        dest = f'{args.drive_path}/{args.filename}'
        rclone_put(csv_path, dest, args.config, args.remote)

        manifest = {
            'schema': 'quiet-window-arcgis-feature-layer-v1',
            'state': 'ACQUIRED',
            'dataset_id': args.dataset_id,
            'name': args.name,
            'source': args.layer_url,
            'tier': 'A/S',
            'object_id_count': len(object_ids),
            'downloaded_rows': written,
            'fields': fieldnames,
            'file': args.filename,
            'bytes': csv_path.stat().st_size,
            'sha256': digest,
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'github_run_id': os.environ.get('GITHUB_RUN_ID'),
        }
        mp = td / f'MANIFEST_{args.dataset_id}_{os.environ.get("GITHUB_RUN_ID", "manual")}.json'
        mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        rclone_put(mp, f'{args.drive_path}/{mp.name}', args.config, args.remote)

    print(json.dumps({'dataset_id': args.dataset_id, 'rows': written, 'sha256': digest}), flush=True)


if __name__ == '__main__':
    main()
