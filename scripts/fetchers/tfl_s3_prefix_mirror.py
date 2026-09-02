#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests

BUCKET = 'crowding.data.tfl.gov.uk'
S3_LIST = f'https://s3-eu-west-1.amazonaws.com/{BUCKET}/'
PUBLIC_ROOT = f'https://{BUCKET}/'
NS = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def run(cmd):
    subprocess.run(cmd, check=True)


def remote_stat(dest: str, cfg: str, remote: str):
    p = subprocess.run(
        ['rclone', 'lsjson', f'{remote}:{dest}', '--config', cfg, '--files-only'],
        text=True, capture_output=True,
    )
    if p.returncode != 0:
        return None
    try:
        rows = json.loads(p.stdout)
        return rows[0] if rows else None
    except Exception:
        return None


def visible_remote_size(dest: str, cfg: str, remote: str) -> int:
    for n in range(10):
        stat = remote_stat(dest, cfg, remote)
        if stat and isinstance(stat.get('Size'), int):
            return int(stat['Size'])
        if n < 9:
            time.sleep(min(30, 2 + n * 3))
    raise RuntimeError(f'remote object not visible after upload: {dest}')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def list_objects(session: requests.Session, prefix: str) -> list[dict]:
    rows = []
    token = None
    while True:
        params = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000'}
        if token:
            params['continuation-token'] = token
        r = session.get(S3_LIST, params=params, timeout=(20, 120))
        r.raise_for_status()
        doc = ET.fromstring(r.content)
        for c in doc.findall('.//s3:Contents', NS):
            key = c.findtext('s3:Key', default='', namespaces=NS) or ''
            size = int(c.findtext('s3:Size', default='0', namespaces=NS) or '0')
            if key != prefix and size > 0:
                rows.append({'key': key, 'size': size})
        if doc.findtext('s3:IsTruncated', default='false', namespaces=NS) != 'true':
            break
        token = doc.findtext('s3:NextContinuationToken', default='', namespaces=NS)
        if not token:
            raise RuntimeError('TfL S3 pagination stalled')
    return rows


def safe_rel(key: str, prefix: str) -> str:
    if not key.startswith(prefix):
        raise RuntimeError(f'key outside prefix: {key}')
    rel = key[len(prefix):].lstrip('/')
    parts = PurePosixPath(rel).parts
    if not parts or any(p in ('', '.', '..') for p in parts):
        raise RuntimeError(f'unsafe key: {key}')
    return '/'.join(parts)


def fetch(session: requests.Session, url: str, path: Path, expected: int):
    for attempt in range(8):
        try:
            with session.get(url, stream=True, timeout=(30, 1800), allow_redirects=True) as r:
                r.raise_for_status()
                with path.open('wb') as f:
                    for block in r.iter_content(8 * 1024 * 1024):
                        if block:
                            f.write(block)
            if path.stat().st_size != expected:
                raise RuntimeError(f'source size mismatch: {path.stat().st_size} != {expected}')
            return
        except Exception:
            path.unlink(missing_ok=True)
            if attempt == 7:
                raise
            time.sleep(min(120, 2 ** attempt))


def put(path: Path, dest: str, cfg: str, remote: str):
    run([
        'rclone', 'copyto', str(path), f'{remote}:{dest}', '--config', cfg,
        '--retries', '8', '--low-level-retries', '20', '--transfers', '1', '--checkers', '4',
        '--drive-chunk-size', '128M', '--stats', '60s', '--stats-one-line',
    ])
    size = visible_remote_size(dest, cfg, remote)
    if size != path.stat().st_size:
        raise RuntimeError(f'remote size mismatch: {size} != {path.stat().st_size}: {dest}')


def verified(dest: str, expected: int, cfg: str, remote: str) -> bool:
    raw = remote_stat(dest, cfg, remote)
    side = remote_stat(dest + '.sha256', cfg, remote)
    return bool(raw and raw.get('Size') == expected and side and isinstance(side.get('Size'), int) and side['Size'] > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--dataset-id', required=True)
    ap.add_argument('--drive-path', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--remote', required=True)
    args = ap.parse_args()

    prefix = args.prefix if args.prefix.endswith('/') else args.prefix + '/'
    session = requests.Session()
    session.headers.update({'User-Agent': 'quiet-window-public-data-worker/3.0'})
    rows = list_objects(session, prefix)
    if not rows:
        raise RuntimeError(f'no objects for TfL prefix: {prefix}')

    records = []
    with tempfile.TemporaryDirectory(prefix='qw-tfl-prefix-') as tmp:
        td = Path(tmp)
        for index, row in enumerate(rows, 1):
            key = row['key']
            expected = int(row['size'])
            rel = safe_rel(key, prefix)
            dest = f"{args.drive_path.rstrip('/')}/{rel}"
            source_url = PUBLIC_ROOT + quote(key, safe='/')
            if verified(dest, expected, args.config, args.remote):
                records.append({'key': key, 'file': rel, 'bytes': expected, 'state': 'SKIPPED_REMOTE_VERIFIED'})
                print(json.dumps({'state': 'SKIP', 'done': index, 'total': len(rows), 'key': key}), flush=True)
                continue

            local = td / Path(rel).name
            fetch(session, source_url, local, expected)
            digest = sha256(local)
            put(local, dest, args.config, args.remote)
            sidecar = td / (Path(rel).name + '.sha256')
            sidecar.write_text(f'{digest}  {Path(rel).name}\n', encoding='utf-8')
            put(sidecar, dest + '.sha256', args.config, args.remote)
            records.append({
                'key': key, 'file': rel, 'source_url': source_url, 'bytes': expected,
                'sha256': digest, 'state': 'ACQUIRED',
            })
            local.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            print(json.dumps({'state': 'ACQUIRED', 'done': index, 'total': len(rows), 'key': key, 'bytes': expected}), flush=True)

        manifest = {
            'schema': 'quiet-window-tfl-s3-prefix-v1',
            'state': 'ACQUIRED',
            'dataset_id': args.dataset_id,
            'source_bucket': BUCKET,
            'source_prefix': prefix,
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'github_run_id': os.environ.get('GITHUB_RUN_ID'),
            'files': records,
            'total_files': len(records),
            'total_bytes': sum(int(x.get('bytes', 0)) for x in records),
        }
        mp = td / f'MANIFEST_{args.dataset_id}_{os.environ.get("GITHUB_RUN_ID", "manual")}.json'
        mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        put(mp, f"{args.drive_path.rstrip('/')}/{mp.name}", args.config, args.remote)

    print(json.dumps({
        'dataset_id': args.dataset_id,
        'files': len(records),
        'bytes': sum(int(x.get('bytes', 0)) for x in records),
        'acquired': sum(1 for x in records if x['state'] == 'ACQUIRED'),
        'skipped_verified': sum(1 for x in records if x['state'] == 'SKIPPED_REMOTE_VERIFIED'),
    }), flush=True)


if __name__ == '__main__':
    main()
