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
from urllib.parse import quote, urlencode

import requests

BUCKET = 'crowding.data.tfl.gov.uk'
S3_LIST = f'https://s3-eu-west-1.amazonaws.com/{BUCKET}/'
PUBLIC_ROOT = f'https://{BUCKET}/'
NS = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
DRIVE_ROOT = '01_rail/UK_London_TfL/04_demand_od/NUMBAT'


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
    last = 'not visible'
    for n in range(10):
        stat = remote_stat(dest, cfg, remote)
        if stat and isinstance(stat.get('Size'), int):
            return int(stat['Size'])
        last = str(stat)
        if n < 9:
            time.sleep(min(30, 2 + n * 3))
    raise RuntimeError(f'remote object not visible after upload: {dest}: {last}')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def list_objects(session: requests.Session, prefix: str, *, delimiter: str | None = None) -> list[dict]:
    rows: list[dict] = []
    token = None
    while True:
        params = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000'}
        if delimiter is not None:
            params['delimiter'] = delimiter
        if token:
            params['continuation-token'] = token
        r = session.get(S3_LIST, params=params, timeout=(20, 120))
        r.raise_for_status()
        doc = ET.fromstring(r.content)
        for c in doc.findall('.//s3:Contents', NS):
            key = c.findtext('s3:Key', default='', namespaces=NS) or ''
            size_text = c.findtext('s3:Size', default='0', namespaces=NS) or '0'
            try:
                size = int(size_text)
            except ValueError:
                size = 0
            if key and size > 0:
                rows.append({'key': key, 'size': size})
        if doc.findtext('s3:IsTruncated', default='false', namespaces=NS) != 'true':
            break
        token = doc.findtext('s3:NextContinuationToken', default='', namespaces=NS)
        if not token:
            raise RuntimeError('TfL S3 listing pagination stalled')
    return rows


def safe_relative(key: str, prefix: str) -> str:
    if not key.startswith(prefix):
        raise RuntimeError(f'key outside prefix: {key}')
    rel = key[len(prefix):].lstrip('/')
    parts = PurePosixPath(rel).parts
    if not parts or any(p in ('', '.', '..') for p in parts):
        raise RuntimeError(f'unsafe relative key: {key}')
    return '/'.join(parts)


def fetch(session: requests.Session, url: str, path: Path, expected_size: int) -> None:
    for attempt in range(8):
        try:
            with session.get(url, stream=True, timeout=(30, 1800), allow_redirects=True) as r:
                r.raise_for_status()
                with path.open('wb') as f:
                    for block in r.iter_content(8 * 1024 * 1024):
                        if block:
                            f.write(block)
            got = path.stat().st_size
            if got != expected_size:
                raise RuntimeError(f'source size mismatch: {got} != {expected_size}')
            return
        except Exception:
            path.unlink(missing_ok=True)
            if attempt == 7:
                raise
            time.sleep(min(120, 2 ** attempt))


def put_verified(local: Path, dest: str, cfg: str, remote: str) -> None:
    run([
        'rclone', 'copyto', str(local), f'{remote}:{dest}', '--config', cfg,
        '--retries', '8', '--low-level-retries', '20', '--transfers', '1', '--checkers', '4',
        '--drive-chunk-size', '128M', '--stats', '60s', '--stats-one-line',
    ])
    remote_size = visible_remote_size(dest, cfg, remote)
    if remote_size != local.stat().st_size:
        raise RuntimeError(f'remote size mismatch: {remote_size} != {local.stat().st_size}: {dest}')


def already_verified(dest: str, expected_size: int, cfg: str, remote: str) -> bool:
    raw = remote_stat(dest, cfg, remote)
    side = remote_stat(dest + '.sha256', cfg, remote)
    return bool(
        raw and raw.get('Size') == expected_size
        and side and isinstance(side.get('Size'), int) and side['Size'] > 0
    )


def acquire_group(session: requests.Session, rows: list[dict], prefix: str, dest_base: str, cfg: str, remote: str, td: Path) -> list[dict]:
    records: list[dict] = []
    for index, row in enumerate(rows, 1):
        key = row['key']
        expected_size = int(row['size'])
        rel = safe_relative(key, prefix)
        dest = f'{dest_base}/{rel}'
        source_url = PUBLIC_ROOT + quote(key, safe='/')

        if already_verified(dest, expected_size, cfg, remote):
            records.append({
                'key': key, 'file': rel, 'source_url': source_url, 'bytes': expected_size,
                'state': 'SKIPPED_REMOTE_VERIFIED',
            })
            print(json.dumps({'state': 'SKIP', 'key': key, 'bytes': expected_size}), flush=True)
            continue

        local = td / Path(rel).name
        fetch(session, source_url, local, expected_size)
        digest = sha256(local)
        put_verified(local, dest, cfg, remote)

        sidecar = td / (Path(rel).name + '.sha256')
        sidecar.write_text(f'{digest}  {Path(rel).name}\n', encoding='utf-8')
        put_verified(sidecar, dest + '.sha256', cfg, remote)
        records.append({
            'key': key, 'file': rel, 'source_url': source_url, 'bytes': expected_size,
            'sha256': digest, 'state': 'ACQUIRED',
        })
        local.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        print(json.dumps({
            'state': 'ACQUIRED', 'done': index, 'total': len(rows), 'key': key,
            'bytes': expected_size, 'sha256': digest,
        }), flush=True)
    return records


def write_manifest(records: list[dict], scope: str, dest_base: str, cfg: str, remote: str, td: Path) -> None:
    manifest = {
        'schema': 'quiet-window-acquisition-manifest-v2',
        'state': 'ACQUIRED',
        'source': 'Transport for London NUMBAT public S3 bucket',
        'source_bucket': BUCKET,
        'tier': 'S',
        'scope': scope,
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'github_run_id': os.environ.get('GITHUB_RUN_ID'),
        'files': records,
        'total_files': len(records),
        'total_bytes': sum(int(x.get('bytes', 0)) for x in records),
    }
    name = f'MANIFEST_NUMBAT_{scope}_{os.environ.get("GITHUB_RUN_ID", "manual")}.json'.replace('/', '_')
    mp = td / name
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    put_verified(mp, f'{dest_base}/{name}', cfg, remote)


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--year', type=int)
    group.add_argument('--root-assets', action='store_true')
    ap.add_argument('--config', required=True)
    ap.add_argument('--remote', required=True)
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({'User-Agent': 'quiet-window-public-data-worker/3.0'})

    if args.root_assets:
        prefix = 'NUMBAT/'
        rows = [x for x in list_objects(session, prefix, delimiter='/') if x['key'] != prefix]
        scope = 'root_assets'
        dest_base = DRIVE_ROOT
    else:
        if args.year < 2016 or args.year > 2100:
            raise SystemExit('unsupported NUMBAT year')
        prefix = f'NUMBAT/NUMBAT {args.year}/'
        rows = [x for x in list_objects(session, prefix) if x['key'] != prefix]
        scope = str(args.year)
        dest_base = f'{DRIVE_ROOT}/{args.year}'

    if not rows:
        raise RuntimeError(f'no NUMBAT objects resolved for {scope}')

    with tempfile.TemporaryDirectory(prefix='qw-numbat-') as tmp:
        td = Path(tmp)
        records = acquire_group(session, rows, prefix, dest_base, args.config, args.remote, td)
        write_manifest(records, scope, dest_base, args.config, args.remote, td)

    print(json.dumps({
        'scope': scope,
        'files': len(records),
        'bytes': sum(int(x.get('bytes', 0)) for x in records),
        'acquired': sum(1 for x in records if x['state'] == 'ACQUIRED'),
        'skipped_verified': sum(1 for x in records if x['state'] == 'SKIPPED_REMOTE_VERIFIED'),
    }), flush=True)


if __name__ == '__main__':
    main()
