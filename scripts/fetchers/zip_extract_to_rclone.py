#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import requests

TEXT_SUFFIXES = {'.csv', '.tsv', '.txt', '.json', '.jsonl'}


def run(cmd):
    subprocess.run(cmd, check=True)


def output(cmd):
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
    rows = json.loads(output(['rclone', 'lsjson', f'{remote}:{dest}', '--config', cfg, '--files-only']))
    remote_size = int(rows[0]['Size']) if rows else -1
    if remote_size != path.stat().st_size:
        raise RuntimeError(f'remote size mismatch: {remote_size} != {path.stat().st_size}: {dest}')


def safe_zip_name(name: str) -> str:
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts:
        raise RuntimeError(f'unsafe zip member: {name}')
    return str(p)


def download(session: requests.Session, url: str, dest: Path) -> None:
    for attempt in range(8):
        try:
            with session.get(url, stream=True, timeout=(30, 7200), allow_redirects=True) as r:
                r.raise_for_status()
                with dest.open('wb') as f:
                    for block in r.iter_content(8 * 1024 * 1024):
                        if block:
                            f.write(block)
            if dest.stat().st_size <= 0:
                raise RuntimeError('zero-byte zip')
            return
        except Exception:
            dest.unlink(missing_ok=True)
            if attempt == 7:
                raise
            time.sleep(min(120, 2 ** attempt))


def split_binary(path: Path, out_dir: Path, target_bytes: int) -> list[dict]:
    rows = []
    with path.open('rb') as src:
        idx = 0
        while True:
            part = out_dir / f'{path.name}.part-{idx:05d}'
            remaining = target_bytes
            wrote = 0
            with part.open('wb') as dst:
                while remaining > 0:
                    block = src.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        break
                    dst.write(block)
                    wrote += len(block)
                    remaining -= len(block)
            if wrote == 0:
                part.unlink(missing_ok=True)
                break
            rows.append({'file': part.name, 'bytes': wrote, 'sha256': sha256(part)})
            idx += 1
    return rows


def split_text_with_header(path: Path, out_dir: Path, target_bytes: int) -> list[dict]:
    rows = []
    with path.open('rb') as src:
        header = src.readline()
        idx = 0
        part = None
        dst = None
        current = 0
        try:
            for line in src:
                if dst is None or (current + len(line) > target_bytes and current > len(header)):
                    if dst is not None:
                        dst.close()
                        rows.append({'file': part.name, 'bytes': part.stat().st_size, 'sha256': sha256(part)})
                    part = out_dir / f'{path.stem}.part-{idx:05d}{path.suffix}'
                    dst = part.open('wb')
                    dst.write(header)
                    current = len(header)
                    idx += 1
                dst.write(line)
                current += len(line)
            if dst is None:
                part = out_dir / f'{path.stem}.part-00000{path.suffix}'
                part.write_bytes(header)
                rows.append({'file': part.name, 'bytes': part.stat().st_size, 'sha256': sha256(part)})
            else:
                dst.close()
                dst = None
                rows.append({'file': part.name, 'bytes': part.stat().st_size, 'sha256': sha256(part)})
        finally:
            if dst is not None:
                dst.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--dataset-id', required=True)
    ap.add_argument('--drive-path', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--remote', required=True)
    ap.add_argument('--direct-file-max-mib', type=int, default=220)
    ap.add_argument('--partition-target-mib', type=int, default=160)
    args = ap.parse_args()

    direct_max = args.direct_file_max_mib * 1024 * 1024
    target = args.partition_target_mib * 1024 * 1024
    session = requests.Session()
    session.headers['User-Agent'] = 'quiet-window-public-data-worker/3.1'

    with tempfile.TemporaryDirectory(prefix='qw-zip-extract-') as tmp:
        td = Path(tmp)
        zip_path = td / (Path(urlparse(args.url).path).name or 'source.zip')
        download(session, args.url, zip_path)
        zip_digest = sha256(zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f'zip CRC failure: {bad}')
            infos = [x for x in zf.infolist() if not x.is_dir()]
            inventory = [{
                'name': safe_zip_name(x.filename), 'bytes_uncompressed': x.file_size,
                'bytes_compressed': x.compress_size, 'crc32': f'{x.CRC:08x}'
            } for x in infos]
            inv = {
                'schema': 'quiet-window-zip-inventory-v1', 'state': 'INVENTORIED',
                'dataset_id': args.dataset_id, 'source_url': args.url,
                'zip_bytes': zip_path.stat().st_size, 'zip_sha256': zip_digest,
                'member_count': len(infos), 'members': inventory,
                'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'github_run_id': os.environ.get('GITHUB_RUN_ID'),
            }
            inv_path = td / f'ZIP_INVENTORY_{args.dataset_id}_{os.environ.get("GITHUB_RUN_ID", "manual")}.json'
            inv_path.write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding='utf-8')
            rclone_put(inv_path, f'{args.drive_path}/{inv_path.name}', args.config, args.remote)

            records = []
            for i, info in enumerate(infos, 1):
                member_name = safe_zip_name(info.filename)
                local = td / 'member'
                local.unlink(missing_ok=True)
                with zf.open(info) as src, local.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                if local.stat().st_size != info.file_size:
                    raise RuntimeError(f'extracted size mismatch: {member_name}')
                digest = sha256(local)
                base_name = Path(member_name).name
                safe_dir = str(PurePosixPath(member_name).parent)
                if safe_dir == '.':
                    safe_dir = ''
                remote_base = f'{args.drive_path}/extracted' + (f'/{safe_dir}' if safe_dir else '')

                if local.stat().st_size <= direct_max:
                    dest = f'{remote_base}/{base_name}'
                    rclone_put(local, dest, args.config, args.remote)
                    rec = {
                        'member': member_name, 'bytes': local.stat().st_size, 'sha256': digest,
                        'storage_mode': 'exact_extracted_file', 'drive_path': dest,
                    }
                else:
                    part_dir = td / 'parts'
                    if part_dir.exists():
                        shutil.rmtree(part_dir)
                    part_dir.mkdir()
                    suffix = Path(base_name).suffix.lower()
                    if suffix in TEXT_SUFFIXES:
                        parts = split_text_with_header(local, part_dir, target)
                        mode = 'text_rows_header_repeated'
                    else:
                        parts = split_binary(local, part_dir, target)
                        mode = 'exact_binary_concat'
                    for part in parts:
                        p = part_dir / part['file']
                        dest = f'{remote_base}/{base_name}.parts/{part["file"]}'
                        rclone_put(p, dest, args.config, args.remote)
                        part['drive_path'] = dest
                        p.unlink()
                    rec = {
                        'member': member_name, 'bytes': local.stat().st_size, 'sha256': digest,
                        'storage_mode': mode, 'partition_target_bytes': target, 'parts': parts,
                        'canonical_note': 'Original corrected ZIP remains the byte-exact canonical source.'
                    }
                records.append(rec)
                local.unlink(missing_ok=True)
                print(json.dumps({'dataset_id': args.dataset_id, 'done': i, 'total': len(infos), 'member': member_name, 'mode': rec['storage_mode']}), flush=True)

        manifest = {
            'schema': 'quiet-window-zip-extraction-manifest-v1', 'state': 'ACQUIRED',
            'dataset_id': args.dataset_id, 'source_url': args.url,
            'zip_sha256': zip_digest, 'zip_bytes': zip_path.stat().st_size,
            'member_count': len(records), 'members': records,
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'github_run_id': os.environ.get('GITHUB_RUN_ID'),
        }
        mp = td / f'MANIFEST_EXTRACTED_{args.dataset_id}_{os.environ.get("GITHUB_RUN_ID", "manual")}.json'
        mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        rclone_put(mp, f'{args.drive_path}/{mp.name}', args.config, args.remote)

    print(json.dumps({'dataset_id': args.dataset_id, 'members': len(records), 'state': 'ACQUIRED'}), flush=True)


if __name__ == '__main__':
    main()
