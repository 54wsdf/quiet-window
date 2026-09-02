#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests


class LinkParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != 'a':
            return
        for key, value in attrs:
            if key.lower() == 'href' and value:
                self.hrefs.append(value)


def run(cmd):
    subprocess.run(cmd, check=True)


def remote_stat(dest, cfg, remote):
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


def visible_remote_size(dest, cfg, remote):
    for n in range(10):
        stat = remote_stat(dest, cfg, remote)
        if stat and isinstance(stat.get('Size'), int):
            return int(stat['Size'])
        if n < 9:
            time.sleep(min(30, 2 + n * 3))
    raise RuntimeError(f'remote object not visible after upload: {dest}')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def put(path, dest, cfg, remote):
    run([
        'rclone', 'copyto', str(path), f'{remote}:{dest}', '--config', cfg,
        '--retries', '8', '--low-level-retries', '20', '--transfers', '1', '--checkers', '4',
        '--drive-chunk-size', '128M', '--stats', '60s', '--stats-one-line',
    ])
    rs = visible_remote_size(dest, cfg, remote)
    if rs != path.stat().st_size:
        raise RuntimeError(f'remote size mismatch: {rs} != {path.stat().st_size}: {dest}')


def download(session, url, path):
    for n in range(8):
        try:
            with session.get(url, stream=True, timeout=(30, 7200), allow_redirects=True) as r:
                r.raise_for_status()
                ct = (r.headers.get('content-type') or '').lower()
                with path.open('wb') as f:
                    for block in r.iter_content(8 * 1024 * 1024):
                        if block:
                            f.write(block)
            if path.stat().st_size <= 0 or ('text/html' in ct and path.stat().st_size < 100000):
                raise RuntimeError('invalid payload')
            return ct
        except Exception:
            path.unlink(missing_ok=True)
            if n == 7:
                raise
            time.sleep(min(120, 2 ** n))


def safe(value):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value or '').strip('_') or 'resource'


def discover_package(session, base, package):
    api = base.rstrip('/') + '/api/3/action/package_show'
    api_error = None
    try:
        r = session.get(api, params={'id': package}, timeout=(30, 180))
        r.raise_for_status()
        payload = r.json()
        if payload.get('success') and isinstance(payload.get('result'), dict):
            pkg = payload['result']
            resources = [
                x for x in pkg.get('resources', [])
                if isinstance(x, dict) and x.get('url') and x.get('url_type', '') != 'upload-disabled'
            ]
            if resources:
                return pkg, resources, 'ckan_api'
    except Exception as exc:
        api_error = str(exc)
        print(json.dumps({'state': 'API_FALLBACK', 'package': package, 'error': api_error}), flush=True)

    page_url = base.rstrip('/') + '/en/dataset/' + package
    r = session.get(page_url, timeout=(30, 180), allow_redirects=True)
    r.raise_for_status()
    parser = LinkParser()
    parser.feed(r.text)

    urls = []
    seen = set()
    for href in parser.hrefs:
        absolute = urljoin(page_url, href)
        path = urlparse(absolute).path
        if '/resource/' not in path or '/download/' not in path:
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)

    if not urls:
        raise RuntimeError(f'CKAN API unavailable and no HTML download resources found: {package}: {api_error}')

    resources = []
    for url in urls:
        path = urlparse(url).path
        name = unquote(Path(path).name) or 'resource'
        m = re.search(r'/resource/([^/]+)/download/', path)
        resources.append({
            'id': m.group(1) if m else None,
            'name': name,
            'url': url,
            'format': Path(name).suffix.lstrip('.').upper(),
            'last_modified': None,
        })
    return {'title': package, 'license_title': None, 'license_url': None}, resources, 'html_fallback'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--package', required=True)
    ap.add_argument('--dataset-id', required=True)
    ap.add_argument('--drive-path', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--remote', required=True)
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; quiet-window-public-data-worker/3.0)',
        'Accept': 'text/html,application/xhtml+xml,application/zip,*/*;q=0.8',
    })
    pkg, resources, discovery_mode = discover_package(session, args.base, args.package)
    print(json.dumps({
        'dataset_id': args.dataset_id, 'package': args.package,
        'discovery_mode': discovery_mode, 'resources': len(resources),
    }), flush=True)

    records = []
    with tempfile.TemporaryDirectory(prefix='qw-ckan-') as tmp:
        td = Path(tmp)
        for i, resource in enumerate(resources, 1):
            url = resource['url']
            raw = resource.get('name') or unquote(Path(urlparse(url).path).name) or f'resource_{i}'
            name = safe(raw)
            ext = Path(unquote(urlparse(url).path)).suffix
            if ext and not name.lower().endswith(ext.lower()):
                name += ext
            if any(x['file'] == name for x in records):
                name = f'{i:03d}_{name}'

            path = td / name
            ct = download(session, url, path)
            size = path.stat().st_size
            digest = sha(path)
            dest = f'{args.drive_path}/{name}'
            put(path, dest, args.config, args.remote)

            sidecar = td / (name + '.sha256')
            sidecar.write_text(f'{digest}  {name}\n', encoding='utf-8')
            put(sidecar, dest + '.sha256', args.config, args.remote)

            records.append({
                'resource_id': resource.get('id'), 'name': resource.get('name'), 'file': name,
                'url': url, 'format': resource.get('format'), 'last_modified': resource.get('last_modified'),
                'bytes': size, 'sha256': digest, 'content_type': ct, 'state': 'ACQUIRED',
            })
            path.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            print(json.dumps({
                'dataset_id': args.dataset_id, 'done': i, 'total': len(resources),
                'file': name, 'bytes': size, 'sha256': digest,
            }), flush=True)

        manifest = {
            'schema': 'quiet-window-ckan-package-v2', 'state': 'ACQUIRED',
            'dataset_id': args.dataset_id, 'package': args.package, 'title': pkg.get('title'),
            'source': args.base, 'tier': 'S/A', 'discovery_mode': discovery_mode,
            'license_title': pkg.get('license_title'), 'license_url': pkg.get('license_url'),
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'github_run_id': os.environ.get('GITHUB_RUN_ID'), 'files': records,
        }
        mp = td / f'MANIFEST_{safe(args.dataset_id)}_{os.environ.get("GITHUB_RUN_ID", "manual")}.json'
        mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        put(mp, f'{args.drive_path}/{mp.name}', args.config, args.remote)

    print(json.dumps({
        'dataset_id': args.dataset_id, 'discovery_mode': discovery_mode,
        'files': len(records), 'bytes': sum(x['bytes'] for x in records),
    }), flush=True)


if __name__ == '__main__':
    main()
