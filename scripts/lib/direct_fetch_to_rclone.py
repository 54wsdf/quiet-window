#!/usr/bin/env python3
"""下载一个公开文件，校验并生成 manifest，供 workflow 直接写入长期存储。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


def safe_filename(url: str, requested: str | None) -> str:
    if requested:
        name = requested.strip()
    else:
        name = Path(urlparse(url).path).name or "payload.bin"
    if not name or name in {".", ".."}:
        raise SystemExit("无效文件名")
    if "/" in name or "\\" in name or "\x00" in name:
        raise SystemExit("文件名不得包含路径分隔符")
    return name


def validate_public_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme != "https" or not p.netloc:
        raise SystemExit("仅允许公开 HTTPS URL")
    if p.username or p.password:
        raise SystemExit("URL 不得内嵌用户名或密码")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--filename")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--expected-sha256", default="")
    ap.add_argument("--terms-url", default="")
    ap.add_argument("--time-coverage", default="")
    args = ap.parse_args()

    validate_public_url(args.url)
    dataset_id = args.dataset_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", dataset_id):
        raise SystemExit("dataset-id 仅允许字母、数字、点、下划线和连字符")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(args.url, args.filename)
    target = out / filename

    session = requests.Session()
    session.headers["User-Agent"] = "quiet-window-public-data-worker/1.0"

    sha = hashlib.sha256()
    total = 0
    content_type = ""
    with session.get(args.url, stream=True, timeout=(30, 1800), allow_redirects=True) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        with target.open("wb") as f:
            for chunk in r.iter_content(8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                sha.update(chunk)
                total += len(chunk)

    if total <= 0:
        raise SystemExit("下载结果为零字节")

    digest = sha.hexdigest()
    expected = args.expected_sha256.strip().lower()
    if expected and digest != expected:
        raise SystemExit(f"SHA256 不匹配: expected={expected} actual={digest}")

    manifest = {
        "schema": "quiet-window-acquisition-manifest-v1",
        "state": "DOWNLOADED_TEMP",
        "dataset_id": dataset_id,
        "provider_url": args.url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "terms_url": args.terms_url.strip() or None,
        "time_coverage": args.time_coverage.strip() or None,
        "original_filename": filename,
        "bytes": total,
        "sha256": digest,
        "content_type": content_type,
        "github_repository": os.environ.get("GITHUB_REPOSITORY"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }
    manifest_path = out / f"{filename}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 只向日志输出非敏感统计，不回显 URL。
    print(json.dumps({
        "dataset_id": dataset_id,
        "filename": filename,
        "bytes": total,
        "sha256": digest,
        "manifest": manifest_path.name,
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"网络获取失败: {exc.__class__.__name__}", file=sys.stderr)
        raise SystemExit(2)
