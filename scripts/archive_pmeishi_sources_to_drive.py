#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import re
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

UA = "quiet-window-pmeishi-archive/1.0"
IMG_RE = re.compile(r"https?://[^\s\"'<>]+?\.(?:jpe?g|png|webp|gif)(?:\?[^\s\"'<>]*)?", re.I)
SKIP_HINTS = ("avatar", "icon", "favicon", "emoji", "logo", "profile", "ogp_default")


def safe_name(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", text).strip("-._")
    return text or "item"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ext_for(url: str, ctype: str | None) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ext
    if ctype:
        guessed = mimetypes.guess_extension(ctype.split(";", 1)[0].strip())
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed
    return ".img"


def extract_urls(page_url: str, text: str) -> list[str]:
    soup = BeautifulSoup(text, "html.parser")
    out: list[str] = []
    for meta in soup.select('meta[property="og:image"], meta[name="twitter:image"]'):
        u = meta.get("content")
        if u:
            out.append(urljoin(page_url, u))
    for tag in soup.find_all(["img", "source"]):
        for attr in ("src", "data-src", "data-original"):
            u = tag.get(attr)
            if u:
                out.append(urljoin(page_url, u))
        for attr in ("srcset", "data-srcset"):
            s = tag.get(attr)
            if s:
                for part in s.split(","):
                    u = part.strip().split(" ", 1)[0]
                    if u:
                        out.append(urljoin(page_url, u))
    raw = html.unescape(text).replace("\\/", "/")
    out.extend(IMG_RE.findall(raw))
    seen, cleaned = set(), []
    for u in out:
        if u.startswith("//"):
            u = "https:" + u
        if not u.startswith("https://"):
            continue
        lu = u.lower()
        if any(h in lu for h in SKIP_HINTS):
            continue
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


def image_info(data: bytes):
    try:
        with Image.open(BytesIO(data)) as im:
            return im.width, im.height, (im.format or "").lower()
    except Exception:
        return None


def download_image(session: requests.Session, url: str):
    r = session.get(url, timeout=40, headers={"Referer": url}, allow_redirects=True)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if not ctype.lower().startswith("image/"):
        return None
    data = r.content
    if len(data) < 10_000:
        return None
    info = image_info(data)
    if not info:
        return None
    w, h, fmt = info
    if max(w, h) < 450 or w * h < 180_000:
        return None
    return data, ctype, w, h, fmt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    cfg = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept-Language": "ja,en;q=0.8"})

    archive = {
        "schema_version": 1,
        "dataset_id": cfg["dataset_id"],
        "drive_path": cfg["drive_path"],
        "rights_note": cfg.get("rights_note"),
        "entries": [],
    }
    global_sha: set[str] = set()

    for entry in cfg["entries"]:
        eid = entry["id"]
        entry_dir = out_root / safe_name(eid)
        entry_dir.mkdir(parents=True, exist_ok=True)
        candidate_urls = list(entry.get("direct_image_urls", []))
        page_results = []

        for page_url in entry.get("page_urls", []):
            pr = {"url": page_url, "status": None, "found_image_urls": 0}
            try:
                resp = session.get(page_url, timeout=40)
                pr["status"] = resp.status_code
                resp.raise_for_status()
                found = extract_urls(page_url, resp.text)
                pr["found_image_urls"] = len(found)
                candidate_urls.extend(found)
            except Exception as exc:
                pr["error"] = f"{type(exc).__name__}: {exc}"
            page_results.append(pr)

        uniq = []
        seen_url = set()
        for u in candidate_urls:
            if u not in seen_url:
                seen_url.add(u)
                uniq.append(u)

        files = []
        failures = []
        for u in uniq[:80]:
            try:
                got = download_image(session, u)
                if not got:
                    continue
                data, ctype, w, h, fmt = got
                digest = sha256(data)
                if digest in global_sha:
                    continue
                global_sha.add(digest)
                ext = ext_for(u, ctype)
                idx = len(files) + 1
                fn = f"{idx:03d}_{digest[:12]}{ext}"
                path = entry_dir / fn
                path.write_bytes(data)
                files.append({
                    "filename": fn,
                    "source_url": u,
                    "bytes": len(data),
                    "sha256": digest,
                    "width": w,
                    "height": h,
                    "format": fmt,
                    "content_type": ctype,
                })
            except Exception as exc:
                failures.append({"url": u, "error": f"{type(exc).__name__}: {exc}"})

        meta = {
            "id": eid,
            "tier": entry.get("tier"),
            "creator": entry.get("creator"),
            "title": entry.get("title"),
            "page_urls": entry.get("page_urls", []),
            "direct_image_urls": entry.get("direct_image_urls", []),
            "page_results": page_results,
            "files": files,
            "failures": failures[:20],
        }
        (entry_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        archive["entries"].append(meta)
        print(f"{eid}: {len(files)} images")

    all_files = [p for p in out_root.rglob("*") if p.is_file()]
    archive["summary"] = {
        "entry_count": len(archive["entries"]),
        "image_count": sum(len(x["files"]) for x in archive["entries"]),
        "file_count_before_index": len(all_files),
        "bytes_before_index": sum(p.stat().st_size for p in all_files),
    }
    (out_root / "archive-index.json").write_text(json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(archive["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
