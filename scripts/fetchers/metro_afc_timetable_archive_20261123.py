#!/usr/bin/env python3
"""Archive public metro timetable and aggregate AFC datasets directly to Google Drive.
Idempotent, independently verifiable acquisition: source URL + SHA256 + remote byte size.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = "transport_data_lake/01_rail"
TAIPEI_OD_INDEX = "https://data.taipei/api/frontstage/tpeod/dataset/resource.download?rid=eb481f58-1238-4cff-8caa-fa7bb20cb4f4"
TAIPEI_TT_PAGE = "https://data.taipei/dataset/detail?id=91cc11bb-40d6-4837-9303-84e6a666c568"
BART_OD_BASE = "https://afcweb.bart.gov/ridership/origin-destination/"
DATA_NY = "https://data.ny.gov"
S = requests.Session()
S.headers.update({"User-Agent": "MetroResearchPublicDataArchiver/1.0 (academic reproducibility)", "Accept": "*/*"})
results = []
RCLONE_CONFIG = None
RCLONE_REMOTE = None

def utc():
    return datetime.now(timezone.utc).isoformat()

def call(*args, capture=False, allow_fail=False):
    cmd = ["rclone", *args, "--config", RCLONE_CONFIG]
    p = subprocess.run(cmd, check=False, capture_output=capture, text=True)
    if p.returncode and not allow_fail:
        raise RuntimeError("rclone operation failed: " + " ".join(args[:3]) + ((" " + p.stderr[-600:]) if capture else ""))
    return p

def remote_size(dest):
    p = call("lsjson", f"{RCLONE_REMOTE}:{dest}", "--files-only", capture=True, allow_fail=True)
    if p.returncode:
        return None
    try:
        js = json.loads(p.stdout)
        if len(js) == 1 and not js[0].get("IsDir"):
            return int(js[0]["Size"])
    except (ValueError, KeyError, TypeError):
        pass
    return None

def push(local, dest):
    size = local.stat().st_size
    if size < 32:
        raise ValueError("Unexpectedly empty payload")
    call("copyto", str(local), f"{RCLONE_REMOTE}:{dest}", "--retries", "6", "--low-level-retries", "15", "--transfers", "1", "--stats", "30s", "--stats-one-line")
    remote = remote_size(dest)
    if remote != size:
        raise RuntimeError(f"remote size validation failed {dest}: {remote} != {size}")
    return size

def put_json(obj, dest):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "document.json"
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        push(p, dest)

def sha_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def transfer(url, dest, *, compression="gzip", min_bytes=75, optional=False):
    """Fetch validated response once, compress to disk, upload, and write per-file manifest."""
    manifest_dest = dest + ".manifest.json"
    old_size = remote_size(dest)
    if old_size and old_size > 32 and remote_size(manifest_dest):
        item = dict(path=dest, status="EXISTING_VERIFIED_SIZE", size_bytes=old_size, url=url)
        results.append(item)
        print("EXISTS", dest, old_size, flush=True)
        return True
    exc = None
    for attempt in range(1, 5):
        try:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / Path(dest).name
                with S.get(url, stream=True, timeout=(30, 150), allow_redirects=True) as resp:
                    if resp.status_code == 404 and optional:
                        results.append(dict(path=dest, status="UNAVAILABLE_404", url=url))
                        print("NOT PUBLISHED", url, flush=True)
                        return False
                    resp.raise_for_status()
                    ctype = resp.headers.get("Content-Type", "")
                    first = True
                    raw_bytes = 0
                    if compression == "gzip":
                        writer = gzip.open(p, "wb", compresslevel=6)
                    else:
                        writer = p.open("wb")
                    with writer as out:
                        for block in resp.iter_content(chunk_size=1024 * 1024):
                            if not block:
                                continue
                            if first:
                                first = False
                                if block.lstrip()[:32].lower().startswith((b"<!doctype", b"<html", b"<?xml", b"{\"error\"")):
                                    raise ValueError(f"Not a raw data payload ({ctype}): {url}")
                                if compression == "keep_gzip" and block[:2] != b"\x1f\x8b":
                                    raise ValueError(f"Expected gzip magic: {url}")
                            raw_bytes += len(block)
                            out.write(block)
                    if raw_bytes < min_bytes:
                        raise ValueError(f"Too small: {raw_bytes} bytes {url}")
                if compression in ("gzip", "keep_gzip"):
                    with gzip.open(p, "rb") as test:
                        if not test.read(64):
                            raise ValueError("Gzip decompression yielded empty data")
                digest = sha_file(p)
                n = push(p, dest)
                entry = dict(path=dest, status="ACQUIRED", size_bytes=n, raw_bytes=raw_bytes, sha256=digest,
                             url=url, retrieved_utc=utc(), content_type=ctype, remote_check="byte_size_equal",
                             compression=compression)
                put_json(entry, manifest_dest)
                results.append(entry)
                print("ACQUIRED", dest, n, "sha256="+digest[:12], flush=True)
                return True
        except Exception as e:
            exc = str(e)
            print(f"RETRY {attempt}/4: {dest} {exc[-300:]}", flush=True)
            time.sleep(min(4 * attempt, 20))
    entry = dict(path=dest, status="FAILED", url=url, error=exc)
    results.append(entry)
    print("FAILED", dest, exc, flush=True)
    return False

def request_bytes(url):
    r = S.get(url, timeout=(30, 120))
    r.raise_for_status()
    return r.content

def taipei_od():
    base = ROOT + "/TW_Taipei_Metro/04_demand_od/hourly_by_entry"
    raw = request_bytes(TAIPEI_OD_INDEX)
    s = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(s))
    rows = list(reader)
    print("TAIPEI INDEX", len(rows), flush=True)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "index.csv"
        p.write_bytes(raw)
        push(p, base+"/source_index_20261123.csv")
    if len(rows) < 100:
        raise RuntimeError(f"Taipei OD index suspiciously short: {len(rows)}")
    for row in rows:
        year = row.get("西元年", "").strip()
        month = row.get("月", "").strip().zfill(2)
        url = (row.get("URL", "") or "").strip()
        if not (re.fullmatch(r"20\d\d", year) and month.isdigit() and url.startswith("http")):
            results.append(dict(status="FAILED_INVALID_INDEX_ROW", row=row))
            continue
        url = "https://" + url[len("http://"):] if url.startswith("http://") else url
        # CSV payloads are often large. Compress to reduce Drive traffic and storage.
        transfer(url, f"{base}/{year}/taipei_od_{year}{month}.csv.gz")
    return len(rows)

def taipei_timetables():
    base = ROOT + "/TW_Taipei_Metro/02_planned_operations/current_station_timetables"
    html = request_bytes(TAIPEI_TT_PAGE).decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].replace("&amp;", "&")
        if "resource.download" in href and "rid=" in href:
            full = urljoin("https://data.taipei/", href)
            if full not in [t[0] for t in links]:
                label = re.sub(r"\s+", " ", a.parent.get_text(" ", strip=True))[:180] if a.parent else a.get_text(" ", strip=True)
                links.append((full, label))
    # The page may expose just the first five in server-rendered HTML. Record the
    # exact discovery count rather than claiming historic timetable coverage.
    print("TAIPEI timetable links", len(links), flush=True)
    put_json(dict(source_page=TAIPEI_TT_PAGE, discovered_links=links, count=len(links),
                  note="These are current station timetables, not historical 2017-2025 schedules.",
                  checked_at_utc=utc()), base+"/source_resources.json")
    for i, (url, label) in enumerate(links, 1):
        rid = re.search(r"rid=([\w-]+)", url)
        name = f"timetable_{i:02d}_{rid.group(1) if rid else 'unknown'}.csv.gz"
        transfer(url, f"{base}/{name}")
    if not links:
        raise RuntimeError("Taipei official HTML had no downloadable timetable links")
    return len(links)

def bart_od():
    base = ROOT + "/US_CA_BART/04_demand_od/hourly_exit_2011_2026"
    years = set()
    try:
        html = request_bytes(BART_OD_BASE).decode("utf-8", errors="replace")
        years = {int(y) for y in re.findall(r"date-hour-soo-dest-(20\d{2})\.csv\.gz", html)}
    except Exception as e:
        print("BART listing not accessible, testing canonical official year URLs:", str(e)[:180], flush=True)
    # Test every possibly released annual file from 2011, including years not
    # linked in a server-side directory listing.
    years.update(range(2011, 2027))
    available = 0
    for y in sorted(years):
        u = f"{BART_OD_BASE}date-hour-soo-dest-{y}.csv.gz"
        if transfer(u, f"{base}/date-hour-soo-dest-{y}.csv.gz", compression="keep_gzip", optional=True):
            available += 1
    return available

def static(name):
    sources = {
        "bart_gtfs": (
            "https://www.bart.gov/dev/schedules/google_transit.zip",
            ROOT + "/US_CA_BART/02_planned_operations/snapshot_20261123/bart_gtfs.zip"),
        "nyc_gtfs": (
            "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip",
            ROOT + "/US_NY_MTA/02_planned_operations/snapshot_20261123/gtfs_subway.zip"),
    }
    url, dest = sources[name]
    return transfer(url, dest, compression="raw", min_bytes=1000)

def nyc_export(task):
    source = {
        "nyc_od_2025": ("y2qv-fytt", "US_NY_MTA/04_demand_od/estimated_typical_2025", "od_2025_all.csv.gz"),
        "nyc_hourly_2025on": ("5wq4-mkjj", "US_NY_MTA/04_demand_od/station_hourly_from_2025", "hourly_2025on_asof_20261123.csv.gz"),
        "nyc_hourly_2020_2024": ("wujg-7c2s", "US_NY_MTA/04_demand_od/station_hourly_2020_2024", "hourly_2020_2024.csv.gz")
    }
    dataset, d, filename = source[task]
    url = f"https://data.ny.gov/api/v3/views/{dataset}/export.csv?accessType=DOWNLOAD"
    return transfer(url, ROOT + "/" + d + "/" + filename, min_bytes=1000)

def nyc_schedule(month):
    import calendar
    from urllib.parse import urlencode
    year = 2025
    start = f"{year}-{month:02d}-01T00:00:00"
    if month == 12:
        end = f"2026-01-01T00:00:00"
    else:
        end = f"{year}-{month+1:02d}-01T00:00:00"
    # Socrata SoQL CSV, paginated to avoid memory and file-size limits.
    base = ROOT + f"/US_NY_MTA/02_planned_operations/2025_stop_schedule/{year}-{month:02d}"
    where = f"service_date >= '{start}' and service_date < '{end}'"
    dest = base + f"/mta_schedule_{year}_{month:02d}.csv.gz"
    if remote_size(dest) and remote_size(dest+".manifest.json"):
        results.append(dict(path=dest, status="EXISTING_VERIFIED_SIZE"))
        return True
    rows = 0
    chunks = 0
    h = hashlib.sha256()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / Path(dest).name
        first_line = None
        with gzip.open(p, "wb", compresslevel=5) as out:
            while True:
                params = {"$where": where, "$limit": "50000", "$offset": str(chunks * 50000),
                          "$order": "service_date,train_id"}
                url = f"https://data.ny.gov/resource/q9nv-uegs.csv?" + urlencode(params)
                err = None
                for attempt in range(5):
                    try:
                        with S.get(url, timeout=(30, 180), stream=True) as r:
                            r.raise_for_status()
                            body = r.content
                        if body.lstrip().lower().startswith((b"<!doctype", b"<html", b"{\"error\"")):
                            raise ValueError("Socrata replied with HTML/error")
                        break
                    except Exception as e:
                        err = e
                        print("NYC page retry", month, chunks, str(e)[:180], flush=True)
                        time.sleep(5*(attempt+1))
                else:
                    raise RuntimeError(f"NYC schedule page month={month}, page={chunks}: {err}")
                if len(body) < 16:
                    break
                lines = body.splitlines(keepends=True)
                if first_line is None:
                    first_line = lines[0]
                    if b"service_date" not in first_line.lower():
                        raise ValueError("Unexpected NYC schedule CSV header: "+repr(first_line[:150]))
                    out.write(first_line if first_line.endswith(b"\n") else first_line+b"\n")
                if lines and lines[0].strip() == first_line.strip():
                    lines = lines[1:]
                if not lines:
                    break
                for line in lines:
                    out.write(line if line.endswith(b"\n") else line+b"\n")
                n = len(lines)
                rows += n
                chunks += 1
                if chunks % 10 == 0:
                    print("NYC SCHEDULE", month, "pages", chunks, "rows", rows, flush=True)
                if n < 50000:
                    break
                if chunks >= 400:
                    raise RuntimeError(f"Pagination guard 20 million rows reached for {month}")
        if rows < 1000:
            raise RuntimeError(f"Suspicious NYC monthly row count: {rows} (month={month})")
        digest = sha_file(p)
        size = push(p, dest)
        entry = dict(path=dest, status="ACQUIRED", dataset_id="q9nv-uegs",
                     service_month=f"2025-{month:02d}", rows=rows, pages=chunks,
                     sha256=digest, size_bytes=size, source="https://data.ny.gov/d/q9nv-uegs",
                     query_where=where, retrieved_utc=utc(), remote_check="byte_size_equal")
        put_json(entry, dest+".manifest.json")
        results.append(entry)
        print("NYC MONTH COMPLETE", month, rows, size, flush=True)
        return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--remote", required=True)
    a=ap.parse_args()
    global RCLONE_CONFIG,RCLONE_REMOTE
    RCLONE_CONFIG,RCLONE_REMOTE=a.config,a.remote
    task=a.task
    print("START", task, utc(), flush=True)
    errors=[]
    try:
        if task=="taipei_od": taipei_od()
        elif task=="taipei_timetables": taipei_timetables()
        elif task=="bart_od": bart_od()
        elif task in ("bart_gtfs", "nyc_gtfs"): static(task)
        elif task in ("nyc_od_2025","nyc_hourly_2025on","nyc_hourly_2020_2024"): nyc_export(task)
        elif re.fullmatch(r"nyc_schedule_2025_(0[1-9]|1[012])",task):
            nyc_schedule(int(task.rsplit("_",1)[1]))
        else: raise ValueError("unknown task: "+task)
    except Exception as e:
        errors.append(str(e))
        print("TASK ERROR",task,str(e),flush=True)
    if any(r.get("status") in ("FAILED","FAILED_INVALID_INDEX_ROW") for r in results):
        errors.append("One or more source downloads failed")
    summary=dict(task=task, status="PARTIAL_OR_FAILED" if errors else "COMPLETED",
                 started_by="GitHub Actions Metro AFC and Timetable Public Archive", ended_utc=utc(),
                 attempted=len(results), acquired=sum(r.get("status")=="ACQUIRED" for r in results),
                 existing=sum(r.get("status")=="EXISTING_VERIFIED_SIZE" for r in results),
                 errors=errors, results=results,
                 disclaimer="Timetables and OD may NOT cover identical historical dates; no invented matched timestamps.")
    dest=f"{ROOT}/99_manifests/metro_afc_timetable_20261123/{task}.json"
    try:
        put_json(summary,dest)
        print("SUMMARY UPLOADED", dest, flush=True)
    except Exception as e:
        print("SUMMARY UPLOAD FAILED",str(e),flush=True)
        errors.append(str(e))
    if errors: sys.exit(1)

if __name__=="__main__":
    main()
