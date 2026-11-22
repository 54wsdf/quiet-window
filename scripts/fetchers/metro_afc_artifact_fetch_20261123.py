#!/usr/bin/env python3
"""Public metro/AFC acquisition to GitHub artifact ZIPs (no long-lived Drive token).
Artifact names and per-file manifests are suitable for upload through the connected
Google Drive application. Only original public URLs are used.
"""
import argparse, csv, gzip, hashlib, io, json, os, re, sys, time
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode, urljoin
import requests
from bs4 import BeautifulSoup

INDEX = "https://data.taipei/api/frontstage/tpeod/dataset/resource.download?rid=eb481f58-1238-4cff-8caa-fa7bb20cb4f4"
TT_PAGE = "https://data.taipei/dataset/detail?id=91cc11bb-40d6-4837-9303-84e6a666c568"
BART_BASE = "https://afcweb.bart.gov/ridership/origin-destination/"
S=requests.Session()
S.headers.update({"User-Agent":"PublicTransitResearchDataAcquisition/1.1","Accept":"*/*"})
items=[]
out=None
task=None

def stamp(): return datetime.now(timezone.utc).isoformat()
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(2*1024*1024),b""): h.update(b)
    return h.hexdigest()

def fetch(url, name, *, mode="gzip", optional=False, min_bytes=64):
    dest=out/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(1,5):
        try:
            with S.get(url, stream=True, timeout=(30,180), allow_redirects=True) as r:
                if optional and r.status_code in (403,404,410):
                    items.append({"file":name,"source":url,"status":f"NOT_PUBLISHED_{r.status_code}"})
                    print("NOT_PUBLISHED",name,r.status_code,flush=True)
                    return False
                r.raise_for_status()
                raw_size=0
                first=True
                with (gzip.open(dest, "wb",compresslevel=5) if mode=="gzip" else dest.open("wb")) as o:
                    for block in r.iter_content(1024*1024):
                        if not block:continue
                        if first:
                            first=False
                            if block.lstrip()[:24].lower().startswith((b"<!doctype",b"<html",b"{\"error\"",b"<?xml")):
                                raise ValueError("Provider returned HTML or error document")
                            if mode=="keep_gzip" and not block.startswith(b"\x1f\x8b"):
                                raise ValueError("Expected gzip magic")
                            if mode=="raw_zip" and not block.startswith(b"PK"):
                                raise ValueError("Expected ZIP magic")
                        raw_size+=len(block)
                        o.write(block)
            if raw_size<min_bytes:raise ValueError(f"Too small: {raw_size}")
            if mode in ("gzip","keep_gzip"):
                with gzip.open(dest,"rb") as g:
                    assert g.read(32), "Empty gzip payload"
            if mode=="raw_zip":
                import zipfile
                with zipfile.ZipFile(dest) as z:
                    assert len(z.namelist())>2
                    assert z.testzip() is None
            info={"file":name,"source":url,"status":"ACQUIRED","raw_bytes":raw_size,"archive_bytes":dest.stat().st_size,
                  "sha256":digest(dest),"fetched_at_utc":stamp()}
            items.append(info)
            print("ACQUIRED",name,info["archive_bytes"],flush=True)
            return True
        except Exception as e:
            if dest.exists():dest.unlink()
            print("RETRY",name,attempt,str(e)[:250],flush=True)
            if attempt==4:
                items.append({"file":name,"source":url,"status":"FAILED","error":str(e)})
                return False
            time.sleep(3*attempt)

def get(url):
    r=S.get(url,timeout=(30,120))
    r.raise_for_status()
    return r.content

def taipei_od(year):
    raw=get(INDEX)
    rows=list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig",errors="replace"))))
    (out/"official_od_index.csv").write_bytes(raw)
    selected=[]
    for r in rows:
        if r.get("西元年","").strip()==str(year):
            selected.append(r)
    print("INDEX",len(rows),"YEAR",year,"MONTHS",len(selected),flush=True)
    expected=10 if year==2026 else 12
    if len(selected)!=expected:
        items.append({"status":"FAILED_INDEX_MONTH_COUNT","year":year,"observed":len(selected),"expected":expected})
    for r in selected:
        m=int(r["月"])
        u=r["URL"].strip()
        if u.startswith("http://"):u="https://"+u[7:]
        fetch(u,f"taipei_od_{year}_{m:02d}.csv.gz")
    return expected

def taipei_tt():
    html=get(TT_PAGE).decode("utf-8",errors="replace")
    soup=BeautifulSoup(html,"html.parser")
    refs={}
    for a in soup.find_all("a",href=True):
        href=a["href"]
        if "resource.download" in href and "rid=" in href:
            url=urljoin("https://data.taipei/",href)
            rid=re.search(r"rid=([\w-]+)",url)
            if rid:
                text=a.parent.get_text(" ",strip=True) if a.parent else ""
                refs[rid.group(1)]=dict(url=url,label=text[:150])
    (out/"source_resources.json").write_text(json.dumps(refs,ensure_ascii=False,indent=2),encoding="utf-8")
    print("CURRENT TIMETABLE RESOURCES",len(refs),flush=True)
    if not refs:items.append({"status":"FAILED_NO_TIMETABLE_LINKS"})
    for i,(rid,d) in enumerate(refs.items(),1):
        fetch(d["url"],f"taipei_timetable_{i:02d}_{rid}.csv.gz")
    return len(refs)

def bart(start,end):
    for y in range(start,end+1):
        url=f"{BART_BASE}date-hour-soo-dest-{y}.csv.gz"
        fetch(url,f"bart_hourly_od_{y}.csv.gz",mode="keep_gzip",optional=True)

def nyc_schedule(month):
    start=f"2025-{month:02d}-01T00:00:00"
    end=f"2026-01-01T00:00:00" if month==12 else f"2025-{month+1:02d}-01T00:00:00"
    where=f"service_date >= '{start}' and service_date < '{end}'"
    dest=out/f"nyc_2025_schedule_{month:02d}.csv.gz"
    total=0; pages=0; first_line=None
    try:
        with gzip.open(dest,"wb",compresslevel=5) as f:
            while True:
                params={"$where":where,"$limit":50000,"$offset":pages*50000,"$order":":id"}
                url="https://data.ny.gov/resource/q9nv-uegs.csv?"+urlencode(params)
                for attempt in range(5):
                    try:
                        r=S.get(url,timeout=(30,180))
                        r.raise_for_status()
                        body=r.content
                        if body.lstrip()[:30].lower().startswith((b"<!doctype",b"<html",b"{\"error\"")):
                            raise ValueError(body[:120])
                        break
                    except Exception as e:
                        print("PAGE RETRY",month,pages,attempt+1,str(e)[:150],flush=True)
                        if attempt==4:raise
                        time.sleep(5*(attempt+1))
                if len(body)<8:break
                lines=body.splitlines(keepends=True)
                if first_line is None:
                    first_line=lines[0]
                    if b"service_date" not in first_line.lower():
                        raise RuntimeError("Unexpected CSV schema: "+repr(first_line[:150]))
                    f.write(first_line if first_line.endswith(b"\n") else first_line+b"\n")
                if lines and lines[0].strip()==first_line.strip(): lines=lines[1:]
                if not lines:break
                for line in lines:
                    f.write(line if line.endswith(b"\n") else line+b"\n")
                total+=len(lines)
                pages+=1
                if pages%10==0:print("SCHEDULE",month,"pages",pages,"rows",total,flush=True)
                if len(lines)<50000:break
                if pages>=400:raise RuntimeError("Monthly pagination cap reached")
        if total<1000:raise RuntimeError(f"Suspiciously few rows: {total}")
        items.append({"file":dest.name,"status":"ACQUIRED","rows":total,"pages":pages,"source":"https://data.ny.gov/d/q9nv-uegs",
                      "service_month":f"2025-{month:02d}","query_where":where,"sha256":digest(dest),
                      "archive_bytes":dest.stat().st_size,"fetched_at_utc":stamp()})
        print("SCHEDULE COMPLETE",month,total,dest.stat().st_size,flush=True)
    except Exception as e:
        if dest.exists():dest.unlink()
        items.append({"file":dest.name,"status":"FAILED","error":str(e),"source":"https://data.ny.gov/d/q9nv-uegs"})
        print("SCHEDULE FAILED",month,str(e),flush=True)

def nyc_export(name):
    specs={
        "nyc_od_2025":("y2qv-fytt","nyc_od_2025.csv.gz"),
        "nyc_hourly_2025on":("5wq4-mkjj","nyc_station_hourly_2025on.csv.gz"),
        "nyc_hourly_2020_2024":("wujg-7c2s","nyc_station_hourly_2020_2024.csv.gz"),
    }
    ds,fn=specs[name]
    fetch(f"https://data.ny.gov/api/v3/views/{ds}/export.csv?accessType=DOWNLOAD",fn,min_bytes=1000)

def main():
    global out,task
    p=argparse.ArgumentParser();p.add_argument("--task",required=True);p.add_argument("--output-root",default="out")
    a=p.parse_args()
    task=a.task;out=Path(a.output_root)/task;out.mkdir(parents=True,exist_ok=True)
    print("START",task,stamp(),flush=True)
    fatal=None
    try:
        m=re.fullmatch(r"taipei_od_(20\d{2})",task)
        b=re.fullmatch(r"bart_od_(20\d\d)_(20\d\d)",task)
        n=re.fullmatch(r"nyc_schedule_2025_(0[1-9]|1[012])",task)
        if m: taipei_od(int(m.group(1)))
        elif b:bart(int(b.group(1)),int(b.group(2)))
        elif n:nyc_schedule(int(n.group(1)))
        elif task=="taipei_timetables":taipei_tt()
        elif task=="nyc_gtfs":
            fetch("https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip","nyc_gtfs_20261123.zip",mode="raw_zip",min_bytes=1000)
        elif task in ("nyc_od_2025","nyc_hourly_2025on","nyc_hourly_2020_2024"):nyc_export(task)
        else: raise ValueError("Unexpected task "+task)
    except Exception as e:
        fatal=str(e);print("FATAL",fatal,flush=True)
    failures=[i for i in items if str(i.get("status","")).startswith("FAILED")]
    manifest={"task":task,"retrieved_at_utc":stamp(),"status":"PARTIAL_OR_FAILED" if failures or fatal else "COLLECTED",
              "files_acquired":sum(i.get("status")=="ACQUIRED" for i in items),
              "files_unavailable":sum(i.get("status","").startswith("NOT_PUBLISHED") for i in items),
              "failed_count":len(failures),"fatal":fatal,"items":items,
              "notes":["Aggregate AFC products only, not individual farecard records.",
                       "Time-matching historical AFC and current timetables requires explicit validity checks.",
                       "Source manifest accompanies every Runner artifact; transfer to Google Drive is a separate verified step."]}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY",json.dumps({k:v for k,v in manifest.items() if k!="items"},ensure_ascii=False),flush=True)
    if failures or fatal:sys.exit(1)

if __name__=="__main__":main()
