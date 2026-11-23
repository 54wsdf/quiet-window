#!/usr/bin/env python3
"""Public ATS-adjacent data discovery, recent MTA feed snapshots, and smoke tests.
No credential or non-public data is required; never claim ATS interlocking logs.
"""
import csv, hashlib, io, json, os, re, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
import requests

BASE=Path("out/ats_discovery_20261123");BASE.mkdir(parents=True,exist_ok=True)
s=requests.Session();s.headers.update({"User-Agent":"Academic-ATS-Public-Data-Archive/2026 (research)", "Accept":"*/*"})
rows=[]
def utc(): return datetime.now(timezone.utc).isoformat()
def fetch(url, filename, min_size=50, expected=None, attempts=3):
    p=BASE/filename;p.parent.mkdir(parents=True,exist_ok=True);error=""
    for attempt in range(1,attempts+1):
        try:
            with s.get(url,timeout=(20,100),stream=True) as r:
                r.raise_for_status()
                ct=r.headers.get("Content-Type","")
                with p.open("wb") as f:
                    for b in r.iter_content(1024*1024):
                        if b:f.write(b)
            b=p.read_bytes()[:64]
            if p.stat().st_size<min_size: raise ValueError("small response")
            if b.lstrip().lower().startswith((b"<!doctype",b"<html",b"{\"error\"")) and expected not in ("html",):
                raise ValueError("unexpected HTML/error instead of raw data")
            if expected=="parquet" and b[:4]!=b"PAR1":raise ValueError("invalid PAR1")
            if expected=="protobuf" and (ct.startswith("text/") or b[:4]==b"PAR1"):raise ValueError("invalid protobuf response")
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            rows.append({"status":"ACQUIRED","url":url,"file":str(p.relative_to(BASE)),"bytes":p.stat().st_size,"sha256":digest,"type":ct,"fetched_utc":utc()})
            print("OK",filename,p.stat().st_size,flush=True)
            return p
        except Exception as e:
            error=str(e)
            print("RETRY",filename,attempt,error[:150],flush=True)
            if p.exists():p.unlink()
            time.sleep(attempt*2)
    rows.append({"status":"FAILED","url":url,"file":filename,"error":error})
    return None
mb="https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/index.csv"
fetch(mb,"MBTA/subway_on_time_performance_index.csv",expected="csv")
p=BASE/"MBTA/subway_on_time_performance_index.csv"
if p.exists():
    t=p.read_text(encoding="utf-8-sig")
    rs=list(csv.DictReader(io.StringIO(t)))
    print("MBTA INDEX",len(rs),"HEAD",json.dumps(rs[:2])[:800],flush=True)
    (BASE/"MBTA/index_info.json").write_text(json.dumps({"count":len(rs),"fields":list(rs[0]) if rs else [],"first":rs[:2],"last":rs[-2:]},indent=2),encoding="utf8")
    # download a representative official daily parquet from index.
    if rs:
        x=rs[len(rs)//2]
        filename=next((str(v) for v in x.values() if str(v).endswith(".parquet")),"")
        if filename:
            url=filename if filename.startswith("http") else urljoin("https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/",filename)
            fetch(url,"MBTA/sample_daily.parquet",expected="parquet")
# MBTA auxiliary datasets (small to medium)
for key in ["LAMP_service_id_by_date_and_route","LAMP_static_stops","LAMP_static_routes"]:
    fetch("https://performancedata.mbta.com/lamp/tableau/rail/"+key+".parquet",f"MBTA/tableau/{key}.parquet",expected="parquet")
fetch("https://data.transitlab.nyc/v1/catalog.json","TransitLab/catalog.json",expected="json",min_size=200)
fetch("https://data.transitlab.nyc/v1/subway_stop_events/year=2025/month=01/2025-01-08.parquet","TransitLab/sample_2025-01-08.parquet",expected="parquet")
fetch("https://data.transitlab.nyc/v1/subway_segment_times.parquet","TransitLab/subway_segment_times.parquet",expected="parquet")
feeds=["gtfs","gtfs-ace","gtfs-bdfm","gtfs-g","gtfs-jz","gtfs-l","gtfs-nqrw","gtfs-si"]
for f in feeds:
    u="https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct/"+f
    fetch(u,"MTA_realtime/"+f+".pb",expected="protobuf",min_size=30)
# read public documentation only, NEVER attempt to bypass authentication.
for key,u in [
("networkrail_open_data.html","https://www.networkrail.co.uk/who-we-are/transparency-and-ethics/transparency/open-data-feeds/"),
("networkrail_td_wiki.html","https://wiki.openraildata.com/index.php?title=TD"),
("networkrail_train_describers.html","https://wiki.openraildata.com/index.php/List_of_Train_Describers"),
("511_open_data.html","https://511.org/open-data/transit"),
("511_faq.html","https://511.org/open-data/faqs")
]:
    fetch(u,"AUTH_REQUIRED_DOCS/"+key,expected="html",min_size=100)
result={"run_utc":utc(),"collector":"official public sources","records":rows,
        "caveats":["No historical 511 stop_observations without a valid 511 API key.",
                   "Network Rail TD live feed requires registration/approval; only documentation is collected.",
                   "MTA raw protobuf messages are current snapshots, NOT a historical ATS database.",
                   "TransitLab departure events are inferred from MTA GTFS-RT, not original track-circuit/interlocking logs."]}
(BASE/"manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print("SUMMARY",json.dumps({"acquired":sum(x["status"]=="ACQUIRED" for x in rows),"failed":sum(x["status"]=="FAILED" for x in rows)}),flush=True)
