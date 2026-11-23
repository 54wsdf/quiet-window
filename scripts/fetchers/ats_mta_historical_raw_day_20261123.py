#!/usr/bin/env python3
"""Fetch one date of archived MTA raw GTFS-RT and stop CSV from Subway Data NYC."""
import hashlib,json,lzma,os,requests,time
from pathlib import Path
from datetime import datetime,timezone
DAY="2025-01-08";OUT=Path("out/subwaydata_20250108");OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session();S.headers.update({"User-Agent":"PublicAcademicArchive/1.0"})
manifest={"service_date":DAY,"source":"https://subwaydata.nyc/programmatic-access",
 "note":"Third-party capture of MTA original GTFS realtime protobuf feed messages; not CI/CBTC raw signal logs",
 "retrieved":datetime.now(timezone.utc).isoformat(),"items":[]}
for typ in ("csv","gtfsrt"):
 name=f"subwaydatanyc_{DAY}_{typ}.tar.xz"
 url=f"https://subwaydata.nyc/data/{name}"
 p=OUT/name
 error=None
 for i in range(3):
  try:
   with S.get(url,stream=True,timeout=(30,150)) as r:
    r.raise_for_status()
    with p.open("wb") as f:
     for b in r.iter_content(2*1024*1024):
      if b:f.write(b)
   if p.stat().st_size<10000 or p.open("rb").read(6)!=bytes.fromhex("FD377A585A00"):
    raise RuntimeError("No valid compressed tar.xz")
   with lzma.open(p) as f:
    first=f.read(512)
    if len(first)<512:raise RuntimeError("Cannot read xz tar header")
   h=hashlib.sha256()
   with p.open("rb") as f:
    for b in iter(lambda:f.read(2*1024*1024),b""):h.update(b)
   manifest["items"].append(dict(file=name,source_url=url,sha256=h.hexdigest(),bytes=p.stat().st_size,status="ACQUIRED"))
   print("ACQUIRED",name,p.stat().st_size,flush=True)
   error=None;break
  except Exception as e:
   error=str(e)
   if p.exists():p.unlink()
   print("RETRY",name,i+1,error,flush=True);time.sleep((i+1)*3)
 if error: manifest["items"].append(dict(file=name,source_url=url,status="FAILED",error=error))
(OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
if not all(i["status"]=="ACQUIRED" for i in manifest["items"]):raise SystemExit(1)
