#!/usr/bin/env python3
"""Bounded research sample of MTA schedules and AFC-derived aggregated counts.
Daily schedule + daily hourly station ridership, and typical-hour estimated OD.
Sample is not presented as full-year archive.
"""
import csv,gzip,hashlib,io,json,requests,time,sys
from pathlib import Path
from urllib.parse import urlencode
from datetime import datetime,timezone
DAY="2025-01-08"
S=requests.Session();S.headers.update({"User-Agent":"PublicMetroScheduleArchive/1.0","Accept":"text/csv"})
OUT=Path("out/nyc_day_sample_20250108");OUT.mkdir(parents=True,exist_ok=True)
entries=[]
SPECS=[
("daily_train_scheduled_stop_20250108","q9nv-uegs", f"service_date >= '{DAY}T00:00:00' and service_date < '2025-01-09T00:00:00'",200000),
("daily_station_hourly_entries_20250108","5wq4-mkjj",f"transit_timestamp >= '{DAY}T00:00:00' and transit_timestamp < '2025-01-09T00:00:00'",60000),
("estimated_typical_wednesday_od_2025_january_08hour","y2qv-fytt","year = 2025 and month = 1 and day_of_week = 'Wednesday' and hour_of_day = 8",200000),
]
def download(name,ds,where,maxrows):
 p=OUT/(name+".csv.gz");limit=50000;offset=0;num=0;head=None
 with gzip.open(p,"wb",compresslevel=6) as f:
  while offset<maxrows:
   q={"$where":where,"$limit":min(limit,maxrows-offset),"$offset":offset,"$order":":id"}
   url=f"https://data.ny.gov/resource/{ds}.csv?"+urlencode(q)
   for attempt in range(4):
    try:
     r=S.get(url,timeout=(25,180));r.raise_for_status()
     b=r.content
     if b.lstrip()[:20].lower().startswith((b"<html",b"<!doctype",b"{\"error\"")):raise ValueError(b[:150])
     break
    except Exception as e:
     print("Retry",name,offset,attempt,str(e)[:160],flush=True)
     if attempt==3:raise
     time.sleep(3*(attempt+1))
   lines=b.splitlines(keepends=True)
   if not lines:break
   if head is None:
    head=lines[0];f.write(head if head.endswith(b"\n") else head+b"\n")
   if lines[0].strip()==head.strip():lines=lines[1:]
   if not lines:break
   for line in lines:f.write(line if line.endswith(b"\n") else line+b"\n")
   num+=len(lines);offset+=len(lines)
   print(name,"records",num,flush=True)
   if len(lines)<limit:break
 if num<10:raise RuntimeError(f"Empty or invalid response for {name}: {num}")
 with p.open("rb") as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 entries.append({"file":p.name,"dataset_id":ds,"source":f"https://data.ny.gov/d/{ds}","where":where,
                 "rows":num,"truncated":num>=maxrows,"size_bytes":p.stat().st_size,"sha256":h})
def main():
 for name,ds,where,cap in SPECS:
  try:download(name,ds,where,cap)
  except Exception as e:entries.append({"file":name+".csv.gz","error":str(e),"status":"FAILED"})
 summary={"retrieved_utc":datetime.now(timezone.utc).isoformat(),"sample_day":DAY,"coverage":"SAMPLE, not full-year",
          "day_od_note":"Estimated OD is a monthly day-of-week typical hour, not the actual January 8 OD.",
          "files":entries}
 (OUT/"manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
 if any("error" in e for e in entries):sys.exit(1)
if __name__=="__main__":main()
