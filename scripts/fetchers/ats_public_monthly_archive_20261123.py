#!/usr/bin/env python3
"""Archive MBTA LAMP actual subway trips or TransitLab NYC stop events by calendar month.
Each downloaded Parquet is kept byte-for-byte. SHA-256 per file, source manifest,
and missing-day audit. No data fabrication or authentication bypass.
"""
import argparse, calendar, csv, hashlib, io, json, re, sys, time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse, urljoin
import requests
BASE_MBTA="https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/"
INDEX_MBTA=BASE_MBTA+"index.csv"
BASE_TL="https://data.transitlab.nyc/v1/subway_stop_events/"
s=requests.Session()
s.headers.update({"User-Agent":"RailTransitATSOpenArchive/1.0 (Academic data reproducibility)","Accept":"*/*"})
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def dayrange(y,m):
 end=min(date(y,m,calendar.monthrange(y,m)[1]), datetime.now(timezone.utc).date()-timedelta(days=2))
 if date(y,m,1)>end:return []
 return [date(y,m,i) for i in range(1,end.day+1)]
def url_of_mbta_path(val):
 if val.startswith("https://"):return val
 v=val.lstrip("/")
 if v.startswith("lamp/"):return "https://performancedata.mbta.com/"+v
 if v.startswith("subway-on-time-performance-v1/"):return "https://performancedata.mbta.com/lamp/"+v
 if "subway-on-time-performance-v1/" in v:return "https://performancedata.mbta.com/lamp/"+v[v.index("subway-on-time-performance-v1/"):]
 return urljoin(BASE_MBTA,v)
def list_mbta():
 r=s.get(INDEX_MBTA,timeout=(30,120));r.raise_for_status()
 raw=r.content.decode("utf-8-sig",errors="replace")
 mapping={}
 # CSV may have column names or raw path; we search all cells for actual published URLs.
 for row in csv.reader(io.StringIO(raw)):
  for cell in row:
   v=cell.strip().strip('"')
   if ".parquet" not in v:continue
   m=re.search(r"(20\d{2})-(\d{2})-(\d{2})",v)
   if not m:
    m=re.search(r"(20\d{2})-(\d{2})-(\d{2})",','.join(row))
   if not m:continue
   iso="-".join(m.groups())
   if v.lower().endswith(".parquet"):mapping[iso]=url_of_mbta_path(v)
 return mapping,r.content
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--provider",choices=["mbta","transitlab"],required=True)
 ap.add_argument("--month",required=True,help="YYYY-MM");ap.add_argument("--output-root",default="out")
 args=ap.parse_args();y,m=map(int,args.month.split("-"));assert 2018<=y<=2027 and 1<=m<=12
 out=Path(args.output_root)/f"{args.provider}_{y}-{m:02d}";out.mkdir(parents=True,exist_ok=True)
 index_urls={}
 index_error=None
 if args.provider=="mbta":
  try:
   index_urls,raw=list_mbta()
   (out/"official_index.csv").write_bytes(raw)
   print("MBTA indexed official dates",len(index_urls),flush=True)
  except Exception as e:
   index_error=str(e)
   print("MBTA INDEX FAILED",index_error,flush=True)
 dates=dayrange(y,m)
 results=[]
 for d in dates:
  iso=d.isoformat()
  if args.provider=="mbta":
   if iso not in index_urls:
    results.append({"service_date":iso,"status":"NOT_PUBLISHED_IN_INDEX"})
    continue
   url=index_urls[iso]
  else:
   url=f"{BASE_TL}year={y}/month={m:02d}/{iso}.parquet"
  f=out/f"{iso}_{args.provider}.parquet"
  ok=False;error=None
  for attempt in range(3):
   try:
    with s.get(url,stream=True,timeout=(25,150)) as r:
     if r.status_code in (404,410):
      results.append({"service_date":iso,"status":f"UNAVAILABLE_{r.status_code}","source":url})
      ok=True;break
     r.raise_for_status()
     with f.open("wb") as w:
      for c in r.iter_content(1024*1024):
       if c:w.write(c)
    size=f.stat().st_size
    with f.open("rb") as b:begin=b.read(4);b.seek(max(0,size-4));end=b.read(4)
    if size<150 or begin!=b"PAR1" or end!=b"PAR1":
     raise ValueError(f"Invalid parquet header/trailer, size={size}")
    results.append({"service_date":iso,"status":"ACQUIRED","source":url,
                    "file":f.name,"bytes":size,"sha256":sha(f)})
    ok=True
    print("ACQUIRED",args.provider,iso,size,flush=True)
    break
   except Exception as e:
    error=str(e)
    if f.exists():f.unlink()
    print("RETRY",args.provider,iso,attempt+1,error[:130],flush=True)
    time.sleep((attempt+1)*2)
  if not ok:results.append({"service_date":iso,"status":"FAILED","source":url,"error":error})
 status_counts={status:sum(z["status"]==status for z in results) for status in sorted(set(z["status"] for z in results))}
 output={"provider":args.provider,"month":args.month,"dates_checked":len(dates),"results":results,
         "statuses":status_counts,"published_index_error":index_error,"created_utc":datetime.now(timezone.utc).isoformat(),
         "license":("TransitLab CC BY 4.0" if args.provider=="transitlab" else "MBTA public data"),
         "semantic_type":("inferred subway station departure events (MTA GTFS-RT)" if args.provider=="transitlab"
                          else "MBTA actual/estimated stop performance, GTFS-RT-derived"),
         "limitations":["Date coverage is explicit, never infer missing dates.","These are not raw CBTC/CI/ATS signal interlocking logs."]}
 (out/"manifest.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print("MONTH SUMMARY",args.provider,args.month,json.dumps(status_counts),flush=True)
 if index_error or not results or (status_counts.get("ACQUIRED",0)==0 and sum(v for k,v in status_counts.items() if k.startswith("NOT_PUBLISHED") or k.startswith("UNAVAILABLE"))==0):
  sys.exit(1)
if __name__=="__main__":main()
