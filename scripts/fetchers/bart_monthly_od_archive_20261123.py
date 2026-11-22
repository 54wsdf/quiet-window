#!/usr/bin/env python3
"""Download BART's public monthly OD matrices with official HTML URL discovery."""
import argparse,hashlib,json,re,time,requests
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import urljoin,urlparse
from bs4 import BeautifulSoup
BASE="https://www.bart.gov/about/reports/ridership"
S=requests.Session()
S.headers.update({"User-Agent":"Mozilla/5.0 (compatible; AcademicResearchArchive/1.0)"})
def timestamp():return datetime.now(timezone.utc).isoformat()
def main():
 p=argparse.ArgumentParser();p.add_argument("--group",required=True);p.add_argument("--output-root",default="out")
 args=p.parse_args();group=args.group
 out=Path(args.output_root)/group;out.mkdir(parents=True,exist_ok=True)
 if group.startswith("bart_annual_od_"):
  m=re.fullmatch(r"bart_annual_od_(20\d\d)_(20\d\d)",group)
  if not m:raise ValueError("invalid annual group")
  begin,end=map(int,m.groups())
 elif group=="bart_monthly_od_2026":
  begin,end=2026,2026
 else:raise ValueError("unknown group")
 r=S.get(BASE,timeout=90);r.raise_for_status()
 links=[]
 for a in BeautifulSoup(r.text,"html.parser").find_all("a",href=True):
  href=urljoin(BASE,a["href"]); name=a.get_text(" ",strip=True)
  if urlparse(href).hostname not in ("www.bart.gov","bart.gov"):continue
  if group=="bart_monthly_od_2026":
   if re.search(r"2026.*Ridership OD",name,re.I):
    links.append((href,name))
  else:
   match=re.search(r"(20\d{2})\s+Monthly Ridership Reports",name,re.I)
   if match and begin<=int(match.group(1))<=end:links.append((href,name))
 seen=set();links=[x for x in links if not (x[0] in seen or seen.add(x[0]))]
 print("DISCOVERED",group,len(links),flush=True)
 entries=[]
 for url,label in links:
  fname=urlparse(url).path.rsplit("/",1)[-1]
  if not fname or not re.search(r"\.(zip|xls|xlsx)$",fname,re.I):
   entries.append(dict(source=url,label=label,status="SKIPPED_NONFILE"));continue
  target=out/fname
  err=None
  for attempt in range(4):
   try:
    with S.get(url,timeout=(30,180),stream=True) as resp:
     resp.raise_for_status()
     with target.open("wb") as f:
      for block in resp.iter_content(2*1024*1024):
       if block:f.write(block)
    header=target.open("rb").read(12)
    if target.stat().st_size<2000 or not (header.startswith(b"PK") or header.startswith(bytes.fromhex("D0CF11E0A1B11AE1"))):
     raise ValueError("Invalid or suspiciously small spreadsheet/archive")
    with target.open("rb") as f: sha=hashlib.file_digest(f,"sha256").hexdigest()
    entries.append(dict(file=fname,source=url,label=label,status="ACQUIRED",bytes=target.stat().st_size,sha256=sha,downloaded_at=timestamp()))
    print("ACQUIRED",fname,target.stat().st_size,flush=True)
    err=None;break
   except Exception as e:
    err=str(e);print("RETRY",fname,attempt,err[:200],flush=True)
    if target.exists():target.unlink()
    time.sleep(2*(attempt+1))
  if err:entries.append(dict(file=fname,source=url,label=label,status="FAILED",error=err))
 m=dict(task=group,source_page=BASE,found=len(links),files_acquired=sum(x["status"]=="ACQUIRED" for x in entries),
        failed=sum(x["status"]=="FAILED" for x in entries),finished_utc=timestamp(),entries=entries,
        note="BART monthly origin-destination spreadsheets show aggregated total/day-type OD, not hourly exit-time OD.")
 (out/"manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
 if m["files_acquired"]==0:raise RuntimeError("No valid files obtained")
if __name__=="__main__":main()
