#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import requests

ROOT='https://crowding.data.tfl.gov.uk/NUMBAT'

def run(c): subprocess.run(c,check=True)
def out(c): return subprocess.check_output(c,text=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def put(p,d,cfg,remote):
    run(['rclone','copyto',str(p),f'{remote}:{d}','--config',cfg,'--retries','8','--low-level-retries','20','--transfers','1','--checkers','4'])
    rs=int(json.loads(out(['rclone','lsjson',f'{remote}:{d}','--config',cfg,'--files-only']))[0]['Size'])
    if rs!=p.stat().st_size: raise RuntimeError('remote size mismatch')
def candidates(year):
    yy=str(year)[2:]
    names=[]
    for d in ['MON','MTT','TWT','FRI','SAT','SUN']:
        for suf in ['_Outputs.xlsx','_Output.xlsx','_OUTPUTS.xlsx','_OUTPUT.xlsx']:
            names.append(f'NBT{yy}{d}{suf}')
            names.append(f'NB{yy}{d}{suf}')
    return list(dict.fromkeys(names))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--year',type=int,required=True); ap.add_argument('--config',required=True); ap.add_argument('--remote',required=True); args=ap.parse_args()
    s=requests.Session(); s.headers['User-Agent']='Mozilla/5.0 quiet-window-public-data-worker/2.0'; rec=[]
    dirs=[f'NUMBAT {args.year}',f'NUMBAT%20{args.year}',str(args.year)]
    with tempfile.TemporaryDirectory(prefix='qw-numbat-') as td:
        td=Path(td); seen=set()
        for name in candidates(args.year):
            found=None
            for d in dirs:
                u=f'{ROOT}/{d}/{quote(name)}'
                try:
                    r=s.get(u,stream=True,timeout=(20,120),allow_redirects=True)
                    if r.status_code==404: continue
                    r.raise_for_status()
                    ct=(r.headers.get('content-type') or '').lower()
                    p=td/name
                    with p.open('wb') as f:
                        for b in r.iter_content(4*1024*1024):
                            if b: f.write(b)
                    if p.stat().st_size<1000 or ('text/html' in ct): p.unlink(missing_ok=True); continue
                    found=(u,p,ct); break
                except requests.RequestException:
                    continue
            if not found or name in seen: continue
            seen.add(name); u,p,ct=found; size=p.stat().st_size; digest=sha(p); dest=f'01_rail/UK_London_TfL/04_demand_od/NUMBAT/{args.year}/{name}'; put(p,dest,args.config,args.remote)
            rec.append({'file':name,'source_url':u,'bytes':size,'sha256':digest,'content_type':ct,'state':'ACQUIRED'}); p.unlink(); print(json.dumps({'year':args.year,'file':name,'bytes':size}))
        if not rec: raise RuntimeError(f'no NUMBAT workbooks resolved for {args.year}')
        m={'schema':'quiet-window-acquisition-manifest-v1','state':'ACQUIRED','source':'Transport for London NUMBAT','tier':'S','year':args.year,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'github_run_id':os.environ.get('GITHUB_RUN_ID'),'files':rec}
        mp=td/f'MANIFEST_NUMBAT_{args.year}_{os.environ.get("GITHUB_RUN_ID","manual")}.json'; mp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8'); put(mp,f'01_rail/UK_London_TfL/04_demand_od/NUMBAT/{args.year}/{mp.name}',args.config,args.remote)
    print(json.dumps({'year':args.year,'files':len(rec),'bytes':sum(x['bytes'] for x in rec)}))
if __name__=='__main__': main()
