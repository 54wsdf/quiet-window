#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, tempfile, time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import requests

INV='https://storage.googleapis.com/parquet.gtfsrt.io/inventory.json'; BUCKET='https://storage.googleapis.com/parquet.gtfsrt.io'
def run(c): subprocess.run(c,check=True)
def out(c): return subprocess.check_output(c,text=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def put(p,d,cfg,remote):
    run(['rclone','copyto',str(p),f'{remote}:{d}','--config',cfg,'--retries','8','--low-level-retries','20','--transfers','1','--checkers','4','--stats','60s','--stats-one-line'])
    rs=int(json.loads(out(['rclone','lsjson',f'{remote}:{d}','--config',cfg,'--files-only']))[0]['Size'])
    if rs!=p.stat().st_size: raise RuntimeError('remote size mismatch')
def dates(a,b):
    d=date.fromisoformat(a); e=date.fromisoformat(b)
    while d<=e: yield d.isoformat(); d+=timedelta(days=1)
def fetch(s,u,p):
    for n in range(6):
        try:
            with s.get(u,stream=True,timeout=(30,3600)) as r:
                if r.status_code==404: return False
                r.raise_for_status()
                with p.open('wb') as f:
                    for b in r.iter_content(8*1024*1024):
                        if b: f.write(b)
            return p.stat().st_size>0
        except Exception:
            p.unlink(missing_ok=True)
            if n==5: raise
            time.sleep(min(60,2**n))
def safe(v): return re.sub(r'[^A-Za-z0-9._-]+','_',v or '').strip('_') or 'system'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--agency-id',required=True); ap.add_argument('--system-id',default=''); ap.add_argument('--label',required=True); ap.add_argument('--drive-path',required=True); ap.add_argument('--config',required=True); ap.add_argument('--remote',required=True); args=ap.parse_args()
    s=requests.Session(); s.headers['User-Agent']='quiet-window-public-data-worker/2.0'; r=s.get(INV,timeout=(30,180)); r.raise_for_status()
    rows=[x for x in r.json() if x.get('agency_id')==args.agency_id and (not args.system_id or x.get('system_id')==args.system_id)]
    if not rows: raise RuntimeError('no matching inventory rows')
    man={'schema':'quiet-window-gtfsrt-system-v1','state':'ACQUIRED','source':'gtfsrt.io','label':args.label,'agency_id':args.agency_id,'system_id':args.system_id or None,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'github_run_id':os.environ.get('GITHUB_RUN_ID'),'feeds':[]}
    with tempfile.TemporaryDirectory(prefix='qw-gtrt-') as td:
        td=Path(td)
        for row in rows:
            feed={k:row.get(k) for k in ['url','base64url','agency_name','system_name','feed_type','date_min','date_max','total_records','total_bytes']}; feed['files']=[]; enc=row['base64url']; ft=row['feed_type']
            for ds in dates(row['date_min'],row['date_max']):
                u=f'{BUCKET}/{ft}/date={ds}/base64url={enc}/data.parquet'; fn=f'{ft}_{ds}_{enc[:12]}.parquet'; p=td/fn
                if not fetch(s,u,p): continue
                size=p.stat().st_size; digest=sha(p); dest=f'{args.drive_path}/{ft}/{ds}/{fn}'; put(p,dest,args.config,args.remote); feed['files'].append({'date':ds,'file':fn,'bytes':size,'sha256':digest,'state':'ACQUIRED'}); p.unlink(); print(json.dumps({'label':args.label,'feed_type':ft,'date':ds,'bytes':size}))
            man['feeds'].append(feed)
        mp=td/f'MANIFEST_{safe(args.label)}_{os.environ.get("GITHUB_RUN_ID","manual")}.json'; mp.write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding='utf-8'); put(mp,f'{args.drive_path}/{mp.name}',args.config,args.remote)
    print(json.dumps({'label':args.label,'feeds':len(rows),'files':sum(len(x['files']) for x in man['feeds']),'bytes':sum(f['bytes'] for x in man['feeds'] for f in x['files'])}))
if __name__=='__main__': main()
