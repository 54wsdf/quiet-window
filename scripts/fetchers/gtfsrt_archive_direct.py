#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, tempfile, time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import requests

INVENTORY='https://storage.googleapis.com/parquet.gtfsrt.io/inventory.json'
BUCKET='https://storage.googleapis.com/parquet.gtfsrt.io'

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
def daterange(a,b):
    d=date.fromisoformat(a); e=date.fromisoformat(b)
    while d<=e:
        yield d.isoformat(); d+=timedelta(days=1)
def download(s,url,p):
    for n in range(6):
        try:
            with s.get(url,stream=True,timeout=(30,3600)) as r:
                if r.status_code==404: return False
                r.raise_for_status()
                with p.open('wb') as f:
                    for b in r.iter_content(8*1024*1024):
                        if b: f.write(b)
            if p.stat().st_size<=0: return False
            return True
        except Exception:
            p.unlink(missing_ok=True)
            if n==5: raise
            time.sleep(min(60,2**n))
def belongs(row,group):
    if row.get('agency_name')!='Metropolitan Transportation Authority': return False
    s=(row.get('system_name') or '').lower()
    if group=='subway': return 'subway' in s
    if group=='lirr': return 'long island rail road' in s
    if group=='mnr': return 'metro-north railroad' in s
    return False
def safe(v): return re.sub(r'[^A-Za-z0-9._-]+','_',v).strip('_') or 'unknown'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--group',choices=['subway','lirr','mnr'],required=True); ap.add_argument('--config',required=True); ap.add_argument('--remote',required=True); args=ap.parse_args()
    s=requests.Session(); s.headers['User-Agent']='quiet-window-public-data-worker/2.0'; inv=s.get(INVENTORY,timeout=(30,180)); inv.raise_for_status(); rows=[r for r in inv.json() if belongs(r,args.group)]
    if not rows: raise RuntimeError('no matching MTA rail feeds in inventory')
    manifest={'schema':'quiet-window-gtfsrt-archive-v1','state':'ACQUIRED','source':'gtfsrt.io / parquet.gtfsrt.io','license_note':'public archive; upstream feed terms may vary','group':args.group,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'github_run_id':os.environ.get('GITHUB_RUN_ID'),'feeds':[]}
    with tempfile.TemporaryDirectory(prefix='qw-gtfsrt-') as td:
        td=Path(td)
        for fi,row in enumerate(rows,1):
            feed={'url':row.get('url'),'base64url':row['base64url'],'agency_name':row.get('agency_name'),'system_name':row.get('system_name'),'feed_type':row['feed_type'],'date_min':row['date_min'],'date_max':row['date_max'],'inventory_records':row.get('total_records'),'inventory_bytes':row.get('total_bytes'),'files':[]}
            sys=safe(row.get('system_name') or args.group); ft=row['feed_type']; enc=row['base64url']
            for di,ds in enumerate(daterange(row['date_min'],row['date_max']),1):
                url=f'{BUCKET}/{ft}/date={ds}/base64url={enc}/data.parquet'; fn=f'{ft}_{ds}_{enc[:12]}.parquet'; p=td/fn
                ok=download(s,url,p)
                if not ok: continue
                size=p.stat().st_size; digest=sha(p); dest=f'01_rail/US_NY_MTA/03_realtime/gtfsrt_io/{sys}/{ft}/{ds}/{fn}'; put(p,dest,args.config,args.remote)
                feed['files'].append({'date':ds,'file':fn,'bytes':size,'sha256':digest,'path':dest,'state':'ACQUIRED'}); p.unlink()
                print(json.dumps({'group':args.group,'feed':fi,'feeds':len(rows),'date':ds,'bytes':size}))
            manifest['feeds'].append(feed)
        mp=td/f'MANIFEST_gtfsrt_io_mta_{args.group}_{os.environ.get("GITHUB_RUN_ID","manual")}.json'; mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); put(mp,f'01_rail/US_NY_MTA/03_realtime/gtfsrt_io/{mp.name}',args.config,args.remote)
    print(json.dumps({'group':args.group,'feeds':len(rows),'files':sum(len(x['files']) for x in manifest['feeds']),'bytes':sum(f['bytes'] for x in manifest['feeds'] for f in x['files'])}))
if __name__=='__main__': main()
