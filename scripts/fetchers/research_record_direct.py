#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import requests

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
    if rs!=p.stat().st_size: raise RuntimeError(f'remote size mismatch {rs} != {p.stat().st_size}')
def download(s,url,p):
    for n in range(8):
        try:
            with s.get(url,stream=True,timeout=(30,7200),allow_redirects=True) as r:
                r.raise_for_status()
                with p.open('wb') as f:
                    for b in r.iter_content(8*1024*1024):
                        if b: f.write(b)
            if p.stat().st_size<=0: raise RuntimeError('zero byte')
            return
        except Exception:
            p.unlink(missing_ok=True)
            if n==7: raise
            time.sleep(min(120,2**n))
def zenodo(s,rid):
    r=s.get(f'https://zenodo.org/api/records/{rid}',timeout=(30,180)); r.raise_for_status(); d=r.json(); rows=[]
    for f in d.get('files',[]):
        links=f.get('links') or {}; url=links.get('content') or links.get('self') or links.get('download')
        if url: rows.append({'name':f.get('key') or Path(urlparse(url).path).name,'url':url,'provider_size':f.get('size'),'provider_checksum':f.get('checksum')})
    return d,rows
def figshare(s,aid):
    r=s.get(f'https://api.figshare.com/v2/articles/{aid}',timeout=(30,180)); r.raise_for_status(); d=r.json(); rows=[]
    for f in d.get('files',[]):
        url=f.get('download_url')
        if url: rows.append({'name':f.get('name') or Path(urlparse(url).path).name,'url':url,'provider_size':f.get('size'),'provider_checksum':f.get('computed_md5') or f.get('supplied_md5')})
    return d,rows
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--provider',choices=['zenodo','figshare'],required=True); ap.add_argument('--record-id',required=True); ap.add_argument('--dataset-id',required=True); ap.add_argument('--drive-path',required=True); ap.add_argument('--config',required=True); ap.add_argument('--remote',required=True); args=ap.parse_args()
    s=requests.Session(); s.headers['User-Agent']='quiet-window-public-data-worker/2.0'
    meta,files=(zenodo(s,args.record_id) if args.provider=='zenodo' else figshare(s,args.record_id))
    if not files: raise RuntimeError('record exposes no downloadable files')
    rec=[]
    with tempfile.TemporaryDirectory(prefix='qw-record-') as td:
        td=Path(td)
        for i,f in enumerate(files,1):
            name=f['name'].replace('/','_').replace('\\','_'); p=td/name; download(s,f['url'],p); size=p.stat().st_size
            if f.get('provider_size') not in (None,0) and int(f['provider_size'])!=size: raise RuntimeError(f'provider size mismatch {name}')
            digest=sha(p); dest=f'{args.drive_path}/{name}'; put(p,dest,args.config,args.remote); rec.append({**f,'name':name,'bytes':size,'sha256':digest,'state':'ACQUIRED'}); p.unlink(); print(json.dumps({'dataset_id':args.dataset_id,'done':i,'total':len(files),'file':name,'bytes':size}))
        manifest={'schema':'quiet-window-research-record-v1','state':'ACQUIRED','provider':args.provider,'record_id':args.record_id,'dataset_id':args.dataset_id,'tier':'A/S','title':meta.get('title') or (meta.get('metadata') or {}).get('title'),'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'github_run_id':os.environ.get('GITHUB_RUN_ID'),'files':rec}
        mp=td/f'MANIFEST_{args.dataset_id}_{os.environ.get("GITHUB_RUN_ID","manual")}.json'; mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); put(mp,f'{args.drive_path}/{mp.name}',args.config,args.remote)
    print(json.dumps({'dataset_id':args.dataset_id,'files':len(rec),'bytes':sum(x['bytes'] for x in rec)}))
if __name__=='__main__': main()
