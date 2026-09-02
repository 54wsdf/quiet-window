#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, tempfile, time
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
    if rs!=p.stat().st_size: raise RuntimeError('remote size mismatch')
def download(s,url,p):
    for n in range(8):
        try:
            with s.get(url,stream=True,timeout=(30,7200),allow_redirects=True) as r:
                r.raise_for_status()
                ct=(r.headers.get('content-type') or '').lower()
                with p.open('wb') as f:
                    for b in r.iter_content(8*1024*1024):
                        if b: f.write(b)
            if p.stat().st_size<=0 or ('text/html' in ct and p.stat().st_size<100000): raise RuntimeError('invalid payload')
            return ct
        except Exception:
            p.unlink(missing_ok=True)
            if n==7: raise
            time.sleep(min(120,2**n))
def safe(v): return re.sub(r'[^A-Za-z0-9._-]+','_',v or '').strip('_') or 'resource'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',required=True); ap.add_argument('--package',required=True); ap.add_argument('--dataset-id',required=True); ap.add_argument('--drive-path',required=True); ap.add_argument('--config',required=True); ap.add_argument('--remote',required=True); args=ap.parse_args()
    s=requests.Session(); s.headers['User-Agent']='quiet-window-public-data-worker/2.0'; api=args.base.rstrip('/')+'/api/3/action/package_show'; r=s.get(api,params={'id':args.package},timeout=(30,180)); r.raise_for_status(); payload=r.json();
    if not payload.get('success'): raise RuntimeError('CKAN package_show failed')
    pkg=payload['result']; resources=[x for x in pkg.get('resources',[]) if x.get('url') and x.get('url_type','')!='upload-disabled']; rec=[]
    with tempfile.TemporaryDirectory(prefix='qw-ckan-') as td:
        td=Path(td)
        for i,x in enumerate(resources,1):
            url=x['url']; raw=x.get('name') or Path(urlparse(url).path).name or f'resource_{i}'; name=safe(raw)
            ext=Path(urlparse(url).path).suffix
            if ext and not name.lower().endswith(ext.lower()): name+=ext
            if any(y['file']==name for y in rec): name=f'{i:03d}_{name}'
            p=td/name; ct=download(s,url,p); size=p.stat().st_size; digest=sha(p); dest=f'{args.drive_path}/{name}'; put(p,dest,args.config,args.remote)
            rec.append({'resource_id':x.get('id'),'name':x.get('name'),'file':name,'url':url,'format':x.get('format'),'last_modified':x.get('last_modified'),'bytes':size,'sha256':digest,'content_type':ct,'state':'ACQUIRED'}); p.unlink(); print(json.dumps({'dataset_id':args.dataset_id,'done':i,'total':len(resources),'file':name,'bytes':size}))
        m={'schema':'quiet-window-ckan-package-v1','state':'ACQUIRED','dataset_id':args.dataset_id,'package':args.package,'title':pkg.get('title'),'source':args.base,'tier':'S/A','license_title':pkg.get('license_title'),'license_url':pkg.get('license_url'),'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'github_run_id':os.environ.get('GITHUB_RUN_ID'),'files':rec}
        mp=td/f'MANIFEST_{safe(args.dataset_id)}_{os.environ.get("GITHUB_RUN_ID","manual")}.json'; mp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8'); put(mp,f'{args.drive_path}/{mp.name}',args.config,args.remote)
    print(json.dumps({'dataset_id':args.dataset_id,'files':len(rec),'bytes':sum(x['bytes'] for x in rec)}))
if __name__=='__main__': main()
