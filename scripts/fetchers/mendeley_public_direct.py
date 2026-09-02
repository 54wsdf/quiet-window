#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
import requests

API='https://data.mendeley.com/public-api/datasets/{dataset}/files'
def run(c): subprocess.run(c,check=True)
def out(c): return subprocess.check_output(c,text=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def remote_size(remote,d,cfg):
    last=''
    for n in range(12):
        r=subprocess.run(['rclone','lsjson',f'{remote}:{d}','--config',cfg,'--files-only'],text=True,capture_output=True)
        if r.returncode==0:
            try:
                x=json.loads(r.stdout)
                if x: return int(x[0]['Size'])
            except Exception as e: last=str(e)
        else: last=(r.stderr or r.stdout or '').strip()
        time.sleep(min(30,2+n*2))
    raise RuntimeError(f'remote not visible: {last}')
def put(p,d,cfg,remote):
    run(['rclone','copyto',str(p),f'{remote}:{d}','--config',cfg,'--retries','8','--low-level-retries','20','--transfers','1','--checkers','4','--stats','60s','--stats-one-line'])
    rs=remote_size(remote,d,cfg)
    if rs!=p.stat().st_size: raise RuntimeError(f'remote size mismatch {rs} != {p.stat().st_size}')
def safe(v): return re.sub(r'[^A-Za-z0-9._-]+','_',v or '').strip('_') or 'file'
def fetch_listing(s,dataset,version,folder='root'):
    u=API.format(dataset=dataset); r=s.get(u,params={'folder_id':folder,'version':version},timeout=(30,180)); r.raise_for_status(); return r.json()
def flatten(s,dataset,version,folder='root',prefix=''):
    rows=[]
    for x in fetch_listing(s,dataset,version,folder):
        kind=(x.get('type') or x.get('object_type') or '').lower()
        fid=x.get('id') or x.get('folder_id')
        name=x.get('filename') or x.get('name') or fid or 'item'
        if kind=='folder' or ('content_details' not in x and x.get('children')):
            rows.extend(flatten(s,dataset,version,fid,prefix+safe(name)+'/')); continue
        cd=x.get('content_details') or {}; url=cd.get('download_url') or x.get('download_url')
        if url: rows.append({'file':safe(name),'relative_path':prefix+safe(name),'url':url,'provider_size':x.get('size') or x.get('file_size'),'provider_hash':x.get('hash') or cd.get('hash'),'mime_type':x.get('mime_type')})
    return rows
def download(s,u,p):
    for n in range(8):
        try:
            with s.get(u,stream=True,timeout=(30,7200),allow_redirects=True) as r:
                r.raise_for_status()
                with p.open('wb') as f:
                    for b in r.iter_content(8*1024*1024):
                        if b: f.write(b)
            if p.stat().st_size<=0: raise RuntimeError('zero-byte file')
            return
        except Exception:
            p.unlink(missing_ok=True)
            if n==7: raise
            time.sleep(min(120,2**n))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--version',type=int,default=1); ap.add_argument('--dataset-id',required=True); ap.add_argument('--drive-path',required=True); ap.add_argument('--config',required=True); ap.add_argument('--remote',required=True); args=ap.parse_args()
    s=requests.Session(); s.headers['User-Agent']='Mozilla/5.0 quiet-window-public-data-worker/2.3'; files=flatten(s,args.dataset,args.version)
    if not files: raise RuntimeError('no public Mendeley files resolved')
    rec=[]
    with tempfile.TemporaryDirectory(prefix='qw-mendeley-') as td:
        td=Path(td)
        for i,x in enumerate(files,1):
            p=td/x['file']; download(s,x['url'],p); size=p.stat().st_size
            if x.get('provider_size') and int(x['provider_size'])!=size: raise RuntimeError(f'provider size mismatch {x["file"]}')
            digest=sha(p); dest=f'{args.drive_path}/{x["relative_path"]}'; put(p,dest,args.config,args.remote); rec.append({**{k:v for k,v in x.items() if k!='url'},'bytes':size,'sha256':digest,'state':'ACQUIRED'}); p.unlink(); print(json.dumps({'dataset_id':args.dataset_id,'done':i,'total':len(files),'file':x['relative_path'],'bytes':size}))
        m={'schema':'quiet-window-mendeley-public-v1','state':'ACQUIRED','source':'Mendeley Data','short_id':args.dataset,'version':args.version,'dataset_id':args.dataset_id,'tier':'A/S','retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'github_run_id':os.environ.get('GITHUB_RUN_ID'),'files':rec}
        mp=td/f'MANIFEST_{safe(args.dataset_id)}_{os.environ.get("GITHUB_RUN_ID","manual")}.json'; mp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8'); put(mp,f'{args.drive_path}/{mp.name}',args.config,args.remote)
    print(json.dumps({'dataset_id':args.dataset_id,'files':len(rec),'bytes':sum(x['bytes'] for x in rec)}))
if __name__=='__main__': main()
