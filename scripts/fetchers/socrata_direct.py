#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
import requests

UA='quiet-window-public-data-worker/2.0'

def run(cmd): subprocess.run(cmd,check=True)
def output(cmd): return subprocess.check_output(cmd,text=True)
def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def rclone_put(p:Path,dest:str,cfg:str,remote:str):
    run(['rclone','copyto',str(p),f'{remote}:{dest}','--config',cfg,'--retries','8','--low-level-retries','20','--transfers','1','--checkers','4','--stats','60s','--stats-one-line'])
    rs=int(json.loads(output(['rclone','lsjson',f'{remote}:{dest}','--config',cfg,'--files-only']))[0]['Size'])
    if rs!=p.stat().st_size: raise RuntimeError(f'remote size mismatch: {rs} != {p.stat().st_size}')

def get_count(s,domain,did):
    u=f'https://{domain}/resource/{did}.json'
    for n in range(8):
        try:
            r=s.get(u,params={'$select':'count(*)'},timeout=(30,180)); r.raise_for_status(); return int(r.json()[0]['count'])
        except Exception:
            if n==7: raise
            time.sleep(min(60,2**n))

def fetch_chunk(s,url,params,dst):
    for n in range(8):
        try:
            with s.get(url,params=params,stream=True,timeout=(30,1800)) as r:
                r.raise_for_status()
                with dst.open('wb') as f:
                    for b in r.iter_content(8*1024*1024):
                        if b: f.write(b)
            if dst.stat().st_size<=0: raise RuntimeError('zero-byte chunk')
            return
        except Exception:
            dst.unlink(missing_ok=True)
            if n==7: raise
            time.sleep(min(120,2**n))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--domain',default='data.ny.gov'); ap.add_argument('--dataset-id',required=True); ap.add_argument('--name',required=True); ap.add_argument('--drive-path',required=True); ap.add_argument('--config',required=True); ap.add_argument('--remote',required=True); ap.add_argument('--chunk-rows',type=int,default=100000); args=ap.parse_args()
    s=requests.Session(); s.headers['User-Agent']=UA
    total=get_count(s,args.domain,args.dataset_id); records=[]; base=f'https://{args.domain}/resource/{args.dataset_id}.csv'
    with tempfile.TemporaryDirectory(prefix='qw-socrata-') as td:
        td=Path(td)
        part=0
        for offset in range(0,total,args.chunk_rows):
            limit=min(args.chunk_rows,total-offset); fn=f'{args.dataset_id}.part-{part:05d}.csv'; p=td/fn
            params={'$limit':str(limit),'$offset':str(offset),'$order':':id'}
            try: fetch_chunk(s,base,params,p)
            except requests.HTTPError:
                params.pop('$order',None); fetch_chunk(s,base,params,p)
            size=p.stat().st_size; digest=sha256(p); dest=f'{args.drive_path}/parts/{fn}'
            rclone_put(p,dest,args.config,args.remote)
            records.append({'part':part,'offset':offset,'limit':limit,'rows_expected':limit,'file':fn,'bytes':size,'sha256':digest,'state':'ACQUIRED'})
            p.unlink(); part+=1
            print(json.dumps({'dataset_id':args.dataset_id,'part':part,'offset':offset,'total_rows':total,'bytes':size}))
        manifest={'schema':'quiet-window-socrata-manifest-v1','state':'ACQUIRED','source':'Socrata Open Data','domain':args.domain,'dataset_id':args.dataset_id,'name':args.name,'tier':'A/S','row_count':total,'chunk_rows':args.chunk_rows,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'github_run_id':os.environ.get('GITHUB_RUN_ID'),'parts':records}
        mp=td/f'MANIFEST_{args.dataset_id}_{os.environ.get("GITHUB_RUN_ID","manual")}.json'; mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        rclone_put(mp,f'{args.drive_path}/{mp.name}',args.config,args.remote)
    print(json.dumps({'dataset_id':args.dataset_id,'rows':total,'parts':len(records),'bytes':sum(x['bytes'] for x in records)}))

if __name__=='__main__': main()
