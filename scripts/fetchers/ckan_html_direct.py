#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, html.parser, json, os, re, subprocess, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
import requests

class Links(html.parser.HTMLParser):
    def __init__(self): super().__init__(); self.hrefs=[]
    def handle_starttag(self,tag,attrs):
        if tag.lower()!='a': return
        for k,v in attrs:
            if k.lower()=='href' and v: self.hrefs.append(v.replace('&amp;','&').strip())

def run(c): subprocess.run(c,check=True)
def visible_size(remote,d,cfg):
    last='missing'
    for n in range(12):
        r=subprocess.run(['rclone','lsjson',f'{remote}:{d}','--config',cfg,'--files-only'],text=True,capture_output=True)
        if r.returncode==0:
            try:
                x=json.loads(r.stdout)
                if x and isinstance(x[0].get('Size'),int): return int(x[0]['Size'])
            except Exception as e: last=str(e)
        else: last=(r.stderr or r.stdout or '').strip()
        time.sleep(min(30,2+n*2))
    raise RuntimeError(f'remote not visible: {d}: {last}')
def put(p,d,cfg,remote):
    run(['rclone','copyto',str(p),f'{remote}:{d}','--config',cfg,'--retries','8','--low-level-retries','20','--transfers','1','--checkers','4','--stats','60s','--stats-one-line'])
    rs=visible_size(remote,d,cfg)
    if rs!=p.stat().st_size: raise RuntimeError(f'remote size mismatch {rs} != {p.stat().st_size}')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def download(s,u,p):
    for n in range(8):
        try:
            with s.get(u,stream=True,timeout=(30,7200),allow_redirects=True) as r:
                r.raise_for_status(); ct=(r.headers.get('content-type') or '').lower()
                with p.open('wb') as f:
                    for b in r.iter_content(8*1024*1024):
                        if b: f.write(b)
            if p.stat().st_size<=0 or ('text/html' in ct and p.stat().st_size<100000): raise RuntimeError('invalid payload')
            return ct
        except Exception:
            p.unlink(missing_ok=True)
            if n==7: raise
            time.sleep(min(120,2**n))
def safe(v): return re.sub(r'[^A-Za-z0-9._-]+','_',v).strip('_') or 'resource'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--page',required=True); ap.add_argument('--dataset-id',required=True); ap.add_argument('--drive-path',required=True); ap.add_argument('--config',required=True); ap.add_argument('--remote',required=True); args=ap.parse_args()
    s=requests.Session(); s.headers.update({'User-Agent':'Mozilla/5.0 quiet-window-public-data-worker/2.2','Accept':'text/html,application/xhtml+xml,*/*'})
    r=s.get(args.page,timeout=(30,180)); r.raise_for_status(); p=Links(); p.feed(r.text)
    urls=[]
    for h in p.hrefs:
        u=urljoin(args.page,h)
        path=urlparse(u).path.lower()
        if '/download/' in path and u not in urls: urls.append(u)
    if not urls: raise RuntimeError('no public download links found on dataset page')
    rec=[]
    with tempfile.TemporaryDirectory(prefix='qw-ckanhtml-') as td:
        td=Path(td)
        for i,u in enumerate(urls,1):
            raw=unquote(urlparse(u).path.rsplit('/',1)[-1]) or f'resource_{i}'; name=safe(raw)
            if any(x['file']==name for x in rec): name=f'{i:03d}_{name}'
            fp=td/name; ct=download(s,u,fp); size=fp.stat().st_size; digest=sha(fp); dest=f'{args.drive_path}/{name}'; put(fp,dest,args.config,args.remote)
            rec.append({'file':name,'url':u,'bytes':size,'sha256':digest,'content_type':ct,'state':'ACQUIRED'}); fp.unlink(); print(json.dumps({'dataset_id':args.dataset_id,'done':i,'total':len(urls),'file':name,'bytes':size}))
        m={'schema':'quiet-window-public-page-resources-v1','state':'ACQUIRED','dataset_id':args.dataset_id,'source_page':args.page,'tier':'S/A','retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'github_run_id':os.environ.get('GITHUB_RUN_ID'),'files':rec}
        mp=td/f'MANIFEST_{safe(args.dataset_id)}_{os.environ.get("GITHUB_RUN_ID","manual")}.json'; mp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8'); put(mp,f'{args.drive_path}/{mp.name}',args.config,args.remote)
    print(json.dumps({'dataset_id':args.dataset_id,'files':len(rec),'bytes':sum(x['bytes'] for x in rec)}))
if __name__=='__main__': main()
