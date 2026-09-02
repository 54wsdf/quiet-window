#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, tempfile, time, xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlparse
import requests

ROOTS={
 'trains':'https://rata.digitraffic.fi/api/v1/trains/dumps/',
 'compositions':'https://rata.digitraffic.fi/api/v1/compositions/dumps/',
 'train_locations':'https://rata.digitraffic.fi/api/v1/train-locations/dumps/',
}
DATE_RE=re.compile(r'(?P<y>20\d{2})[-_/]?(?P<m>0[1-9]|1[0-2])[-_/]?(?P<d>[0-3]\d)')
YM_RE=re.compile(r'(?P<y>20\d{2})[-_/]?(?P<m>0[1-9]|1[0-2])')

def strip(tag): return tag.rsplit('}',1)[-1]
def texts(root,w): return [(e.text or '').strip() for e in root.iter() if strip(e.tag)==w and (e.text or '').strip()]
def list_objects(s,base):
    for mode in ('v2','v1'):
        try:
            keys=[]; token=None; seen=set()
            while True:
                params={'max-keys':'1000'}
                if mode=='v2': params['list-type']='2'
                if token: params['continuation-token' if mode=='v2' else 'marker']=token
                r=s.get(base+'?'+urlencode(params),timeout=(30,180)); r.raise_for_status(); root=ET.fromstring(r.content)
                page=texts(root,'Key'); keys.extend(page); trunc=(texts(root,'IsTruncated') or ['false'])[0].lower()=='true'
                if not trunc: return sorted(set(keys)),mode
                nxt=(texts(root,'NextContinuationToken') if mode=='v2' else texts(root,'NextMarker'))
                token=nxt[0] if nxt else (page[-1] if page else None)
                if not token or token in seen: raise RuntimeError('pagination stalled')
                seen.add(token)
        except Exception:
            continue
    raise RuntimeError('provider listing failed')
def date_parts(v):
    m=DATE_RE.search(v) or YM_RE.search(v)
    if not m: return None
    return int(m.group('y')),int(m.group('m')),int(m.groupdict().get('d') or 0)
def payload_url(base,key):
    if key.startswith('http'): return key
    if key.startswith('api/v1/'):
        return urljoin(base.split('/api/v1/',1)[0]+'/',key)
    return urljoin(base,key.rsplit('/',1)[-1] if '/dumps/' in key else key)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def run(c): subprocess.run(c,check=True)
def out(c): return subprocess.check_output(c,text=True)
def put(p,d,cfg,remote):
    run(['rclone','copyto',str(p),f'{remote}:{d}','--config',cfg,'--retries','8','--low-level-retries','20','--transfers','1','--checkers','4','--stats','60s','--stats-one-line'])
    rs=int(json.loads(out(['rclone','lsjson',f'{remote}:{d}','--config',cfg,'--files-only']))[0]['Size'])
    if rs!=p.stat().st_size: raise RuntimeError('remote size mismatch')
def fetch(s,url,p):
    for n in range(8):
        try:
            with s.get(url,stream=True,timeout=(30,1800)) as r:
                r.raise_for_status()
                with p.open('wb') as f:
                    for b in r.iter_content(8*1024*1024):
                        if b: f.write(b)
            if p.stat().st_size<=0: raise RuntimeError('zero bytes')
            return
        except Exception:
            p.unlink(missing_ok=True)
            if n==7: raise
            time.sleep(min(120,2**n))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--kind',choices=ROOTS,required=True); ap.add_argument('--year',type=int,required=True); ap.add_argument('--month',type=int,required=True); ap.add_argument('--config',required=True); ap.add_argument('--remote',required=True); args=ap.parse_args()
    s=requests.Session(); s.headers['User-Agent']='quiet-window-public-data-worker/2.0'; base=ROOTS[args.kind]; keys,mode=list_objects(s,base)
    selected=[]
    for key in keys:
        dp=date_parts(key)
        if dp and dp[0]==args.year and dp[1]==args.month: selected.append((dp[2],key,payload_url(base,key)))
    selected.sort(); layer={'trains':'02_actual_operations','compositions':'07_rolling_stock_composition','train_locations':'03_realtime'}[args.kind]
    destbase=f'01_rail/FI_digitraffic/{layer}/{args.year}/{args.month:02d}'; rec=[]
    with tempfile.TemporaryDirectory(prefix='qw-fi-') as td:
        td=Path(td)
        for i,(day,key,url) in enumerate(selected,1):
            fn=urlparse(url).path.rsplit('/',1)[-1] or key.rsplit('/',1)[-1]; p=td/fn; fetch(s,url,p); size=p.stat().st_size; digest=sha(p); put(p,f'{destbase}/{fn}',args.config,args.remote)
            rec.append({'day':day,'key':key,'file':fn,'bytes':size,'sha256':digest,'state':'ACQUIRED'}); p.unlink(); print(json.dumps({'kind':args.kind,'year':args.year,'month':args.month,'done':i,'total':len(selected),'bytes':size}))
        m={'schema':'quiet-window-acquisition-manifest-v1','state':'ACQUIRED','source':'Fintraffic Digitraffic Rail','tier':'S','license':'CC BY 4.0','kind':args.kind,'year':args.year,'month':args.month,'listing_mode':mode,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'github_run_id':os.environ.get('GITHUB_RUN_ID'),'files':rec}
        mp=td/f'MANIFEST_{args.kind}_{args.year}_{args.month:02d}_{os.environ.get("GITHUB_RUN_ID","manual")}.json'; mp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8'); put(mp,f'{destbase}/{mp.name}',args.config,args.remote)
    print(json.dumps({'kind':args.kind,'year':args.year,'month':args.month,'files':len(rec),'bytes':sum(x['bytes'] for x in rec)}))
if __name__=='__main__': main()
