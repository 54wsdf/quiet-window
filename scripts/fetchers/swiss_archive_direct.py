#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, html.parser, json, os, re, subprocess, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

INDEX={
 'actual':'https://archive.opentransportdata.swiss/istdaten.php',
 'gtfs':'https://archive.opentransportdata.swiss/timetable_gtfs.php',
}

class Links(html.parser.HTMLParser):
    def __init__(self): super().__init__(); self.hrefs=[]
    def handle_starttag(self, tag, attrs):
        if tag.lower()!='a': return
        for k,v in attrs:
            if k.lower()=='href' and v: self.hrefs.append(v.replace('&amp;','&').strip())

def run(cmd): subprocess.run(cmd, check=True)
def output(cmd): return subprocess.check_output(cmd, text=True)

def listing(kind:str)->list[dict]:
    text=output(['curl','--compressed','-fsSL','--retry','5','--retry-all-errors','-A','curl/8.5.0',INDEX[kind]])
    p=Links(); p.feed(text); rows=[]; seen=set()
    for href in p.hrefs:
        if '.zip' not in href.lower(): continue
        url=urljoin(INDEX[kind],href)
        if url in seen: continue
        seen.add(url)
        path=urlparse(url).path; name=path.rsplit('/',1)[-1]
        if kind=='actual':
            ym=re.search(r'/istdaten/(20\d{2})/',path)
            if not ym: continue
            year=int(ym.group(1)); month=None
            for pat in [r'(?:^|/)(\d{2})_(0[1-9]|1[0-2])\.zip$',r'ist-daten(?:-v2)?-(20\d{2})-(0[1-9]|1[0-2])\.zip$']:
                m=re.search(pat,path,re.I)
                if m:
                    month=int(m.group(2)); break
            if month is None: continue
            rows.append({'url':url,'name':name,'year':year,'month':month})
        else:
            sy=re.search(r'/timetable_gtfs/timetable-(20\d{2})-gtfs(?:2020)?/',path)
            if not sy: continue
            year=int(sy.group(1)); dt=None
            m=re.search(r'(20\d{2})-(0[1-9]|1[0-2])-([0-3]\d)',name)
            if m: dt=(int(m.group(1)),int(m.group(2)),int(m.group(3)))
            else:
                m=re.search(r'(20\d{2})(0[1-9]|1[0-2])([0-3]\d)',name)
                if m: dt=(int(m.group(1)),int(m.group(2)),int(m.group(3)))
            if not dt: continue
            rows.append({'url':url,'name':name,'year':year,'snapshot':dt})
    return sorted(rows,key=lambda x:(x['year'],x.get('month',0),x.get('snapshot',(0,0,0)),x['name']))

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def remote_size(dest:str, config:str, remote:str)->int:
    last='not visible'
    for n in range(10):
        p=subprocess.run(['rclone','lsjson',f'{remote}:{dest}','--config',config,'--files-only'],text=True,capture_output=True)
        if p.returncode==0:
            try:
                rows=json.loads(p.stdout)
                if rows and isinstance(rows[0].get('Size'),int): return int(rows[0]['Size'])
            except Exception as exc:
                last=str(exc)
        else:
            last=(p.stderr or p.stdout or f'exit {p.returncode}').strip()
        if n<9: time.sleep(min(30,2+n*3))
    raise RuntimeError(f'remote object not visible after upload: {dest}: {last}')
def rclone_put(local:Path, dest:str, config:str, remote:str):
    run(['rclone','copyto',str(local),f'{remote}:{dest}','--config',config,'--retries','8','--low-level-retries','20','--transfers','1','--checkers','4','--stats','60s','--stats-one-line'])
    rs=remote_size(dest,config,remote)
    local_size=local.stat().st_size
    if rs!=local_size: raise RuntimeError(f'remote size mismatch {local_size} != {rs}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--kind',choices=['actual','gtfs'],required=True); ap.add_argument('--year',type=int,required=True); ap.add_argument('--config',required=True); ap.add_argument('--remote',required=True); args=ap.parse_args()
    selected=[x for x in listing(args.kind) if x['year']==args.year]
    if not selected:
        print(json.dumps({'kind':args.kind,'year':args.year,'files':0})); return
    layer='02_actual_operations' if args.kind=='actual' else '01_schedule'
    base=f'01_rail/CH_swiss/{layer}/{args.year}'
    records=[]
    with tempfile.TemporaryDirectory(prefix='qw-swiss-') as td:
        td=Path(td)
        for i,item in enumerate(selected,1):
            local=td/item['name']
            run(['curl','--compressed','-fL','--retry','8','--retry-all-errors','--connect-timeout','30','--max-time','7200',item['url'],'-o',str(local)])
            size=local.stat().st_size
            if size<=0: raise RuntimeError('zero-byte payload')
            digest=sha256(local)
            rclone_put(local,f'{base}/{item["name"]}',args.config,args.remote)
            records.append({**item,'bytes':size,'sha256':digest,'state':'ACQUIRED'})
            local.unlink()
            print(json.dumps({'kind':args.kind,'year':args.year,'done':i,'total':len(selected),'name':item['name'],'bytes':size}))
        manifest={
            'schema':'quiet-window-acquisition-manifest-v1','state':'ACQUIRED','source':'Open Transport Data Switzerland archive','tier':'S','kind':args.kind,'year':args.year,
            'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'terms_url':'https://opentransportdata.swiss/terms-of-use/','github_run_id':os.environ.get('GITHUB_RUN_ID'),'files':records,
        }
        mp=td/f'MANIFEST_{args.kind}_{args.year}_{os.environ.get("GITHUB_RUN_ID","manual")}.json'; mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        rclone_put(mp,f'{base}/{mp.name}',args.config,args.remote)
    print(json.dumps({'kind':args.kind,'year':args.year,'files':len(records),'bytes':sum(x['bytes'] for x in records)}))

if __name__=='__main__': main()
