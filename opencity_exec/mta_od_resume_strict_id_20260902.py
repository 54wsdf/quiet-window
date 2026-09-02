#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import requests

DATASET='28vm-gjqr'
CHUNK=100000
BASE=f'https://data.ny.gov/resource/{DATASET}.csv'
META=f'https://data.ny.gov/api/views/{DATASET}'
COUNT=f'https://data.ny.gov/resource/{DATASET}.json'
UA='OpenCity-MTA-OD-strict-id-resume/20260902'
PART_RE=re.compile(r'^28vm-gjqr\.part-(\d{5})\.csv$')
SAMPLE_PARTS=(0,80,160,240,318,397,476,554,561)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def output(cmd):
    return subprocess.check_output(cmd, text=True)


def rc(args, cfg, remote, root_id, **kw):
    cmd=['rclone',*args,'--config',cfg,'--drive-root-folder-id',root_id]
    return run(cmd, **kw)


def rc_output(args, cfg, remote, root_id):
    cmd=['rclone',*args,'--config',cfg,'--drive-root-folder-id',root_id]
    return output(cmd)


def sha256_file(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def remote_sha256(name:str,cfg:str,remote:str,root_id:str):
    p=subprocess.Popen(['rclone','cat',f'{remote}:parts/{name}','--config',cfg,'--drive-root-folder-id',root_id],stdout=subprocess.PIPE)
    h=hashlib.sha256(); n=0
    assert p.stdout is not None
    for b in iter(lambda:p.stdout.read(8*1024*1024),b''):
        h.update(b); n+=len(b)
    rcod=p.wait()
    if rcod!=0: raise RuntimeError(f'rclone cat failed for {name}: {rcod}')
    return name,n,h.hexdigest()


def session():
    s=requests.Session(); s.headers['User-Agent']=UA; return s


def get_meta(s):
    r=s.get(META,timeout=(30,180)); r.raise_for_status(); m=r.json()
    return {k:m.get(k) for k in ('id','name','rowsUpdatedAt','dataUpdatedAt','metadataUpdatedAt','publicationDate')}


def get_count(s):
    r=s.get(COUNT,params={'$select':'count(*)'},timeout=(30,180)); r.raise_for_status(); return int(r.json()[0]['count'])


def retry_get(s,url,params,stream=False,attempts=8):
    last=None
    for n in range(attempts):
        try:
            r=s.get(url,params=params,stream=stream,timeout=(30,1800 if stream else 300))
            r.raise_for_status(); return r
        except Exception as e:
            last=e
            if n==attempts-1: raise
            time.sleep(min(90,2**n))
    raise last


def fetch_one_row(off:int):
    s=session()
    with retry_get(s,BASE,{'$limit':'1','$offset':str(off),'$order':':id'}) as r:
        rows=list(csv.reader(r.text.splitlines()))
    if len(rows)!=2: raise RuntimeError(f'expected one data row at {off}, got {max(0,len(rows)-1)}')
    return rows[0],rows[1]


def file_bounds(p:Path):
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        rd=csv.reader(f); header=next(rd); first=next(rd); last=first; n=1
        for row in rd: last=row; n+=1
    return header,first,last,n


def validate_existing_parts(cfg,remote,root_id,tmp:Path):
    listing=rc_output(['lsf',f'{remote}:parts','--files-only'],cfg,remote,root_id).splitlines()
    idx=[]
    for name in listing:
        m=PART_RE.match(name.strip())
        if m: idx.append(int(m.group(1)))
    idx=sorted(set(idx))
    if not idx: raise RuntimeError('no existing MTA OD parts')
    missing=sorted(set(range(idx[-1]+1))-set(idx))
    if idx[0]!=0 or missing: raise RuntimeError(f'existing multipart gaps: first={idx[0]}, missing={missing[:20]}')
    if idx[-1] < 561: raise RuntimeError(f'existing chain regressed: max={idx[-1]}')
    sample=[x for x in SAMPLE_PARTS if x<=idx[-1]]
    stored={}
    for i in sample:
        fn=f'{DATASET}.part-{i:05d}.csv'; p=tmp/fn
        rc(['copyto',f'{remote}:parts/{fn}',str(p),'--retries','5','--low-level-retries','10'],cfg,remote,root_id)
        h,a,b,n=file_bounds(p)
        if n!=CHUNK: raise RuntimeError(f'{fn} has {n} data rows, expected {CHUNK}')
        stored[i]={'header':h,'first':a,'last':b,'rows':n,'bytes':p.stat().st_size,'sha256':sha256_file(p)}
        p.unlink()
    jobs={}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i in sample:
            jobs[ex.submit(fetch_one_row,i*CHUNK)]=(i,'first')
            jobs[ex.submit(fetch_one_row,i*CHUNK+CHUNK-1)]=(i,'last')
        live={}
        for fut in as_completed(jobs):
            i,side=jobs[fut]; live.setdefault(i,{})[side]=fut.result()
    checks=[]
    for i in sample:
        sh,sf,sl=live[i]['first'][0],live[i]['first'][1],live[i]['last'][1]
        ok=(stored[i]['header']==sh and stored[i]['first']==sf and stored[i]['last']==sl)
        checks.append({'part':i,'first_offset':i*CHUNK,'last_offset':i*CHUNK+CHUNK-1,'exact_header_first_last_match':ok,
                       'stored_bytes':stored[i]['bytes'],'stored_sha256':stored[i]['sha256']})
        if not ok: raise RuntimeError(f':id contract mismatch at archived part {i}')
    return idx,checks


def count_rows(p:Path):
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        rd=csv.reader(f); next(rd); return sum(1 for _ in rd)


def fetch_chunk_strict(s,off:int,limit:int,dst:Path):
    params={'$limit':str(limit),'$offset':str(off),'$order':':id'}
    with retry_get(s,BASE,params,stream=True) as r:
        with dst.open('wb') as f:
            for b in r.iter_content(8*1024*1024):
                if b: f.write(b)
    if dst.stat().st_size<=0: raise RuntimeError(f'zero-byte chunk at {off}')


def upload_part(p:Path,name:str,digest:str,rows:int,off:int,cfg,remote,root_id):
    rc(['copyto',str(p),f'{remote}:parts/{name}','--retries','8','--low-level-retries','20','--transfers','1','--checkers','4'],cfg,remote,root_id)
    meta={'file':name,'offset':off,'rows':rows,'bytes':p.stat().st_size,'sha256':digest,'order':':id','state':'ACQUIRED'}
    side=p.parent/(name+'.sha256.json'); side.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    rc(['copyto',str(side),f'{remote}:parts/{side.name}','--retries','5','--low-level-retries','10'],cfg,remote,root_id)
    side.unlink()


def existing_remote_csv(cfg,remote,root_id):
    ls=json.loads(rc_output(['lsjson',f'{remote}:parts','--files-only'],cfg,remote,root_id))
    out={}
    for x in ls:
        n=x.get('Name',''); m=PART_RE.match(n)
        if m: out[int(m.group(1))]={'file':n,'bytes':int(x.get('Size',0))}
    return out


def load_sidecars(cfg,remote,root_id,tmp:Path):
    rc(['copy',f'{remote}:parts',str(tmp/'sidecars'),'--include','*.sha256.json','--retries','5','--low-level-retries','10'],cfg,remote,root_id)
    out={}
    for p in (tmp/'sidecars').glob('*.sha256.json'):
        d=json.loads(p.read_text()); m=PART_RE.match(d.get('file',''))
        if m: out[int(m.group(1))]=d
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--remote',required=True); ap.add_argument('--root-id',required=True); a=ap.parse_args()
    s=session(); start_meta=get_meta(s); total=get_count(s)
    expected_parts=(total+CHUNK-1)//CHUNK
    if total!=72639113: raise RuntimeError(f'source row count changed from frozen acquisition count: {total}')
    with tempfile.TemporaryDirectory(prefix='opencity-mta-od-resume-') as td0:
        td=Path(td0)
        existing_idx,contract_checks=validate_existing_parts(a.config,a.remote,a.root_id,td)
        initial_max=max(existing_idx)
        source_version={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'start_metadata':start_meta,'row_count':total,
                        'ordering_contract':'$order=:id','original_downloader_commit':'fbc63b8efdc63c632ca2fdafe1eee9bc010367e0',
                        'original_run_id':'33586650747','original_job_id':'100112095351','preserved_part_max_before_resume':initial_max,
                        'contract_samples':contract_checks}
        vp=td/'SOURCE_VERSION_AND_ORDER_CONTRACT_20260902.json'; vp.write_text(json.dumps(source_version,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        rc(['copyto',str(vp),f'{a.remote}:SOURCE_VERSION_AND_ORDER_CONTRACT_20260902.json','--retries','5','--low-level-retries','10'],a.config,a.remote,a.root_id)

        remote_files=existing_remote_csv(a.config,a.remote,a.root_id)
        new_records={}; start_idx=initial_max+1
        for i in range(start_idx,expected_parts):
            # Recheck immutable source version every 10 new parts.
            if (i-start_idx)%10==0:
                now_meta=get_meta(s); now_total=get_count(s)
                if now_meta.get('rowsUpdatedAt')!=start_meta.get('rowsUpdatedAt') or now_total!=total:
                    raise RuntimeError(f'source version changed during resume: {start_meta} -> {now_meta}, rows {total}->{now_total}')
            off=i*CHUNK; limit=min(CHUNK,total-off); name=f'{DATASET}.part-{i:05d}.csv'; p=td/name
            if i in remote_files:
                rc(['copyto',f'{a.remote}:parts/{name}',str(p),'--retries','5','--low-level-retries','10'],a.config,a.remote,a.root_id)
                rows=count_rows(p)
                if rows!=limit: raise RuntimeError(f'existing resumed {name} row mismatch {rows}!={limit}')
                # Existing post-561 part on a retry must still match current strict :id at both ends.
                h,first,last,n=file_bounds(p); lh,lf=fetch_one_row(off); _,ll=fetch_one_row(off+limit-1)
                if h!=lh or first!=lf or last!=ll: raise RuntimeError(f'existing resumed {name} fails strict :id boundary validation')
            else:
                fetch_chunk_strict(s,off,limit,p); rows=count_rows(p)
                if rows!=limit: raise RuntimeError(f'downloaded {name} row mismatch {rows}!={limit}')
            dig=sha256_file(p); size=p.stat().st_size
            if i not in remote_files: upload_part(p,name,dig,rows,off,a.config,a.remote,a.root_id)
            new_records[i]={'part':i,'offset':off,'limit':limit,'rows_expected':limit,'file':name,'bytes':size,'sha256':dig,'state':'ACQUIRED','order':':id'}
            p.unlink()
            print(json.dumps({'state':'RESUMED_PART_ACQUIRED','part':i,'offset':off,'rows':limit,'bytes':size,'sha256':dig}),flush=True)

        end_meta=get_meta(s); end_total=get_count(s)
        if end_meta.get('rowsUpdatedAt')!=start_meta.get('rowsUpdatedAt') or end_total!=total:
            raise RuntimeError(f'source version changed before closure: {start_meta} -> {end_meta}, rows {total}->{end_total}')
        final_files=existing_remote_csv(a.config,a.remote,a.root_id)
        final_idx=sorted(final_files)
        if final_idx!=list(range(expected_parts)):
            missing=sorted(set(range(expected_parts))-set(final_idx)); extra=sorted(set(final_idx)-set(range(expected_parts)))
            raise RuntimeError(f'final part index mismatch missing={missing[:20]} extra={extra[:20]}')

        sidecars=load_sidecars(a.config,a.remote,a.root_id,td)
        records={i:new_records[i] for i in new_records}
        for i,d in sidecars.items(): records.setdefault(i,{'part':i,'offset':i*CHUNK,'limit':d['rows'],'rows_expected':d['rows'],'file':d['file'],'bytes':d['bytes'],'sha256':d['sha256'],'state':'ACQUIRED','order':':id'})
        need_hash=[i for i in final_idx if i not in records]
        print(json.dumps({'state':'HASHING_PRESERVED_PARTS','count':len(need_hash)}),flush=True)
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs={ex.submit(remote_sha256,final_files[i]['file'],a.config,a.remote,a.root_id):i for i in need_hash}
            for fut in as_completed(futs):
                i=futs[fut]; name,n,dig=fut.result(); expected=min(CHUNK,total-i*CHUNK)
                records[i]={'part':i,'offset':i*CHUNK,'limit':expected,'rows_expected':expected,'file':name,'bytes':n,'sha256':dig,'state':'ACQUIRED','order':':id'}
                print(json.dumps({'state':'PRESERVED_PART_HASHED','part':i,'bytes':n,'sha256':dig}),flush=True)
        parts=[records[i] for i in range(expected_parts)]
        # Size metadata must agree with what was hashed/read.
        for r in parts:
            if r['bytes']!=final_files[r['part']]['bytes']:
                raise RuntimeError(f"remote size mismatch in manifest for {r['file']}")
        manifest={
          'schema':'opencity-mta-od-strict-id-manifest-v2','state':'ACQUIRED','evidence_class':'public_data registry / official public source',
          'source':'MTA Open Data via NY Open Data Socrata','domain':'data.ny.gov','dataset_id':DATASET,
          'name':'MTA Subway Origin-Destination Ridership Estimate: Beginning 2026','source_resource_url':BASE,'source_metadata_url':META,
          'source_metadata':start_meta,'source_row_count':total,'ordering_contract':'$order=:id','chunk_rows':CHUNK,'part_count':expected_parts,
          'preserved_parts':start_idx,'resumed_parts':expected_parts-start_idx,'contract_validation':contract_checks,
          'source_version_unchanged_through_close':True,'retrieved_closed_at_utc':datetime.now(timezone.utc).isoformat(),
          'github_run_id':os.environ.get('GITHUB_RUN_ID'),'executor_commit':os.environ.get('GITHUB_SHA'),'parts':parts}
        mp=td/f'MANIFEST_{DATASET}_{os.environ.get("GITHUB_RUN_ID","manual")}.json'; mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        sums=td/'SHA256SUMS.txt'; sums.write_text(''.join(f"{r['sha256']}  parts/{r['file']}\n" for r in parts),encoding='utf-8')
        close=td/'ACQUISITION_CLOSED_20260902.json'; close.write_text(json.dumps({k:manifest[k] for k in ('state','evidence_class','dataset_id','name','source_metadata','source_row_count','ordering_contract','chunk_rows','part_count','preserved_parts','resumed_parts','source_version_unchanged_through_close','retrieved_closed_at_utc','github_run_id','executor_commit')},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for p,name in [(mp,mp.name),(sums,'SHA256SUMS.txt'),(close,'ACQUISITION_CLOSED_20260902.json')]:
            rc(['copyto',str(p),f'{a.remote}:{name}','--retries','8','--low-level-retries','20'],a.config,a.remote,a.root_id)
        print(json.dumps({'state':'ACQUIRED','dataset_id':DATASET,'rows':total,'parts':expected_parts,'preserved':start_idx,'resumed':expected_parts-start_idx,'manifest':mp.name}),flush=True)

if __name__=='__main__': main()
