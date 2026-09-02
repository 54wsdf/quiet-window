#!/usr/bin/env python3
"""Rerun OpenCity T3/T4 on the frozen WebTRIS 20-site x 30-day public-registry acquisition.

Evidence class: public_data registry / official public source.
This is a rehearsal, not an organizer score. T3 reports observational proxies and
threshold sensitivity; T4 uses leave-one-site-out prediction so public counts from
the held-out detector are never used for fitting.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd

THRESHOLDS=(55.0,60.0,65.0)
Q=(0.90,0.95,0.975,0.99)

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load_manifest(root:Path):
    p=root/'ACQUISITION_MANIFEST.json'
    obj=json.loads(p.read_text(encoding='utf-8'))
    exp={x['file']:x['sha256'] for x in obj.get('files',[]) if x.get('file') and x.get('sha256')}
    checked=[]
    for name,expected in exp.items():
        q=root/name
        if q.exists():
            got=sha256(q); checked.append({'file':name,'expected':expected,'observed':got,'match':got==expected,'bytes':q.stat().st_size})
    bad=[x for x in checked if not x['match']]
    if bad: raise SystemExit('SHA mismatch: '+json.dumps(bad))
    return obj,p,checked

def load_site_file(p:Path)->pd.DataFrame:
    raw=json.loads(p.read_text(encoding='utf-8'))
    d=pd.DataFrame(raw.get('Rows',[])).copy()
    if d.empty: return d
    d['speed']=pd.to_numeric(d['Avg mph'],errors='coerce')
    d['y']=pd.to_numeric(d['Total Volume'],errors='coerce')
    d['ts']=pd.to_datetime(d['Report Date'].str[:10]+' '+d['Time Period Ending'],errors='coerce')
    d=d.dropna(subset=['speed','y','ts']).sort_values('ts').reset_index(drop=True)
    d['date']=d.ts.dt.date.astype(str)
    d['slot']=d.ts.dt.hour*4+d.ts.dt.minute//15
    d['weekend']=(d.ts.dt.dayofweek>=5).astype(int)
    d['vph']=4.0*d['y']
    m=re.search(r'site_([^_]+)_',p.name)
    d['site_id']=m.group(1) if m else p.stem
    d['site_name']=d['Site Name'].astype(str) if 'Site Name' in d else d['site_id']
    return d

def episode_stats(d:pd.DataFrame,thr:float)->dict:
    z=(d.speed<thr).to_numpy(bool); v=d.vph.to_numpy(float)
    starts=[]; rec=[]; lens=[]; cur=0
    for i,flag in enumerate(z):
        if flag:
            if cur==0 and i>0: starts.append(v[i-1])
            cur+=1
        elif cur:
            lens.append(cur); rec.append(v[i]); cur=0
    if cur: lens.append(cur)
    return {
      f'lt{int(thr)}_share':float(z.mean()),
      f'lt{int(thr)}_episodes':int(len(lens)),
      f'lt{int(thr)}_median_episode_min':None if not lens else float(np.median(lens)*15),
      f'lt{int(thr)}_pre_episode_flow_median_vph':None if not starts else float(np.median(starts)),
      f'lt{int(thr)}_recovery_flow_median_vph':None if not rec else float(np.median(rec)),
    }

def t3_sites(all_df:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for sid,d in all_df.groupby('site_id',sort=True):
        r={'site_id':sid,'site_name':str(d.site_name.iloc[0]),'rows':int(len(d)),'days':int(d.date.nunique()),
           'speed_mean_mph':float(d.speed.mean()),'speed_flow_corr':float(d[['speed','vph']].corr().iloc[0,1]),
           'max_vph':float(d.vph.max())}
        for q in Q:r[f'q{str(q).replace(".","")}_vph']=float(d.vph.quantile(q))
        for thr in THRESHOLDS:r.update(episode_stats(d,thr))
        rows.append(r)
    return pd.DataFrame(rows)

def design(d:pd.DataFrame,mode:str)->np.ndarray:
    s=d.speed.to_numpy(float); h=d.ts.dt.hour.to_numpy()+d.ts.dt.minute.to_numpy()/60
    if mode=='speed_quadratic': return np.c_[np.ones(len(d)),s,s*s]
    if mode=='speed_plus_time':
        return np.c_[np.ones(len(d)),s,s*s,np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),np.sin(4*np.pi*h/24),np.cos(4*np.pi*h/24),d.weekend.to_numpy(float)]
    raise ValueError(mode)

def pred_group(train:pd.DataFrame,test:pd.DataFrame,key:str)->np.ndarray:
    mp=train.groupby(key).y.mean().to_dict(); g=float(train.y.mean())
    return np.array([mp.get(v,g) for v in test[key]],float)

def fit_lstsq(train:pd.DataFrame,test:pd.DataFrame,mode:str)->np.ndarray:
    b=np.linalg.lstsq(design(train,mode),train.y.to_numpy(float),rcond=None)[0]
    return np.clip(design(test,mode)@b,0,None)

def mae(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);return float(np.mean(np.abs(y-p)))

def t4_loso(all_df:pd.DataFrame):
    rec=[]; regimes=[]
    for sid in sorted(all_df.site_id.unique()):
        tr=all_df[all_df.site_id!=sid]; te=all_df[all_df.site_id==sid].copy()
        preds={
          'global_mean':np.full(len(te),float(tr.y.mean())),
          'speed_integer_mean':pred_group(tr,te,'speed'),
          'time_slot_mean':pred_group(tr,te,'slot'),
          'speed_quadratic':fit_lstsq(tr,te,'speed_quadratic'),
          'speed_plus_time':fit_lstsq(tr,te,'speed_plus_time'),
        }
        for m,p in preds.items():
            p=np.clip(p,0,None)
            rec.append({'held_site_id':sid,'held_site_name':str(te.site_name.iloc[0]),'method':m,'n':int(len(te)),'mae_volume15':mae(te.y,p),'mean_actual_volume15':float(te.y.mean())})
            for thr in THRESHOLDS:
                cong=te.speed<thr
                for label,mask in [('congested',cong),('free_flow',~cong)]:
                    regimes.append({'held_site_id':sid,'method':m,'speed_threshold_mph':thr,'regime':label,'n':int(mask.sum()),'mae_volume15':None if not mask.any() else mae(te.loc[mask,'y'],p[mask.to_numpy()])})
    return pd.DataFrame(rec),pd.DataFrame(regimes)

def geometry_inventory(root:Path):
    p=root/'sites.json'
    if not p.exists(): return {'present':False}
    obj=json.loads(p.read_text(encoding='utf-8')); rows=obj.get('Rows') or obj.get('rows') or []
    keys=sorted({str(k) for r in rows[:100] for k in r})
    geom=[k for k in keys if any(x in k.lower() for x in ('lat','lon','east','north','coord'))]
    return {'present':True,'inventory_rows':len(rows),'geometry_like_fields':geom}

def jclean(x):
    if isinstance(x,float) and (math.isnan(x) or math.isinf(x)): return None
    if isinstance(x,dict): return {k:jclean(v) for k,v in x.items()}
    if isinstance(x,list): return [jclean(v) for v in x]
    return x

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    root=a.input_dir; out=a.output; out.mkdir(parents=True,exist_ok=True)
    manifest,manifest_path,checks=load_manifest(root)
    files=sorted(root.glob('site_*_*.json'))
    dfs=[load_site_file(p) for p in files]; dfs=[d for d in dfs if not d.empty]
    if len(dfs)<20: raise SystemExit(f'need >=20 site files, found {len(dfs)}')
    all_df=pd.concat(dfs,ignore_index=True)
    site_days=all_df.groupby('site_id').date.nunique()
    if int(site_days.min())<30: raise SystemExit(f'need >=30 days/site, min={int(site_days.min())}')
    t3=t3_sites(all_df); t4,reg=t4_loso(all_df)
    t3.to_csv(out/'t3_site_metrics.csv',index=False);t4.to_csv(out/'t4_loso_site_metrics.csv',index=False);reg.to_csv(out/'t4_loso_regime_metrics.csv',index=False)
    (out/'INPUT_SHA256.txt').write_text('\n'.join(f"{x['observed']}  {x['file']}" for x in checks)+'\n',encoding='utf-8')
    t4means=t4.groupby('method').mae_volume15.mean().sort_values()
    best=str(t4means.index[0])
    t3agg={'median_q95_vph':float(t3.q095_vph.median()),'median_q99_vph':float(t3.q099_vph.median()),'median_max_vph':float(t3.max_vph.median())}
    for thr in THRESHOLDS:
        c=f'lt{int(thr)}_recovery_flow_median_vph'; vals=pd.to_numeric(t3[c],errors='coerce').dropna();t3agg[f'median_site_recovery_flow_vph_at_{int(thr)}mph_threshold']=None if vals.empty else float(vals.median())
    summary={
      'evidence_class':'public_data registry / official public source',
      'source':'National Highways WebTRIS API v1.0 frozen 20-site x 30-day acquisition',
      'source_url':manifest.get('source_url'),
      'input_manifest_sha256':sha256(manifest_path),
      'input_manifest_state':manifest.get('state'),
      'input_files_hash_checked':len(checks),'input_hash_mismatches':0,
      'rows_total':int(len(all_df)),'site_count':int(all_df.site_id.nunique()),'min_days_per_site':int(site_days.min()),'max_days_per_site':int(site_days.max()),
      'geometry_inventory':geometry_inventory(root),
      'T3':{
        'scope':'20-detector observational capacity/recovery-flow proxy rehearsal; no organizer capacity/discharge truth',
        'interval_minutes':15,'occupancy_available':False,'multiple_detectors':True,'thirty_day_window':True,
        'aggregate':t3agg,
        'capacity_proxy_definition':'empirical upper quantiles of observed 15-min volume, multiplied by 4 to veh/h',
        'recovery_proxy_definition':'first interval at/above threshold after a consecutive speed-below-threshold episode; threshold sensitivity at 55/60/65 mph',
        'formal_verdict':'PUBLIC_SOURCE_SCALE_CLOSED; ORGANIZER_TRUTH_AND_OCCUPANCY_CONTRACT_STILL_OPEN'
      },
      'T4':{
        'scope':'20-fold leave-one-detector-out speed-to-15min-volume rehearsal',
        'held_out_unit':'detector site','folds':int(all_df.site_id.nunique()),
        'mean_loso_mae_volume15':{str(k):float(v) for k,v in t4means.items()},'best_method_by_mean_mae':best,'best_mean_mae_volume15':float(t4means.iloc[0]),
        'mean_actual_volume15':float(all_df.y.mean()),
        'regime_thresholds_mph':list(THRESHOLDS),
        'formal_verdict':'PUBLIC_SOURCE_SCALE_AND_BLIND_SITE_REHEARSAL_CLOSED; ORGANIZER_HELD_OUT_TRUTH_CONTRACT_STILL_OPEN'
      }
    }
    (out/'summary.json').write_text(json.dumps(jclean(summary),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text('# T3/T4 WebTRIS 20-site x 30-day public-registry rerun — 2026-09-02\n\nEvidence class: **public_data registry / official public source**.\n\nInputs are the hash-verified frozen Drive acquisition. The older one-site x seven-day ASU/website-lineage result remains separate. T3 values are observational proxies and threshold-sensitivity diagnostics. T4 is 20-fold leave-one-site-out; held-out-site counts are never used for model fitting. These results are rehearsal evidence, not organizer scores.\n',encoding='utf-8')
    print(json.dumps(jclean(summary),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
