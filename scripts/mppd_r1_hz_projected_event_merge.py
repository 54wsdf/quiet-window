from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import scripts.mppd_r1_hz_joint_full_day as base
import scripts.mppd_r1_hz_event_level_joint as event

SCHEMA="mppd.r1-hz-projected-event-merge.v1"


def merge(paths:list[Path],old_params:Path,roots:Path,out_params:Path,out_summary:Path):
    stats=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    old=json.loads(old_params.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='mppd-proj-merge-') as td:
        tp=Path(td)/'params.json';ts=Path(td)/'summary.json'
        event.merge_stats(paths,old_params,roots,tp,ts,update_events=False)
        new=json.loads(tp.read_text(encoding='utf-8'));bs=json.loads(ts.read_text(encoding='utf-8'))
    oldev={str(k):float(v) for k,v in old.get('event_offsets_s',{}).items()}
    agg=defaultdict(lambda:{'w':0.0,'sum':0.0})
    for s in stats:
        for k,r in s.get('event_projection_stats',{}).items():
            w=float(r['w']);agg[k]['w']+=w;agg[k]['sum']+=float(r['sum_target_offset'])
    direct={}
    for k,r in agg.items():
        w=r['w']
        if w<=0:continue
        target=r['sum']/w;prev=oldev.get(k,0.0);alpha=w/(w+250.0)
        v=(1-alpha)*prev+alpha*target
        direct[k]=(max(-event.MAX_EVENT_OFFSET_S,min(event.MAX_EVENT_OFFSET_S,v)),w)
    raw=event.load_raw_roots(roots);offsets={};repairs=0;roots_anchor=0
    for r in raw['roots']:
        rid=str(r['root_id'])
        if any(event.event_key(rid,e['station']) in direct for e in r['events']):roots_anchor+=1
        vals,rep=event._interpolate_offsets(r,direct,oldev);repairs+=rep
        for e,v in zip(r['events'],vals.tolist()):
            if abs(float(v))>1e-9:offsets[event.event_key(rid,e['station'])]=float(v)
    new['event_offsets_s']=offsets;new['root_offsets_s']={}
    new['semantics']={**new.get('semantics',{}),'event_times_jointly_updated_from_access_transfer_egress_passenger_constraints':True,'event_projection_targets_are_posterior_mass_weighted':True,'single_root_level_shift_is_not_the_service_model':True}
    out_params.write_text(json.dumps(new,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    vals=np.asarray(list(offsets.values()),float);av=np.abs(vals) if len(vals) else np.asarray([],float)
    summary={**bs,'schema':SCHEMA,'passenger_event_projection_anchor_count':len(direct),'service_roots_with_projection_anchor':roots_anchor,'service_event_offsets_nonzero':len(vals),'service_event_abs_offset_median_s':float(np.median(av)) if len(av) else 0.0,'service_event_abs_offset_p90_s':float(np.quantile(av,.9)) if len(av) else 0.0,'service_event_abs_offset_max_s':float(np.max(av)) if len(av) else 0.0,'service_event_offset_bound_hits':int(np.sum(np.isclose(av,event.MAX_EVENT_OFFSET_S,atol=1e-8))) if len(av) else 0,'service_monotonicity_repairs':repairs}
    out_summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2));return new,summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--stats',type=Path,action='append',required=True);p.add_argument('--old-params',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--out-params',type=Path,required=True);p.add_argument('--out-summary',type=Path,required=True);a=p.parse_args();merge(a.stats,a.old_params,a.roots,a.out_params,a.out_summary)

if __name__=='__main__':main()
