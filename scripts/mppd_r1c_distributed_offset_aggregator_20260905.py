import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import scripts.mppd_g2v2_uncertain_service_full_network_posterior_20260904 as g2
import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--score-root',required=True); ap.add_argument('--e0-root',required=True); ap.add_argument('--service-init',required=True); ap.add_argument('--min-service-usage',type=float,default=25.0); ap.add_argument('--out',required=True); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    score_files=sorted(Path(args.score_root).glob('**/offset_score_sums.json')); summary_files=sorted(Path(args.e0_root).glob('**/shard_summary.json'))
    if not score_files or not summary_files: raise RuntimeError('missing score or E0 summary files')
    score_sum=defaultdict(lambda:defaultdict(lambda:{'ll_sum':0.0,'used_weight':0.0})); grids=None
    for p in score_files:
        x=json.loads(p.read_text(encoding='utf-8')); grid=[int(v) for v in x['grid_sec']]
        if grids is None: grids=grid
        elif grids!=grid: raise RuntimeError('offset grid mismatch')
        for root,ds in x['scores'].items():
            for d,v in ds.items():
                score_sum[root][int(d)]['ll_sum']+=float(v['ll_sum']); score_sum[root][int(d)]['used_weight']+=float(v['used_weight'])
    root_usage=Counter(); shard_count=None; full_seen=set()
    for p in summary_files:
        x=json.loads(p.read_text(encoding='utf-8'))
        if x.get('phase')!='E0': continue
        if shard_count is None: shard_count=int(x['shard_count'])
        full_seen.add((int(x['scan_authority']['full_cohort_count_seen']),int(x['scan_authority']['full_passenger_mass_seen'])))
        for root,w in x.get('root_usage',{}).items(): root_usage[root]+=float(w)
    if len(full_seen)!=1: raise RuntimeError(f'inconsistent full scan authority {full_seen}')
    roots,service_manifest,service_payload=g2.load_uncertain_service(args.service_init); rmeta=base.root_metadata(roots)
    offsets={}; diagnostics={}
    for root,usage in root_usage.items():
        md=rmeta.get(root)
        if not md or usage<args.min_service_usage: continue
        if md['evidence_class']=='PARTIAL_DIRECT_SERVICE_ANCHOR':
            offsets[root]=0; diagnostics[root]={'usage_mass':usage,'evidence_class':md['evidence_class'],'selected_offset_sec':0,'frozen':True}; continue
        ds=score_sum.get(root,{})
        scored=[]
        prior_sd=max(30.0,md['median_event_sd'],90.0 if 'WEAK' in md['evidence_class'] else 60.0)
        for d in grids or []:
            v=ds.get(d); 
            if not v or v['used_weight']<=0: continue
            mean_ll=v['ll_sum']/v['used_weight']; prior=-0.5*(float(d)/prior_sd)**2; scored.append((mean_ll+prior,d,mean_ll,prior,v['used_weight']))
        if not scored: continue
        scored.sort(reverse=True); best=scored[0]; zero=next((x for x in scored if x[1]==0),None); gain=best[0]-zero[0] if zero else None; selected=int(best[1])
        if gain is not None and gain<1e-4: selected=0
        offsets[root]=selected; diagnostics[root]={'usage_mass':usage,'factor_weight':best[4],'evidence_class':md['evidence_class'],'prior_sd_sec':prior_sd,'selected_offset_sec':selected,'score_gain_vs_zero':gain,'best_mean_loglik':best[2],'best_prior_log':best[3],'frozen':False}
    nonzero={k:v for k,v in offsets.items() if int(v)!=0}
    payload={'schema':'mppd.r1c-full-denominator-context-aware-service-offsets.v1','date':'2026-09-05','status':'R1C_FULL_DENOMINATOR_CONTEXT_AWARE_SERVICE_OFFSET_MSTEP_COMPLETED','service_schema':service_payload.get('schema'),'shard_count':shard_count,'full_scan_authority':{'cohort_count':next(iter(full_seen))[0],'passenger_mass':next(iter(full_seen))[1]},'grid_sec':grids,'min_service_usage':args.min_service_usage,'root_count_with_offset_decision':len(offsets),'moved_root_count':len(nonzero),'offsets':offsets,'nonzero_offsets':nonzero,'diagnostics':diagnostics,'scientific_boundary':['PARTIAL_DIRECT_SERVICE_ANCHOR roots are frozen at zero offset.','Offsets are trajectory-level timing shifts only and never add service support.','Shard score contributions are merged before the prior and selection rule are applied, reproducing the global mean-loglikelihood objective.'],'no_email_notification_logic':True}
    (out/'global_offsets.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'root_count_with_offset_decision':len(offsets),'moved_root_count':len(nonzero),'full_scan_authority':payload['full_scan_authority']},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
