import argparse, json, math
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import scripts.mppd_n3_normalized_schedule_likelihood_20260904 as n

m=n.m


def crossing_sets(offsets):
    before={st for st,(arr_off,dep_off,cnt) in offsets.items() if dep_off <= 0}
    after={st for st,(arr_off,dep_off,cnt) in offsets.items() if arr_off >= 0}
    return before,after


def inbound_crosses(row, transfer_pairs, before):
    if row['ol']==m.L2 or row['dl']!=m.L2:
        return False
    return any(l2st in before for ost,l2st,dv in transfer_pairs.get((row['ol'],m.L2),[]))


def outbound_crosses(row, transfer_pairs, after):
    if row['ol']!=m.L2 or row['dl']==m.L2:
        return False
    return any(l2st in after for l2st,dst,dv in transfer_pairs.get((m.L2,row['dl']),[]))


def run():
    ap=argparse.ArgumentParser(); ap.add_argument('--taims',required=True); ap.add_argument('--p1c',required=True); ap.add_argument('--service',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    transfer_pairs,train_events,by_line,rows=m.load_inputs(args.taims,args.p1c,args.service)
    tt=m.target_trains(train_events); selected=m.select_targets(tt)
    out={'schema':'mppd.n3-segment-conditioned-normalized-likelihood.v1','date':'2026-09-04','status':'N3_SEGMENT_CONDITIONED_NORMALIZED_LIKELIHOOD_COMPLETED','segment':{'line':m.L2,'station_a':m.A_ST,'station_b':m.B_ST,'names':['당산','합정']},'task_class':'KNOWN_NEIGHBOR_GAP_CONDITIONAL_DIAGNOSTIC_NOT_FREE_SEARCH','scientific_boundary':['Only OD observations whose structural service/transfer chain crosses the hidden Dangsan-Hapjeong segment are admitted to the target likelihood.','Before/after station sets are derived from visible same-direction train trajectories, not from the held target full route.','Each candidate hidden service is inserted into the complete visible Line-2 schedule and first-feasible access/transfer intervals are recomputed.','Waiting is induced by discrete service gates and is not an independent free kernel.','Route alternatives are averaged at physical-transfer level; shortest/unique observed passenger route is not claimed.','Headway prior is fitted from visible same-direction gaps and reported separately.','Direct and one-transfer factors only; double-transfer and pulse factors remain future expansion.','BMS remains partial service-anchor evidence, not exhaustive ATS.','No card identifiers are retained.'],'targets':[]}
    for truth_ref,tr,direction in selected:
        excluded={(m.L2,tr)}; base_get=m.make_path_cache(train_events,by_line,excluded)
        A,E,ae_hist=m.fit_access_egress(rows,base_get); Kg,Kb,k_hist=n.fit_transfer_targeted(rows,transfer_pairs,base_get,A,E); offsets=m.trajectory_offsets(train_events,direction,excluded); before,after=crossing_sets(offsets); H=n.headway_prior(tt,direction,tr)
        direct=[r for r in rows if r['ol']==m.L2 and r['dl']==m.L2 and r['os'] in before and r['ds'] in after and r['os']!=r['ds']]
        inbound=[r for r in rows if inbound_crosses(r,transfer_pairs,before) and r['ds'] in after]
        outbound=[r for r in rows if outbound_crosses(r,transfer_pairs,after) and r['os'] in before]
        rowsets={'direct':m.stable_sample(direct,2500,f'seg-{tr}-d'),'inbound':m.stable_sample(inbound,2500,f'seg-{tr}-i'),'outbound':m.stable_sample(outbound,2500,f'seg-{tr}-o')}
        dr=[x for x in tt if x[2]==direction and x[1]!=tr]; b=[x for x in dr if x[0]<truth_ref]; a=[x for x in dr if x[0]>truth_ref]
        if not b or not a: continue
        prev=max(b,key=lambda x:x[0]); nxt=min(a,key=lambda x:x[0]); start=prev[0]+timedelta(seconds=30); stop=nxt[0]-timedelta(seconds=30)
        if start>=stop: continue
        grid=[]; cur=start
        while cur<=stop: grid.append(cur); cur+=timedelta(seconds=30)
        vals=defaultdict(list)
        for cand in grid:
            gp=n.candidate_provider(base_get,offsets,cand); sc=n.score_schedule(rowsets,gp,transfer_pairs,A,E,Kg,Kb)
            hlog=m.log_point_density(H,m.sec(cand-prev[0]))+m.log_point_density(H,m.sec(nxt[0]-cand))
            for tier,v in sc.items(): vals[tier].append({'candidate':cand.isoformat(),'passenger_loglik':v['loglik'],'used':v['used'],'headway_prior_log':hlog,'posterior_score':v['loglik']+hlog})
        summaries={}
        for tier,arr in vals.items():
            bp=max(arr,key=lambda x:x['passenger_loglik']); bpost=max(arr,key=lambda x:x['posterior_score'])
            ep=abs(m.sec(datetime.fromisoformat(bp['candidate'])-truth_ref)); epost=abs(m.sec(datetime.fromisoformat(bpost['candidate'])-truth_ref))
            summaries[tier]={'passenger_only':{'best_candidate':bp['candidate'],'abs_error_sec':ep,'within_60':ep<=60,'within_120':ep<=120,'best_loglik':bp['passenger_loglik'],'used':bp['used']},'plus_visible_headway_prior':{'best_candidate':bpost['candidate'],'abs_error_sec':epost,'within_60':epost<=60,'within_120':epost<=120,'best_score':bpost['posterior_score'],'passenger_loglik':bpost['passenger_loglik'],'headway_prior_log':bpost['headway_prior_log'],'used':bpost['used']}}
        midpoint=prev[0]+(nxt[0]-prev[0])/2
        out['targets'].append({'train':tr,'direction':direction,'truth_ref':truth_ref.isoformat(),'neighbor_gap':{'prev':prev[0].isoformat(),'next':nxt[0].isoformat(),'gap_selection_uses_target_truth':True,'midpoint_abs_error_sec':abs(m.sec(midpoint-truth_ref))},'crossing_station_sets':{'before_n':len(before),'after_n':len(after)},'candidate_cohort_counts':{'direct':len(direct),'inbound':len(inbound),'outbound':len(outbound)},'sampled_cohort_sizes':{k:len(v) for k,v in rowsets.items()},'kernels':{'A_median_sec':math.exp(A['mu']),'E_median_sec':math.exp(E['mu']),'K_global_median_sec':math.exp(Kg['mu']),'K_specific_count':len(Kb),'headway_median_sec':math.exp(H['mu']),'headway_sigma':H['sigma'],'top_specific':sorted([{'key':k,'n':v['n'],'median_sec':math.exp(v['mu']),'sigma':v['sigma']} for k,v in Kb.items()],key=lambda x:x['n'],reverse=True)[:20]},'tiers':summaries})
    agg={}
    for tier in ('DIRECT','DIRECT_PLUS_INBOUND','DIRECT_PLUS_BOTH'):
        for mode in ('passenger_only','plus_visible_headway_prior'):
            errs=[t['tiers'][tier][mode]['abs_error_sec'] for t in out['targets'] if tier in t['tiers']]
            agg[f'{tier}::{mode}']={'n':len(errs),'median_abs_error_sec':float(np.median(errs)) if errs else None,'within_60_share':float(np.mean([e<=60 for e in errs])) if errs else None,'within_120_share':float(np.mean([e<=120 for e in errs])) if errs else None}
    out['aggregate']=agg; out['next_gate']='If segment-conditioned passenger-only network tiers outperform direct-only across multiple targets, qualify preliminary H-NET evidence under the known-gap task and add double-transfer/pulse factors. Otherwise inspect route mixture, station-specific A/E, and partial-background-service misspecification before free-search.'
    (outdir/'n3_segment_conditioned_normalized_likelihood_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': run()
