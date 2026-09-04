import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.special import logsumexp

import scripts.mppd_n3_shared_kernel_joint_likelihood_20260904 as m


def fit_transfer_targeted(rows, transfer_pairs, get_paths, A_par, E_par):
    eligible=[r for r in rows if r['ol']!=r['dl'] and transfer_pairs.get((r['ol'],r['dl']))]
    line2=m.stable_sample([r for r in eligible if r['ol']==m.L2 or r['dl']==m.L2],16000,'norm-k-l2')
    other=m.stable_sample([r for r in eligible if r['ol']!=m.L2 and r['dl']!=m.L2],4000,'norm-k-other')
    cross=line2+other
    K_global={'mu':math.log(180.0),'sigma':0.65,'n':0,'fitted':False}; K_by={}; history=[]
    for it in range(3):
        by=defaultdict(list); allints=[]; assigned=0; line2_assigned=0
        for r in cross:
            chains=m.candidate_transfer_chains(r,transfer_pairs,get_paths,A_par,E_par,K_global,K_by)
            if not chains:
                continue
            best=max(chains,key=lambda x:x[0]); assigned+=1
            if r['ol']==m.L2 or r['dl']==m.L2: line2_assigned+=1
            l,u=best[2]; by[best[1]].append((l,u,1.0)); allints.append((l,u,1.0))
        K_global=m.fit_lognorm_intervals(allints,(K_global['mu'],K_global['sigma']))
        K_by={k:m.fit_lognorm_intervals(ints,(K_global['mu'],K_global['sigma'])) for k,ints in by.items() if len(ints)>=20}
        history.append({'iter':it+1,'sample_total':len(cross),'sample_line2_adjacent':len(line2),'assigned':assigned,'line2_assigned':line2_assigned,'global':K_global.copy(),'specific_count':len(K_by),'top_specific':sorted([(k,v['n'],math.exp(v['mu']),v['sigma']) for k,v in K_by.items()],key=lambda x:x[1],reverse=True)[:30]})
    return K_global,K_by,history


def direct_logsum(row,get_paths,A_par,E_par):
    paths=get_paths(row['ol'],row['os'],row['ds']); vals=[]
    for i,(dep,arr,tr) in enumerate(paths):
        if dep<=row['t0'] or arr>=row['t1']: continue
        upper=m.sec(dep-row['t0'])
        if upper<=0 or upper>2400: continue
        pd=m.prev_dep(paths,i,row['t0']); lower=max(0.0,m.sec(pd-row['t0'])) if pd else 0.0
        eg=m.sec(row['t1']-arr)
        lp=m.log_interval_prob(A_par,lower,upper)+m.log_point_density(E_par,eg)
        if np.isfinite(lp): vals.append(lp)
    return float(logsumexp(sorted(vals,reverse=True)[:30])) if vals else -np.inf


def transfer_logsum(row,transfer_pairs,get_paths,A_par,E_par,K_global,K_by):
    chains=m.candidate_transfer_chains(row,transfer_pairs,get_paths,A_par,E_par,K_global,K_by)
    if not chains: return -np.inf
    by_route=defaultdict(list)
    for c in chains:
        dv=c[5][2]
        by_route[dv].append(c[0])
    route_vals=[float(logsumexp(sorted(v,reverse=True)[:30])) for v in by_route.values() if v]
    if not route_vals: return -np.inf
    return float(logsumexp(route_vals)-math.log(len(route_vals)))


def candidate_provider(base_get,offsets,ref):
    cache={}
    def get(line,o,d):
        key=(line,o,d)
        if key in cache: return cache[key]
        paths=list(base_get(line,o,d))
        if line==m.L2:
            tp=m.target_path(offsets,ref,o,d)
            if tp:
                paths.append((tp[0],tp[1],'__HIDDEN_CANDIDATE__'))
                paths.sort(key=lambda x:x[0])
        cache[key]=paths
        return paths
    return get


def score_schedule(rowsets,get_paths,transfer_pairs,A,E,Kg,Kb):
    group_scores={}
    for g,rows in rowsets.items():
        total=0.0; used=0
        for r in rows:
            if g=='direct': lp=direct_logsum(r,get_paths,A,E)
            else: lp=transfer_logsum(r,transfer_pairs,get_paths,A,E,Kg,Kb)
            if np.isfinite(lp): total+=lp; used+=1
        group_scores[g]=(total,used)
    tiers={}
    combos={'DIRECT':['direct'],'DIRECT_PLUS_INBOUND':['direct','inbound'],'DIRECT_PLUS_BOTH':['direct','inbound','outbound']}
    for tier,groups in combos.items():
        tiers[tier]={'loglik':float(sum(group_scores[g][0] for g in groups)),'used':int(sum(group_scores[g][1] for g in groups))}
    return tiers


def headway_prior(tt,direction,excluded_train):
    refs=sorted(x[0] for x in tt if x[2]==direction and x[1]!=excluded_train)
    gaps=[m.sec(refs[i]-refs[i-1]) for i in range(1,len(refs)) if 60<=m.sec(refs[i]-refs[i-1])<=1800]
    return m.fit_lognorm_points(gaps,(math.log(360.0),0.45))


def run():
    import argparse, json
    from pathlib import Path
    ap=argparse.ArgumentParser(); ap.add_argument('--taims',required=True); ap.add_argument('--p1c',required=True); ap.add_argument('--service',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    transfer_pairs,train_events,by_line,rows=m.load_inputs(args.taims,args.p1c,args.service)
    tt=m.target_trains(train_events); selected=m.select_targets(tt)
    out={'schema':'mppd.n3-normalized-schedule-likelihood.v1','date':'2026-09-04','status':'N3_NORMALIZED_SCHEDULE_LIKELIHOOD_COMPLETED','task_class':'KNOWN_NEIGHBOR_GAP_CONDITIONAL_DIAGNOSTIC_NOT_FREE_SEARCH','scientific_boundary':['Each candidate hidden service is inserted into the complete visible Line-2 schedule before passenger likelihood is evaluated.','Access and transfer first-feasible intervals are recomputed under each candidate schedule, so candidate service gates redistribute rather than only add probability mass.','Route alternatives are averaged at the physical-transfer level to reduce route-count inflation.','Waiting is service-induced and is not fitted as an independent kernel.','Headway prior uses visible same-direction service gaps only and is reported separately from passenger-only likelihood.','Direct and one-transfer factors only; double-transfer and pulse factors remain future expansions.','BMS is partial service-anchor evidence, not exhaustive ATS truth.','No card identifiers are retained.'],'targets':[]}
    for truth_ref,tr,direction in selected:
        excluded={(m.L2,tr)}; base_get=m.make_path_cache(train_events,by_line,excluded)
        A,E,ae_hist=m.fit_access_egress(rows,base_get); Kg,Kb,k_hist=fit_transfer_targeted(rows,transfer_pairs,base_get,A,E); offsets=m.trajectory_offsets(train_events,direction,excluded); H=headway_prior(tt,direction,tr)
        direct=[r for r in rows if r['ol']==m.L2 and r['dl']==m.L2 and m.target_path(offsets,truth_ref,r['os'],r['ds'])]
        inbound=[r for r in rows if r['ol']!=m.L2 and r['dl']==m.L2 and transfer_pairs.get((r['ol'],m.L2))]
        outbound=[r for r in rows if r['ol']==m.L2 and r['dl']!=m.L2 and transfer_pairs.get((m.L2,r['dl']))]
        rowsets={'direct':m.stable_sample(direct,3000,f'norm-{tr}-d'),'inbound':m.stable_sample(inbound,3000,f'norm-{tr}-i'),'outbound':m.stable_sample(outbound,3000,f'norm-{tr}-o')}
        dr=[x for x in tt if x[2]==direction and x[1]!=tr]; before=[x for x in dr if x[0]<truth_ref]; after=[x for x in dr if x[0]>truth_ref]
        if not before or not after: continue
        prev=max(before,key=lambda x:x[0]); nxt=min(after,key=lambda x:x[0]); start=prev[0]+timedelta(seconds=30); stop=nxt[0]-timedelta(seconds=30)
        if start>=stop: continue
        grid=[]; cur=start
        while cur<=stop: grid.append(cur); cur+=timedelta(seconds=30)
        vals=defaultdict(list)
        for cand in grid:
            gp=candidate_provider(base_get,offsets,cand); sc=score_schedule(rowsets,gp,transfer_pairs,A,E,Kg,Kb)
            hlog=m.log_point_density(H,m.sec(cand-prev[0]))+m.log_point_density(H,m.sec(nxt[0]-cand))
            for tier,v in sc.items(): vals[tier].append({'candidate':cand.isoformat(),'passenger_loglik':v['loglik'],'used':v['used'],'headway_prior_log':hlog,'posterior_score':v['loglik']+hlog})
        summaries={}
        for tier,arr in vals.items():
            bp=max(arr,key=lambda x:x['passenger_loglik']); bpost=max(arr,key=lambda x:x['posterior_score'])
            ep=abs(m.sec(datetime.fromisoformat(bp['candidate'])-truth_ref)); epost=abs(m.sec(datetime.fromisoformat(bpost['candidate'])-truth_ref))
            summaries[tier]={'passenger_only':{'best_candidate':bp['candidate'],'abs_error_sec':ep,'within_60':ep<=60,'within_120':ep<=120,'best_loglik':bp['passenger_loglik'],'used':bp['used']},'plus_visible_headway_prior':{'best_candidate':bpost['candidate'],'abs_error_sec':epost,'within_60':epost<=60,'within_120':epost<=120,'best_score':bpost['posterior_score'],'passenger_loglik':bpost['passenger_loglik'],'headway_prior_log':bpost['headway_prior_log'],'used':bpost['used']}}
        midpoint=prev[0]+(nxt[0]-prev[0])/2
        out['targets'].append({'train':tr,'direction':direction,'truth_ref':truth_ref.isoformat(),'neighbor_gap':{'prev':prev[0].isoformat(),'next':nxt[0].isoformat(),'gap_selection_uses_target_truth':True,'midpoint_abs_error_sec':abs(m.sec(midpoint-truth_ref))},'kernels':{'A_median_sec':math.exp(A['mu']),'E_median_sec':math.exp(E['mu']),'K_global_median_sec':math.exp(Kg['mu']),'K_specific_count':len(Kb),'headway_median_sec':math.exp(H['mu']),'headway_sigma':H['sigma'],'top_specific':sorted([{'key':k,'n':v['n'],'median_sec':math.exp(v['mu']),'sigma':v['sigma']} for k,v in Kb.items()],key=lambda x:x['n'],reverse=True)[:20],'transfer_history':k_hist},'cohort_sizes':{k:len(v) for k,v in rowsets.items()},'tiers':summaries})
    agg={}
    for tier in ('DIRECT','DIRECT_PLUS_INBOUND','DIRECT_PLUS_BOTH'):
        for mode in ('passenger_only','plus_visible_headway_prior'):
            errs=[t['tiers'][tier][mode]['abs_error_sec'] for t in out['targets'] if tier in t['tiers']]
            agg[f'{tier}::{mode}']={'n':len(errs),'median_abs_error_sec':float(np.median(errs)) if errs else None,'within_60_share':float(np.mean([e<=60 for e in errs])) if errs else None,'within_120_share':float(np.mean([e<=120 for e in errs])) if errs else None}
    out['aggregate']=agg; out['next_gate']='Compare normalized passenger-only and prior-assisted local-to-network gains. If network tiers improve multiple targets, proceed to double-transfer/pulse expansion; if not, inspect route-mixture and service-trajectory factors before free-search.'
    (outdir/'n3_normalized_schedule_likelihood_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    run()
