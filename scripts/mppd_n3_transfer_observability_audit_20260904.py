import argparse, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np

import scripts.mppd_n3_shared_kernel_joint_likelihood_20260904 as m
import scripts.mppd_n3_normalized_schedule_likelihood_20260904 as n
import scripts.mppd_n3_segment_conditioned_normalized_likelihood_20260904 as s


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--taims',required=True); ap.add_argument('--p1c',required=True); ap.add_argument('--service',required=True); ap.add_argument('--out',required=True)
    args=ap.parse_args(); outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    transfer_pairs,train_events,by_line,rows=m.load_inputs(args.taims,args.p1c,args.service)
    tt=m.target_trains(train_events)
    target=[x for x in tt if x[1]=='3060']
    if not target: target=m.select_targets(tt)[:1]
    truth_ref,tr,direction=target[0]
    excluded={(m.L2,tr)}; get_paths=m.make_path_cache(train_events,by_line,excluded)
    A,E,ae_hist=m.fit_access_egress(rows,get_paths)
    offsets=m.trajectory_offsets(train_events,direction,excluded); before,after=s.crossing_sets(offsets)
    broadK={'mu':math.log(180.0),'sigma':1.0,'n':0,'fitted':False}

    train_station_keys=defaultdict(set)
    for (line,tno),sts in train_events.items():
        if (line,tno) in excluded: continue
        for st in sts: train_station_keys[(line,st)].add(tno)

    rec=defaultdict(lambda:{'side':None,'line_pair':None,'physical_station':None,'l2_station':None,'other_station':None,'structural_rows':0,'feasible_rows':0,'intervals':[],'chain_count':0,'other_line_anchor_trains':0,'l2_anchor_trains':0})

    for r in rows:
        if r['ol']!=m.L2 and r['dl']==m.L2 and r['ds'] in after:
            groups=[(ost,l2st,dv) for ost,l2st,dv in transfer_pairs.get((r['ol'],m.L2),[]) if l2st in before]
            side='INBOUND'
        elif r['ol']==m.L2 and r['dl']!=m.L2 and r['os'] in before:
            groups=[(l2st,dst,dv) for l2st,dst,dv in transfer_pairs.get((m.L2,r['dl']),[]) if l2st in after]
            side='OUTBOUND'
        else:
            continue
        if not groups: continue
        for a,b,dv in groups:
            if side=='INBOUND':
                key=f"{r['ol']}->{m.L2}@{dv}"; other_line=r['ol']; other_st=a; l2st=b
            else:
                key=f"{m.L2}->{r['dl']}@{dv}"; other_line=r['dl']; l2st=a; other_st=b
            q=rec[key]; q['side']=side; q['line_pair']=key.split('@')[0]; q['physical_station']=dv; q['l2_station']=l2st; q['other_station']=other_st; q['structural_rows']+=1
            q['other_line_anchor_trains']=len(train_station_keys[(other_line,other_st)]); q['l2_anchor_trains']=len(train_station_keys[(m.L2,l2st)])

        chains=m.candidate_transfer_chains(r,transfer_pairs,get_paths,A,E,broadK,{})
        bykey=defaultdict(list)
        for c in chains:
            bykey[c[1]].append(c)
        for key,vals in bykey.items():
            if key not in rec: continue
            q=rec[key]; q['feasible_rows']+=1; q['chain_count']+=len(vals)
            best=max(vals,key=lambda x:x[0]); q['intervals'].append((best[2][0],best[2][1],1.0))

    rows_out=[]
    for key,q in rec.items():
        ints=q.pop('intervals'); nint=len(ints)
        kfit=m.fit_lognorm_intervals(ints,(math.log(180.0),0.8)) if nint>=20 else {'mu':math.log(180.0),'sigma':0.8,'n':nint,'fitted':False}
        row={'key':key,**q,'feasible_share':q['feasible_rows']/q['structural_rows'] if q['structural_rows'] else 0.0,'interval_count':nint,'mean_chains_per_feasible_row':q['chain_count']/q['feasible_rows'] if q['feasible_rows'] else 0.0,'K_fitted':bool(kfit['fitted']),'K_median_sec':math.exp(kfit['mu']) if nint else None,'K_log_sigma':kfit['sigma'] if nint else None}
        rows_out.append(row)
    rows_out.sort(key=lambda x:(x['feasible_rows'],x['feasible_share'],x['interval_count']),reverse=True)

    def summarize(side):
        g=[x for x in rows_out if x['side']==side]
        return {'groups':len(g),'structural_rows_sum':sum(x['structural_rows'] for x in g),'feasible_rows_sum':sum(x['feasible_rows'] for x in g),'groups_with_20_intervals':sum(x['interval_count']>=20 for x in g),'groups_with_50_feasible_rows':sum(x['feasible_rows']>=50 for x in g),'median_feasible_share':float(np.median([x['feasible_share'] for x in g])) if g else None}

    result={'schema':'mppd.n3-transfer-observability-audit.v1','date':'2026-09-04','status':'N3_TRANSFER_OBSERVABILITY_AUDIT_COMPLETED','target':{'train':tr,'direction':direction,'truth_ref':truth_ref.isoformat(),'segment':['당산','합정']},'access_fit':{'median_sec':math.exp(A['mu']),'sigma':A['sigma'],'n':A['n']},'egress_fit':{'median_sec':math.exp(E['mu']),'sigma':E['sigma'],'n':E['n']},'summary':{'INBOUND':summarize('INBOUND'),'OUTBOUND':summarize('OUTBOUND')},'groups':rows_out,'top_groups':rows_out[:30],'boundary':['Structural rows are candidate segment-crossing transfer observations, not observed route assignments.','Feasible rows require at least one complete visible-service chain under the current partial BMS background service field with target train 3060 excluded.','K intervals use first-feasible service-gate censoring and are model-derived, not measured walking times.','This audit diagnoses transfer-factor observability and background-service coverage; it is not hidden-service recovery.','No card identifiers are retained.'],'next_gate':'Use group-level feasible share, interval support, K stability, and service-anchor coverage to define a pre-registered high-confidence transfer-factor subset for the next segment-conditioned joint likelihood. If most structurally eligible groups have low feasible coverage, complete the non-target background service field with explicitly labeled AFC_INFERRED_SERVICE_FIELD before network certification.'}
    (outdir/'n3_transfer_observability_audit_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
