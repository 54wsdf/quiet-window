import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

EXPECTED_COHORTS=562071
EXPECTED_MASS=632143


def phase_merge(root,phase):
    files=sorted(Path(root).glob('**/shard_summary.json'))
    rows=[]
    for p in files:
        x=json.loads(p.read_text(encoding='utf-8'))
        if x.get('phase')==phase: rows.append(x)
    if not rows: raise RuntimeError(f'no {phase} shard summaries')
    shard_count=int(rows[0]['shard_count']); ids=sorted(int(x['shard_id']) for x in rows)
    if ids!=list(range(shard_count)): raise RuntimeError(f'{phase} incomplete shard ids: {ids}')
    cohort_sum=sum(int(x['scope']['shard_cohort_count']) for x in rows); mass_sum=sum(float(x['scope']['shard_passenger_mass']) for x in rows)
    scan={(int(x['scan_authority']['full_cohort_count_seen']),int(x['scan_authority']['full_passenger_mass_seen'])) for x in rows}
    if scan!={(EXPECTED_COHORTS,EXPECTED_MASS)}: raise RuntimeError(f'{phase} scan authority mismatch {scan}')
    if cohort_sum!=EXPECTED_COHORTS or abs(mass_sum-EXPECTED_MASS)>1e-6: raise RuntimeError(f'{phase} partition conservation failed {cohort_sum} {mass_sum}')
    finite=sum(float(x['posterior']['finite_posterior_mass']) for x in rows); processed=sum(float(x['posterior']['processed_passenger_mass']) for x in rows)
    failure=Counter(); status=Counter(); skip=Counter(); missing=Counter(); non_simple=0.0; access_tail=0.0; egress_tail=0.0; h_num=0.0; hc_num=0.0
    finite_coh=0; nonfinite_coh=0
    for x in rows:
        p=x['posterior']; fmass=float(p['finite_posterior_mass'])
        for k,v in p.get('failure_mass',{}).items(): failure[k]+=float(v)
        for k,v in p.get('status_mass',{}).items(): status[k]+=float(v)
        for k,v in p.get('skip_mass_by_count',{}).items(): skip[int(k)]+=float(v)
        for r in p.get('top_missing_service_pressure',[]): missing[r['segment']]+=float(r['pressure_mass'])
        non_simple+=float(p.get('non_simple_route_posterior_mass',0.0)); access_tail+=float(p.get('access_tail_gt_15min_posterior_mass',0.0)); egress_tail+=float(p.get('egress_tail_gt_15min_posterior_mass',0.0))
        if p.get('weighted_mean_route_entropy_nats') is not None: h_num+=fmass*float(p['weighted_mean_route_entropy_nats'])
        if p.get('weighted_mean_first_boarding_route_entropy_contraction_nats') is not None: hc_num+=fmass*float(p['weighted_mean_first_boarding_route_entropy_contraction_nats'])
        finite_coh+=int(p.get('finite_cohort_count',0)); nonfinite_coh+=int(p.get('nonfinite_cohort_count',0))
    nonfinite=processed-finite; fsum=sum(failure.values()); leg=sum(skip.values())
    return {'phase':phase,'shard_count':shard_count,'cohort_count':cohort_sum,'passenger_mass':mass_sum,'finite_posterior_mass':finite,'finite_posterior_share':finite/processed,'no_finite_posterior_mass':nonfinite,'status_mass':dict(status),'failure_mass':dict(failure),'failure_mass_sum':fsum,'failure_mass_conservation_error':fsum-nonfinite,'failure_mass_conservation_pass':abs(fsum-nonfinite)<=1e-5,'weighted_mean_route_entropy_nats':h_num/finite if finite else None,'weighted_mean_first_boarding_route_entropy_contraction_nats':hc_num/finite if finite else None,'skip_mass_by_count':dict(sorted(skip.items())),'skip_positive_leg_share':sum(v for k,v in skip.items() if k>0)/leg if leg else None,'non_simple_route_posterior_mass':non_simple,'non_simple_route_share_of_finite_mass':non_simple/finite if finite else None,'access_tail_gt_15min_posterior_mass':access_tail,'egress_tail_gt_15min_posterior_mass':egress_tail,'top_missing_service_pressure':[{'segment':k,'pressure_mass':v} for k,v in missing.most_common(100)],'finite_cohort_count':finite_coh,'nonfinite_cohort_count':nonfinite_coh}


def compare_shards(e0_root,e1_root):
    e0_files={p.parent.name:p for p in Path(e0_root).glob('**/e0_cohort_posterior.jsonl.gz')}; e1_files={p.parent.name:p for p in Path(e1_root).glob('**/e1_cohort_posterior.jsonl.gz')}
    if set(e0_files)!=set(e1_files): raise RuntimeError('E0/E1 cohort shard folder mismatch')
    common_mass=finite_both=became=lost=route_changed=first_changed=tv_num=0.0; row_count=0
    for shard in sorted(e0_files):
        e0={}
        with gzip.open(e0_files[shard],'rt',encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r=json.loads(line); e0[r['cohort_id']]=r
        with gzip.open(e1_files[shard],'rt',encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                b=json.loads(line); a=e0.pop(b['cohort_id'],None)
                if a is None: raise RuntimeError(f'E1 cohort absent E0 {b["cohort_id"]}')
                row_count+=1; mass=float(a.get('mass',b.get('mass',0.0))); common_mass+=mass; af=a.get('status')=='FINITE_POSTERIOR'; bf=b.get('status')=='FINITE_POSTERIOR'
                if not af and bf: became+=mass
                if af and not bf: lost+=mass
                if not(af and bf): continue
                finite_both+=mass
                if a.get('top_route')!=b.get('top_route'): route_changed+=mass
                if a.get('top_first_root')!=b.get('top_first_root'): first_changed+=mass
                keys=set(a.get('route_probs',{})).union(b.get('route_probs',{})); tv=0.5*sum(abs(float(a.get('route_probs',{}).get(k,0.0))-float(b.get('route_probs',{}).get(k,0.0))) for k in keys); tv_num+=mass*tv
        if e0: raise RuntimeError(f'E0 rows absent E1 in {shard}: {len(e0)}')
    if row_count!=EXPECTED_COHORTS or abs(common_mass-EXPECTED_MASS)>1e-6: raise RuntimeError(f'comparison conservation failed {row_count} {common_mass}')
    return {'common_passenger_mass':common_mass,'finite_in_both_mass':finite_both,'became_finite_mass':became,'lost_finite_mass':lost,'top_route_changed_mass':route_changed,'top_route_changed_share_among_finite_both':route_changed/finite_both if finite_both else None,'top_first_service_changed_mass':first_changed,'top_first_service_changed_share_among_finite_both':first_changed/finite_both if finite_both else None,'mean_route_total_variation_among_finite_both':tv_num/finite_both if finite_both else None}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--e0-root',required=True); ap.add_argument('--e1-root',required=True); ap.add_argument('--kernels',required=True); ap.add_argument('--offsets',required=True); ap.add_argument('--out',required=True); args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    e0=phase_merge(args.e0_root,'E0'); e1=phase_merge(args.e1_root,'E1'); movement=compare_shards(args.e0_root,args.e1_root); kernels=json.loads(Path(args.kernels).read_text(encoding='utf-8')); offsets=json.loads(Path(args.offsets).read_text(encoding='utf-8'))
    result={'schema':'mppd.r1c-full-denominator-distributed-joint-posterior.v1','date':'2026-09-05','status':'R1C_FULL_DENOMINATOR_DISTRIBUTED_JOINT_POSTERIOR_COMPLETED','scientific_unit':'SEOUL_2026-08-29_0700_1000_FULL_NETWORK_CORRECTED_MAPPED_DENOMINATOR','scope':{'cohort_count':EXPECTED_COHORTS,'passenger_mass':EXPECTED_MASS,'full_network_candidate_domain':True,'line_filter':False,'segment_filter':False,'transfer_count_cap':False,'behavioral_regularities_hard_coded':False,'service_support_mutation':False},'E0':e0,'E1':e1,'posterior_redistribution':movement,'kernel_schema':kernels.get('kernels',kernels).get('schema'),'kernel_counts':{'access_context_count':len(kernels.get('kernels',kernels).get('access_by_context',{})),'access_context_hour_count':len(kernels.get('kernels',kernels).get('access_by_context_hour',{})),'egress_context_count':len(kernels.get('kernels',kernels).get('egress_by_context',{})),'egress_context_hour_count':len(kernels.get('kernels',kernels).get('egress_by_context_hour',{})),'transfer_movement_count':len(kernels.get('kernels',kernels).get('transfer_by_movement',{}))},'service_timing':{'schema':offsets.get('schema'),'root_count_with_offset_decision':offsets.get('root_count_with_offset_decision'),'moved_root_count':offsets.get('moved_root_count'),'nonzero_offset_count':len(offsets.get('nonzero_offsets',{}))},'residual_semantics':{'TIME_INCOMPATIBLE_CHAIN':'SERVICE_EXIT_HORIZON_INCOMPATIBLE_UNDER_CURRENT_ENGINE_UNLESS_FUTURE_ENGINE_SEMANTICS_CHANGE','MISSING_SERVICE_SEGMENT':'NO_RIDE_SUPPORT_FOR_REQUIRED_SEGMENT_UNDER_CURRENT_SERVICE_SUBSTRATE'},'qualification':{'denominator_conservation':'PASS','cohort_partition_conservation':'PASS','failure_mass_conservation_E0':e0['failure_mass_conservation_pass'],'failure_mass_conservation_E1':e1['failure_mass_conservation_pass'],'genuine_bidirectional_redistribution':movement['top_route_changed_mass']>0 or movement['top_first_service_changed_mass']>0,'service_support_evidence_frozen':True},'behavioral_diagnostics_first_read':{'skip_positive_leg_share':e1['skip_positive_leg_share'],'non_simple_route_share_of_finite_mass':e1['non_simple_route_share_of_finite_mass'],'first_boarding_route_entropy_contraction_nats':e1['weighted_mean_first_boarding_route_entropy_contraction_nats'],'access_tail_gt_15min_posterior_mass':e1['access_tail_gt_15min_posterior_mass'],'egress_tail_gt_15min_posterior_mass':e1['egress_tail_gt_15min_posterior_mass'],'interpretation':'DIAGNOSTIC_ONLY_NOT_BEHAVIORAL_PRIOR'},'scientific_boundary':['This is the full corrected mapped Seoul 2026-08-29 07:00-10:00 denominator, not a full-day reconstruction.','All 632143 passenger mass and 562071 cohorts must be present exactly once across deterministic hash shards.','Service support remains G1F and is not mutated; G1G2 and service-exit-horizon negative gates remain binding.','Behavioral diagnostics remain posterior-derived observations and are not hard priors.','K=1 remains the first hierarchical pass; transfer-regime mixture selection requires held-out qualification.'],'next_gate':'R1D_SERVICE_AND_PASSENGER_BIDIRECTIONAL_CLOSURE_AND_FULL_DENOMINATOR_BEHAVIORAL_DIAGNOSTICS','no_email_notification_logic':True}
    (out/'r1c_full_denominator_distributed_joint_posterior_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'E0':e0,'E1':e1,'posterior_redistribution':movement,'kernel_counts':result['kernel_counts'],'service_timing':result['service_timing']},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
