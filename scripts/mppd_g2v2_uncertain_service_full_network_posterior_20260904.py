import argparse,csv,gzip,json,math,resource,time
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r0_cached_full_network_support_rescan_20260904 as support
import scripts.mppd_r0_full_network_factor_engine_20260904 as r0
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay,load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch,load_patch

# Discrete approximation to standard-normal integration. Weights sum to 1.
QUAD=[(-2.0,0.0544887),(-1.0,0.2442013),(0.0,0.40262),(1.0,0.2442013),(2.0,0.0544887)]


def logsumexp(vals):
    vals=[x for x in vals if math.isfinite(x)]
    if not vals:return -math.inf
    m=max(vals);return m+math.log(sum(math.exp(x-m) for x in vals))

def lognorm_cdf(x,median,sigma):
    if x<=0:return 0.0
    mu=math.log(max(1e-6,median));z=(math.log(x)-mu)/(max(1e-6,sigma)*math.sqrt(2.0))
    return 0.5*(1+math.erf(z))

def lognorm_pdf(x,median,sigma):
    if x<=0:return 0.0
    mu=math.log(max(1e-6,median));s=max(1e-6,sigma);z=(math.log(x)-mu)/s
    return math.exp(-0.5*z*z)/(x*s*math.sqrt(2*math.pi))

def expected_cdf_normal_difference(mean,sd,median,sigma):
    if sd<=1e-9:return lognorm_cdf(mean,median,sigma)
    return sum(w*lognorm_cdf(mean+z*sd,median,sigma) for z,w in QUAD)

def expected_pdf_normal_difference(mean,sd,median,sigma):
    if sd<=1e-9:return lognorm_pdf(mean,median,sigma)
    return sum(w*lognorm_pdf(mean+z*sd,median,sigma) for z,w in QUAD)

def interval_logprob_uncertain(prev_dep,prev_sd,cur_dep,cur_sd,ready,ready_sd,median,sigma):
    um=(cur_dep-ready).total_seconds();usd=math.sqrt(cur_sd*cur_sd+ready_sd*ready_sd)
    Fu=expected_cdf_normal_difference(um,usd,median,sigma)
    if prev_dep is None:Fl=0.0
    else:
        lm=(prev_dep-ready).total_seconds();lsd=math.sqrt(prev_sd*prev_sd+ready_sd*ready_sd)
        Fl=expected_cdf_normal_difference(lm,lsd,median,sigma)
    return math.log(max(1e-14,Fu-Fl))

def egress_logdensity_uncertain(exit_time,arr,arr_sd,median,sigma):
    mean=(exit_time-arr).total_seconds();v=expected_pdf_normal_difference(mean,arr_sd,median,sigma)
    return math.log(max(v,1e-14))

def evidence_rank(c):
    return {'PARTIAL_DIRECT_SERVICE_ANCHOR':0,'AFC_INFERRED_SERVICE_FIELD':1,'SERVICE_TRAJECTORY_COMPLETION_HYPOTHESIS':2,'AFC_INFERRED_SERVICE_FIELD_WEAK_LATTICE_INITIALIZATION':3}.get(c,4)

def load_uncertain_service(path):
    payload=json.loads(Path(path).read_text(encoding='utf-8'));roots=defaultdict(list)
    for st in payload.get('states',[]):
        root=st.get('root_service_id') or st['service_id'];ev={};state_sd=float(st.get('timing_uncertainty_sec') or 0.0)
        for e in st.get('station_events',[]):
            arr=e.get('arrival') or e.get('time');dep=e.get('departure') or e.get('time')
            if not arr or not dep:continue
            esd=float(e.get('timing_uncertainty_sec') if e.get('timing_uncertainty_sec') is not None else state_sd)
            ev[e['node']]={'arrival':datetime.fromisoformat(arr),'departure':datetime.fromisoformat(dep),'sd':max(0.0,esd),'event_evidence_class':e.get('event_evidence_class') or st.get('evidence_class')}
        if len(ev)>=2:roots[(st['line'],root)].append({'id':st['service_id'],'root':root,'line':st['line'],'direction':st.get('direction'),'evidence_class':st.get('evidence_class'),'events':ev})
    return roots,payload.get('manifest',{}),payload

def build_uncertain_rides(roots):
    cache={}
    def rides(line,o,d):
        key=(line,o,d)
        if key in cache:return cache[key]
        out=[]
        for (ln,root),variants in roots.items():
            if ln!=line:continue
            choices=[]
            for st in variants:
                if o not in st['events'] or d not in st['events']:continue
                eo,ed=st['events'][o],st['events'][d]
                if eo['departure']>=ed['arrival']:continue
                choices.append((evidence_rank(st['evidence_class']),eo['sd']+ed['sd'],st['id'],eo,ed,st['evidence_class']))
            if not choices:continue
            # One segment representation per root service prevents direct/completion
            # variants from inflating probability mass. Prefer strongest evidence,
            # then smallest event uncertainty for this station pair.
            _,_,sid,eo,ed,evc=min(choices,key=lambda x:(x[0],x[1],x[2]))
            out.append({'dep':eo['departure'],'arr':ed['arrival'],'dep_sd':eo['sd'],'arr_sd':ed['sd'],'root':root,'variant_id':sid,'evidence_class':evc})
        out.sort(key=lambda x:(x['dep'],x['root']))
        cache[key]=out;return out
    return rides,cache

def previous_ride(rides,i):
    return rides[i-1] if i>0 else None

def route_beam(cand,meta,rides_fn,tin,tout,beam,priors):
    legs=r0.path_legs(cand['path'],meta);states=[(tin,0.0,0.0,[])]
    for li,(line,o,d) in enumerate(legs):
        if o==d:continue
        rides=rides_fn(line,o,d)
        if not rides:return [],'MISSING_SERVICE_SEGMENT'
        nxt=[]
        for ready,ready_sd,base_lp,chain in states:
            for i,rd in enumerate(rides):
                # Cheap probability support gate only; exact likelihood remains soft.
                if rd['arr']>tout and (rd['arr']-tout).total_seconds()>3*max(1.0,rd['arr_sd']):continue
                prev=previous_ride(rides,i)
                if li==0:med,sig=priors['access_median'],priors['access_sigma']
                else:med,sig=priors['transfer_median'],priors['transfer_sigma']
                lp=interval_logprob_uncertain(prev['dep'] if prev else None,prev['dep_sd'] if prev else 0.0,rd['dep'],rd['dep_sd'],ready,ready_sd,med,sig)
                if not math.isfinite(lp):continue
                nxt.append((rd['arr'],rd['arr_sd'],base_lp+lp,chain+[rd]))
        if not nxt:return [],'TIME_INCOMPATIBLE_CHAIN'
        nxt.sort(key=lambda x:(x[2],-x[0].timestamp()),reverse=True);states=nxt[:beam]
    final=[]
    for arr,arr_sd,lp,chain in states:
        le=egress_logdensity_uncertain(tout,arr,arr_sd,priors['egress_median'],priors['egress_sigma'])
        if math.isfinite(le):final.append((lp+le,chain))
    final.sort(key=lambda x:x[0],reverse=True)
    return final[:beam],None if final else 'EGRESS_INCOMPATIBLE'

def load_cohorts(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):yield r['origin_code'],r['destination_code'],datetime.fromisoformat(r['entry_time']),datetime.fromisoformat(r['exit_time']),int(r['passenger_mass'])

def entropy(ps):return -sum(p*math.log(max(p,1e-15)) for p in ps if p>0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--p1c',required=True);ap.add_argument('--cohorts',required=True);ap.add_argument('--routes',required=True);ap.add_argument('--service-init',required=True);ap.add_argument('--out',required=True);ap.add_argument('--beam',type=int,default=12);ap.add_argument('--topology-patch');ap.add_argument('--gtxa-overlay');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);wall0=time.perf_counter();phase={}
    priors={'access_median':180.0,'access_sigma':0.90,'transfer_median':180.0,'transfer_sigma':0.85,'egress_median':120.0,'egress_sigma':0.80,'evidence_class':'PROVISIONAL_BROAD_INITIAL_PRIOR_FOR_G2V2'}
    t=time.perf_counter();G,meta,code_to_nodes,transfer_groups,ambiguous_seq,ambiguous_codes,graph_build=g0.build_network(args.p1c)
    base_graph={'nodes':G.number_of_nodes(),'edges':G.number_of_edges()}
    topology_result=None;overlay_result=None
    if args.topology_patch:
        topology_result=apply_topology_patch(G,meta,load_patch(args.topology_patch))
    after_patch_graph={'nodes':G.number_of_nodes(),'edges':G.number_of_edges()}
    if args.gtxa_overlay:
        overlay_result=apply_gtxa_overlay(G,meta,code_to_nodes,load_overlay(args.gtxa_overlay))
    final_graph={'nodes':G.number_of_nodes(),'edges':G.number_of_edges()}
    phase['network_build_sec']=time.perf_counter()-t
    t=time.perf_counter();routes=support.load_routes(args.routes);phase['route_cache_load_sec']=time.perf_counter()-t
    t=time.perf_counter();roots,svc_manifest,svc_payload=load_uncertain_service(args.service_init);rides_fn,_=build_uncertain_rides(roots);phase['service_cache_load_sec']=time.perf_counter()-t
    route_mass=Counter();tc_mass=Counter();best_tc=Counter();fail=Counter();status=Counter();processed=finite=cohorts_n=0;ent_sum=0.0
    with gzip.open(out/'g2v2_uncertain_route_posterior_cohorts.jsonl.gz','wt',encoding='utf-8') as fo:
        for oc,dc,tin,tout,mass in load_cohorts(args.cohorts):
            cohorts_n+=1;processed+=mass;cands=routes.get((oc,dc),[])
            if not cands:status['NO_STRUCTURAL_ROUTE']+=mass;continue
            mincost=min(float(c.get('base_cost',0)) for c in cands);scores=[];chains=[];fails=[]
            for c in cands:
                ch,reason=route_beam(c,meta,rides_fn,tin,tout,args.beam,priors);chains.append(ch);fails.append(reason)
                if not ch:scores.append(-math.inf);continue
                lp=logsumexp([x[0] for x in ch])-0.30*max(0.0,float(c.get('base_cost',0))-mincost);scores.append(lp)
            fs=[(i,x) for i,x in enumerate(scores) if math.isfinite(x)]
            if not fs:
                reason=Counter(x or 'UNKNOWN' for x in fails).most_common(1)[0][0];fail[reason]+=mass;status['NO_FINITE_POSTERIOR']+=mass;continue
            z=logsumexp([x for _,x in fs]);post=[(i,math.exp(x-z)) for i,x in fs];finite+=mass;status['FINITE_POSTERIOR']+=mass;ps=[p for _,p in post];ent_sum+=mass*entropy(ps)
            bi,bp=max(post,key=lambda x:x[1]);best_tc[int(cands[bi].get('transfer_count',0))]+=mass;outs=[]
            for i,p in post:
                c=cands[i];tc=int(c.get('transfer_count',0));route_mass[c.get('line_sequence','')]+=mass*p;tc_mass[tc]+=mass*p
                outs.append({'line_sequence':c.get('line_sequence'),'transfer_count':tc,'route_probability':p,'base_cost':c.get('base_cost'),'candidate_set_status':c.get('candidate_set_status')})
            fo.write(json.dumps({'origin_code':oc,'destination_code':dc,'entry_time':tin.isoformat(),'exit_time':tout.isoformat(),'cohort_mass':mass,'route_entropy_nats':entropy(ps),'routes':outs},ensure_ascii=False)+'\n')
    phase['posterior_scan_sec']=time.perf_counter()-t
    representation={
      'base_graph':base_graph,
      'topology_patch_applied':bool(args.topology_patch),'topology_patch':topology_result,'after_topology_patch_graph':after_patch_graph,
      'gtxa_overlay_applied':bool(args.gtxa_overlay),'gtxa_overlay':overlay_result,'final_graph':final_graph,
      'service_init_schema':svc_payload.get('schema'),'service_init_status':svc_payload.get('status')
    }
    result={'schema':'mppd.g2v2-uncertain-service-full-network-posterior.v2','date':'2026-09-04','status':'G2V2_UNCERTAINTY_AWARE_FULL_NETWORK_POSTERIOR_COMPLETED','authority':'00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md',
      'scope_assertions':{'full_network':True,'line_filter':False,'segment_filter':False,'transfer_count_cap':False,'service_timing_uncertainty_integrated':True,'root_service_variant_deduplicated_per_segment':True},
      'network_representation':representation,
      'afc':{'cohort_count':cohorts_n,'passenger_mass':processed,'finite_posterior_mass':finite,'finite_posterior_share':finite/processed if processed else 0},
      'service':{'root_service_count':len(roots),'manifest':svc_manifest},'initial_station_time_priors':priors,
      'posterior':{'mass_by_transfer_count':dict(sorted(tc_mass.items())),'best_route_mass_by_transfer_count':dict(sorted(best_tc.items())),'weighted_mean_route_entropy_nats':ent_sum/finite if finite else None,'status_mass':dict(status),'failure_mass':dict(fail),'top_route_families':[{'line_sequence':k,'posterior_mass':v} for k,v in route_mass.most_common(100)]},
      'performance':{'phase_sec':phase,'total_wall_sec':time.perf_counter()-wall0,'max_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
      'scientific_boundary':['This is an uncertainty-aware G2 initialization, not final joint inversion.','Service-event timing uncertainty is integrated approximately using discrete normal quadrature in access/transfer/egress likelihoods.','Completion variants sharing a root service are deduplicated per segment to avoid route likelihood inflation; G3 still must maintain competing trajectory hypotheses explicitly.','A/K/E are broad initialization priors and not measured station-time truth.','Route posterior quality depends on the supplied full-network candidate set; truncated candidate sets must remain flagged.','Topology/crosswalk overlays are provenance-typed representation hypotheses and never become observed passenger routes or ATS events.'],
      'next_gate':'Qualify this exact denominator/network/service representation, then update shared Theta_A/Theta_K/Theta_E and service-event support in G3.','no_email_notification_logic':True}
    (out/'g2v2_uncertain_service_full_network_posterior_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
