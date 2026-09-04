import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r0_cached_full_network_support_rescan_20260904 as support
import scripts.mppd_g2v2_uncertain_service_full_network_posterior_20260904 as g2
import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base
import scripts.mppd_r1c_hierarchical_temporal_kernels_first_pass_20260904 as r1c
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch

ROUND_DIGITS = 3


def rf(x):
    return round(float(x), ROUND_DIGITS)


def shard_index(origin, destination, tin, tout, shard_count):
    key = f'{origin}|{destination}|{tin.isoformat()}|{tout.isoformat()}'.encode()
    return int(hashlib.sha1(key).hexdigest()[:16], 16) % shard_count


def load_shard_cohorts(path, shard_count, shard_id):
    rows = []
    scanned_count = 0
    scanned_mass = 0
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            tin = datetime.fromisoformat(r['entry_time'])
            tout = datetime.fromisoformat(r['exit_time'])
            oc = str(r['origin_code']); dc = str(r['destination_code']); mass = int(r['passenger_mass'])
            scanned_count += 1; scanned_mass += mass
            if shard_index(oc, dc, tin, tout, shard_count) == shard_id:
                rows.append((oc, dc, tin, tout, mass))
    return rows, scanned_count, scanned_mass


def default_kernels():
    return {
        'schema': 'mppd.r1b-kernels.v1-mixture-capable-k1-first-pass',
        'access': base.kernel_from_median_sigma(180.0, 0.90, 'G2V2_BROAD_INITIAL_PRIOR'),
        'transfer_global': base.kernel_from_median_sigma(180.0, 0.85, 'G2V2_BROAD_INITIAL_PRIOR'),
        'transfer_by_movement': {},
        'egress': base.kernel_from_median_sigma(120.0, 0.80, 'G2V2_BROAD_INITIAL_PRIOR'),
    }


def parse_kernel_payload(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return payload.get('kernels', payload)


def parse_offsets(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return payload.get('offsets', payload.get('nonzero_offsets', payload))


def route_probs_summary(hypotheses):
    route_prob = defaultdict(float)
    first_root_prob = defaultdict(float)
    route_first = defaultdict(lambda: defaultdict(float))
    for h in hypotheses:
        p = h['probability']
        route_prob[h['line_sequence']] += p
        fr = h['chain'][0]['root_key'] if h['chain'] else 'NONE'
        first_root_prob[fr] += p
        route_first[fr][h['line_sequence']] += p
    probs = list(route_prob.values())
    h_before = -sum(p * math.log(max(p, base.EPS)) for p in probs if p > 0)
    h_after = 0.0
    for fr, frp in first_root_prob.items():
        if frp <= 0:
            continue
        conditional = [v / frp for v in route_first[fr].values()]
        h_after += frp * (-sum(p * math.log(max(p, base.EPS)) for p in conditional if p > 0))
    return route_prob, first_root_prob, h_before, max(0.0, h_before - h_after)


def init_stats():
    return {
        'access_global': Counter(), 'transfer_global': Counter(), 'egress_global': Counter(),
        'access_by_context': defaultdict(Counter), 'access_by_context_hour': defaultdict(Counter),
        'egress_by_context': defaultdict(Counter), 'egress_by_context_hour': defaultdict(Counter),
        'transfer_by_movement': defaultdict(Counter),
        'offset': Counter(),
    }


def interval_pair(f):
    lower = 0.0 if f.get('lower') is None else (f['lower'] - f['ready']).total_seconds()
    upper = (f['upper'] - f['ready']).total_seconds()
    if upper <= max(0.0, lower) + 1e-6:
        return None
    return rf(max(0.0, lower)), rf(upper)


def add_mstep_stats(stats, f, w):
    if w <= 0:
        return
    if f.get('type') == 'INTERVAL':
        pair = interval_pair(f)
        if pair is not None:
            kind = f.get('kind')
            if kind == 'ACCESS':
                stats['access_global'][pair] += w
                ctx = f.get('station_context')
                hour = f.get('time_bin_hour')
                if ctx:
                    stats['access_by_context'][str(ctx)][pair] += w
                    if hour is not None:
                        stats['access_by_context_hour'][f'{ctx}|h{int(hour):02d}'][pair] += w
            elif kind == 'TRANSFER':
                stats['transfer_global'][pair] += w
                movement = f.get('movement') or 'UNKNOWN'
                stats['transfer_by_movement'][movement][pair] += w

        touched = {x for x in (f.get('ready_root'), f.get('lower_root'), f.get('upper_root')) if x}
        lower_rel = None if f.get('lower') is None else rf((f['lower'] - f['ready']).total_seconds())
        upper_rel = rf((f['upper'] - f['ready']).total_seconds())
        for root in touched:
            lower_coeff = 0 if f.get('lower') is None else int(f.get('lower_root') == root) - int(f.get('ready_root') == root)
            upper_coeff = int(f.get('upper_root') == root) - int(f.get('ready_root') == root)
            sig = (
                root, 'I', str(f.get('kind') or ''), str(f.get('movement') or ''), str(f.get('station_context') or ''),
                int(f.get('time_bin_hour')) if f.get('time_bin_hour') is not None else -1,
                lower_rel, upper_rel, rf(f.get('lower_sd', 0.0)), rf(f.get('upper_sd', 0.0)), rf(f.get('ready_sd', 0.0)),
                lower_coeff, upper_coeff,
            )
            stats['offset'][sig] += w

    elif f.get('type') == 'EGRESS':
        x = (f['exit_time'] - f['arr']).total_seconds()
        if not f.get('station_only_proxy') and f.get('fit_eligible') is not False and 0 < x <= 3600:
            xv = rf(x)
            stats['egress_global'][(xv,)] += w
            ctx = f.get('station_context')
            hour = f.get('time_bin_hour')
            if ctx:
                stats['egress_by_context'][str(ctx)][(xv,)] += w
                if hour is not None:
                    stats['egress_by_context_hour'][f'{ctx}|h{int(hour):02d}'][(xv,)] += w
        root = f.get('arr_root')
        if root:
            sig = (
                root, 'E', 'EGRESS', '', str(f.get('station_context') or ''),
                int(f.get('time_bin_hour')) if f.get('time_bin_hour') is not None else -1,
                rf(x), rf(f.get('arr_sd', 0.0)), -1,
            )
            stats['offset'][sig] += w


def write_kernel_stats(path, stats):
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        def emit(scope, group, kind, hist):
            for key, weight in hist.items():
                if kind == 'INTERVAL':
                    row = {'scope':scope,'group':group,'kind':kind,'lower_sec':key[0],'upper_sec':key[1],'weight':weight}
                else:
                    row = {'scope':scope,'group':group,'kind':kind,'x_sec':key[0],'weight':weight}
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        emit('ACCESS_GLOBAL', None, 'INTERVAL', stats['access_global'])
        emit('TRANSFER_GLOBAL', None, 'INTERVAL', stats['transfer_global'])
        emit('EGRESS_GLOBAL', None, 'POINT', stats['egress_global'])
        for g,h in stats['access_by_context'].items(): emit('ACCESS_CONTEXT',g,'INTERVAL',h)
        for g,h in stats['access_by_context_hour'].items(): emit('ACCESS_CONTEXT_HOUR',g,'INTERVAL',h)
        for g,h in stats['egress_by_context'].items(): emit('EGRESS_CONTEXT',g,'POINT',h)
        for g,h in stats['egress_by_context_hour'].items(): emit('EGRESS_CONTEXT_HOUR',g,'POINT',h)
        for g,h in stats['transfer_by_movement'].items(): emit('TRANSFER_MOVEMENT',g,'INTERVAL',h)


def write_offset_stats(path, stats):
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for sig, weight in stats['offset'].items():
            if sig[1] == 'I':
                root, _, kind, movement, context, hour, lower_rel, upper_rel, lower_sd, upper_sd, ready_sd, lower_coeff, upper_coeff = sig
                row = {'root':root,'type':'INTERVAL','kind':kind,'movement':movement or None,'station_context':context or None,'time_bin_hour':None if hour < 0 else hour,'lower_rel_sec':lower_rel,'upper_rel_sec':upper_rel,'lower_sd':lower_sd,'upper_sd':upper_sd,'ready_sd':ready_sd,'lower_shift_coeff':lower_coeff,'upper_shift_coeff':upper_coeff,'weight':weight}
            else:
                root, _, kind, movement, context, hour, x_sec, arr_sd, x_coeff = sig
                row = {'root':root,'type':'EGRESS','kind':'EGRESS','movement':None,'station_context':context or None,'time_bin_hour':None if hour < 0 else hour,'x_sec':x_sec,'arr_sd':arr_sd,'x_shift_coeff':x_coeff,'weight':weight}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def posterior_stream(cohorts, routes, meta, roots, kernels, beam, max_skip, phase, cohort_path):
    rides_fn, _ = base.build_joint_rides(roots)
    collect = phase == 'E0'
    stats = init_stats() if collect else None
    result = {
        'processed_mass': 0.0, 'finite_mass': 0.0, 'status_mass': Counter(), 'failure_mass': Counter(),
        'route_entropy_weighted': 0.0, 'skip_mass': Counter(), 'non_simple_mass': 0.0,
        'first_boarding_entropy_contraction_weighted': 0.0, 'access_tail_mass_gt_15min': 0.0, 'egress_tail_mass_gt_15min': 0.0,
        'missing_service_pressure': Counter(), 'root_usage': Counter(), 'hypothesis_factor_visits': 0,
        'finite_cohort_count': 0, 'nonfinite_cohort_count': 0,
    }
    with gzip.open(cohort_path, 'wt', encoding='utf-8') as cf:
        for oc, dc, tin, tout, mass in cohorts:
            result['processed_mass'] += mass
            cands = routes.get((oc, dc), [])
            cohort_id = f'{oc}|{dc}|{tin.isoformat()}|{tout.isoformat()}'
            if not cands:
                result['status_mass']['NO_STRUCTURAL_ROUTE'] += mass; result['failure_mass']['NO_STRUCTURAL_ROUTE'] += mass; result['nonfinite_cohort_count'] += 1
                cf.write(json.dumps({'cohort_id':cohort_id,'mass':mass,'status':'NO_STRUCTURAL_ROUTE','failure_reason_probs':{'NO_STRUCTURAL_ROUTE':1.0}},ensure_ascii=False)+'\n')
                continue
            min_cost = min(float(c.get('base_cost',0.0)) for c in cands)
            hypotheses=[]; fail_details=[]; route_scores=[]
            for ci,cand in enumerate(cands):
                chains,failure=r1c.route_beam_joint_r1c(cand,meta,rides_fn,tin,tout,beam,kernels,max_skip)
                fail_details.append(failure)
                if not chains:
                    route_scores.append(-math.inf); continue
                route_lp=base.logsumexp([x['logp'] for x in chains])-0.30*max(0.0,float(cand.get('base_cost',0.0))-min_cost)
                route_scores.append(route_lp)
                for ch in chains:
                    hypotheses.append({'candidate_index':ci,'line_sequence':cand.get('line_sequence',''),'transfer_count':int(cand.get('transfer_count',0)),'non_simple':base.path_is_non_simple(cand),'logp':ch['logp']-0.30*max(0.0,float(cand.get('base_cost',0.0))-min_cost),'chain':ch['chain'],'factors':ch['factors']})
            finite_routes=[(i,lp) for i,lp in enumerate(route_scores) if math.isfinite(lp)]
            if not hypotheses or not finite_routes:
                sw=base.structural_pressure_weights(cands); reason_weights=Counter()
                for ci,failure in enumerate(fail_details):
                    reason=(failure or {}).get('reason','UNKNOWN'); w=float(sw[ci]) if ci<len(sw) else 0.0; reason_weights[reason]+=w
                    if failure and reason=='MISSING_SERVICE_SEGMENT': result['missing_service_pressure'][f"{failure.get('line')}|{failure.get('origin')}|{failure.get('destination')}"] += mass*w
                z=sum(reason_weights.values()) or 1.0; reason_probs={k:float(v)/z for k,v in reason_weights.items()} if reason_weights else {'UNKNOWN':1.0}
                for reason,p in reason_probs.items(): result['failure_mass'][reason]+=mass*p
                result['status_mass']['NO_FINITE_POSTERIOR']+=mass; result['nonfinite_cohort_count']+=1
                dominant=max(reason_probs.items(),key=lambda x:x[1])[0]
                cf.write(json.dumps({'cohort_id':cohort_id,'mass':mass,'status':dominant,'failure_reason_probs':reason_probs},ensure_ascii=False)+'\n'); continue
            z_h=base.logsumexp([h['logp'] for h in hypotheses])
            for h in hypotheses: h['probability']=math.exp(h['logp']-z_h)
            route_prob,first_root_prob,h_before,contraction=route_probs_summary(hypotheses)
            for h in hypotheses:
                p=h['probability']; wp=mass*p; result['non_simple_mass']+=wp*(1.0 if h['non_simple'] else 0.0)
                for leg in h['chain']:
                    result['skip_mass'][int(leg['n_skip'])]+=wp; result['root_usage'][leg['root_key']]+=wp
                af=next((f for f in h['factors'] if f.get('type')=='INTERVAL' and f.get('kind')=='ACCESS'),None)
                if af and (af['upper']-af['ready']).total_seconds()>900: result['access_tail_mass_gt_15min']+=wp
                ef=next((f for f in reversed(h['factors']) if f.get('type')=='EGRESS'),None)
                if ef and not ef.get('station_only_proxy') and (ef['exit_time']-ef['arr']).total_seconds()>900: result['egress_tail_mass_gt_15min']+=wp
                if collect:
                    for fac in h['factors']:
                        add_mstep_stats(stats,fac,wp); result['hypothesis_factor_visits']+=1
            result['finite_mass']+=mass; result['status_mass']['FINITE_POSTERIOR']+=mass; result['route_entropy_weighted']+=mass*h_before; result['first_boarding_entropy_contraction_weighted']+=mass*contraction; result['finite_cohort_count']+=1
            top_route=max(route_prob.items(),key=lambda x:x[1]); top_first=max(first_root_prob.items(),key=lambda x:x[1])
            cf.write(json.dumps({'cohort_id':cohort_id,'mass':mass,'status':'FINITE_POSTERIOR','route_probs':dict(route_prob),'route_entropy':h_before,'top_route':top_route[0],'top_route_prob':top_route[1],'first_root_probs':dict(first_root_prob),'top_first_root':top_first[0],'top_first_prob':top_first[1]},ensure_ascii=False)+'\n')
    return result,stats


def summarize(result):
    processed=float(result['processed_mass']); finite=float(result['finite_mass']); leg_mass=sum(float(v) for v in result['skip_mass'].values()); failure_sum=sum(float(v) for v in result['failure_mass'].values()); nonfinite=processed-finite
    return {'processed_passenger_mass':processed,'finite_posterior_mass':finite,'finite_posterior_share':finite/processed if processed else 0.0,'status_mass':dict(result['status_mass']),'failure_mass':dict(result['failure_mass']),'failure_mass_sum':failure_sum,'failure_mass_conservation_error':failure_sum-nonfinite,'failure_mass_conservation_pass':abs(failure_sum-nonfinite)<=1e-6,'weighted_mean_route_entropy_nats':result['route_entropy_weighted']/finite if finite else None,'weighted_mean_first_boarding_route_entropy_contraction_nats':result['first_boarding_entropy_contraction_weighted']/finite if finite else None,'skip_mass_by_count':dict(sorted(result['skip_mass'].items())),'skip_positive_leg_share':sum(v for k,v in result['skip_mass'].items() if int(k)>0)/leg_mass if leg_mass else None,'non_simple_route_posterior_mass':result['non_simple_mass'],'non_simple_route_share_of_finite_mass':result['non_simple_mass']/finite if finite else None,'access_tail_gt_15min_posterior_mass':result['access_tail_mass_gt_15min'],'egress_tail_gt_15min_posterior_mass':result['egress_tail_mass_gt_15min'],'top_missing_service_pressure':[{'segment':k,'pressure_mass':v} for k,v in result['missing_service_pressure'].most_common(100)],'finite_cohort_count':result['finite_cohort_count'],'nonfinite_cohort_count':result['nonfinite_cohort_count'],'hypothesis_factor_visits':result['hypothesis_factor_visits']}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['E0','E1'],required=True); ap.add_argument('--shard-count',type=int,required=True); ap.add_argument('--shard-id',type=int,required=True); ap.add_argument('--p1c',required=True); ap.add_argument('--cohorts',required=True); ap.add_argument('--routes',required=True); ap.add_argument('--service-init',required=True); ap.add_argument('--topology-patch',required=True); ap.add_argument('--gtxa-overlay',required=True); ap.add_argument('--kernels'); ap.add_argument('--offsets'); ap.add_argument('--beam',type=int,default=8); ap.add_argument('--max-skip',type=int,default=2); ap.add_argument('--out',required=True); args=ap.parse_args()
    if not 0<=args.shard_id<args.shard_count: raise ValueError('invalid shard id')
    if args.phase=='E1' and (not args.kernels or not args.offsets): raise ValueError('E1 requires kernels and offsets')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    G,meta,code_to_nodes,transfer_groups,ambiguous_seq,ambiguous_codes,graph_build=g0.build_network(args.p1c); apply_topology_patch(G,meta,load_patch(args.topology_patch)); apply_gtxa_overlay(G,meta,code_to_nodes,load_overlay(args.gtxa_overlay))
    routes=support.load_routes(args.routes); roots,service_manifest,service_payload=g2.load_uncertain_service(args.service_init); kernels=parse_kernel_payload(args.kernels) if args.kernels else default_kernels(); offsets=parse_offsets(args.offsets) if args.offsets else {}
    if offsets: base.apply_root_offsets(roots,offsets)
    cohorts,scanned_count,scanned_mass=load_shard_cohorts(args.cohorts,args.shard_count,args.shard_id); cohort_path=out/f'{args.phase.lower()}_cohort_posterior.jsonl.gz'
    result,stats=posterior_stream(cohorts,routes,meta,roots,kernels,args.beam,args.max_skip,args.phase,cohort_path)
    outputs={'cohort_posterior_file':cohort_path.name}
    if args.phase=='E0':
        ks=out/'e0_kernel_sufficient_stats.jsonl.gz'; osf=out/'e0_offset_sufficient_stats.jsonl.gz'; write_kernel_stats(ks,stats); write_offset_stats(osf,stats); outputs.update({'kernel_sufficient_stats_file':ks.name,'offset_sufficient_stats_file':osf.name})
    payload={'schema':'mppd.r1c-distributed-posterior-shard.v2-sufficient-stats','date':'2026-09-05','status':'R1C_DISTRIBUTED_POSTERIOR_SHARD_COMPLETED','phase':args.phase,'shard_count':args.shard_count,'shard_id':args.shard_id,'scan_authority':{'full_cohort_count_seen':scanned_count,'full_passenger_mass_seen':scanned_mass},'scope':{'shard_cohort_count':len(cohorts),'shard_passenger_mass':sum(x[4] for x in cohorts),'full_network_candidate_domain':True,'line_filter':False,'segment_filter':False,'transfer_count_cap':False,'behavioral_regularities_hard_coded':False},'inputs':{'service_schema':service_payload.get('schema'),'service_status':service_payload.get('status'),'route_od_count':len(routes),'graph_nodes':G.number_of_nodes(),'graph_edges':G.number_of_edges(),'offset_count':len(offsets),'kernel_schema':kernels.get('schema')},'posterior':summarize(result),'root_usage':dict(result['root_usage']),'outputs':outputs,'sufficient_statistics':{'relative_time_round_digits':ROUND_DIGITS,'kernel_fit_equivalence':'BASE_KERNEL_FITS_USE_RELATIVE_INTERVAL_OR_POINT_MASS_ONLY; IDENTICAL_VALUES_ARE_WEIGHT_AGGREGATED','service_offset_equivalence':'ROOT_SHIFT_EFFECT_REPRESENTED_BY_RELATIVE_TIME_PLUS_ROOT_TOUCH_COEFFICIENTS'},'scientific_boundary':['Cohorts are partitioned only by deterministic hash; every shard retains the full network route/service candidate domain and arbitrary-transfer support.','Sufficient-statistic aggregation replaces raw factor persistence but does not alter posterior hypothesis probabilities or declared kernel/service-offset likelihood formulas.','This shard output is not independently interpretable as a city result; authority exists only after exact all-shard conservation merge.','No behavioral regularity is hard-coded and no service support is mutated.'],'no_email_notification_logic':True}
    (out/'shard_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'phase':args.phase,'shard':args.shard_id,'scope':payload['scope'],'posterior':payload['posterior'],'outputs':outputs},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
