import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r0_cached_full_network_support_rescan_20260904 as support
import scripts.mppd_g2v2_uncertain_service_full_network_posterior_20260904 as g2
import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base
import scripts.mppd_r1b_g1g2_graph_anchor_existing_root_qualification_20260904 as g1g2
import scripts.mppd_r1b_missing_service_support_proposal_audit_20260904 as support_audit
import scripts.mppd_r1c_hierarchical_temporal_kernels_first_pass_20260904 as r1c
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch


def load_residuals(path):
    out = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def mixture_quantile(kernel, q):
    lo, hi = 0.0, 7200.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if base.kernel_cdf(mid, kernel) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def normalized_probs(logps):
    z = base.logsumexp(logps)
    if not math.isfinite(z):
        return [1.0 / len(logps)] * len(logps) if logps else []
    return [math.exp(x - z) for x in logps]


def route_access_kernel(cand, meta, kernels, tin):
    access_context, _, _, _ = r1c.route_contexts(cand, meta)
    return r1c.choose_context_kernel(kernels, 'ACCESS', access_context, tin.hour)


def replay_to_hard_horizon(cand, meta, rides_fn, tin, tout, beam, kernels, max_skip):
    legs = [x for x in base.r0.path_legs(cand.get('path') or [], meta) if x[1] != x[2]]
    states = [{'ready': tin, 'ready_sd': 0.0, 'ready_root': None, 'logp': 0.0, 'chain': []}]
    access_kernel = route_access_kernel(cand, meta, kernels, tin)

    for leg_index, (line, origin, destination) in enumerate(legs):
        rides = rides_fn(line, origin, destination)
        if not rides:
            return {'reason': 'MISSING_SERVICE_SEGMENT', 'leg_index': leg_index, 'line': line, 'origin': origin, 'destination': destination}

        admissible_rides = []
        late_rides = []
        for rd in rides:
            late_sec = (rd['arr'] - tout).total_seconds()
            gate = 3.0 * max(1.0, float(rd['arr_sd']))
            if late_sec > gate:
                late_rides.append((late_sec - gate, rd))
            else:
                admissible_rides.append(rd)

        if not admissible_rides:
            prev_leg = states[0]['chain'][-1] if states and states[0]['chain'] else None
            movement = None if prev_leg is None else f"{prev_leg['line']}:{prev_leg['destination']}->{line}:{origin}"
            kind = 'ACCESS' if prev_leg is None else 'TRANSFER'
            kern = access_kernel if kind == 'ACCESS' else base.kernel_for(kind, movement, kernels)
            q05 = mixture_quantile(kern, 0.05); q50 = mixture_quantile(kern, 0.50); q95 = mixture_quantile(kern, 0.95)
            probs = normalized_probs([s['logp'] for s in states])
            ready50_ts = sum(p * (s['ready'] + timedelta(seconds=q50)).timestamp() for p, s in zip(probs, states)) if states else None
            closest = min(late_rides, key=lambda x: x[0]) if late_rides else None
            return {
                'reason': 'SERVICE_EXIT_HORIZON_INCOMPATIBLE',
                'leg_index': leg_index,
                'line': line,
                'origin': origin,
                'destination': destination,
                'kind_before_gate': kind,
                'movement_before_gate': movement,
                'ready_q05_min': min((s['ready'] + timedelta(seconds=q05) for s in states), default=None),
                'ready_q50_weighted_mean': datetime.fromtimestamp(ready50_ts) if ready50_ts is not None else None,
                'ready_q95_max': max((s['ready'] + timedelta(seconds=q95) for s in states), default=None),
                'observed_exit': tout,
                'closest_current_arrival': closest[1]['arr'] if closest else None,
                'closest_current_root': closest[1]['root_key'] if closest else None,
                'closest_excess_beyond_exit_3sigma_sec': closest[0] if closest else None,
            }

        nxt = []
        for st in states:
            prev_leg = st['chain'][-1] if st['chain'] else None
            movement = None if prev_leg is None else f"{prev_leg['line']}:{prev_leg['destination']}->{line}:{origin}"
            kind = 'ACCESS' if prev_leg is None else 'TRANSFER'
            kern = access_kernel if kind == 'ACCESS' else base.kernel_for(kind, movement, kernels)
            for i, rd in enumerate(rides):
                late_sec = (rd['arr'] - tout).total_seconds()
                if late_sec > 3.0 * max(1.0, float(rd['arr_sd'])):
                    continue
                for n_skip, lower, upper in base.skip_intervals(rides, i, max_skip):
                    lp = base.interval_logprob_kernel(
                        lower['dep'] if lower else None,
                        lower['dep_sd'] if lower else 0.0,
                        upper['dep'], upper['dep_sd'],
                        st['ready'], st['ready_sd'], kern,
                    ) - math.log(max(1, len(base.skip_intervals(rides, i, max_skip))))
                    if not math.isfinite(lp):
                        continue
                    leg = {'line': line, 'origin': origin, 'destination': destination, 'root_key': rd['root_key'], 'arr': rd['arr']}
                    nxt.append({'ready': rd['arr'], 'ready_sd': rd['arr_sd'], 'ready_root': rd['root_key'], 'logp': st['logp'] + lp, 'chain': st['chain'] + [leg]})
        if not nxt:
            return {'reason': 'UNEXPLAINED_TIME_ENGINE_FAILURE', 'leg_index': leg_index, 'line': line, 'origin': origin, 'destination': destination}
        nxt.sort(key=lambda x: (x['logp'], -x['ready'].timestamp()), reverse=True)
        states = nxt[:beam]
    return {'reason': 'HARD_CHAIN_EXISTS'}


def iso(x):
    return x.isoformat() if isinstance(x, datetime) else x


def shift_root_variants(root_variants, offsets):
    for root_key, variants in root_variants.items():
        delta = int(offsets.get(root_key, 0))
        if delta == 0:
            continue
        for variant in variants:
            for event in variant.get('events', {}).values():
                event['time'] = event['time'] + timedelta(seconds=delta)


def evaluate_candidate(op, dp, records):
    dep = op['time']; arr = dp['time']
    out = {'q05_support_mass': 0.0, 'q50_support_mass': 0.0, 'q95_support_mass': 0.0, 'exit_support_mass': 0.0,
           'q05_supported_cohorts': 0, 'q50_supported_cohorts': 0, 'q95_supported_cohorts': 0}
    for r in records:
        m = float(r['pressure_mass'])
        exit_ok = arr <= r['observed_exit']
        if exit_ok: out['exit_support_mass'] += m
        if exit_ok and dep >= r['ready_q05']:
            out['q05_support_mass'] += m; out['q05_supported_cohorts'] += 1
        if exit_ok and dep >= r['ready_q50']:
            out['q50_support_mass'] += m; out['q50_supported_cohorts'] += 1
        if exit_ok and dep >= r['ready_q95']:
            out['q95_support_mass'] += m; out['q95_supported_cohorts'] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p1c', required=True)
    ap.add_argument('--routes', required=True)
    ap.add_argument('--service-init', required=True)
    ap.add_argument('--r1c-summary', required=True)
    ap.add_argument('--residual-cohorts', required=True)
    ap.add_argument('--topology-patch', required=True)
    ap.add_argument('--gtxa-overlay', required=True)
    ap.add_argument('--beam', type=int, default=8)
    ap.add_argument('--max-skip', type=int, default=2)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    apply_topology_patch(G, meta, load_patch(args.topology_patch)); apply_gtxa_overlay(G, meta, code_to_nodes, load_overlay(args.gtxa_overlay))
    line_graphs = support_audit.line_graphs(G, meta)
    routes = support.load_routes(args.routes)
    roots, service_manifest, service_payload = g2.load_uncertain_service(args.service_init)
    service_json = json.loads(Path(args.service_init).read_text(encoding='utf-8'))
    summary = json.loads(Path(args.r1c_summary).read_text(encoding='utf-8'))
    residuals = load_residuals(args.residual_cohorts)
    kernels = summary['iteration']['M_kernel_update']['kernels_after']
    offsets = summary['iteration']['M_service_timing_update'].get('nonzero_offsets', {})
    base.apply_root_offsets(roots, offsets)
    rides_fn, _ = base.build_joint_rides(roots)
    root_variants = g1g2.build_root_variants(service_json)
    shift_root_variants(root_variants, offsets)

    seg_records = defaultdict(list)
    seg_pressure = Counter(); total_horizon_pressure = 0.0; unexplained = 0.0
    for rr in residuals:
        oc=str(rr['origin_code']); dc=str(rr['destination_code']); tin=datetime.fromisoformat(rr['entry_time']); tout=datetime.fromisoformat(rr['exit_time']); mass=float(rr['passenger_mass'])
        cands=routes.get((oc,dc),[]); sw=base.structural_pressure_weights(cands) if cands else []
        for ci,cand in enumerate(cands):
            w=float(sw[ci]) if ci < len(sw) else 0.0
            if w<=0: continue
            rec=replay_to_hard_horizon(cand,meta,rides_fn,tin,tout,args.beam,kernels,args.max_skip)
            if rec.get('reason')=='SERVICE_EXIT_HORIZON_INCOMPATIBLE':
                seg=f"{rec['line']}|{rec['origin']}|{rec['destination']}"; pm=mass*w
                seg_pressure[seg]+=pm; total_horizon_pressure+=pm
                seg_records[seg].append({'pressure_mass':pm,'od':f'{oc}->{dc}','ready_q05':rec['ready_q05_min'],'ready_q50':rec['ready_q50_weighted_mean'],'ready_q95':rec['ready_q95_max'],'observed_exit':tout,'entry_time':tin,'closest_current_arrival':rec.get('closest_current_arrival'),'closest_current_root':rec.get('closest_current_root'),'closest_excess_sec':rec.get('closest_excess_beyond_exit_3sigma_sec')})
            elif rec.get('reason')=='UNEXPLAINED_TIME_ENGINE_FAILURE':
                unexplained += mass*w

    rows=[]
    for seg, pressure in seg_pressure.items():
        parsed=support_audit.parse_pressure_segment(seg)
        if not parsed: continue
        line, origin, destination=parsed; H=line_graphs.get(line,nx.Graph()); path,unique=g1g2.unique_shortest_path(H,origin,destination)
        records=seg_records[seg]; ods={r['od'] for r in records}
        variants=[]
        for root_key, vlist in root_variants.items():
            if not root_key.startswith(f'{line}||'): continue
            for v in vlist:
                op=g1g2.predict_target_strict(v,origin,H); dp=g1g2.predict_target_strict(v,destination,H)
                if not op or not dp or op['time']>=dp['time']: continue
                strong=op['anchor_grade'] in g1g2.STRONG_GRADES and dp['anchor_grade'] in g1g2.STRONG_GRADES
                if not strong: continue
                ev=evaluate_candidate(op,dp,records)
                row={'root':root_key,'variant_id':v['variant_id'],'variant_evidence_class':v['evidence_class'],
                     'origin_prediction':{**op,'time':op['time'].isoformat()},'destination_prediction':{**dp,'time':dp['time'].isoformat()},**ev}
                key=(ev['q50_support_mass'],ev['q95_support_mass'],ev['exit_support_mass'],int(op['anchor_grade']=='ORIGINAL_EXACT')+int(dp['anchor_grade']=='ORIGINAL_EXACT'),-max(op['bracket_span_edges'],dp['bracket_span_edges']))
                variants.append((key,row))
        variants.sort(key=lambda x:x[0],reverse=True)
        best=variants[0][1] if variants else None
        q50frac=best['q50_support_mass']/pressure if best and pressure>0 else 0.0
        q95frac=best['q95_support_mass']/pressure if best and pressure>0 else 0.0
        trial=bool(unique and best and q50frac>=0.5 and len(ods)>=2)
        rows.append({'segment':seg,'line':line,'pressure_mass':pressure,'cohort_count':len(records),'distinct_od_count':len(ods),'segment_unique_shortest_path':unique,'segment_graph_path_edges':len(path)-1 if path else None,'strong_candidate_count':len(variants),'best_strong_variant':best,'best_strong_q50_support_fraction':q50frac,'best_strong_q95_support_fraction':q95frac,'strong_bounded_repair_trial_eligible':trial,'top_variants':[x[1] for x in variants[:20]]})

    rows.sort(key=lambda r:(int(r['strong_bounded_repair_trial_eligible']),r['best_strong_q50_support_fraction'],r['pressure_mass']),reverse=True)
    trials=[r for r in rows if r['strong_bounded_repair_trial_eligible']]
    result={
        'schema':'mppd.r1c-service-exit-horizon-support-qualification.v1','date':'2026-09-05',
        'status':'R1C_SERVICE_EXIT_HORIZON_SUPPORT_QUALIFICATION_COMPLETED_NO_MUTATION',
        'source_r1c_schema':summary.get('schema'),'source_service_schema':service_payload.get('schema'),
        'scope':{'horizon_pressure_mass':total_horizon_pressure,'segment_count':len(rows),'unexplained_engine_pressure_mass':unexplained,'pressure_with_strong_bounded_repair_trial':sum(r['pressure_mass'] for r in trials),'strong_bounded_repair_trial_segment_count':len(trials)},
        'strong_bounded_repair_trials':trials,'segments':rows,
        'qualification_rules':{'graph_path':'TARGET_SEGMENT_UNIQUE_SAME_LINE_SHORTEST_PATH','endpoint_prediction':'EXACT_SOURCE_ROOT_ORIGINAL_EVENT_OR_UNIQUE_GRAPH_BRACKET_INTERPOLATION_ONLY','temporal_support':'BEST_STRONG_VARIANT_Q50_SUPPORT_FRACTION_AT_LEAST_0_5','network_support':'DISTINCT_OD_COUNT_AT_LEAST_2'},
        'scientific_boundary':['This audit targets service-support holes hidden inside the previous TIME_INCOMPATIBLE_CHAIN label when rides exist only after the AFC exit horizon.','It performs no service mutation and does not convert posterior pressure into observed train truth.','A surviving trial only authorizes a controlled observed-event-preserving completion inside an existing root followed by exact paired rerun.','If no trial survives, the service-exit horizon residual remains explicit and full-denominator R1C may proceed without further service completion.'],
        'next_gate':'Apply only surviving bounded existing-root trials, exact-paired rerun, and reject on scale reversal; if no trials survive, freeze horizon residual and advance distributed full-denominator R1C.',
        'no_email_notification_logic':True,
    }
    (outdir/'r1c_service_exit_horizon_support_qualification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'scope':result['scope'],'trials':trials[:20],'top_segments':rows[:20]},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
