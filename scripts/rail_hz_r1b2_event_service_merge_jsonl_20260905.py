from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "rail.hz-r1b2-event-service-posterior.v3-jsonl"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def event_key(root_id: str, event_index: int) -> str:
    return f"{root_id}::e{event_index:03d}"


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {'p10':None,'median':None,'p90':None}
    xs=sorted(values)
    def q(p: float) -> float:
        pos=p*(len(xs)-1); lo=int(math.floor(pos)); hi=int(math.ceil(pos))
        if lo==hi: return float(xs[lo])
        w=pos-lo
        return float(xs[lo]*(1-w)+xs[hi]*w)
    return {'p10':q(.1),'median':q(.5),'p90':q(.9)}


def iter_jsonl_gz(path: Path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--event-jsonl', type=Path, action='append', required=True)
    ap.add_argument('--summary', type=Path, action='append', required=True)
    ap.add_argument('--roots', type=Path, required=True)
    ap.add_argument('--genealogy-global', type=Path, required=True)
    ap.add_argument('--out-events', type=Path, required=True)
    ap.add_argument('--out-summary', type=Path, required=True)
    a=ap.parse_args()

    shard_summaries=[load_json(p) for p in a.summary]
    if len(shard_summaries)!=8 or len(a.event_jsonl)!=8:
        raise SystemExit(f'expected 8 shard products, got summaries={len(shard_summaries)} events={len(a.event_jsonl)}')
    roots_raw=load_json(a.roots)
    if roots_raw.get('status')!='QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION':
        raise SystemExit('roots not qualified')
    roots=roots_raw['roots']
    genealogy=load_json(a.genealogy_global)

    acc: dict[str, dict[str,float]] = defaultdict(lambda: defaultdict(float))
    source_rows=0
    for path in a.event_jsonl:
        for row in iter_jsonl_gz(path):
            source_rows += 1
            ek=str(row['event_key'])
            for src,dst in [
                ('traversal_lineage_mass','traversal'),('boarding_lineage_mass','boarding'),
                ('alighting_lineage_mass','alighting'),('transfer_arrival_lineage_mass','transfer_arrival'),
                ('transfer_departure_lineage_mass','transfer_departure')]:
                acc[ek][dst]+=float(row[src])
            acc[ek]['edge_count']+=int(row['lineage_edge_contribution_count'])

    expected_event_states=sum(len(r['events']) for r in roots)
    seen=set(); inventory=[]; evidence_counts=Counter(); supported=[]; zero_support=0
    root_max_support=defaultdict(float)
    with gzip.open(a.out_events,'wt',encoding='utf-8') as out:
        for root in roots:
            rid=str(root['root_id']); eclass=str(root.get('evidence_class','UNKNOWN'))
            evidence_counts[eclass]+=len(root['events'])
            for idx,ev in enumerate(root['events']):
                ek=event_key(rid,idx)
                if ek in seen: raise SystemExit(f'duplicate event key {ek}')
                seen.add(ek)
                x=acc.get(ek,{})
                trav=float(x.get('traversal',0.0)); root_max_support[rid]=max(root_max_support[rid],trav)
                if trav>0: supported.append(trav); support_state='GENEALOGY_SUPPORTED_EVENT_SEED'
                else: zero_support+=1; support_state='NO_CURRENT_GENEALOGY_TRAVERSAL_SUPPORT'
                t=float(ev['time_s']); sd=float(ev['sd_s'])
                row={
                    'event_key':ek,'root_id':rid,'path_id':str(root['path_id']),'direction':str(root['direction']),
                    'event_index':idx,'station':int(ev['station']),'prior_time_s':t,'prior_sd_s':sd,
                    'prior_lower_95_s':t-1.96*sd,'prior_upper_95_s':t+1.96*sd,
                    'current_timing_center_s':t,'current_timing_sd_s':sd,
                    'timing_stage':'PRE_BACKWARD_CONSTRAINT_UNCHANGED_FROM_QUALIFIED_SERVICE_INITIALIZATION',
                    'evidence_class':eclass,'support_state':support_state,
                    'traversal_lineage_mass':trav,'boarding_lineage_mass':float(x.get('boarding',0.0)),
                    'alighting_lineage_mass':float(x.get('alighting',0.0)),
                    'transfer_arrival_lineage_mass':float(x.get('transfer_arrival',0.0)),
                    'transfer_departure_lineage_mass':float(x.get('transfer_departure',0.0)),
                    'lineage_edge_contribution_count':int(x.get('edge_count',0)),
                }
                out.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                inventory.append(row)

    mapped=sum(float(x['mapped_chain_mass']) for x in shard_summaries)
    station_only=sum(float(x['station_only_mass']) for x in shard_summaries)
    resolved=sum(float(x['resolved_lineage_mass']) for x in shard_summaries)
    unresolved=sum(float(x['unresolved_mass']) for x in shard_summaries)
    board=sum(float(x['first_boarding_mass']) for x in shard_summaries)
    alight=sum(float(x['final_alighting_mass']) for x in shard_summaries)
    transfer=sum(float(x['transfer_episode_mass']) for x in shard_summaries)
    failure_mass=sum(float(x['mapping_failure_mass']) for x in shard_summaries)
    roots_with=sum(1 for r in roots if root_max_support[str(r['root_id'])]>0)
    roots_without=[str(r['root_id']) for r in roots if root_max_support[str(r['root_id'])]<=0]

    gates={
        'all_candidate_root_event_states_enumerated_once':len(inventory)==expected_event_states and len(seen)==expected_event_states,
        'all_8_shards_qualified':all(x['status']=='QUALIFIED_R1B2_EVENT_EVIDENCE_SHARD' and all(x['qualification_gates'].values()) for x in shard_summaries),
        'resolved_genealogy_mass_matches_r1b1':abs(resolved-float(genealogy['resolved_genealogy_mass']))<=1e-6,
        'resolved_decomposition_matches':abs((mapped+station_only)-resolved)<=1e-6,
        'unresolved_mass_matches_r1b1':abs(unresolved-float(genealogy['unresolved_genealogy_mass']))<=1e-6,
        'first_boarding_mass_conserved_over_service_chains':abs(board-mapped)<=1e-6,
        'final_alighting_mass_conserved_over_service_chains':abs(alight-mapped)<=1e-6,
        'no_event_mapping_failure_mass':abs(failure_mass)<=1e-9,
        'timing_not_silently_changed_in_r1b2':all(abs(float(x['current_timing_center_s'])-float(x['prior_time_s']))<=1e-12 and abs(float(x['current_timing_sd_s'])-float(x['prior_sd_s']))<=1e-12 for x in inventory),
    }
    if not all(gates.values()):
        raise SystemExit('R1B2 pure-python global qualification failed: '+json.dumps(gates))

    result={
        'schema':SCHEMA,'status':'QUALIFIED_R1B2_EVENT_LEVEL_SERVICE_POSTERIOR_SUBSTRATE','service_date':'2019-01-04',
        'scope':'FULL_SERVICE_DAY_0400_TO_NEXT_0400_FULL_NETWORK_FULL_QUALIFIED_PASSENGER_DOMAIN',
        'service_initialization_anchor_event_count':19236,
        'candidate_root_event_state_count':expected_event_states,
        'semantic_note_on_event_counts':'19236 is the qualified AFC passenger-facing initialization anchor count; candidate_root_event_state_count is the complete latent root-event inventory. They are not the same object.',
        'source_event_aggregate_rows':source_rows,'roots_total':len(roots),'roots_with_genealogy_support':roots_with,
        'roots_without_genealogy_support':roots_without,'root_support_share':roots_with/len(roots) if roots else None,
        'events_with_genealogy_traversal_support':expected_event_states-zero_support,
        'events_without_current_genealogy_traversal_support':zero_support,
        'event_support_share':(expected_event_states-zero_support)/expected_event_states if expected_event_states else None,
        'genealogy_traversal_support_mass_quantiles':quantiles(supported),
        'resolved_genealogy_mass':resolved,'mapped_service_chain_mass':mapped,'station_only_mass':station_only,
        'unresolved_genealogy_mass':unresolved,'first_boarding_mass':board,'final_alighting_mass':alight,
        'transfer_episode_mass':transfer,'mapping_failure_mass':failure_mass,
        'evidence_class_event_counts':dict(evidence_counts),'qualification_gates':gates,
        'timing_semantics':'R1B.2 only constructs the event-level service posterior substrate and attaches genealogy support. It does not yet alter event timing, event uncertainty, or event-existence probability; R1B.3 performs backward passenger-evidence updates.',
        'next_stage':'R1B_3_APPLY_GENEALOGY_BASED_BACKWARD_SERVICE_CONSTRAINTS'
    }
    a.out_summary.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
