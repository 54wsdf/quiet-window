import argparse,csv,gzip,hashlib,json,time
from collections import Counter,defaultdict
from pathlib import Path
import networkx as nx
import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r0_full_network_factor_engine_20260904 as r0
from scripts.mppd_topology_patch_20260904 import apply_topology_patch,load_patch
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay,load_overlay

def hunit(text):return int(hashlib.sha1(text.encode()).hexdigest()[:8],16)/0xffffffff

def load_od_mass(path):
    out=Counter()
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        for x in csv.DictReader(f):out[(x['origin_code'],x['destination_code'])]+=int(x['passenger_mass'])
    return out

def policies():
    p=[]
    for tm in (0.65,0.85,1.0,1.2,1.5):
        for um in (1.0,1.35):p.append({'name':f'T{tm:.2f}_U{um:.2f}','transfer':tm,'uncertain':um,'seed':None})
    for seed in range(16):p.append({'name':f'PERTURB_{seed:02d}','transfer':1.0,'uncertain':1.15,'seed':seed})
    return p

def weight_fn(pol,meta):
    def w(u,v,d):
        base=float(d.get('weight',1.0));kind=d.get('kind');mult=1.0
        if kind in ('transfer','transfer_repaired','gtxa_transfer_repaired'):mult*=pol['transfer']
        elif kind=='inline_uncertain':mult*=pol['uncertain']
        seed=pol.get('seed')
        if seed is not None:
            key='|'.join(sorted((u,v)))+f'|{seed}'
            q=hunit(key)
            if kind in ('transfer','transfer_repaired','gtxa_transfer_repaired'):mult*=0.65+0.9*q
            else:
                line=meta.get(u,{}).get('line','')
                lq=hunit(f'{line}|{seed}')
                mult*=(0.88+0.24*lq)*(0.92+0.16*q)
        return base*mult
    return w

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--p1c',required=True);ap.add_argument('--cohorts',required=True);ap.add_argument('--out',required=True);ap.add_argument('--topology-patch');ap.add_argument('--gtxa-overlay');args=ap.parse_args()
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True);wall0=time.perf_counter()
    G,meta,code_to_nodes,transfer_groups,ambiguous_seq,ambiguous_codes,graph_build=g0.build_network(args.p1c)
    base_components=nx.number_connected_components(G);base_edges=G.number_of_edges();base_nodes=G.number_of_nodes();patch_result=None;overlay_result=None
    if args.topology_patch:
        patch=load_patch(args.topology_patch);patch_result=apply_topology_patch(G,meta,patch)
    after_patch={'nodes':G.number_of_nodes(),'edges':G.number_of_edges(),'connected_components':nx.number_connected_components(G)}
    if args.gtxa_overlay:
        overlay=load_overlay(args.gtxa_overlay);overlay_result=apply_gtxa_overlay(G,meta,code_to_nodes,overlay)
    topology_state={
      'base_nodes':base_nodes,'base_edges':base_edges,'base_connected_components':base_components,
      'after_topology_patch':after_patch,
      'final_nodes':G.number_of_nodes(),'final_edges':G.number_of_edges(),'final_connected_components':nx.number_connected_components(G),
      'topology_patch_applied':bool(args.topology_patch),'topology_patch':patch_result,
      'gtxa_overlay_applied':bool(args.gtxa_overlay),'gtxa_overlay':overlay_result
    }
    od_mass=load_od_mass(args.cohorts)
    source_to_ods=defaultdict(list)
    for oc,dc in od_mass:
        for src in code_to_nodes.get(oc,[]):
            if src in G:source_to_ods[src].append((oc,dc))
    by_od=defaultdict(dict);pt={}
    for pol in policies():
        t=time.perf_counter();best={};w=weight_fn(pol,meta)
        for src,ods in source_to_ods.items():
            lengths,paths=nx.single_source_dijkstra(G,src,weight=w)
            for od in ods:
                oc,dc=od;cur=best.get(od)
                for dst in code_to_nodes.get(dc,[]):
                    if dst not in paths:continue
                    path=paths[dst];key=(lengths[dst],len(path),src,dst)
                    if cur is None or key<cur[0]:cur=(key,path)
                if cur is not None:best[od]=cur
        for od,(_,path) in best.items():
            sig=tuple(path);rf,tc=r0.route_signature(path,meta);base=r0.path_cost(G,path)
            obj=by_od[od].setdefault(sig,{'path':path,'line_sequence':rf,'transfer_count':tc,'base_cost':base,'ensemble_policies':[]});obj['ensemble_policies'].append(pol['name'])
        pt[pol['name']]=time.perf_counter()-t
    hist=Counter();tc_mass=Counter();total_mass=0;unrouted=0;unrouted_mass=0
    with gzip.open(outdir/'r0_full_network_route_ensemble_candidates.jsonl.gz','wt',encoding='utf-8') as f:
        for od,mass in sorted(od_mass.items()):
            total_mass+=mass;cands=list(by_od.get(od,{}).values());cands.sort(key=lambda x:(x['base_cost'],x['transfer_count'],len(x['path'])))
            if not cands:unrouted+=1;unrouted_mass+=mass
            hist[len(cands)]+=mass
            for c in cands:tc_mass[c['transfer_count']]+=mass
            f.write(json.dumps({'origin_code':od[0],'destination_code':od[1],'passenger_mass':mass,'candidate_set_status':'ENSEMBLE_STRUCTURAL_HYPOTHESES','topology_patch_applied':bool(args.topology_patch),'gtxa_overlay_applied':bool(args.gtxa_overlay),'candidates':cands},ensure_ascii=False)+'\n')
    result={'schema':'mppd.r0-full-network-route-ensemble.v3','date':'2026-09-04','status':'R0_FULL_NETWORK_ROUTE_ENSEMBLE_COMPLETED','authority':'00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md',
      'scope_assertions':{'full_network':True,'line_filter':False,'segment_filter':False,'transfer_count_cap':False},'topology':topology_state,'policy_count':len(policies()),'policy_runtime_sec':pt,
      'od':{'count':len(od_mass),'passenger_mass':total_mass,'unrouted_od':unrouted,'unrouted_passenger_mass':unrouted_mass,'routed_passenger_mass':total_mass-unrouted_mass,'routed_share':((total_mass-unrouted_mass)/total_mass if total_mass else None)},'passenger_mass_by_candidate_count':dict(sorted(hist.items())),'candidate_support_mass_by_transfer_count':dict(sorted(tc_mass.items())),
      'performance':{'wall_sec':time.perf_counter()-wall0},'scientific_boundary':['Ensemble paths are structural route hypotheses, not observed routes and not guaranteed to exhaust all near-optimal alternatives.','No transfer-count cap is applied.','Topology patches and GTX-A crosswalk overlays remain provenance-typed and do not become observed route or ATS truth.','The route ensemble must be paired with a service-state initialization built on the same date-aware network representation before posterior inference.','The ensemble is an efficient candidate-diversity substrate; KSP shards remain the deeper alternative-set audit.'],
      'next_gate':'Use the exact G0H corrected cohort denominator and date-aware network representation; replace the invalid cross-component line-1032 weak lattice before corrected-denominator G2v2.','no_email_notification_logic':True}
    (outdir/'r0_full_network_route_ensemble_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
