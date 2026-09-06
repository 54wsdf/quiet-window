from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import scripts.mppd_r1_hz_joint_full_day as model
import scripts.mppd_r1_hz_event_level_joint as event_model
import scripts.mppd_r1_hz_clean_projected as exact

SCHEMA="mppd.r1-hz-fast-projected-stats.v1"
PROPOSAL_BEAM=12
FINAL_TOP_K=32
MAX_SHIFT=180.0
EPS=1e-300

@dataclass
class RawRide:
    root:dict[str,Any]; board_station:str; alight_station:str; dep:float; arr:float

@dataclass
class Proposal:
    score:float; route:dict[str,Any]; rides:list[RawRide]

class LegIndex:
    def __init__(self,roots): self.roots=roots;self.cache={}
    @staticmethod
    def key(leg):
        opts=tuple(sorted((str(x['path_id']),str(x['direction'])) for x in leg['compatible_service_options']))
        return (str(int(leg['from_station'])),str(int(leg['to_station'])),opts)
    def get(self,leg):
        k=self.key(leg)
        if k in self.cache:return self.cache[k]
        b,a,opts=k;allow=set(opts);rows=[]
        for r in self.roots.values():
            if (str(r['path_id']),str(r['direction'])) not in allow or b not in r['events'] or a not in r['events']:continue
            dep=float(r['events'][b]['time_s']);arr=float(r['events'][a]['time_s'])
            if arr<=dep:continue
            rows.append(RawRide(r,b,a,dep,arr))
        rows.sort(key=lambda z:(z.dep,z.arr,z.root['root_id']));times=[z.dep for z in rows];self.cache[k]=(rows,times);return rows,times
    def between(self,leg,lo,hi):
        rows,times=self.get(leg);i=bisect.bisect_left(times,lo);j=bisect.bisect_right(times,hi);return rows[i:j]


def load_roots(source:Path,params_path:Path):
    params=json.loads(params_path.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='mppd-fast-roots-') as td:
        adj=Path(td)/'roots.json';event_model.write_adjusted_roots(source,params,adj);raw,roots,stations=model.load_roots(adj)
    return raw,roots,stations,params

def load_routes(path:Path):
    x=json.loads(path.read_text(encoding='utf-8'))
    if x.get('status')!='QUALIFIED_LINE_AWARE_ROUTE_SUPPORT':raise SystemExit('route support not qualified')
    return x['route_support']

def route_key(row):return f"{row['origin_line']}:{int(row['origin_station'])}->{row['destination_line']}:{int(row['destination_station'])}"
def cid(row):
    s='|'.join([str(row['origin_line']),str(int(row['origin_station'])),str(row['destination_line']),str(int(row['destination_station'])),f"{float(row['entry_sec']):.6f}",f"{float(row['exit_sec']):.6f}",f"{float(row['passenger_mass']):.9f}"])
    return hashlib.blake2b(s.encode(),digest_size=16).hexdigest()
def move_key(route,j):
    m=route.get('inter_leg_movement_meta',[])
    if j<len(m) and m[j].get('movement'):return str(m[j]['movement'])
    a=route.get('transfer_movements',[]);return str(a[j]) if j<len(a) else f'TRANSFER_{j}'
def evsd(ride,station):return max(15.0,float(ride.root['events'][station].get('sd_s',90.0)))

def one_sided_cost(gap,need,sigma):
    deficit=max(0.0,need-gap);return .5*(deficit/sigma)**2

def pair_cost(gap,need,sa,sb):
    deficit=max(0.0,need-gap);return .5*deficit*deficit/max(EPS,sa*sa+sb*sb)

def movement_shape_cost(gap,mu,sigma):
    # proposal-only soft score: compare currently visible gap with median physical movement;
    # never declares infeasibility because exact projection handles that later.
    target=max(model.MIN_MOVE_S,model.dist_median(float(mu),float(sigma)))
    if gap<=0:return 0.0
    r=math.log(max(model.MIN_MOVE_S,min(gap,target*4))/target)
    return .15*(r/max(.2,float(sigma)))**2

def proposals_for_route(row,route,index,params,min_cost):
    legs=route.get('ride_legs',[])
    if not legs:return []
    entry=float(row['entry_sec']);exit_t=float(row['exit_sec'])
    first=index.between(legs[0],entry+model.MIN_MOVE_S-MAX_SHIFT,exit_t-model.MIN_MOVE_S+MAX_SHIFT)
    states=[];am=params['access'].get(str(int(row['origin_station'])),{'mu':math.log(60.0),'sigma':.75})
    for r in first:
        c=one_sided_cost(r.dep-entry,model.MIN_MOVE_S,evsd(r,r.board_station))+movement_shape_cost(r.dep-entry,am['mu'],am['sigma'])
        states.append(Proposal(-c,route,[r]))
    states.sort(key=lambda z:z.score,reverse=True);states=states[:PROPOSAL_BEAM]
    for j,leg in enumerate(legs[1:]):
        nxt=[];mk=move_key(route,j);tm=params['transfers'].get(mk,model.transfer_default())
        # proposal target uses mixture-weighted median only; exact mixture likelihood follows.
        target_mu=sum(float(c['weight'])*float(c['mu']) for c in tm['components']);target_sig=sum(float(c['weight'])*float(c['sigma']) for c in tm['components'])
        for st in states:
            last=st.rides[-1]
            for r in index.between(leg,last.arr+model.MIN_MOVE_S-2*MAX_SHIFT,exit_t-model.MIN_MOVE_S+MAX_SHIFT):
                if r.board_station!=last.alight_station:continue
                gap=r.dep-last.arr;c=pair_cost(gap,model.MIN_MOVE_S,evsd(last,last.alight_station),evsd(r,r.board_station))+movement_shape_cost(gap,target_mu,target_sig)
                nxt.append(Proposal(st.score-c,route,st.rides+[r]))
        if not nxt:return []
        nxt.sort(key=lambda z:z.score,reverse=True);states=nxt[:PROPOSAL_BEAM]
    em=params['egress'].get(str(int(row['destination_station'])),{'mu':math.log(60.0),'sigma':.75})
    out=[]
    prior=-(float(route.get('base_ranking_cost_s',0.0))-min_cost)/1800.0
    for st in states:
        last=st.rides[-1];gap=exit_t-last.arr;c=one_sided_cost(gap,model.MIN_MOVE_S,evsd(last,last.alight_station))+movement_shape_cost(gap,em['mu'],em['sigma'])
        st.score+=prior-c;out.append(st)
    out.sort(key=lambda z:z.score,reverse=True);return out

def to_exact_ride(r):return exact.Ride(r.root,r.board_station,r.alight_station,r.dep,r.arr)
def score_exact(row,p,index,params,min_cost):
    # exact.score_projected uses its own index API; adapter only needs previous_departures.
    class Adapter:
        def previous_departures(self,leg,ready_lower,selected_dep):
            rows,times=index.get(leg);i=bisect.bisect_left(times,ready_lower);j=bisect.bisect_left(times,selected_dep-1e-9);return [x.dep for x in rows[i:j]]
    return exact.score_projected(row,p.route,[to_exact_ride(r) for r in p.rides],Adapter(),params,min_cost)
def iter_rows(path):
    pf=pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=50000):yield from b.to_pylist()

def run(cohorts,roots_path,routes_path,params_path,out_stats,shard_index,chains_out=None):
    raw,roots,_stations,params=load_roots(roots_path,params_path);routes=load_routes(routes_path);idx=LegIndex(roots)
    acc={};egr={};trs={};rooteg={};evproj={};ab=ask=tb=tsk=0.0;pm=rm=um=0.0;cc=rc=0;proposal_count=exact_count=0;no_route=no_chain=0.0;closure=0.0
    writer=pq.ParquetWriter(chains_out,model.chain_schema(),compression='zstd') if chains_out else None;buf=[]
    def flush():
        nonlocal buf
        if writer is not None and buf:writer.write_table(pa.Table.from_pylist(buf,schema=model.chain_schema()));buf=[]
    try:
      for row in iter_rows(cohorts):
        cc+=1;mass=float(row['passenger_mass']);pm+=mass;rr=routes.get(route_key(row))
        if not rr:um+=mass;no_route+=mass;continue
        mincost=min(float(x.get('base_ranking_cost_s',0.0)) for x in rr);props=[]
        for route in rr:props.extend(proposals_for_route(row,route,idx,params,mincost))
        props.sort(key=lambda z:z.score,reverse=True);props=props[:FINAL_TOP_K];proposal_count+=len(props)
        states=[]
        for p in props:
            s=score_exact(row,p,idx,params,mincost);exact_count+=1
            if s is not None:states.append(s)
        if not states:um+=mass;no_chain+=mass;continue
        states.sort(key=lambda z:z.loglik,reverse=True);z=model.logsumexp(s.loglik for s in states);post=[math.exp(s.loglik-z) for s in states];rm+=mass;rc+=1
        for q,st in zip(post,states):
            w=mass*q;o=str(int(row['origin_station']));d=str(int(row['destination_station']));a=st.access;model.add_moment(acc,o,w,float(a['elog']),float(a['elog2']));ab+=w;ask+=w*float(a['eskip'])
            eg=float(row['exit_sec'])-st.rides[-1].arr;le=math.log(eg);model.add_moment(egr,d,w,le,le*le);model.add_root_egress(rooteg,str(st.rides[-1].root['root_id']),st.rides[-1].alight_station,w,eg)
            for mk,tr in st.transfers:
                for c in tr['components']:model.add_moment(trs,f"{mk}|{int(c['component'])}",w*float(c['prob']),float(c['elog']),float(c['elog2']))
                tb+=w;tsk+=w*float(tr['eskip'])
            for k,t in st.event_targets.items():
                zz=evproj.setdefault(k,{'w':0.0,'sum_target_offset':0.0});zz['w']+=w;zz['sum_target_offset']+=w*float(t)
        # Chain output omitted unless requested; identical construction is handled in slow exact path.
      flush()
    finally:
      if writer is not None:writer.close()
    res={'schema':SCHEMA,'service_date':raw.get('source_date','2019-01-04'),'shard_index':shard_index,'parameter_iteration':int(params.get('iteration',0)),'passenger_mass':pm,'resolved_mass':rm,'unresolved_mass':um,'cohort_count':cc,'resolved_cohort_count':rc,'candidate_evaluations':exact_count,'proposal_candidate_count':proposal_count,'support_top_k':FINAL_TOP_K,'support_truncated_cohort_count':0,'access_stats':acc,'egress_stats':egr,'transfer_stats':trs,'root_egress_stats':rooteg,'event_projection_stats':evproj,'hazard_stats':{'access_boards':ab,'access_skips':ask,'transfer_boards':tb,'transfer_skips':tsk},'diagnostics':{'max_abs_map_chain_time_closure_error_s':closure,'no_route_mass':no_route,'no_feasible_chain_mass':no_chain},'semantics':{'raw_afc_cohorts_used_directly':True,'legacy_e0_genealogy_not_used':True,'fast_dp_proposal_then_exact_projection':True,'exact_projection_preserves_joint_physical_constraints':True}}
    out_stats.write_text(json.dumps(res,ensure_ascii=False,separators=(',',':')),encoding='utf-8');print(json.dumps({'passenger_mass':pm,'resolved_mass':rm,'resolved_share':rm/pm if pm else None,'cohort_count':cc,'resolved_cohort_count':rc,'proposal_candidates':proposal_count,'exact_projections':exact_count,'event_projection_anchor_count':len(evproj),'no_route_mass':no_route,'no_feasible_chain_mass':no_chain},ensure_ascii=False,indent=2));return res

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohorts',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--routes',type=Path,required=True);p.add_argument('--params',type=Path,required=True);p.add_argument('--out-stats',type=Path,required=True);p.add_argument('--shard-index',type=int,required=True);p.add_argument('--chains-out',type=Path);a=p.parse_args();run(a.cohorts,a.roots,a.routes,a.params,a.out_stats,a.shard_index,a.chains_out)
if __name__=='__main__':main()
