from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import scripts.mppd_r1_hz_joint_full_day as model
import scripts.mppd_r1_hz_event_level_joint as event_model
import scripts.mppd_r1_hz_fast_projected as fast

SCHEMA="mppd.r1-hz-consensus-constraints.v1"


def constraint_schema():
    return pa.schema([
        ("cohort_id",pa.string()),("mass",pa.float64()),("posterior_probability",pa.float64()),
        ("weight",pa.float64()),("kind",pa.string()),("event_a",pa.string()),("event_b",pa.string()),
        ("bound_s",pa.float64()),("route_rank",pa.int16()),("root_chain",pa.string()),
    ])


def emit_constraints(row,st,posterior):
    mass=float(row['passenger_mass']);w=mass*float(posterior);cid=fast.cid(row);rr=int(st.route_rank)
    chain='>'.join(str(r.root['root_id']) for r in st.rides)
    out=[]
    def add(kind,a,b,bound):
        out.append({'cohort_id':cid,'mass':mass,'posterior_probability':float(posterior),'weight':w,'kind':kind,'event_a':a,'event_b':b or '', 'bound_s':float(bound),'route_rank':rr,'root_chain':chain})
    first=st.rides[0]
    add('ABS_LOWER',event_model.event_key(str(first.root['root_id']),first.board_station),None,float(row['entry_sec'])+model.MIN_MOVE_S)
    last=st.rides[-1]
    add('ABS_UPPER',event_model.event_key(str(last.root['root_id']),last.alight_station),None,float(row['exit_sec'])-model.MIN_MOVE_S)
    for r in st.rides:
        ka=event_model.event_key(str(r.root['root_id']),r.alight_station);kb=event_model.event_key(str(r.root['root_id']),r.board_station)
        add('DIFF_LOWER',ka,kb,1.0)
    for a,b in zip(st.rides,st.rides[1:]):
        ka=event_model.event_key(str(b.root['root_id']),b.board_station);kb=event_model.event_key(str(a.root['root_id']),a.alight_station)
        add('DIFF_LOWER',ka,kb,model.MIN_MOVE_S)
    return out


def run(cohorts:Path,roots_path:Path,routes_path:Path,params_path:Path,out_constraints:Path,out_summary:Path,shard_index:int):
    raw,roots,_stations,params=fast.load_roots(roots_path,params_path);routes=fast.load_routes(routes_path);idx=fast.LegIndex(roots)
    out_constraints.parent.mkdir(parents=True,exist_ok=True);writer=pq.ParquetWriter(out_constraints,constraint_schema(),compression='zstd')
    buf=[];pm=rm=0.0;cc=rc=0;constraints=0;topprob_num=0.0;proposal_count=exact_count=0;no_route=no_chain=0.0
    def flush():
        nonlocal buf
        if buf:
            writer.write_table(pa.Table.from_pylist(buf,schema=constraint_schema()));buf=[]
    try:
      for row in fast.iter_rows(cohorts):
        cc+=1;mass=float(row['passenger_mass']);pm+=mass;rr=routes.get(fast.route_key(row))
        if not rr:no_route+=mass;continue
        mincost=min(float(x.get('base_ranking_cost_s',0.0)) for x in rr);props=[]
        for route in rr:props.extend(fast.proposals_for_route(row,route,idx,params,mincost))
        props.sort(key=lambda z:z.score,reverse=True);props=props[:fast.FINAL_TOP_K];proposal_count+=len(props)
        states=[]
        for p in props:
            s=fast.score_exact(row,p,idx,params,mincost);exact_count+=1
            if s is not None:states.append(s)
        if not states:no_chain+=mass;continue
        states.sort(key=lambda z:z.loglik,reverse=True);z=model.logsumexp(s.loglik for s in states)
        if not math.isfinite(z):no_chain+=mass;continue
        post=[math.exp(s.loglik-z) for s in states];bi=max(range(len(states)),key=lambda i:post[i]);st=states[bi];q=post[bi]
        rows=emit_constraints(row,st,q);buf.extend(rows);constraints+=len(rows);rm+=mass;rc+=1;topprob_num+=mass*q
        if len(buf)>=50000:flush()
      flush()
    finally:writer.close()
    summary={'schema':SCHEMA,'shard_index':int(shard_index),'passenger_mass':pm,'resolved_mass':rm,'resolved_share':rm/pm if pm else None,'cohort_count':cc,'resolved_cohort_count':rc,'constraint_row_count':constraints,'mean_map_posterior_probability':topprob_num/rm if rm else None,'proposal_candidate_count':proposal_count,'exact_projection_count':exact_count,'no_route_mass':no_route,'no_projectable_chain_mass':no_chain,'semantics':{'classification_em_map_chain_constraints':True,'constraint_weight_is_passenger_mass_times_map_posterior':True,'all_constraints_reference_train_station_events':True,'five_second_access_transfer_egress_physics_preserved':True}}
    out_summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2));return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohorts',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--routes',type=Path,required=True);p.add_argument('--params',type=Path,required=True);p.add_argument('--out-constraints',type=Path,required=True);p.add_argument('--out-summary',type=Path,required=True);p.add_argument('--shard-index',type=int,required=True);a=p.parse_args();run(a.cohorts,a.roots,a.routes,a.params,a.out_constraints,a.out_summary,a.shard_index)
if __name__=='__main__':main()
