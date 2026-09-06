from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import scripts.mppd_r1_hz_joint_full_day as model
import scripts.mppd_r1_hz_event_level_joint as event_model

SCHEMA = "mppd.r1-hz-clean-projected-stats.v1"
BEAM = 48
FINAL_TOP_K = 64
MAX_EVENT_OFFSET_S = 180.0
MIN_RIDE_S = 1.0
PROJECTION_ITERS = 50
TOL = 1e-7
EPS = 1e-300


@dataclass
class Ride:
    root: dict[str, Any]
    board_station: str
    alight_station: str
    dep: float
    arr: float


@dataclass
class State:
    loglik: float
    route_rank: int
    rides: list[Ride]
    access: dict[str, float]
    transfers: list[tuple[str, dict[str, Any]]]
    event_targets: dict[str, float]
    projection_penalty: float


class LegIndex:
    def __init__(self, roots: dict[str, dict[str, Any]]):
        self.roots = roots
        self.cache: dict[tuple[Any, ...], tuple[list[Ride], list[float]]] = {}

    @staticmethod
    def key(leg: dict[str, Any]) -> tuple[Any, ...]:
        opts = tuple(sorted((str(x["path_id"]), str(x["direction"])) for x in leg["compatible_service_options"]))
        return (str(int(leg["from_station"])), str(int(leg["to_station"])), opts)

    def get(self, leg: dict[str, Any]) -> tuple[list[Ride], list[float]]:
        k = self.key(leg)
        if k in self.cache:
            return self.cache[k]
        b, a, opts = k
        allowed = set(opts); rows: list[Ride] = []
        for r in self.roots.values():
            if (str(r["path_id"]), str(r["direction"])) not in allowed:
                continue
            if b not in r["events"] or a not in r["events"]:
                continue
            dep = float(r["events"][b]["time_s"]); arr = float(r["events"][a]["time_s"])
            if arr <= dep:
                continue
            rows.append(Ride(r,b,a,dep,arr))
        rows.sort(key=lambda x:(x.dep,x.arr,x.root["root_id"])); times=[x.dep for x in rows]
        self.cache[k]=(rows,times); return rows,times

    def between(self, leg: dict[str, Any], lo: float, hi: float) -> list[Ride]:
        rows,times=self.get(leg); i=bisect.bisect_left(times,lo); j=bisect.bisect_right(times,hi); return rows[i:j]

    def previous_departures(self, leg: dict[str, Any], ready_lower: float, selected_dep: float) -> list[float]:
        rows,times=self.get(leg); i=bisect.bisect_left(times,ready_lower); j=bisect.bisect_left(times,selected_dep-1e-9)
        return [x.dep for x in rows[i:j]]


def cohort_id(row: dict[str, Any]) -> str:
    text="|".join([str(row["origin_line"]),str(int(row["origin_station"])),str(row["destination_line"]),str(int(row["destination_station"])),f"{float(row['entry_sec']):.6f}",f"{float(row['exit_sec']):.6f}",f"{float(row['passenger_mass']):.9f}"])
    return hashlib.blake2b(text.encode(),digest_size=16).hexdigest()


def route_key(row: dict[str, Any]) -> str:
    return f"{row['origin_line']}:{int(row['origin_station'])}->{row['destination_line']}:{int(row['destination_station'])}"


def movement_key(route: dict[str, Any], j: int) -> str:
    meta=route.get("inter_leg_movement_meta",[])
    if j<len(meta) and meta[j].get("movement"): return str(meta[j]["movement"])
    m=route.get("transfer_movements",[]); return str(m[j]) if j<len(m) else f"TRANSFER_{j}"


def load_routes(path: Path):
    x=json.loads(path.read_text(encoding="utf-8"))
    if x.get("status")!="QUALIFIED_LINE_AWARE_ROUTE_SUPPORT": raise SystemExit("route support not qualified")
    return x["route_support"]


def load_roots(source: Path, params_path: Path):
    params=json.loads(params_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="mppd-projected-roots-") as td:
        adj=Path(td)/"roots.json"; event_model.write_adjusted_roots(source,params,adj); raw,roots,stations=model.load_roots(adj)
    return raw,roots,stations,params


def _event_key(r: Ride, station: str) -> str:
    return event_model.event_key(str(r.root["root_id"]),station)


def _event_sd(r: Ride, station: str) -> float:
    return max(15.0,float(r.root["events"][station].get("sd_s",90.0)))


def project_chain(rides: list[Ride], entry: float, exit_t: float, current_offsets: dict[str,float]):
    # Variables are incremental event shifts around the current shared timetable.
    curtime: dict[str,float]={}; sigma: dict[str,float]={}; current: dict[str,float]={}
    for r in rides:
        for station,t in ((r.board_station,r.dep),(r.alight_station,r.arr)):
            k=_event_key(r,station); curtime[k]=float(t); sigma[k]=_event_sd(r,station); current[k]=float(current_offsets.get(k,0.0))
    delta={k:0.0 for k in curtime}
    constraints: list[tuple[dict[str,float],float]]=[]
    # a^T delta >= rhs
    k0=_event_key(rides[0],rides[0].board_station)
    constraints.append(({k0:1.0}, entry+model.MIN_MOVE_S-curtime[k0]))
    for r in rides:
        kb=_event_key(r,r.board_station); ka=_event_key(r,r.alight_station)
        constraints.append(({ka:1.0,kb:-1.0}, MIN_RIDE_S-(curtime[ka]-curtime[kb])))
    for a,b in zip(rides,rides[1:]):
        if a.alight_station!=b.board_station: return None
        ka=_event_key(a,a.alight_station); kb=_event_key(b,b.board_station)
        constraints.append(({kb:1.0,ka:-1.0}, model.MIN_MOVE_S-(curtime[kb]-curtime[ka])))
    kl=_event_key(rides[-1],rides[-1].alight_station)
    constraints.append(({kl:-1.0}, -(exit_t-model.MIN_MOVE_S-curtime[kl])))

    for _ in range(PROJECTION_ITERS):
        changed=False
        for k in delta:
            lo=-MAX_EVENT_OFFSET_S-current[k]; hi=MAX_EVENT_OFFSET_S-current[k]
            nv=max(lo,min(hi,delta[k]))
            if abs(nv-delta[k])>TOL: delta[k]=nv; changed=True
        for coeff,rhs in constraints:
            lhs=sum(a*delta[k] for k,a in coeff.items()); gap=rhs-lhs
            if gap<=TOL: continue
            denom=sum((a*a)*(sigma[k]**2) for k,a in coeff.items())
            if denom<=0: return None
            for k,a in coeff.items(): delta[k]+=gap*(sigma[k]**2)*a/denom
            changed=True
        if not changed: break
    for k in delta:
        total=current[k]+delta[k]
        if total < -MAX_EVENT_OFFSET_S-1e-6 or total > MAX_EVENT_OFFSET_S+1e-6: return None
    for coeff,rhs in constraints:
        if sum(a*delta[k] for k,a in coeff.items()) < rhs-1e-5: return None
    targets={k:current[k]+delta[k] for k in delta}
    penalty=.5*sum((delta[k]/sigma[k])**2 for k in delta)
    out=[]
    for r in rides:
        kd=_event_key(r,r.board_station); ka=_event_key(r,r.alight_station)
        out.append(Ride(r.root,r.board_station,r.alight_station,r.dep+delta[kd],r.arr+delta[ka]))
    return out,targets,penalty


def route_prior(route:dict[str,Any],min_cost:float)->float:
    return -(float(route.get("base_ranking_cost_s",0.0))-min_cost)/1800.0


def score_projected(row:dict[str,Any],route:dict[str,Any],raw_rides:list[Ride],index:LegIndex,params:dict[str,Any],min_cost:float):
    entry=float(row["entry_sec"]); exit_t=float(row["exit_sec"]); origin=str(int(row["origin_station"])); dest=str(int(row["destination_station"]))
    pr=project_chain(raw_rides,entry,exit_t,{str(k):float(v) for k,v in params.get("event_offsets_s",{}).items()})
    if pr is None:return None
    rides,targets,penalty=pr
    am=params["access"].get(origin,{"mu":math.log(60.0),"sigma":.75}); em=params["egress"].get(dest,{"mu":math.log(60.0),"sigma":.75})
    first_leg=route["ride_legs"][0]
    prev=index.previous_departures(first_leg,entry+model.MIN_MOVE_S,rides[0].dep)
    access=model.boarding_component(entry,rides[0].dep,prev,float(am["mu"]),float(am["sigma"]),float(params["access_hazard"]))
    if access is None:return None
    transfers=[]; tll=0.0
    for j,(a,b) in enumerate(zip(rides,rides[1:])):
        mk=movement_key(route,j); tm=params["transfers"].get(mk,model.transfer_default())
        leg=route["ride_legs"][j+1]; prev_times=index.previous_departures(leg,a.arr+model.MIN_MOVE_S,b.dep)
        tr=model.transfer_mixture(a.arr,b.dep,prev_times,tm,float(params["transfer_hazard"]))
        if tr is None:return None
        transfers.append((mk,tr));tll+=float(tr["loglik"])
    e=exit_t-rides[-1].arr; ell=model.truncated_logpdf(e,float(em["mu"]),float(em["sigma"]))
    if not math.isfinite(ell):return None
    return State(float(access["loglik"])+tll+ell+route_prior(route,min_cost)-penalty,int(route["rank"]),rides,access,transfers,targets,penalty)


def evaluate_route(row:dict[str,Any],route:dict[str,Any],index:LegIndex,params:dict[str,Any],min_cost:float):
    legs=route.get("ride_legs",[])
    if not legs:return []
    entry=float(row["entry_sec"]);exit_t=float(row["exit_sec"])
    # Both endpoint events may move inside +/-180 s; use a wide proposal window and let projection decide feasibility.
    first=index.between(legs[0],entry+model.MIN_MOVE_S-2*MAX_EVENT_OFFSET_S,exit_t-model.MIN_MOVE_S+MAX_EVENT_OFFSET_S)
    states=[]
    for r in first:
        st=score_projected(row,route,[r],index,params,min_cost)
        if st is not None:states.append(st)
    states.sort(key=lambda x:x.loglik,reverse=True);states=states[:BEAM]
    for j,leg in enumerate(legs[1:]):
        nxt=[]
        for st in states:
            last=st.rides[-1]
            # Reconstruct raw current ride for the existing chain from root event times, not projected times.
            raw_chain=[]
            for rr in st.rides:
                dep=float(rr.root["events"][rr.board_station]["time_s"]);arr=float(rr.root["events"][rr.alight_station]["time_s"])
                raw_chain.append(Ride(rr.root,rr.board_station,rr.alight_station,dep,arr))
            lo=last.arr+model.MIN_MOVE_S-2*MAX_EVENT_OFFSET_S;hi=exit_t-model.MIN_MOVE_S+MAX_EVENT_OFFSET_S
            for cand in index.between(leg,lo,hi):
                if cand.board_station!=last.alight_station:continue
                ns=score_projected(row,route,raw_chain+[cand],index,params,min_cost)
                if ns is not None:nxt.append(ns)
        if not nxt:return []
        nxt.sort(key=lambda x:x.loglik,reverse=True);states=nxt[:BEAM]
    return states


def iter_rows(path:Path):
    pf=pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=50000):
        yield from batch.to_pylist()


def run_shard(cohorts_path:Path,roots_path:Path,routes_path:Path,params_path:Path,out_stats:Path,shard_index:int,chains_out:Path|None):
    raw,roots,_stations,params=load_roots(roots_path,params_path);routes=load_routes(routes_path);index=LegIndex(roots)
    access_stats={};egress_stats={};transfer_stats={};root_egress={};event_proj={}
    ab=ask=tb=tsk=0.0;pm=rm=um=0.0;cc=rc=ce=0;closure_max=0.0;no_route=no_chain=0.0
    writer=pq.ParquetWriter(chains_out,model.chain_schema(),compression="zstd") if chains_out else None;buf=[]
    def flush():
        nonlocal buf
        if writer is not None and buf:writer.write_table(pa.Table.from_pylist(buf,schema=model.chain_schema()));buf=[]
    try:
        for row in iter_rows(cohorts_path):
            cc+=1;mass=float(row["passenger_mass"]);pm+=mass;rk=route_key(row);rcands=routes.get(rk)
            if not rcands:um+=mass;no_route+=mass;continue
            min_cost=min(float(r.get("base_ranking_cost_s",0.0)) for r in rcands);states=[]
            for route in rcands:
                ss=evaluate_route(row,route,index,params,min_cost);ce+=len(ss);states.extend(ss)
            if not states:um+=mass;no_chain+=mass;continue
            states.sort(key=lambda x:x.loglik,reverse=True);states=states[:FINAL_TOP_K];z=model.logsumexp(s.loglik for s in states)
            if not math.isfinite(z):um+=mass;no_chain+=mass;continue
            post=[math.exp(s.loglik-z) for s in states];rm+=mass;rc+=1
            for q,st in zip(post,states):
                w=mass*q;o=str(int(row["origin_station"]));d=str(int(row["destination_station"]));a=st.access
                model.add_moment(access_stats,o,w,float(a["elog"]),float(a["elog2"]));ab+=w;ask+=w*float(a["eskip"])
                e=float(row["exit_sec"])-st.rides[-1].arr;le=math.log(e);model.add_moment(egress_stats,d,w,le,le*le)
                model.add_root_egress(root_egress,str(st.rides[-1].root["root_id"]),st.rides[-1].alight_station,w,e)
                for mk,tr in st.transfers:
                    for c in tr["components"]:model.add_moment(transfer_stats,f"{mk}|{int(c['component'])}",w*float(c["prob"]),float(c["elog"]),float(c["elog2"]))
                    tb+=w;tsk+=w*float(tr["eskip"])
                for k,target in st.event_targets.items():
                    zz=event_proj.setdefault(k,{"w":0.0,"sum_target_offset":0.0});zz["w"]+=w;zz["sum_target_offset"]+=w*float(target)
            if writer is not None:
                bi=max(range(len(states)),key=lambda i:post[i]);st=states[bi];q=post[bi];acc=float(st.access["etime"]);iw=st.rides[0].dep-float(row["entry_sec"])-acc
                trs=[];tsum=wsum=0.0
                for j,(mk,tr) in enumerate(st.transfers):
                    comp=max(tr["components"],key=lambda x:float(x["prob"]));kt=max(model.MIN_MOVE_S,float(comp["etime"]));wait=st.rides[j+1].dep-st.rides[j].arr-kt
                    tm=params["transfers"].get(mk,model.transfer_default());pid=str(tm["components"][int(comp["component"])]["path_id"]);trs.append({"movement":mk,"path_id":pid,"transfer_time_s":kt,"wait_time_s":wait});tsum+=kt;wsum+=wait
                rides=[];rt=0.0
                for r in st.rides:rides.append({"service_id":str(r.root["root_id"]),"board_station":int(r.board_station),"alight_station":int(r.alight_station),"board_time_s":r.dep,"alight_time_s":r.arr});rt+=r.arr-r.dep
                egr=float(row["exit_sec"])-st.rides[-1].arr;total=acc+iw+rt+tsum+wsum+egr;obs=float(row["exit_sec"])-float(row["entry_sec"]);cerr=total-obs;closure_max=max(closure_max,abs(cerr))
                buf.append({"cohort_id":cohort_id(row),"origin_station":int(row["origin_station"]),"destination_station":int(row["destination_station"]),"entry_sec":float(row["entry_sec"]),"exit_sec":float(row["exit_sec"]),"passenger_mass":mass,"route_rank":int(st.route_rank),"root_chain":">".join(str(r.root["root_id"]) for r in st.rides),"posterior_probability":float(q),"access_time_s":acc,"initial_wait_s":iw,"egress_time_s":egr,"transfer_assignments_json":json.dumps(trs,ensure_ascii=False,separators=(",",":")),"ride_assignments_json":json.dumps(rides,ensure_ascii=False,separators=(",",":")),"travel_time_closure_error_s":cerr})
                if len(buf)>=20000:flush()
        flush()
    finally:
        if writer is not None:writer.close()
    result={"schema":SCHEMA,"service_date":raw.get("source_date","2019-01-04"),"shard_index":shard_index,"parameter_iteration":int(params.get("iteration",0)),"passenger_mass":pm,"resolved_mass":rm,"unresolved_mass":um,"cohort_count":cc,"resolved_cohort_count":rc,"candidate_evaluations":ce,"support_top_k":FINAL_TOP_K,"support_truncated_cohort_count":0,"access_stats":access_stats,"egress_stats":egress_stats,"transfer_stats":transfer_stats,"root_egress_stats":root_egress,"event_projection_stats":event_proj,"hazard_stats":{"access_boards":ab,"access_skips":ask,"transfer_boards":tb,"transfer_skips":tsk},"diagnostics":{"max_abs_map_chain_time_closure_error_s":closure_max,"no_route_mass":no_route,"no_feasible_chain_mass":no_chain},"semantics":{"raw_afc_cohorts_used_directly":True,"legacy_e0_genealogy_not_used":True,"passenger_constraints_projected_to_event_level_service_times":True,"projected_event_offsets_are_shared_m_step_evidence_not_final_passenger_specific_truth":True}}
    out_stats.write_text(json.dumps(result,ensure_ascii=False,separators=(",",":")),encoding="utf-8");print(json.dumps({"shard_index":shard_index,"passenger_mass":pm,"resolved_mass":rm,"resolved_share":rm/pm if pm else None,"cohort_count":cc,"resolved_cohort_count":rc,"candidate_evaluations":ce,"event_projection_anchor_count":len(event_proj),"no_route_mass":no_route,"no_feasible_chain_mass":no_chain},ensure_ascii=False,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--cohorts",type=Path,required=True);p.add_argument("--roots",type=Path,required=True);p.add_argument("--routes",type=Path,required=True);p.add_argument("--params",type=Path,required=True);p.add_argument("--out-stats",type=Path,required=True);p.add_argument("--shard-index",type=int,required=True);p.add_argument("--chains-out",type=Path);a=p.parse_args();run_shard(a.cohorts,a.roots,a.routes,a.params,a.out_stats,a.shard_index,a.chains_out)

if __name__=="__main__":main()
