from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA="mppd.r1-joint-reconstruction.v1"
REPORT_SCHEMA="mppd.r1-joint-reconstruction-validation.v1"
MIN_MOVE_S=5.0
MIN_PATHS=2
TOL=1e-6

def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(float(x))
def interval(x):
    if not isinstance(x,(list,tuple)) or len(x)!=2 or not all(finite(v) for v in x): return None
    return float(x[0]),float(x[1])
def ekey(service,station): return str(service),str(station)
def mkey(station,fl,fd,tl,td): return tuple(map(str,(station,fl,fd,tl,td)))

def validate(doc:dict[str,Any], tol=TOL):
    vc=defaultdict(int); ex=defaultdict(list)
    def bad(code,msg):
        vc[code]+=1
        if len(ex[code])<20: ex[code].append(msg)

    if doc.get("schema")!=SCHEMA: bad("schema",f"expected {SCHEMA}")
    sem=doc.get("semantics",{})
    for k,v in {
        "jointly_estimated":True,
        "actual_timetable_is_latent":True,
        "planned_timetable_is_soft_prior":True,
        "r1_split_into_bcd":False}.items():
        if sem.get(k) is not v: bad("semantics",f"{k} must be {v}")

    con=doc.get("constraints",{})
    min_move=max(MIN_MOVE_S,float(con.get("minimum_physical_move_s",MIN_MOVE_S)))
    min_paths=max(MIN_PATHS,int(con.get("minimum_transfer_paths_per_movement",MIN_PATHS)))

    inv=doc.get("inventory",{})
    stations={str(x) for x in inv.get("station_ids",[])}
    events={ekey(x["service_id"],x["station_id"]) for x in inv.get("service_event_keys",[])
            if isinstance(x,dict) and {"service_id","station_id"}<=x.keys()}
    moves={mkey(x["station_id"],x["from_line_id"],x["from_direction_id"],x["to_line_id"],x["to_direction_id"])
           for x in inv.get("transfer_movements",[]) if isinstance(x,dict) and
           {"station_id","from_line_id","from_direction_id","to_line_id","to_direction_id"}<=x.keys()}
    if not stations: bad("inventory","complete station_ids required")
    if not events: bad("inventory","complete service_event_keys required")
    if not moves: bad("inventory","complete transfer_movements required")

    # Realized timetable: every declared train/station event must be reconstructed.
    ei={}; smeta={}; seqs=defaultdict(list)
    for i,r in enumerate(doc.get("realized_service_timetable",[])):
        req={"service_id","station_id","line_id","direction_id","arrival_time_s","departure_time_s"}
        if not isinstance(r,dict) or not req<=r.keys():
            bad("service_event",f"row {i} missing fields"); continue
        key=ekey(r["service_id"],r["station_id"])
        if key in ei: bad("service_duplicate",str(key)); continue
        if not finite(r["arrival_time_s"]) or not finite(r["departure_time_s"]):
            bad("service_time",f"{key} non-finite"); continue
        if float(r["departure_time_s"])+tol<float(r["arrival_time_s"]):
            bad("service_time",f"{key} dep<arr")
        ei[key]=r; seqs[str(r["service_id"])].append(r)
        meta=(str(r["line_id"]),str(r["direction_id"])); sid=str(r["service_id"])
        if sid in smeta and smeta[sid]!=meta: bad("service_metadata",f"{sid} line/direction changed")
        smeta[sid]=meta
    for x in sorted(events-set(ei)): bad("service_coverage",f"missing {x}")
    for x in sorted(set(ei)-events): bad("service_coverage",f"undeclared {x}")
    for sid,rows in seqs.items():
        if rows and all("sequence_index" in r for r in rows):
            rows=sorted(rows,key=lambda r:int(r["sequence_index"]))
            for a,b in zip(rows,rows[1:]):
                if float(b["arrival_time_s"])+tol<float(a["arrival_time_s"]): bad("service_monotonicity",sid)
                if float(b["departure_time_s"])+tol<float(a["departure_time_s"]): bad("service_monotonicity",sid)

    # Every station must have its own access and egress interval.
    si={}
    for i,r in enumerate(doc.get("station_movement_intervals",[])):
        if not isinstance(r,dict) or "station_id" not in r: bad("station_interval",f"row {i}"); continue
        s=str(r["station_id"])
        if s in si: bad("station_duplicate",s); continue
        si[s]=r
        for f in ("access_interval_s","egress_interval_s"):
            iv=interval(r.get(f))
            if iv is None: bad("station_interval",f"{s}.{f} invalid"); continue
            lo,hi=iv
            if lo<min_move-tol: bad("physical_lower_bound",f"{s}.{f} lower={lo}")
            if hi+tol<lo: bad("station_interval",f"{s}.{f} upper<lower")
    for s in sorted(stations-set(si)): bad("station_coverage",f"missing {s}")
    for s in sorted(set(si)-stations): bad("station_coverage",f"undeclared {s}")

    # Every transfer movement must expose multiple internal paths, each with its own interval.
    ti={}
    for i,r in enumerate(doc.get("transfer_path_intervals",[])):
        req={"station_id","from_line_id","from_direction_id","to_line_id","to_direction_id"}
        if not isinstance(r,dict) or not req<=r.keys(): bad("transfer_interval",f"row {i}"); continue
        k=mkey(r["station_id"],r["from_line_id"],r["from_direction_id"],r["to_line_id"],r["to_direction_id"])
        if k in ti: bad("transfer_duplicate",str(k)); continue
        paths={}
        for p in r.get("paths",[]):
            if not isinstance(p,dict) or "path_id" not in p: bad("transfer_path",f"{k} missing path_id"); continue
            pid=str(p["path_id"]); iv=interval(p.get("transfer_interval_s"))
            if pid in paths: bad("transfer_path_duplicate",f"{k}/{pid}")
            if iv is None: bad("transfer_path",f"{k}/{pid} invalid interval")
            else:
                lo,hi=iv
                if lo<min_move-tol: bad("physical_lower_bound",f"{k}/{pid} lower={lo}")
                if hi+tol<lo: bad("transfer_path",f"{k}/{pid} upper<lower")
            paths[pid]=p
        if len(paths)<min_paths: bad("transfer_path_multiplicity",f"{k} has {len(paths)} < {min_paths}")
        ti[k]=paths
    for k in sorted(moves-set(ti)): bad("transfer_coverage",f"missing {k}")
    for k in sorted(set(ti)-moves): bad("transfer_coverage",f"undeclared {k}")

    # Passenger chains must jointly reconcile timetable, station intervals and transfer paths.
    n=0; mass=0.0; maxerr=0.0; minwait=math.inf
    for i,p in enumerate(doc.get("passenger_chains",[])):
        if not isinstance(p,dict): bad("passenger_chain",f"row {i}"); continue
        n+=1; pid=str(p.get("passenger_id",p.get("cohort_id",i)))
        w=p.get("mass",1.0)
        if not finite(w) or float(w)<=0: bad("passenger_mass",pid); w=0
        mass+=float(w)
        req={"origin_station_id","destination_station_id","entry_time_s","exit_time_s","access_time_s","egress_time_s","rides"}
        if not req<=p.keys(): bad("passenger_chain",f"{pid} missing fields"); continue
        vals=[p["entry_time_s"],p["exit_time_s"],p["access_time_s"],p["egress_time_s"]]
        if not all(finite(x) for x in vals): bad("passenger_time",pid); continue
        ent,out,acc,egr=map(float,vals); o=str(p["origin_station_id"]); d=str(p["destination_station_id"])
        if out+tol<ent: bad("passenger_time",f"{pid} exit<entry")
        if acc<min_move-tol: bad("physical_lower_bound",f"{pid} access={acc}")
        if egr<min_move-tol: bad("physical_lower_bound",f"{pid} egress={egr}")
        for s,f,x in ((o,"access_interval_s",acc),(d,"egress_interval_s",egr)):
            if s not in si: bad("station_reference",f"{pid}:{s}")
            else:
                iv=interval(si[s].get(f))
                if iv and not(iv[0]-tol<=x<=iv[1]+tol): bad("station_membership",f"{pid}:{s}:{x} not in {iv}")

        rides=p["rides"]; trs=p.get("transfers",[])
        if not isinstance(rides,list) or not rides: bad("passenger_chain",f"{pid} no rides"); continue
        if not isinstance(trs,list) or len(trs)!=len(rides)-1: bad("passenger_chain",f"{pid} ride/transfer count mismatch"); continue
        rr=[]
        for j,r in enumerate(rides):
            if not isinstance(r,dict) or not {"service_id","board_station_id","alight_station_id"}<=r.keys():
                bad("ride",f"{pid}:{j}"); continue
            sid,b,a=map(str,(r["service_id"],r["board_station_id"],r["alight_station_id"]))
            be,ae=ei.get(ekey(sid,b)),ei.get(ekey(sid,a))
            if be is None or ae is None: bad("ride_reference",f"{pid}:{sid}:{b}->{a}"); continue
            dep,arr=float(be["departure_time_s"]),float(ae["arrival_time_s"])
            if arr+tol<dep: bad("ride_time",f"{pid}:{j}")
            rr.append((sid,b,a,str(be["line_id"]),str(be["direction_id"]),dep,arr))
        if len(rr)!=len(rides): continue

        fw=rr[0][5]-(ent+acc); minwait=min(minwait,fw)
        if fw<-tol: bad("negative_wait",f"{pid}:first={fw}")
        tsum=wsum=0.0
        for j,(prev,nxt) in enumerate(zip(rr,rr[1:])):
            if prev[2]!=nxt[1]: bad("route_continuity",f"{pid}:{prev[2]}!={nxt[1]}"); continue
            tr=trs[j]
            if not isinstance(tr,dict) or "path_id" not in tr or not finite(tr.get("transfer_time_s")):
                bad("transfer_assignment",f"{pid}:{j}"); continue
            kt=float(tr["transfer_time_s"]); path=str(tr["path_id"])
            if kt<min_move-tol: bad("physical_lower_bound",f"{pid}:transfer{j}={kt}")
            mk=mkey(prev[2],prev[3],prev[4],nxt[3],nxt[4]); paths=ti.get(mk)
            if paths is None: bad("transfer_reference",f"{pid}:{mk}")
            elif path not in paths: bad("transfer_reference",f"{pid}:{mk}/{path}")
            else:
                iv=interval(paths[path].get("transfer_interval_s"))
                if iv and not(iv[0]-tol<=kt<=iv[1]+tol): bad("transfer_membership",f"{pid}:{mk}/{path}:{kt}")
            ww=nxt[5]-prev[6]-kt; minwait=min(minwait,ww)
            if ww<-tol: bad("negative_wait",f"{pid}:transfer{j}={ww}")
            tsum+=kt; wsum+=ww
        expected=out-rr[-1][6]
        if abs(expected-egr)>tol: bad("egress_consistency",f"{pid}:{egr}!={expected}")
        ride=sum(x[6]-x[5] for x in rr)
        err=acc+fw+ride+tsum+wsum+egr-(out-ent); maxerr=max(maxerr,abs(err))
        if abs(err)>tol: bad("time_closure",f"{pid}:{err}")

    if n==0: bad("passenger_coverage","no passenger chains")

    gates={
      "joint_semantics":vc["schema"]==vc["semantics"]==0,
      "complete_realized_service_timetable":sum(vc[x] for x in ("inventory","service_event","service_duplicate","service_time","service_coverage","service_monotonicity"))==0,
      "complete_station_access_egress_intervals":sum(vc[x] for x in ("station_interval","station_duplicate","station_coverage"))==0,
      "complete_multi_path_transfer_intervals":sum(vc[x] for x in ("transfer_interval","transfer_duplicate","transfer_path","transfer_path_duplicate","transfer_path_multiplicity","transfer_coverage"))==0,
      "five_second_physical_lower_bound":vc["physical_lower_bound"]==0,
      "nonnegative_waiting":vc["negative_wait"]==0,
      "passenger_chain_references":sum(vc[x] for x in ("station_reference","ride_reference","transfer_reference","route_continuity"))==0,
      "interval_membership":vc["station_membership"]==vc["transfer_membership"]==0,
      "complete_travel_time_closure":vc["time_closure"]==vc["egress_consistency"]==vc["passenger_time"]==0,
    }
    passed=all(gates.values()) and sum(vc.values())==0
    return {
      "schema":REPORT_SCHEMA,
      "status":"QUALIFIED_R1_JOINT_RECONSTRUCTION" if passed else "R1_JOINT_RECONSTRUCTION_NOT_QUALIFIED",
      "constraints":{"minimum_physical_move_s":min_move,"minimum_transfer_paths_per_movement":min_paths,"time_tolerance_s":tol},
      "coverage":{"expected_station_count":len(stations),"reconstructed_station_count":len(si),
                  "expected_service_event_count":len(events),"reconstructed_service_event_count":len(ei),
                  "expected_transfer_movement_count":len(moves),"reconstructed_transfer_movement_count":len(ti),
                  "reconstructed_transfer_path_count":sum(map(len,ti.values())),
                  "passenger_chain_count":n,"passenger_mass":mass},
      "diagnostics":{"max_abs_travel_time_closure_error_s":maxerr,
                     "minimum_derived_wait_s":None if math.isinf(minwait) else minwait},
      "qualification_gates":gates,
      "violation_counts":dict(sorted((k,v) for k,v in vc.items() if v)),
      "violation_examples":dict(sorted(ex.items()))
    }

def fixture():
    return {
      "schema":SCHEMA,
      "semantics":{"jointly_estimated":True,"actual_timetable_is_latent":True,
                   "planned_timetable_is_soft_prior":True,"r1_split_into_bcd":False},
      "constraints":{"minimum_physical_move_s":5.0,"minimum_transfer_paths_per_movement":2},
      "inventory":{"station_ids":["O","X","D"],
        "service_event_keys":[{"service_id":"S1","station_id":"O"},{"service_id":"S1","station_id":"X"},
                              {"service_id":"S2","station_id":"X"},{"service_id":"S2","station_id":"D"}],
        "transfer_movements":[{"station_id":"X","from_line_id":"L1","from_direction_id":"UP",
                               "to_line_id":"L2","to_direction_id":"DOWN"}]},
      "realized_service_timetable":[
        {"service_id":"S1","station_id":"O","line_id":"L1","direction_id":"UP","sequence_index":0,"arrival_time_s":90,"departure_time_s":100},
        {"service_id":"S1","station_id":"X","line_id":"L1","direction_id":"UP","sequence_index":1,"arrival_time_s":200,"departure_time_s":210},
        {"service_id":"S2","station_id":"X","line_id":"L2","direction_id":"DOWN","sequence_index":0,"arrival_time_s":215,"departure_time_s":230},
        {"service_id":"S2","station_id":"D","line_id":"L2","direction_id":"DOWN","sequence_index":1,"arrival_time_s":330,"departure_time_s":340}],
      "station_movement_intervals":[
        {"station_id":"O","access_interval_s":[5,40],"egress_interval_s":[5,60]},
        {"station_id":"X","access_interval_s":[5,50],"egress_interval_s":[5,50]},
        {"station_id":"D","access_interval_s":[5,40],"egress_interval_s":[5,60]}],
      "transfer_path_intervals":[{"station_id":"X","from_line_id":"L1","from_direction_id":"UP",
        "to_line_id":"L2","to_direction_id":"DOWN","paths":[
          {"path_id":"P1","transfer_interval_s":[5,20]},{"path_id":"P2","transfer_interval_s":[12,35]}]}],
      "passenger_chains":[{"passenger_id":"P","mass":1,"origin_station_id":"O","destination_station_id":"D",
        "entry_time_s":70,"exit_time_s":350,"access_time_s":20,"egress_time_s":20,
        "rides":[{"service_id":"S1","board_station_id":"O","alight_station_id":"X"},
                 {"service_id":"S2","board_station_id":"X","alight_station_id":"D"}],
        "transfers":[{"path_id":"P1","transfer_time_s":10}]}]
    }

def main():
    ap=argparse.ArgumentParser()
    sp=ap.add_subparsers(dest="cmd",required=True)
    v=sp.add_parser("validate"); v.add_argument("--input",type=Path,required=True); v.add_argument("--out",type=Path)
    sp.add_parser("self-test")
    a=ap.parse_args()
    if a.cmd=="self-test":
        r=validate(fixture()); assert r["status"]=="QUALIFIED_R1_JOINT_RECONSTRUCTION",r
        x=fixture(); x["transfer_path_intervals"][0]["paths"][0]["transfer_interval_s"][0]=1
        r=validate(x); assert r["status"]!="QUALIFIED_R1_JOINT_RECONSTRUCTION",r
        print("R1 joint reconstruction validator self-test PASS"); return 0
    doc=json.loads(a.input.read_text(encoding="utf-8")); r=validate(doc)
    text=json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True)
    if a.out: a.out.write_text(text+"\n",encoding="utf-8")
    print(text); return 0 if r["status"]=="QUALIFIED_R1_JOINT_RECONSTRUCTION" else 2

if __name__=="__main__": raise SystemExit(main())
