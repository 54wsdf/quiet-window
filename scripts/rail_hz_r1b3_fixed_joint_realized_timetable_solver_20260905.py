from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.linalg import splu

import scripts.rail_hz_r1b3_direct_realized_timetable_20260905 as prior

SCHEMA = "rail.hz-r1b3-fixed-joint-realized-timetable.v1"
MIN_PROGRESS_S = 5.0
Z90 = 1.6448536269514722
Z95 = 1.959963984540054
TAU_ACCESS_S = 180.0
TAU_EGRESS_S = 120.0
TAU_TRANSFER_S = 180.0
HUTCHINSON_PROBES = 32
RNG_SEED = 20260905


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_root_mass(path: Path) -> dict[str, float]:
    table = pq.read_table(path, columns=["root_id", "lineage_usage_mass"])
    return {str(r["root_id"]): float(r["lineage_usage_mass"]) for r in table.to_pylist()}


def physical_key(root: dict[str, Any]) -> tuple[str, str, int, float]:
    return (
        str(root["afc_line"]), str(root["direction"]), int(root["terminal_station"]),
        round(float(root["terminal_anchor_s"]), 6),
    )


def build_model(roots: list[dict[str, Any]], root_mass: dict[str, float]):
    groups: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for r in roots:
        groups[physical_key(r)].append(r)

    root_to_gid: dict[str, str] = {}
    family_weight: dict[str, float] = {}
    group_members: dict[str, list[dict[str, Any]]] = {}
    group_meta: dict[str, dict[str, Any]] = {}

    for key, members in sorted(groups.items(), key=lambda kv: kv[0]):
        gid = prior.stable_id("phys_", key)
        masses = np.asarray([max(0.0, root_mass.get(str(r["root_id"]), 0.0)) for r in members], dtype=float)
        if masses.sum() > 0:
            probs = masses / masses.sum()
        else:
            probs = np.full(len(members), 1.0 / len(members))
        group_members[gid] = members
        for r, p in zip(members, probs):
            rid = str(r["root_id"])
            root_to_gid[rid] = gid
            family_weight[rid] = float(p)
        group_meta[gid] = {
            "key": key,
            "member_count": len(members),
            "max_family_weight": float(probs.max()),
            "family_entropy_nats": float(-sum(float(p) * math.log(max(float(p), 1e-300)) for p in probs)),
        }

    # Each physical-service union is a tree: a singleton line or the two B-Up branches
    # converging into one shared trunk. Parameterize by one free anchor time plus positive edge gaps.
    node_keys: list[tuple[str, int]] = []
    node_index: dict[tuple[str, int], int] = {}
    root_station_to_node: dict[tuple[str, int], int] = {}
    row_coeffs: list[dict[int, float]] = []
    p0: list[float] = []
    bounds: list[tuple[float | None, float | None]] = []
    gap_param_meta: list[dict[str, Any]] = []
    node_components: dict[tuple[str, int], list[str]] = defaultdict(list)

    for gid, members in sorted(group_members.items()):
        stations: set[int] = set()
        directed_edges: dict[frozenset[int], tuple[int, int, float]] = {}
        seed_by_station: dict[int, list[float]] = defaultdict(list)
        for r in members:
            rid = str(r["root_id"])
            evs = list(r["events"])
            for ev in evs:
                st = int(ev["station"])
                stations.add(st)
                seed_by_station[st].append(float(ev["time_s"]))
                node_components[(gid, st)].append(rid)
            for left, right in zip(evs[:-1], evs[1:]):
                u, v = int(left["station"]), int(right["station"])
                gap = float(right["time_s"]) - float(left["time_s"])
                if gap < MIN_PROGRESS_S - 1e-6:
                    raise SystemExit(f"seed root violates forward time: {rid} {u}->{v} gap={gap}")
                k = frozenset((u, v))
                if k in directed_edges:
                    old = directed_edges[k]
                    if old[0] != u or old[1] != v:
                        raise SystemExit(f"contradictory physical edge orientation in {gid}: {old} vs {(u,v,gap)}")
                else:
                    directed_edges[k] = (u, v, gap)
        if len(directed_edges) != len(stations) - 1:
            raise SystemExit(f"physical service union is not a tree: {gid} nodes={len(stations)} edges={len(directed_edges)}")

        anchor_station = int(group_meta[gid]["key"][2])
        if anchor_station not in stations:
            raise SystemExit(f"terminal anchor station missing from physical group {gid}")
        anchor_param = len(p0)
        anchor_seed = float(np.mean(seed_by_station[anchor_station]))
        p0.append(anchor_seed)
        bounds.append((None, None))

        edge_param: dict[frozenset[int], int] = {}
        adjacency: dict[int, list[int]] = defaultdict(list)
        for k, (u, v, gap) in sorted(directed_edges.items(), key=lambda kv: tuple(sorted(kv[0]))):
            pi = len(p0)
            edge_param[k] = pi
            p0.append(max(MIN_PROGRESS_S, gap))
            bounds.append((MIN_PROGRESS_S, None))
            gap_param_meta.append({"param_index": pi, "physical_service_id": gid, "from_station": u, "to_station": v})
            adjacency[u].append(v); adjacency[v].append(u)

        coeff_by_station: dict[int, dict[int, float]] = {anchor_station: {anchor_param: 1.0}}
        q = deque([anchor_station])
        seen = {anchor_station}
        while q:
            cur = q.popleft()
            for nbr in adjacency[cur]:
                if nbr in seen:
                    continue
                k = frozenset((cur, nbr))
                pi = edge_param[k]
                u, v, _gap = directed_edges[k]
                sign = 1.0 if (cur == u and nbr == v) else -1.0
                c = dict(coeff_by_station[cur]); c[pi] = c.get(pi, 0.0) + sign
                coeff_by_station[nbr] = c
                seen.add(nbr); q.append(nbr)
        if seen != stations:
            raise SystemExit(f"physical service tree disconnected: {gid}")

        for st in sorted(stations):
            ni = len(node_keys)
            node_keys.append((gid, st)); node_index[(gid, st)] = ni; row_coeffs.append(coeff_by_station[st])
        for r in members:
            rid = str(r["root_id"])
            for ev in r["events"]:
                root_station_to_node[(rid, int(ev["station"]))] = node_index[(gid, int(ev["station"]))]

    if len(node_keys) != len(p0):
        raise SystemExit(f"tree parameterization should be square: nodes={len(node_keys)} params={len(p0)}")
    rows=[]; cols=[]; vals=[]
    for ri, c in enumerate(row_coeffs):
        for ci, v in c.items(): rows.append(ri); cols.append(ci); vals.append(v)
    A = csr_matrix((vals, (rows, cols)), shape=(len(node_keys), len(p0)), dtype=float)
    return {
        "groups": groups, "group_members": group_members, "group_meta": group_meta,
        "root_to_gid": root_to_gid, "family_weight": family_weight,
        "node_keys": node_keys, "node_index": node_index, "root_station_to_node": root_station_to_node,
        "node_components": node_components, "A": A, "p0": np.asarray(p0, dtype=float), "bounds": bounds,
        "gap_param_meta": gap_param_meta,
    }


def append_batches(paths: list[Path], columns: list[str]):
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=columns, batch_size=100000):
            yield batch.to_pydict()


def map_unary(paths: list[Path], model, kind: str):
    idx=[]; obs=[]; mass=[]
    time_col = "entry_sec" if kind == "access" else "exit_sec"
    for b in append_batches(paths, ["root_id", "station", time_col, "lineage_mass"]):
        n=len(b["root_id"])
        for k in range(n):
            key=(str(b["root_id"][k]), int(b["station"][k]))
            ni=model["root_station_to_node"].get(key)
            if ni is None: raise SystemExit(f"unmapped {kind} factor event: {key}")
            idx.append(ni); obs.append(float(b[time_col][k])); mass.append(float(b["lineage_mass"][k]))
    return np.asarray(idx,np.int32), np.asarray(obs,float), np.asarray(mass,float)


def map_transfer(paths: list[Path], model):
    lo=[]; hi=[]; mass=[]
    for b in append_batches(paths, ["lower_root","lower_station","upper_root","upper_station","lineage_mass"]):
        n=len(b["lower_root"])
        for k in range(n):
            lk=(str(b["lower_root"][k]),int(b["lower_station"][k])); hk=(str(b["upper_root"][k]),int(b["upper_station"][k]))
            li=model["root_station_to_node"].get(lk); hi_i=model["root_station_to_node"].get(hk)
            if li is None or hi_i is None: raise SystemExit(f"unmapped transfer factor: {lk}->{hk}")
            lo.append(li); hi.append(hi_i); mass.append(float(b["lineage_mass"][k]))
    return np.asarray(lo,np.int32),np.asarray(hi,np.int32),np.asarray(mass,float)


def structural_factors(roots: list[dict[str,Any]], service_init: dict[str,Any], model):
    ctx=prior.contexts_by_edge(service_init)
    pu_i=[]; pu_y=[]; pu_c=[]
    tr_i=[]; tr_j=[]; tr_lag=[]; tr_c=[]
    for r in roots:
        rid=str(r["root_id"]); fw=float(model["family_weight"][rid]); evs=list(r["events"])
        for ev in evs:
            if bool(ev.get("matched_observed_pulse",False)):
                ni=model["root_station_to_node"][(rid,int(ev["station"]))]
                sd=max(1e-6,float(ev["sd_s"])); pu_i.append(ni); pu_y.append(float(ev["time_s"])); pu_c.append(fw/(sd*sd))
        for left,right in zip(evs[:-1],evs[1:]):
            lag,sd,_eclass=prior.transition_factor(str(r["path_id"]),str(r["direction"]),left,right,ctx)
            i=model["root_station_to_node"][(rid,int(left["station"]))]; j=model["root_station_to_node"][(rid,int(right["station"]))]
            tr_i.append(i); tr_j.append(j); tr_lag.append(lag); tr_c.append(fw/(sd*sd))
    return tuple(np.asarray(x,dtype=(np.int32 if n in (0,3) else float)) for n,x in enumerate((pu_i,pu_y,pu_c,tr_i,tr_j,tr_lag,tr_c)))


def add_at(v: np.ndarray, idx: np.ndarray, values: np.ndarray) -> None:
    np.add.at(v, idx, values)


def soft_order_obj_grad(t, access, egress, transfer):
    gi=np.zeros_like(t); obj=0.0
    ai,entry,am=access
    g=t[ai]-entry; z=-g/TAU_ACCESS_S; obj += float(np.sum(am*np.logaddexp(0.0,z))); d=-(am/TAU_ACCESS_S)*expit(z); add_at(gi,ai,d)
    ei,exit_t,em=egress
    g=exit_t-t[ei]; z=-g/TAU_EGRESS_S; obj += float(np.sum(em*np.logaddexp(0.0,z))); d=(em/TAU_EGRESS_S)*expit(z); add_at(gi,ei,d)
    li,ui,tm=transfer
    g=t[ui]-t[li]; z=-g/TAU_TRANSFER_S; obj += float(np.sum(tm*np.logaddexp(0.0,z))); s=(tm/TAU_TRANSFER_S)*expit(z); add_at(gi,li,s); add_at(gi,ui,-s)
    return obj,gi


def solve(a: argparse.Namespace):
    roots_payload=load_json(a.roots); service_init=load_json(a.service_init)
    if roots_payload.get("status")!="QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION": raise SystemExit("roots not qualified")
    if service_init.get("status")!="QUALIFIED_SINGLE_FULL_SERVICE_DAY_SERVICE_INIT": raise SystemExit("service init not qualified")
    roots=list(roots_payload["roots"]); root_mass=load_root_mass(a.root_mass); model=build_model(roots,root_mass)
    if len(model["groups"])!=1438 or len(model["node_keys"])!=38649: raise SystemExit("physical service model inventory drift")

    access=map_unary(a.access,model,"access"); egress=map_unary(a.egress,model,"egress"); transfer=map_transfer(a.transfer,model)
    pu_i,pu_y,pu_c,tr_i,tr_j,tr_lag,tr_c=structural_factors(roots,service_init,model)
    A=model["A"]

    factor_mass={"access":float(access[2].sum()),"egress":float(egress[2].sum()),"transfer":float(transfer[2].sum())}
    if abs(factor_mass["access"]-932464.0)>1e-5 or abs(factor_mass["egress"]-932464.0)>1e-5 or abs(factor_mass["transfer"]-319931.0069536726)>1e-4:
        raise SystemExit("global fixed factor mass mismatch: "+json.dumps(factor_mass))

    eval_counter={"n":0}
    def objective(p):
        eval_counter["n"]+=1; t=A@p; gt=np.zeros_like(t); obj=0.0
        r=t[pu_i]-pu_y; obj += 0.5*float(np.sum(pu_c*r*r)); add_at(gt,pu_i,pu_c*r)
        r=t[tr_j]-t[tr_i]-tr_lag; obj += 0.5*float(np.sum(tr_c*r*r)); add_at(gt,tr_i,-tr_c*r); add_at(gt,tr_j,tr_c*r)
        oo,go=soft_order_obj_grad(t,access,egress,transfer); obj += oo; gt += go
        gp=A.T@gt
        return obj,np.asarray(gp).ravel()

    p0=model["p0"].copy(); initial_obj=float(objective(p0)[0])
    opt=minimize(lambda p: objective(p),p0,jac=True,method="L-BFGS-B",bounds=model["bounds"],options={"maxiter":400,"ftol":1e-12,"gtol":1e-5,"maxls":50,"maxcor":20})
    if not opt.success and opt.status not in (1,2): raise SystemExit(f"fixed joint optimization failed: {opt.status} {opt.message}")
    p=np.asarray(opt.x,float); t=np.asarray(A@p).ravel(); final_obj=float(objective(p)[0])
    active_gaps=[m for m in model["gap_param_meta"] if p[int(m["param_index"])]<=MIN_PROGRESS_S+1e-5]

    # Local curvature in physical event-time coordinates. It is sparse even though the positive-gap
    # parameterization is path-cumulative. Hard-bound conditioning is left to R1D calibration.
    hr=[]; hc=[]; hv=[]
    def diag_add(idx,c):
        hr.extend(idx.tolist()); hc.extend(idx.tolist()); hv.extend(c.tolist())
    diag_add(pu_i,pu_c)
    hr.extend(tr_i.tolist());hc.extend(tr_i.tolist());hv.extend(tr_c.tolist());hr.extend(tr_j.tolist());hc.extend(tr_j.tolist());hv.extend(tr_c.tolist());hr.extend(tr_i.tolist());hc.extend(tr_j.tolist());hv.extend((-tr_c).tolist());hr.extend(tr_j.tolist());hc.extend(tr_i.tolist());hv.extend((-tr_c).tolist())
    ai,entry,am=access; g=t[ai]-entry; s=expit(-g/TAU_ACCESS_S); diag_add(ai,am/(TAU_ACCESS_S**2)*s*(1-s))
    ei,exit_t,em=egress; g=exit_t-t[ei]; s=expit(-g/TAU_EGRESS_S); diag_add(ei,em/(TAU_EGRESS_S**2)*s*(1-s))
    li,ui,tm=transfer; g=t[ui]-t[li]; s=expit(-g/TAU_TRANSFER_S); c=tm/(TAU_TRANSFER_S**2)*s*(1-s)
    hr.extend(li.tolist());hc.extend(li.tolist());hv.extend(c.tolist());hr.extend(ui.tolist());hc.extend(ui.tolist());hv.extend(c.tolist());hr.extend(li.tolist());hc.extend(ui.tolist());hv.extend((-c).tolist());hr.extend(ui.tolist());hc.extend(li.tolist());hv.extend((-c).tolist())
    n=len(t); H=coo_matrix((np.asarray(hv,float),(np.asarray(hr,int),np.asarray(hc,int))),shape=(n,n)).tocsr(); H=H+eye(n,format="csr")*1e-10
    Hdiag=np.asarray(H.diagonal()).ravel(); lu=splu(H.tocsc())
    rng=np.random.default_rng(RNG_SEED); diag_est=np.zeros(n,float)
    for _ in range(HUTCHINSON_PROBES):
        z=rng.choice(np.array([-1.0,1.0]),size=n); x=lu.solve(z); diag_est += z*x
    diag_est/=HUTCHINSON_PROBES
    lower=np.where(Hdiag>0,1.0/Hdiag,0.0); var=np.maximum.reduce([diag_est,lower,np.full(n,1e-8)]); sd=np.sqrt(var)

    # Build path-family/existence metadata and outputs.
    node_existence=np.zeros(n,float); node_afc_weight=np.zeros(n,float); node_components: dict[int,list[tuple[str,float,str,bool]]]=defaultdict(list)
    for r in roots:
        rid=str(r["root_id"]); fw=float(model["family_weight"][rid])
        for ev in r["events"]:
            ni=model["root_station_to_node"][(rid,int(ev["station"]))]
            node_components[ni].append((rid,fw,str(r["path_id"]),bool(ev.get("matched_observed_pulse",False))))
    for ni, comps in node_components.items():
        byroot={rid:(w,path,matched) for rid,w,path,matched in comps}; node_existence[ni]=min(1.0,sum(x[0] for x in byroot.values())); node_afc_weight[ni]=min(1.0,sum(x[0] for x in byroot.values() if x[2]))

    physical_rows=[]
    for ni,(gid,station) in enumerate(model["node_keys"]):
        comps=node_components[ni]; byroot={rid:(w,path,matched) for rid,w,path,matched in comps}
        physical_rows.append({"schema":SCHEMA,"service_date":"2019-01-04","physical_service_id":gid,"station":station,"event_existence_weight":float(node_existence[ni]),"realized_time_mean_s":float(t[ni]),"realized_time_sd_laplace_s":float(sd[ni]),"realized_time_q05_s":float(t[ni]-Z90*sd[ni]),"realized_time_q50_s":float(t[ni]),"realized_time_q95_s":float(t[ni]+Z90*sd[ni]),"realized_time_lower95_s":float(t[ni]-Z95*sd[ni]),"realized_time_upper95_s":float(t[ni]+Z95*sd[ni]),"afc_anchor_component_weight":float(node_afc_weight[ni]),"component_root_ids":sorted(byroot),"component_path_weights":{path:float(w) for _rid,(w,path,_m) in byroot.items()},"planned_absolute_timestamp_used":False})
    root_rows=[]
    for r in roots:
        rid=str(r["root_id"]); gid=model["root_to_gid"][rid]; fw=float(model["family_weight"][rid])
        for idx,ev in enumerate(r["events"]):
            ni=model["root_station_to_node"][(rid,int(ev["station"]))]
            root_rows.append({"schema":SCHEMA,"service_date":"2019-01-04","physical_service_id":gid,"root_id":rid,"path_id":str(r["path_id"]),"direction":str(r["direction"]),"path_family_weight_within_physical_service":fw,"event_key":prior.event_key(rid,idx),"event_index":idx,"station":int(ev["station"]),"realized_time_mean_s":float(t[ni]),"realized_time_sd_laplace_s":float(sd[ni]),"realized_time_lower95_s":float(t[ni]-Z95*sd[ni]),"realized_time_upper95_s":float(t[ni]+Z95*sd[ni]),"matched_afc_passenger_facing_pulse":bool(ev.get("matched_observed_pulse",False)),"source_seed_time_s":float(ev["time_s"]),"source_seed_is_absolute_factor":bool(ev.get("matched_observed_pulse",False)),"planned_absolute_timestamp_used":False})
    write_jsonl_gz(a.out_physical_events,physical_rows);write_jsonl_gz(a.out_root_events,root_rows)

    widths=[2*Z95*x for x in sd]; weights=node_existence.tolist(); shifts=[]; shiftw=[]
    for row in root_rows:
        shifts.append(float(row["realized_time_mean_s"])-float(row["source_seed_time_s"]));shiftw.append(float(row["path_family_weight_within_physical_service"]))
    dup=[gid for gid,m in model["group_meta"].items() if m["member_count"]>1]
    maxw=[model["group_meta"][gid]["max_family_weight"] for gid in dup]
    gates={"fixed_original_factor_mass_complete":True,"full_physical_service_inventory":len(model["groups"])==1438,"full_physical_event_inventory":len(model["node_keys"])==38649,"all_candidate_root_events_mapped":len(root_rows)==43584,"planned_absolute_timestamp_absent":True,"audit_residuals_not_used_as_input":True,"scientific_posterior_recycling_absent":True,"hard_forward_time_domain_enforced":all(p[int(m["param_index"])]>=MIN_PROGRESS_S-1e-7 for m in model["gap_param_meta"]),"optimizer_objective_improved":final_obj<=initial_obj+1e-6,"laplace_intervals_finite":bool(np.all(np.isfinite(sd)))}
    if not all(gates.values()):raise SystemExit("fixed joint solve gates failed: "+json.dumps(gates))
    result={"schema":SCHEMA,"status":"QUALIFIED_R1B3_FIXED_JOINT_REALIZED_TIMETABLE_CANDIDATE","service_date":"2019-01-04","scope":"FULL_SERVICE_DAY_0400_TO_NEXT_0400_FULL_NETWORK_FULL_QUALIFIED_PASSENGER_DOMAIN","fixed_factor_mass":factor_mass,"inventory":{"candidate_root_hypotheses":1673,"physical_service_hypotheses":1438,"physical_event_nodes":38649,"candidate_root_event_states":43584,"ambiguous_two_family_physical_services":len(dup)},"ordering_likelihood":{"family":"SOFT_LOGISTIC_ORDERING_POTENTIAL","access_softness_s":TAU_ACCESS_S,"egress_softness_s":TAU_EGRESS_S,"transfer_softness_s":TAU_TRANSFER_S,"interpretation":"Broad nuisance ordering likelihood only; it does not equate entry-to-departure with access time or train-to-train gap with physical transfer time."},"optimization":{"method":"L_BFGS_B_ON_PHYSICAL_SERVICE_TREE_ANCHORS_AND_POSITIVE_GAPS","initial_objective":initial_obj,"final_objective":final_obj,"iterations":int(opt.nit),"function_evaluations":int(opt.nfev),"message":str(opt.message),"active_minimum_progress_gap_count":len(active_gaps),"minimum_progress_s":MIN_PROGRESS_S},"realized_time_interval_width95_s":{"p10":prior.weighted_quantile(widths,weights,.1),"median":prior.weighted_quantile(widths,weights,.5),"p90":prior.weighted_quantile(widths,weights,.9)},"posterior_minus_source_seed_center_s":{"p10":prior.weighted_quantile(shifts,shiftw,.1),"median":prior.weighted_quantile(shifts,shiftw,.5),"p90":prior.weighted_quantile(shifts,shiftw,.9)},"path_family_ambiguity":{"max_family_weight_median":prior.weighted_quantile(maxw,[1.0]*len(maxw),.5),"decisive_ge_0_75_count":sum(1 for x in maxw if x>=.75),"near_tie_le_0_60_count":sum(1 for x in maxw if x<=.60)},"uncertainty":{"method":"LOCAL_LAPLACE_EVENT_TIME_HESSIAN_WITH_DETERMINISTIC_HUTCHINSON_DIAGONAL_ESTIMATOR","hutchinson_probes":HUTCHINSON_PROBES,"hard_bound_conditioning_included":False,"calibration_deferred_to_r1d":True},"qualification_gates":gates,"scientific_semantics":{"one_fixed_joint_objective":True,"source_factors":"ORIGINAL_AFC_PLUS_FROZEN_R1B1_PROBABILISTIC_GENEALOGY_PLUS_WEAK_RELATIVE_STRUCTURE","r1b4_audit_summary_used_as_evidence":False,"previous_r1b3_timetable_used_as_evidence":False,"planned_absolute_timetable_used":False,"scientific_self_iteration":False,"waiting_time_not_equated_to_access_time":True,"transfer_connection_gap_not_equated_to_physical_transfer_time":True},"next_stage":"READ_ONLY_REAUDIT_BEFORE_R1C"}
    a.out_summary.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8");print(json.dumps(result,ensure_ascii=False,indent=2));return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--roots",type=Path,required=True);ap.add_argument("--service-init",type=Path,required=True);ap.add_argument("--root-mass",type=Path,required=True);ap.add_argument("--access",type=Path,action="append",required=True);ap.add_argument("--egress",type=Path,action="append",required=True);ap.add_argument("--transfer",type=Path,action="append",required=True);ap.add_argument("--out-physical-events",type=Path,required=True);ap.add_argument("--out-root-events",type=Path,required=True);ap.add_argument("--out-summary",type=Path,required=True);solve(ap.parse_args())

if __name__=="__main__":main()
