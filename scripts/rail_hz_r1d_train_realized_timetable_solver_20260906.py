from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu

import rail_hz_r1b3_fixed_joint_realized_timetable_solver_20260905 as b

SCHEMA = "rail.hz-r1d-train-realized-timetable-refit.v1"


def train_structural_factors(roots, service_init, model):
    pu_i, pu_y, pu_c, tr_i, tr_j, tr_lag, tr_c = b.structural_factors(roots, service_init, model)
    # Full-data AFC-matched absolute pulse anchors are discovery-time information.
    # They are forbidden from the train-only timing objective. Relative transition
    # factors remain as the frozen weak structural prior for conditional cross-fit.
    return (
        np.asarray([], dtype=np.int32), np.asarray([], dtype=float), np.asarray([], dtype=float),
        tr_i, tr_j, tr_lag, tr_c,
    )


def solve(a):
    roots_payload=b.load_json(a.roots); service_init=b.load_json(a.service_init)
    if roots_payload.get("status")!="QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION": raise SystemExit("roots not qualified")
    if service_init.get("status")!="QUALIFIED_SINGLE_FULL_SERVICE_DAY_SERVICE_INIT": raise SystemExit("service init not qualified")
    roots=list(roots_payload["roots"]); root_mass=b.load_root_mass(a.root_mass); model=b.build_model(roots,root_mass)
    if len(model["groups"])!=1438 or len(model["node_keys"])!=38649: raise SystemExit("conditional candidate service inventory drift")

    access=b.map_unary(a.access,model,"access"); egress=b.map_unary(a.egress,model,"egress"); transfer=b.map_transfer(a.transfer,model)
    pu_i,pu_y,pu_c,tr_i,tr_j,tr_lag,tr_c=train_structural_factors(roots,service_init,model)
    A=model["A"]
    factor_mass={"access":float(access[2].sum()),"egress":float(egress[2].sum()),"transfer":float(transfer[2].sum())}
    if factor_mass["access"]<=0 or factor_mass["egress"]<=0 or factor_mass["transfer"]<=0: raise SystemExit("train passenger factor mass is not positive")
    if abs(factor_mass["access"]-factor_mass["egress"])>1e-6: raise SystemExit("train endpoint factor mass mismatch")

    eval_counter={"n":0}
    def objective(p):
        eval_counter["n"]+=1; t=A@p; gt=np.zeros_like(t); obj=0.0
        # No full-data absolute AFC pulse anchors in R1D train objective.
        r=t[tr_j]-t[tr_i]-tr_lag; obj += 0.5*float(np.sum(tr_c*r*r)); b.add_at(gt,tr_i,-tr_c*r); b.add_at(gt,tr_j,tr_c*r)
        oo,go=b.soft_order_obj_grad(t,access,egress,transfer); obj += oo; gt += go
        return obj,np.asarray(A.T@gt).ravel()

    p0=model["p0"].copy(); initial_obj=float(objective(p0)[0])
    opt=minimize(lambda p:objective(p),p0,jac=True,method="L-BFGS-B",bounds=model["bounds"],options={"maxiter":500,"ftol":1e-12,"gtol":1e-5,"maxls":50,"maxcor":20})
    if not opt.success and opt.status not in (1,2): raise SystemExit(f"train-only timetable optimization failed: {opt.status} {opt.message}")
    p=np.asarray(opt.x,float); t=np.asarray(A@p).ravel(); final_obj=float(objective(p)[0])
    active_gaps=[m for m in model["gap_param_meta"] if p[int(m["param_index"])]<=b.MIN_PROGRESS_S+1e-5]

    # Train-only local curvature. No full-data absolute-pulse curvature enters here.
    hr=[];hc=[];hv=[]
    def diag_add(idx,c): hr.extend(idx.tolist());hc.extend(idx.tolist());hv.extend(c.tolist())
    hr.extend(tr_i.tolist());hc.extend(tr_i.tolist());hv.extend(tr_c.tolist());hr.extend(tr_j.tolist());hc.extend(tr_j.tolist());hv.extend(tr_c.tolist());hr.extend(tr_i.tolist());hc.extend(tr_j.tolist());hv.extend((-tr_c).tolist());hr.extend(tr_j.tolist());hc.extend(tr_i.tolist());hv.extend((-tr_c).tolist())
    ai,entry,am=access; g=t[ai]-entry; s=expit(-g/b.TAU_ACCESS_S);diag_add(ai,am/(b.TAU_ACCESS_S**2)*s*(1-s))
    ei,exit_t,em=egress; g=exit_t-t[ei]; s=expit(-g/b.TAU_EGRESS_S);diag_add(ei,em/(b.TAU_EGRESS_S**2)*s*(1-s))
    li,ui,tm=transfer;g=t[ui]-t[li];s=expit(-g/b.TAU_TRANSFER_S);c=tm/(b.TAU_TRANSFER_S**2)*s*(1-s)
    hr.extend(li.tolist());hc.extend(li.tolist());hv.extend(c.tolist());hr.extend(ui.tolist());hc.extend(ui.tolist());hv.extend(c.tolist());hr.extend(li.tolist());hc.extend(ui.tolist());hv.extend((-c).tolist());hr.extend(ui.tolist());hc.extend(li.tolist());hv.extend((-c).tolist())
    n=len(t);H=coo_matrix((np.asarray(hv,float),(np.asarray(hr,int),np.asarray(hc,int))),shape=(n,n)).tocsr()+eye(n,format="csr")*1e-10
    Hdiag=np.asarray(H.diagonal()).ravel();lu=splu(H.tocsc());rng=np.random.default_rng(20260906);diag_est=np.zeros(n,float)
    for _ in range(b.HUTCHINSON_PROBES):
        z=rng.choice(np.array([-1.0,1.0]),size=n);x=lu.solve(z);diag_est+=z*x
    diag_est/=b.HUTCHINSON_PROBES;lower=np.where(Hdiag>0,1.0/Hdiag,0.0);var=np.maximum.reduce([diag_est,lower,np.full(n,1e-8)]);sd=np.sqrt(var)

    node_existence=np.zeros(n,float);node_components=defaultdict(list)
    for r in roots:
        rid=str(r["root_id"]);fw=float(model["family_weight"][rid])
        for ev in r["events"]:
            ni=model["root_station_to_node"][(rid,int(ev["station"]))];node_components[ni].append((rid,fw,str(r["path_id"])))
    for ni,comps in node_components.items(): node_existence[ni]=min(1.0,sum(v[1] for v in {x[0]:x for x in comps}.values()))

    physical_rows=[]
    for ni,(gid,station) in enumerate(model["node_keys"]):
        comps=node_components[ni];byroot={rid:(w,path) for rid,w,path in comps}
        physical_rows.append({"schema":SCHEMA,"service_date":"2019-01-04","physical_service_id":gid,"station":station,"event_existence_weight_train":float(node_existence[ni]),"realized_time_mean_s":float(t[ni]),"realized_time_sd_laplace_s":float(sd[ni]),"realized_time_q05_s":float(t[ni]-b.Z90*sd[ni]),"realized_time_q50_s":float(t[ni]),"realized_time_q95_s":float(t[ni]+b.Z90*sd[ni]),"realized_time_lower95_s":float(t[ni]-b.Z95*sd[ni]),"realized_time_upper95_s":float(t[ni]+b.Z95*sd[ni]),"component_root_ids":sorted(byroot),"component_path_weights_train":{path:float(w) for _rid,(w,path) in byroot.items()},"full_data_afc_absolute_anchor_used":False,"heldout_passenger_evidence_used":False})
    root_rows=[]
    for r in roots:
        rid=str(r["root_id"]);gid=model["root_to_gid"][rid];fw=float(model["family_weight"][rid])
        for idx,ev in enumerate(r["events"]):
            ni=model["root_station_to_node"][(rid,int(ev["station"]))]
            root_rows.append({"schema":SCHEMA,"service_date":"2019-01-04","physical_service_id":gid,"root_id":rid,"path_id":str(r["path_id"]),"direction":str(r["direction"]),"path_family_weight_within_physical_service_train":fw,"event_key":b.prior.event_key(rid,idx),"event_index":idx,"station":int(ev["station"]),"realized_time_mean_s":float(t[ni]),"realized_time_sd_laplace_s":float(sd[ni]),"realized_time_lower95_s":float(t[ni]-b.Z95*sd[ni]),"realized_time_upper95_s":float(t[ni]+b.Z95*sd[ni]),"source_seed_time_s_initialization_only":float(ev["time_s"]),"source_seed_used_as_objective_factor":False,"heldout_passenger_evidence_used":False})
    b.write_jsonl_gz(a.out_physical_events,physical_rows);b.write_jsonl_gz(a.out_root_events,root_rows)

    widths=[2*b.Z95*x for x in sd];weights=node_existence.tolist();dup=[gid for gid,m in model["group_meta"].items() if m["member_count"]>1];maxw=[model["group_meta"][gid]["max_family_weight"] for gid in dup]
    gates={"train_endpoint_factor_mass_balanced":abs(factor_mass["access"]-factor_mass["egress"])<=1e-6,"train_transfer_factor_mass_positive":factor_mass["transfer"]>0,"conditional_full_candidate_inventory_retained":len(model["groups"])==1438 and len(model["node_keys"])==38649,"all_candidate_root_events_mapped":len(root_rows)==43584,"heldout_passenger_evidence_absent_from_objective":True,"full_data_afc_absolute_pulse_anchors_absent_from_objective":len(pu_i)==0,"hard_forward_time_domain_enforced":all(p[int(m["param_index"])]>=b.MIN_PROGRESS_S-1e-7 for m in model["gap_param_meta"]),"optimizer_objective_improved":final_obj<=initial_obj+1e-6,"intervals_finite":bool(np.all(np.isfinite(sd)))}
    if not all(gates.values()):raise SystemExit("R1D train timetable gates failed: "+json.dumps(gates))
    result={"schema":SCHEMA,"status":"QUALIFIED_R1D_TRAIN_REALIZED_TIMETABLE_REFIT_CONDITIONAL","service_date":"2019-01-04","validation_scope":"CONDITIONAL_CROSSFIT_WITH_FROZEN_FULL_DAY_CANDIDATE_SERVICE_INVENTORY","train_factor_mass":factor_mass,"inventory":{"candidate_root_hypotheses":1673,"physical_service_hypotheses":1438,"physical_event_nodes":38649,"candidate_root_event_states":43584},"optimization":{"initial_objective":initial_obj,"final_objective":final_obj,"iterations":int(opt.nit),"function_evaluations":int(opt.nfev),"message":str(opt.message),"active_minimum_progress_gap_count":len(active_gaps)},"realized_time_interval_width95_s":{"p10":b.prior.weighted_quantile(widths,weights,.1),"median":b.prior.weighted_quantile(widths,weights,.5),"p90":b.prior.weighted_quantile(widths,weights,.9)},"path_family_ambiguity_train":{"max_family_weight_median":b.prior.weighted_quantile(maxw,[1.0]*len(maxw),.5),"decisive_ge_0_75_count":sum(1 for x in maxw if x>=.75),"near_tie_le_0_60_count":sum(1 for x in maxw if x<=.60)},"qualification_gates":gates,"scientific_semantics":{"train_passenger_evidence_only_for_absolute_timing":True,"heldout_passengers_used_for_parameter_estimation":False,"full_data_matched_afc_pulse_absolute_anchors_used":False,"relative_structural_transition_prior_frozen":True,"candidate_service_inventory_frozen_from_prequalification":True,"therefore_validation_is_conditional_not_end_to_end_candidate_discovery_validation":True,"planned_absolute_timetable_used":False,"scientific_self_iteration":False},"next_stage":"FIT_R1C_TRANSFER_POSTERIOR_ON_TRAIN_ONLY_THEN_SCORE_HELDOUT"}
    a.out_summary.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8");print(json.dumps(result,ensure_ascii=False,indent=2));return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--roots",type=Path,required=True);ap.add_argument("--service-init",type=Path,required=True);ap.add_argument("--root-mass",type=Path,required=True);ap.add_argument("--access",type=Path,action="append",required=True);ap.add_argument("--egress",type=Path,action="append",required=True);ap.add_argument("--transfer",type=Path,action="append",required=True);ap.add_argument("--out-physical-events",type=Path,required=True);ap.add_argument("--out-root-events",type=Path,required=True);ap.add_argument("--out-summary",type=Path,required=True);solve(ap.parse_args())

if __name__=="__main__": main()
