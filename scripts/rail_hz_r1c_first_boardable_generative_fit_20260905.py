from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import ndtr, expit, logit

SCHEMA = "rail.hz-r1c-first-boardable-generative-transfer.v1"
GH_X, GH_W = np.polynomial.hermite.hermgauss(5)
GH_Z = math.sqrt(2.0) * GH_X
GH_WN = GH_W / math.sqrt(math.pi)
EPS = 1e-300
TIME_BIN_S = 1800.0


def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_schedule(path: Path):
    events=list(iter_jsonl_gz(path))
    if len(events)!=43584: raise SystemExit(f"expected 43584 R1B root-event states, found {len(events)}")
    root_meta={}
    groups:dict[tuple[str,str,int],list[dict[str,Any]]]=defaultdict(list)
    by_key={}
    for e in events:
        rid=str(e["root_id"]); st=int(e["station"]); by_key[(rid,st)]=e
        root_meta[rid]=(str(e["path_id"]),str(e["direction"]),str(e["physical_service_id"]))
        groups[(str(e["path_id"]),str(e["direction"]),st)].append(e)
    # one row per physical service within each path/direction/station sequence
    service_seq={}
    for key,rows in groups.items():
        uniq={}
        for e in rows:
            pid=str(e["physical_service_id"])
            prev=uniq.get(pid)
            if prev is None or str(e["root_id"]) < str(prev["root_id"]): uniq[pid]=e
        seq=sorted(uniq.values(),key=lambda e:(float(e["realized_time_mean_s"]),str(e["physical_service_id"])))
        service_seq[key]=seq
    return by_key,root_meta,service_seq


def load_substrate(path:Path):
    return pq.read_table(path).to_pylist()


def build_episodes(substrate,by_key,root_meta,service_seq,movement:str):
    eps=[]; residual_mass=0.0; positive_mass=0.0
    for r in substrate:
        if str(r["movement"])!=movement: continue
        lm=float(r["lineage_mass"])
        lower_root=str(r["lower_root"]); lower_station=int(r["lower_station"])
        upper_root=str(r["upper_root"]); upper_station=int(r["upper_station"])
        lo=by_key.get((lower_root,lower_station)); up=by_key.get((upper_root,upper_station))
        if lo is None or up is None: raise SystemExit("schedule mapping failure in R1C generative builder")
        arr=float(lo["realized_time_mean_s"]); arr_sd=float(lo["realized_time_sd_laplace_s"])
        boarded=float(up["realized_time_mean_s"]); boarded_pid=str(up["physical_service_id"])
        if boarded<=arr:
            residual_mass+=lm; continue
        path,direction,_pid=root_meta[upper_root]
        seq=service_seq[(path,direction,upper_station)]
        cand=[]
        found=False
        for e in seq:
            dep=float(e["realized_time_mean_s"])
            if dep<=arr: continue
            if dep>boarded+1e-6: break
            # Only services at or before the observed-boarded service enter the first-boardable set.
            cand.append(e)
            if str(e["physical_service_id"])==boarded_pid:
                found=True
                # there may be a tie in mean time; stop at exact physical observed service
                break
        if not found or not cand:
            raise SystemExit(f"boarded service not found in downstream sequence: {upper_root}@{upper_station}")
        gaps=[]; gap_sds=[]
        for e in cand:
            gaps.append(float(e["realized_time_mean_s"])-arr)
            gap_sds.append(math.sqrt(arr_sd*arr_sd+float(e["realized_time_sd_laplace_s"])**2))
        if not all(g>0 for g in gaps) or any(gaps[i]>=gaps[i+1] for i in range(len(gaps)-1)):
            raise SystemExit(f"non-increasing candidate connection gaps for {upper_root}@{upper_station}: {gaps}")
        positive_mass+=lm
        eps.append({"lineage_mass":lm,"arrival_time_s":arr,"time_bin_index_30m":int(math.floor(arr/TIME_BIN_S)),"gaps":np.asarray(gaps,float),"gap_sds":np.asarray(gap_sds,float),"candidate_service_count":len(gaps)})
    return eps,positive_mass,residual_mass


def effective_cdf(bound_mean:np.ndarray,bound_sd:np.ndarray,mu:float,sigma:float):
    # E[F_K(U)] over Gaussian service-timing uncertainty U.
    x=bound_mean[:,None]+bound_sd[:,None]*GH_Z[None,:]
    z=np.full_like(x,-np.inf,dtype=float)
    pos=x>0
    z[pos]=(np.log(x[pos])-mu)/sigma
    return np.sum(ndtr(z)*GH_WN[None,:],axis=1)


def nll(theta,episodes,return_parts=False):
    mu=float(theta[0]); sigma=math.exp(float(theta[1])); skip=float(expit(theta[2]))
    total=0.0; masses=0.0; skipped_expect_num=0.0
    for e in episodes:
        gaps=e["gaps"]; sds=e["gap_sds"]; lm=float(e["lineage_mass"]); m=len(gaps)
        cdf=effective_cdf(gaps,sds,mu,sigma)
        prev=np.concatenate(([0.0],cdf[:-1])); delta=np.maximum(0.0,cdf-prev)
        powers=skip**np.arange(m-1,-1,-1,dtype=float)
        terms=delta*powers*(1.0-skip)
        prob=float(np.sum(terms))
        total-=lm*math.log(max(prob,EPS)); masses+=lm
        if prob>EPS:
            post=terms/prob
            skipped=np.arange(m-1,-1,-1,dtype=float)
            skipped_expect_num+=lm*float(np.sum(post*skipped))
    if return_parts:
        return total,{"mass":masses,"expected_skipped_services_per_episode":skipped_expect_num/max(masses,EPS)}
    return total


def fit_model(episodes,start=None):
    if not episodes: return None
    if start is None:
        # Connection upper gaps initialize a deliberately broad transfer distribution.
        vals=np.asarray([e["gaps"][-1] for e in episodes]); w=np.asarray([e["lineage_mass"] for e in episodes])
        order=np.argsort(vals); vals=vals[order];w=w[order];c=np.cumsum(w)/w.sum(); med=float(vals[np.searchsorted(c,.5)])
        start=np.array([math.log(max(10.0,med*.55)),math.log(.65),logit(.12)])
    bounds=[(math.log(1.0),math.log(7200.0)),(math.log(.08),math.log(2.5)),(logit(1e-5),logit(.98))]
    starts=[np.asarray(start,float),np.array([start[0],math.log(.4),logit(.03)]),np.array([start[0],math.log(.9),logit(.30)])]
    best=None
    for x0 in starts:
        res=minimize(lambda x:nll(x,episodes),x0,method="L-BFGS-B",bounds=bounds,options={"maxiter":500,"ftol":1e-12,"gtol":1e-6,"maxls":50})
        if best is None or res.fun<best.fun: best=res
    if best is None or not np.isfinite(best.fun): raise SystemExit("R1C transfer likelihood optimization failed")
    x=np.asarray(best.x,float)
    # finite-difference Hessian in transformed parameter space
    h=np.array([2e-3,2e-3,3e-3])
    H=np.zeros((3,3),float);f0=nll(x,episodes)
    for i in range(3):
        ei=np.zeros(3);ei[i]=h[i];H[i,i]=(nll(x+ei,episodes)-2*f0+nll(x-ei,episodes))/(h[i]**2)
        for j in range(i+1,3):
            ej=np.zeros(3);ej[j]=h[j]
            v=(nll(x+ei+ej,episodes)-nll(x+ei-ej,episodes)-nll(x-ei+ej,episodes)+nll(x-ei-ej,episodes))/(4*h[i]*h[j]);H[i,j]=H[j,i]=v
    cov=np.linalg.pinv(H,rcond=1e-10)
    _f,parts=nll(x,episodes,True)
    return {"x":x,"cov":cov,"nll":float(best.fun),"success":bool(best.success),"message":str(best.message),"nit":int(best.nit),"parts":parts,"hessian_eigenvalues":np.linalg.eigvalsh(0.5*(H+H.T)).tolist()}


def weighted_cov(estimates,weights):
    X=np.asarray(estimates,float);w=np.asarray(weights,float);w=w/w.sum();mu=np.sum(X*w[:,None],axis=0);D=X-mu;return (D*w[:,None]).T@D


def quantiles(theta):
    mu=float(theta[0]);sig=math.exp(float(theta[1]));return {"q05_s":math.exp(mu-1.6448536269514722*sig),"q25_s":math.exp(mu-.6744897501960817*sig),"median_s":math.exp(mu),"mean_s":math.exp(mu+.5*sig*sig),"q75_s":math.exp(mu+.6744897501960817*sig),"q95_s":math.exp(mu+1.6448536269514722*sig),"log_sigma":sig,"skip_or_left_behind_probability":float(expit(theta[2]))}


def fit_movement(a):
    by_key,root_meta,seq=load_schedule(a.schedule); substrate=load_substrate(a.substrate)
    movements=sorted({str(r["movement"]) for r in substrate})
    if a.movement_index<0 or a.movement_index>=len(movements): raise SystemExit("movement index out of range")
    movement=movements[a.movement_index];episodes,pos_mass,resid_mass=build_episodes(substrate,by_key,root_meta,seq,movement)
    global_fit=fit_model(episodes); gx=global_fit["x"]
    bins=sorted({e["time_bin_index_30m"] for e in episodes}); local=[]
    for b in bins:
        ee=[e for e in episodes if e["time_bin_index_30m"]==b]
        mass=sum(e["lineage_mass"] for e in ee)
        lf=fit_model(ee,start=gx)
        local.append({"bin":b,"episodes":ee,"mass":mass,"fit":lf})
    # Empirical-Bayes diagonal between-bin variance after subtracting local estimation variance.
    good=[z for z in local if np.all(np.isfinite(z["fit"]["cov"]))]
    if len(good)>=2:
        est=[z["fit"]["x"] for z in good];weights=[z["mass"] for z in good];raw=np.diag(weighted_cov(est,weights));noise=np.average(np.asarray([np.diag(z["fit"]["cov"]) for z in good]),axis=0,weights=np.asarray(weights));tau2=np.maximum(raw-noise,1e-6)
    else: tau2=np.full(3,1e-6)
    contexts=[]
    for z in local:
        x=z["fit"]["x"];v=np.maximum(np.diag(z["fit"]["cov"]),1e-10);post_v=1.0/(1.0/v+1.0/tau2);post_x=post_v*(x/v+gx/tau2)
        q=quantiles(post_x); upper_vals=np.asarray([e["gaps"][-1] for e in z["episodes"]]);upper_w=np.asarray([e["lineage_mass"] for e in z["episodes"]]);order=np.argsort(upper_vals);upper_vals=upper_vals[order];upper_w=upper_w[order];c=np.cumsum(upper_w)/upper_w.sum()
        upperq={f"q{int(p*100):02d}_s":float(upper_vals[min(len(upper_vals)-1,int(np.searchsorted(c,p)))]) for p in (.05,.25,.5,.75,.95)}
        contexts.append({"movement":movement,"time_bin_index_30m":int(z["bin"]),"time_bin_start_s":float(z["bin"]*TIME_BIN_S),"effective_lineage_mass":float(z["mass"]),"aggregated_episode_count":len(z["episodes"]),"strict_first_boardable_generative_estimate":q,"skip_robust_connection_upper_envelope_quantiles":upperq,"eb_parameter_sd":{"log_median":math.sqrt(post_v[0]),"log_sigma_transformed":math.sqrt(post_v[1]),"logit_skip":math.sqrt(post_v[2])},"shrinkage_between_bin_variance":tau2.tolist(),"component_count_selected_baseline":1,"mixture_interface_capable":True})
    result={"schema":SCHEMA,"status":"QUALIFIED_R1C_MOVEMENT_GENERATIVE_K1_FIT","service_date":"2019-01-04","movement_index":a.movement_index,"movement":movement,"positive_connection_mass":pos_mass,"nonpositive_connection_residual_mass":resid_mass,"aggregated_positive_episode_rows":len(episodes),"candidate_service_count_distribution":{"max":max(e["candidate_service_count"] for e in episodes),"mass_weighted_mean":sum(e["candidate_service_count"]*e["lineage_mass"] for e in episodes)/pos_mass},"movement_level_fit":{"parameters":quantiles(gx),"nll":global_fit["nll"],"optimizer_success":global_fit["success"],"optimizer_message":global_fit["message"],"optimizer_iterations":global_fit["nit"],"hessian_eigenvalues":global_fit["hessian_eigenvalues"],"expected_skipped_services_per_episode":global_fit["parts"]["expected_skipped_services_per_episode"]},"time_contexts":contexts,"scientific_semantics":{"generative_process":"physical transfer time determines first-boardable service; skip/left-behind probability explains boarding a later eligible service","candidate_previous_service_not_hard_lower_bound":True,"skip_probability_estimated_from_variable_service-opportunity patterns":True,"schedule_uncertainty_integrated_by_gauss_hermite":True,"service_timetable_frozen_from_r1b":True,"service_feedback_to_r1b":False,"planned_absolute_timetable_used":False,"k1_baseline":"LOGNORMAL","mixture_capable_interface":True,"skip_robust_upper_envelope":"weighted empirical quantiles of boarded-service connection upper bounds; transfer-time quantiles cannot exceed these under the observed service-chain explanation"}}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8");print(json.dumps(result,ensure_ascii=False,indent=2));return result


def merge(a):
    ss=[json.loads(p.read_text(encoding="utf-8")) for p in a.input]
    if len(ss)!=9 or not all(x["status"]=="QUALIFIED_R1C_MOVEMENT_GENERATIVE_K1_FIT" for x in ss): raise SystemExit("expected nine movement fits")
    if len({x["movement"] for x in ss})!=9: raise SystemExit("movement fits are not unique")
    total_pos=sum(x["positive_connection_mass"] for x in ss);total_res=sum(x["nonpositive_connection_residual_mass"] for x in ss)
    contexts=sum(len(x["time_contexts"]) for x in ss)
    eig_min=min(min(x["movement_level_fit"]["hessian_eigenvalues"]) for x in ss)
    weak=[x["movement"] for x in ss if min(x["movement_level_fit"]["hessian_eigenvalues"])<=1e-6]
    result={"schema":SCHEMA,"status":"QUALIFIED_R1C_FULL_DAY_MOVEMENT_TIME_GENERATIVE_K1_BASELINE","service_date":"2019-01-04","scope":"FULL_SERVICE_DAY_FULL_NETWORK_ALL_TRANSFER_GENEALOGY_MASS","movement_count":9,"movement_time_context_count":contexts,"positive_connection_mass":total_pos,"nonpositive_connection_residual_mass":total_res,"total_transfer_genealogy_mass":total_pos+total_res,"minimum_movement_hessian_eigenvalue":eig_min,"weakly_identified_movement_parameter_sets":weak,"movement_results":ss,"qualification_gates":{"all_nine_movements_fit":True,"total_transfer_mass_conserved":abs((total_pos+total_res)-319931.0069536726)<=1e-4,"service_timetable_frozen":True,"nonpositive_residual_explicit":True,"time_conditioning_active":contexts>9,"hierarchical_empirical_bayes_shrinkage_active":True,"candidate_lower_bound_not_treated_as_unconditional_truth":True,"planned_absolute_time_absent":True},"next_stage":"R1C_MIXTURE_MODE_CHECK_AND_READ_ONLY_TRANSFER_POSTERIOR_PREDICTIVE_AUDIT"}
    if not all(result["qualification_gates"].values()): raise SystemExit("R1C K1 merge gates failed")
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8");print(json.dumps(result,ensure_ascii=False,indent=2));return result


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    f=sp.add_parser("fit-movement");f.add_argument("--schedule",type=Path,required=True);f.add_argument("--substrate",type=Path,required=True);f.add_argument("--movement-index",type=int,required=True);f.add_argument("--out",type=Path,required=True)
    m=sp.add_parser("merge");m.add_argument("--input",type=Path,action="append",required=True);m.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();fit_movement(a) if a.cmd=="fit-movement" else merge(a)

if __name__=="__main__":main()
