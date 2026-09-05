from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr

SCHEMA = "rail.hz-r1d-heldout-service-order-score.v1"
TAU_ACCESS_S = 45.0
TAU_EGRESS_S = 45.0
TAU_TRANSFER_S = 30.0


def iter_jsonl_gz(path: Path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def load_schedule(path: Path):
    rows=list(iter_jsonl_gz(path))
    if len(rows)!=43584: raise SystemExit(f"expected 43584 root-event states, found {len(rows)}")
    out={}
    for r in rows:
        key=(str(r["root_id"]),int(r["station"]))
        out[key]=(float(r["realized_time_mean_s"]),float(r["realized_time_sd_laplace_s"]))
    return out


def softplus_neg_margin(margin: float, tau: float) -> float:
    x=-margin/tau
    if x>50: return x
    return math.log1p(math.exp(x))


def score_unary(paths, schedule, kind):
    mass=feasible=weighted_prob=loss=0.0;mapping=0.0
    for p in paths:
        pf=pq.ParquetFile(p)
        cols=["root_id","station",("entry_sec" if kind=="access" else "exit_sec"),"lineage_mass"]
        for b in pf.iter_batches(batch_size=100000,columns=cols):
            d=b.to_pydict();n=len(d["root_id"])
            for i in range(n):
                lm=float(d["lineage_mass"][i]);mass+=lm
                ev=schedule.get((str(d["root_id"][i]),int(d["station"][i])))
                if ev is None: mapping+=lm;continue
                mu,sd=ev;obs=float(d[cols[2]][i]);sd=max(sd,1e-6)
                margin=(mu-obs) if kind=="access" else (obs-mu)
                feasible+=lm*(margin>=0)
                weighted_prob+=lm*float(ndtr(margin/sd))
                loss+=lm*softplus_neg_margin(margin,TAU_ACCESS_S if kind=="access" else TAU_EGRESS_S)
    return {"mass":mass,"mapping_failure_mass":mapping,"center_feasible_share":feasible/mass if mass else None,"mean_ordering_probability":weighted_prob/mass if mass else None,"soft_order_loss_per_mass":loss/mass if mass else None}


def score_transfer(paths,schedule):
    mass=feasible=weighted_prob=loss=0.0;mapping=0.0
    for p in paths:
        pf=pq.ParquetFile(p)
        cols=["lower_root","lower_station","upper_root","upper_station","lineage_mass"]
        for b in pf.iter_batches(batch_size=100000,columns=cols):
            d=b.to_pydict();n=len(d["lower_root"])
            for i in range(n):
                lm=float(d["lineage_mass"][i]);mass+=lm
                lo=schedule.get((str(d["lower_root"][i]),int(d["lower_station"][i])));up=schedule.get((str(d["upper_root"][i]),int(d["upper_station"][i])))
                if lo is None or up is None: mapping+=lm;continue
                margin=up[0]-lo[0];sd=math.sqrt(max(lo[1],1e-6)**2+max(up[1],1e-6)**2)
                feasible+=lm*(margin>0)
                weighted_prob+=lm*float(ndtr(margin/sd))
                loss+=lm*softplus_neg_margin(margin,TAU_TRANSFER_S)
    return {"mass":mass,"mapping_failure_mass":mapping,"center_positive_connection_share":feasible/mass if mass else None,"mean_ordering_probability":weighted_prob/mass if mass else None,"soft_order_loss_per_mass":loss/mass if mass else None}


def compare_schedules(train,full):
    keys=sorted(set(train)&set(full));absd=[];cover=[]
    for k in keys:
        mt,st=train[k];mf,_=full[k];absd.append(abs(mt-mf));cover.append(abs(mt-mf)<=1.959963984540054*st)
    a=np.asarray(absd,float)
    return {"event_count":len(keys),"absolute_mean_shift_s":{"median":float(np.quantile(a,.5)),"p90":float(np.quantile(a,.9)),"p99":float(np.quantile(a,.99)),"mean":float(a.mean())},"full_data_event_mean_inside_train95_interval_share":float(np.mean(cover))}


def main():
    p=argparse.ArgumentParser();p.add_argument("--train-schedule",type=Path,required=True);p.add_argument("--full-schedule",type=Path,required=True);p.add_argument("--access",type=Path,action="append",required=True);p.add_argument("--egress",type=Path,action="append",required=True);p.add_argument("--transfer",type=Path,action="append",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    train=load_schedule(a.train_schedule);full=load_schedule(a.full_schedule)
    acc=score_unary(a.access,train,"access");egr=score_unary(a.egress,train,"egress");tr=score_transfer(a.transfer,train);st=compare_schedules(train,full)
    gates={"heldout_mass_positive":min(acc["mass"],egr["mass"],tr["mass"])>0,"zero_mapping_failure":max(acc["mapping_failure_mass"],egr["mapping_failure_mass"],tr["mapping_failure_mass"])<=1e-9,"finite_scores":all(math.isfinite(x) for x in [acc["mean_ordering_probability"],egr["mean_ordering_probability"],tr["mean_ordering_probability"],acc["soft_order_loss_per_mass"],egr["soft_order_loss_per_mass"],tr["soft_order_loss_per_mass"]])}
    out={"schema":SCHEMA,"status":"QUALIFIED_R1D_HELDOUT_SERVICE_ORDER_SCORE" if all(gates.values()) else "FAILED_R1D_HELDOUT_SERVICE_ORDER_SCORE","service_date":"2019-01-04","validation_scope":"CONDITIONAL_CROSSFIT_FOLD0_HELDOUT_ONLY_SCORING","heldout_access":acc,"heldout_egress":egr,"heldout_transfer_order":tr,"train_vs_full_data_schedule_stability":st,"qualification_gates":gates,"scientific_semantics":{"train_schedule_estimated_without_heldout_passengers":True,"heldout_factors_scoring_only":True,"candidate_chain_support_and_inventory_frozen_from_prequalification":True,"schedule_stability_against_full_data_is_robustness_not_ground_truth_accuracy":True,"ordering_probability_is_passenger_compatibility_not_ATS_coverage":True}}
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8");print(json.dumps(out,ensure_ascii=False,indent=2))
    if out["status"].startswith("FAILED"):raise SystemExit("heldout service score failed")

if __name__=="__main__":main()
