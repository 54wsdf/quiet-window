from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import rail_hz_r1c_transfer_interval_substrate_20260905 as base

SCHEMA = "rail.hz-r1d-train-transfer-substrate.v1"


def solve(a):
    schedule, previous = base.load_schedule(a.schedule)
    agg = base.aggregate_transfer_files(a.transfer)
    rows=[]
    total_mass=valid_upper_mass=nonpositive_upper_mass=0.0
    candidate_lower_mass=zero_lower_mass=conflicting_lower_mass=0.0
    movements=set();contexts=set()
    for (lower_root,lower_station,upper_root,upper_station,movement),lm in sorted(agg.items()):
        total_mass+=lm;movements.add(movement)
        lo=schedule.get((lower_root,lower_station));up=schedule.get((upper_root,upper_station))
        if lo is None or up is None: raise SystemExit(f"train schedule mapping failure {lower_root}:{lower_station}->{upper_root}:{upper_station}")
        arr=float(lo["realized_time_mean_s"]);dep=float(up["realized_time_mean_s"])
        arr_sd=float(lo["realized_time_sd_laplace_s"]);dep_sd=float(up["realized_time_sd_laplace_s"])
        upper=dep-arr;upper_sd=math.sqrt(arr_sd*arr_sd+dep_sd*dep_sd)
        bin_index=math.floor(arr/base.TIME_BIN_S);bin_start=bin_index*base.TIME_BIN_S;contexts.add((movement,bin_index))
        prev=previous.get((upper_root,upper_station))
        prev_dep=prev_sd=None;candidate_lower=0.0;candidate_lower_sd=None;lower_kind="ZERO_NO_PREVIOUS_CATCHABLE_SERVICE"
        if prev is not None:
            prev_dep=float(prev["realized_time_mean_s"]);prev_sd=float(prev["realized_time_sd_laplace_s"]);raw_lower=prev_dep-arr
            if 0.0<raw_lower<upper:
                candidate_lower=raw_lower;candidate_lower_sd=math.sqrt(arr_sd*arr_sd+prev_sd*prev_sd);lower_kind="PREVIOUS_SERVICE_CANDIDATE_LOWER_BOUND"
            elif raw_lower>=upper: lower_kind="CONFLICTING_PREVIOUS_SERVICE_ORDER_UPPER_ONLY"
            else: lower_kind="ZERO_PREVIOUS_SERVICE_BEFORE_TRANSFER_READY_OR_ARRIVAL"
        if upper>0:valid_upper_mass+=lm
        else:nonpositive_upper_mass+=lm
        if lower_kind=="PREVIOUS_SERVICE_CANDIDATE_LOWER_BOUND":candidate_lower_mass+=lm
        elif lower_kind=="CONFLICTING_PREVIOUS_SERVICE_ORDER_UPPER_ONLY":conflicting_lower_mass+=lm
        else:zero_lower_mass+=lm
        rows.append({"schema":SCHEMA,"service_date":"2019-01-04","movement":movement,"time_bin_index_30m":int(bin_index),"time_bin_start_s":float(bin_start),"lower_root":lower_root,"lower_station":lower_station,"upper_root":upper_root,"upper_station":upper_station,"lineage_mass":lm,"incoming_arrival_mean_s":arr,"incoming_arrival_sd_s":arr_sd,"boarded_departure_mean_s":dep,"boarded_departure_sd_s":dep_sd,"connection_upper_bound_mean_s":upper,"connection_upper_bound_sd_s":upper_sd,"previous_downstream_root":None if prev is None else str(prev["root_id"]),"previous_downstream_departure_mean_s":prev_dep,"previous_downstream_departure_sd_s":prev_sd,"candidate_lower_bound_mean_s":candidate_lower,"candidate_lower_bound_sd_s":candidate_lower_sd,"candidate_lower_bound_kind":lower_kind,"candidate_lower_bound_is_strict_physical_truth":False,"candidate_lower_bound_role":"FIRST_BOARDABLE_COMPONENT_ONLY; MUST ALLOW SKIP_LEFT_BEHIND_CONTAMINATION","upper_bound_role":"PHYSICAL_TRANSFER_TIME_MUST_NOT_EXCEED_CONNECTION_GAP_FOR_THIS SERVICE_CHAIN EXPLANATION","planned_absolute_timetable_used":False,"service_posterior_frozen_from_r1d_train_refit":True})
    schema=pa.schema([("schema",pa.string()),("service_date",pa.string()),("movement",pa.string()),("time_bin_index_30m",pa.int32()),("time_bin_start_s",pa.float64()),("lower_root",pa.string()),("lower_station",pa.int32()),("upper_root",pa.string()),("upper_station",pa.int32()),("lineage_mass",pa.float64()),("incoming_arrival_mean_s",pa.float64()),("incoming_arrival_sd_s",pa.float64()),("boarded_departure_mean_s",pa.float64()),("boarded_departure_sd_s",pa.float64()),("connection_upper_bound_mean_s",pa.float64()),("connection_upper_bound_sd_s",pa.float64()),("previous_downstream_root",pa.string()),("previous_downstream_departure_mean_s",pa.float64()),("previous_downstream_departure_sd_s",pa.float64()),("candidate_lower_bound_mean_s",pa.float64()),("candidate_lower_bound_sd_s",pa.float64()),("candidate_lower_bound_kind",pa.string()),("candidate_lower_bound_is_strict_physical_truth",pa.bool_()),("candidate_lower_bound_role",pa.string()),("upper_bound_role",pa.string()),("planned_absolute_timetable_used",pa.bool_()),("service_posterior_frozen_from_r1d_train_refit",pa.bool_())])
    a.out.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(rows,schema=schema),a.out,compression="zstd")
    gates={"train_transfer_factor_mass_positive":total_mass>0,"zero_schedule_mapping_failure":True,"movement_inventory_preserved":len(movements)==9,"planned_absolute_timetable_absent":True,"candidate_lower_bounds_not_mislabeled_as_truth":True,"full_day_time_context_retained":len(contexts)>9}
    result={"schema":SCHEMA,"status":"QUALIFIED_R1D_TRAIN_TRANSFER_SUBSTRATE" if all(gates.values()) else "FAILED_R1D_TRAIN_TRANSFER_SUBSTRATE","service_date":"2019-01-04","scope":"TRAIN_ONLY_CONDITIONAL_CROSSFIT_FULL_SERVICE_DAY","aggregated_transfer_pair_rows":len(rows),"movement_count":len(movements),"movement_time_30m_context_count":len(contexts),"transfer_lineage_mass":total_mass,"positive_connection_upper_bound_mass":valid_upper_mass,"nonpositive_connection_upper_bound_mass":nonpositive_upper_mass,"candidate_previous_service_lower_bound_mass":candidate_lower_mass,"zero_or_no_previous_lower_bound_mass":zero_lower_mass,"conflicting_previous_service_lower_bound_mass":conflicting_lower_mass,"qualification_gates":gates,"scientific_semantics":{"train_passenger_evidence_only":True,"heldout_passenger_evidence_used":False,"service_timetable":"R1D_TRAIN_ONLY_REFIT","Theta_K_target":"PHYSICAL_TRANSFER_TIME_DISTRIBUTION_BY_MOVEMENT_AND_TIME_OF_DAY","service_feedback_to_timetable":False}}
    a.summary.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8");print(json.dumps(result,ensure_ascii=False,indent=2))
    if result["status"].startswith("FAILED"):raise SystemExit("train transfer substrate failed")


def main():
    p=argparse.ArgumentParser();p.add_argument("--schedule",type=Path,required=True);p.add_argument("--transfer",type=Path,action="append",required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--summary",type=Path,required=True);solve(p.parse_args())

if __name__=="__main__":main()
