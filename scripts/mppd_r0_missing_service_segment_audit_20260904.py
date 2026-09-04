import argparse
import csv
import gzip
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r0_cached_full_network_support_rescan_20260904 as support
import scripts.mppd_r0_full_network_factor_engine_20260904 as r0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--p1c',required=True)
    ap.add_argument('--cohorts',required=True)
    ap.add_argument('--routes',required=True)
    ap.add_argument('--service-init',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    G,meta,code_to_nodes,transfer_groups,ambiguous_seq,ambiguous_codes,graph_build=g0.build_network(args.p1c)
    routes=support.load_routes(args.routes)
    trajectories,evidence_counts,manifest=support.load_service_trajectories(args.service_init)
    rides_fn,_=r0.build_segment_ride_cache(trajectories)
    line_missing=Counter(); seg_missing=Counter(); time_incompat=Counter(); eval_mass=Counter()
    cohort_mass=0
    for oc,dc,tin,tout,mass in support.load_cohorts(args.cohorts):
        cohort_mass+=mass
        for cand in routes.get((oc,dc),[]):
            tc=int(cand.get('transfer_count',0)); eval_mass[tc]+=mass
            chain,fail=r0.first_feasible_chain(cand,meta,rides_fn,tin,tout)
            if chain is not None or not fail: continue
            line=str(fail.get('failed_line'))
            if int(fail.get('available_rides',0) or 0)==0:
                line_missing[line]+=mass
                key=f"{line}|{fail.get('failed_origin')}|{fail.get('failed_destination')}"
                seg_missing[key]+=mass
            else:
                time_incompat[line]+=mass
    line_rows=[]
    for line,m in line_missing.most_common():
        line_rows.append({'line':line,'missing_service_candidate_mass':m,'time_incompatible_candidate_mass':time_incompat[line]})
    seg_rows=[{'segment':k,'missing_service_candidate_mass':m} for k,m in seg_missing.most_common()]
    for name,rows in [('missing_service_by_line.csv',line_rows),('missing_service_segments.csv',seg_rows)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            fields=list(rows[0]) if rows else ['empty']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    result={
      'schema':'mppd.r0-missing-service-segment-audit.v1','date':'2026-09-04','status':'R0_MISSING_SERVICE_SEGMENT_AUDIT_COMPLETED',
      'authority':'00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md',
      'scope_assertions':{'full_network':True,'line_filter':False,'segment_filter':False,'transfer_count_cap':False},
      'mapped_cohort_passenger_mass':cohort_mass,
      'service_initialization_manifest':manifest,
      'missing_service_candidate_mass_total':sum(line_missing.values()),
      'top_missing_lines':line_rows[:30],
      'top_missing_segments':seg_rows[:50],
      'performance':{'wall_sec':time.perf_counter()-t0},
      'scientific_boundary':['Candidate mass counts route-candidate evaluations and is not unique passenger mass.','The audit identifies spatial service-trajectory support gaps under the current initialization; it does not estimate service truth.','Repair must complete latent service trajectories or state support, not remove affected lines/routes.'],
      'next_gate':'Complete partial service trajectories across qualified line station sequences and retain event-level evidence/uncertainty, then rerun full-network support.',
      'no_email_notification_logic':True
    }
    (out/'r0_missing_service_segment_audit_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
