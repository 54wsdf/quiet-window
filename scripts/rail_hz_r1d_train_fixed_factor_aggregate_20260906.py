from __future__ import annotations

import argparse
import json
from pathlib import Path

import rail_hz_r1b3_fixed_joint_factor_aggregate_20260905 as base


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--genealogy",type=Path,required=True);ap.add_argument("--routes",type=Path,required=True);ap.add_argument("--shard-index",type=int,required=True)
    ap.add_argument("--out-access",type=Path,required=True);ap.add_argument("--out-egress",type=Path,required=True);ap.add_argument("--out-transfer",type=Path,required=True);ap.add_argument("--out-summary",type=Path,required=True)
    a=ap.parse_args(); result=base.aggregate_shard(a)
    result["schema"]="rail.hz-r1d-train-fixed-factor-aggregate.v1"
    result["status"]="QUALIFIED_R1D_TRAIN_FIXED_FACTOR_AGGREGATE_SHARD"
    result["scientific_semantics"].update({"source":"TRAIN_ONLY_R1B1_GENEALOGY_EDGES","heldout_passenger_evidence_used":False,"parameter_refit_role":"R1D_CONDITIONAL_CROSSFIT_TRAIN_EVIDENCE"})
    a.out_summary.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
