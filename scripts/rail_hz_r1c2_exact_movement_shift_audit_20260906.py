from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from rail_hz_r1c2_transfer_posterior_chain_reweight_v2_20260906 import parse_transfer_chain

SCHEMA = "rail.hz-r1c2-exact-movement-shift-audit.v1"
MOVEMENTS = (
    "A:15->B:15", "A:16->B:16", "A:46->C:46", "A:5->B:5",
    "B:15->A:15", "B:16->A:16", "B:5->A:5", "C:46->A:46",
    "SAME_LINE_SERVICE_CHANGE:B:20",
)


def shard(a):
    before=defaultdict(float);after=defaultdict(float);unsupported=defaultdict(float);parse_failure=0.0
    pf=pq.ParquetFile(a.edges)
    cols=["descendant_state_type","transfer_count","transfer_chain","passenger_mass","posterior_probability_r1b1","posterior_probability","r1c2_transfer_update_supported","r1c2_transfer_update_reason"]
    for b in pf.iter_batches(batch_size=100000,columns=cols):
        for r in b.to_pylist():
            tc=int(r["transfer_count"])
            if str(r["descendant_state_type"])=="UNRESOLVED" or tc<=0: continue
            ms=parse_transfer_chain(str(r["transfer_chain"]),tc,MOVEMENTS)
            mass=float(r["passenger_mass"])
            p0=float(r["posterior_probability_r1b1"]);p1=float(r["posterior_probability"])
            if ms is None:
                parse_failure += mass*p0
                continue
            for m in ms:
                before[m]+=mass*p0;after[m]+=mass*p1
                if not bool(r["r1c2_transfer_update_supported"]) and str(r["r1c2_transfer_update_reason"]).startswith("NEW:NONPOSITIVE_REALIZED_CONNECTION"):
                    unsupported[m]+=mass*p0
    out={"schema":SCHEMA,"status":"QUALIFIED_R1C2_EXACT_MOVEMENT_SHIFT_AUDIT_SHARD" if parse_failure<=1e-9 else "FAILED_R1C2_EXACT_MOVEMENT_PARSE","shard_index":a.shard_index,"parse_failure_lineage_mass":parse_failure,"before":dict(before),"after":dict(after),"unsupported_edge_incidence_mass":dict(unsupported)}
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8");print(json.dumps(out,ensure_ascii=False,indent=2))
    if out["status"].startswith("FAILED"): raise SystemExit("movement audit parse failure")


def merge(a):
    before=defaultdict(float);after=defaultdict(float);unsupported=defaultdict(float);pfail=0.0
    for p in a.input:
        x=json.loads(p.read_text(encoding="utf-8"));
        if x.get("status")!="QUALIFIED_R1C2_EXACT_MOVEMENT_SHIFT_AUDIT_SHARD": raise SystemExit("unqualified audit shard")
        pfail+=float(x["parse_failure_lineage_mass"])
        for k,v in x["before"].items():before[k]+=float(v)
        for k,v in x["after"].items():after[k]+=float(v)
        for k,v in x["unsupported_edge_incidence_mass"].items():unsupported[k]+=float(v)
    rows=[]
    for m in MOVEMENTS:
        b=before.get(m,0.0);c=after.get(m,0.0)
        rows.append({"movement":m,"before_lineage_mass":b,"after_lineage_mass":c,"delta_lineage_mass":c-b,"relative_delta":None if b<=0 else (c-b)/b,"unsupported_nonpositive_connection_edge_incidence_mass":unsupported.get(m,0.0)})
    out={"schema":SCHEMA,"status":"QUALIFIED_R1C2_EXACT_MOVEMENT_SHIFT_AUDIT","service_date":"2019-01-04","parse_failure_lineage_mass":pfail,"movement_count":len(rows),"movement_rows":rows,"note":"Unsupported edge-incidence mass counts an edge on every movement in its chain; it is diagnostic incidence, not an additive partition of residual passenger mass."}
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8");print(json.dumps(out,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest="cmd",required=True)
    s=sp.add_parser("shard");s.add_argument("--edges",type=Path,required=True);s.add_argument("--shard-index",type=int,required=True);s.add_argument("--out",type=Path,required=True)
    m=sp.add_parser("merge");m.add_argument("--input",type=Path,action="append",required=True);m.add_argument("--out",type=Path,required=True)
    a=p.parse_args();shard(a) if a.cmd=="shard" else merge(a)

if __name__=="__main__":main()
