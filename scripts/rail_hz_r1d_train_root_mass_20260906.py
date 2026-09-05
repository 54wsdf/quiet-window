from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = "rail.hz-r1d-train-root-lineage-mass.v1"


def shard(a):
    mass=defaultdict(float); total_usage=0.0; resolved_mass=0.0
    pf=pq.ParquetFile(a.genealogy)
    for b in pf.iter_batches(batch_size=50000,columns=["descendant_state_type","root_chain","lineage_mass"]):
        for r in b.to_pylist():
            if str(r["descendant_state_type"])=="UNRESOLVED": continue
            lm=float(r["lineage_mass"]); resolved_mass+=lm
            roots=[x for x in str(r["root_chain"]).split(">") if x]
            for rid in roots:
                mass[rid]+=lm; total_usage+=lm
    out={"schema":SCHEMA,"status":"QUALIFIED_R1D_TRAIN_ROOT_MASS_SHARD","shard_index":a.shard_index,"resolved_edge_lineage_mass":resolved_mass,"root_usage_mass_total":total_usage,"root_usage_mass":dict(sorted(mass.items()))}
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k!="root_usage_mass"},ensure_ascii=False,indent=2))


def merge(a):
    total=defaultdict(float); resolved=usage=0.0
    for p in a.input:
        x=json.loads(p.read_text(encoding="utf-8"))
        if x.get("status")!="QUALIFIED_R1D_TRAIN_ROOT_MASS_SHARD": raise SystemExit("unqualified train root mass shard")
        resolved+=float(x["resolved_edge_lineage_mass"]); usage+=float(x["root_usage_mass_total"])
        for k,v in x["root_usage_mass"].items(): total[k]+=float(v)
    rows=[{"root_id":k,"lineage_usage_mass":v} for k,v in sorted(total.items())]
    pq.write_table(pa.Table.from_pylist(rows,schema=pa.schema([("root_id",pa.string()),("lineage_usage_mass",pa.float64())])),a.out_parquet,compression="zstd")
    out={"schema":SCHEMA,"status":"QUALIFIED_R1D_TRAIN_ROOT_MASS_GLOBAL","resolved_edge_lineage_mass":resolved,"root_usage_mass_total":usage,"positive_root_count":len(rows),"scientific_semantics":{"heldout_lineage_mass_used":False,"root_family_weights_train_only":True}}
    a.out_summary.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8");print(json.dumps(out,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest="cmd",required=True)
    s=sp.add_parser("shard");s.add_argument("--genealogy",type=Path,required=True);s.add_argument("--shard-index",type=int,required=True);s.add_argument("--out",type=Path,required=True)
    m=sp.add_parser("merge");m.add_argument("--input",type=Path,action="append",required=True);m.add_argument("--out-parquet",type=Path,required=True);m.add_argument("--out-summary",type=Path,required=True)
    a=p.parse_args(); shard(a) if a.cmd=="shard" else merge(a)

if __name__=="__main__": main()
