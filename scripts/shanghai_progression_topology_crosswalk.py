#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prior",type=Path,required=True)
    ap.add_argument("--connections",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    connections=json.loads(a.connections.read_text(encoding="utf-8"))
    ride_edges=set()
    for x,y,line in connections:
        ln=str(line).zfill(2)
        ride_edges.add((ln,*sorted((x.strip(),y.strip()))))
    with a.prior.open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    matched=[]; unmatched=[]; per=defaultdict(Counter)
    for r in rows:
        ln=r["line"].zfill(2)
        key=(ln,*sorted((r["a_name"].strip(),r["b_name"].strip())))
        ok=key in ride_edges
        (matched if ok else unmatched).append(r)
        per[ln]["prior_rows"]+=1
        per[ln]["matched_ride_edges"]+=int(ok)
        per[ln]["unmatched_serial_pairs"]+=int(not ok)
    result={
      "schema":"mppd.shanghai.soda2015.progression-topology-crosswalk.v1",
      "prior_rows":len(rows),
      "matched_ride_edge_rows":len(matched),
      "unmatched_serial_pair_rows":len(unmatched),
      "per_line":{k:dict(v) for k,v in sorted(per.items())},
      "unmatched":[{"line":r["line"],"a_name":r["a_name"],"b_name":r["b_name"],"accepted_observations":int(r["accepted_observations"])} for r in unmatched],
      "first_science":{
        "L02":{"prior_rows":per["02"]["prior_rows"],"matched_ride_edges":per["02"]["matched_ride_edges"]},
        "L11":{"prior_rows":per["11"]["prior_rows"],"matched_ride_edges":per["11"]["matched_ride_edges"],"unmatched_serial_pairs":per["11"]["unmatched_serial_pairs"]}
      },
      "decision":"ONLY_MATCHED_TOPOLOGY_RIDE_EDGES_MAY_CONTRIBUTE_TO_T_PROG",
      "boundaries":[
        "The frozen progression prior is source-sequence evidence; it is not itself a topology authority.",
        "Unmatched SERIAL_NO pairs remain source diagnostics and are excluded from route-time accumulation.",
        "A matched row remains scheduled 2015 prior evidence, not observed 2016 ATS."
      ]
    }
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
