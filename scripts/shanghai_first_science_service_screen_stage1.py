#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path

def state_parts(x):
    line,name=x.split("::",1)
    return line,name

def prior_lookup(path):
    out={}
    with path.open(encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            line=r["line"].zfill(2)
            key=(line,frozenset((r["a_name"].strip(),r["b_name"].strip())))
            if key in out:
                raise ValueError(f"duplicate prior edge {key}")
            out[key]={
              "p10":float(r["p10_progression_s"]),
              "median":float(r["median_progression_s"]),
              "p90":float(r["p90_progression_s"]),
              "n":int(r["accepted_observations"])
            }
    return out

def progression(path,prior):
    p10=med=p90=0.0; rides=transfers=covered=0; missing=[]
    for a,b in zip(path,path[1:]):
        la,na=state_parts(a); lb,nb=state_parts(b)
        if la!=lb:
            transfers+=1
            continue
        if na==nb:
            raise ValueError(f"same-line zero-movement edge {a} {b}")
        rides+=1
        key=(la,frozenset((na,nb)))
        rec=prior.get(key)
        if rec is None:
            missing.append({"line":la,"a":na,"b":nb})
        else:
            covered+=1; p10+=rec["p10"]; med+=rec["median"]; p90+=rec["p90"]
    return {
      "ride_edges":rides,"transfer_edges":transfers,"prior_covered_ride_edges":covered,
      "prior_missing_ride_edges":len(missing),"missing_prior_edges":missing,
      "t_prog_p10_s":p10 if not missing else None,
      "t_prog_median_s":med if not missing else None,
      "t_prog_p90_s":p90 if not missing else None
    }

def classify_vector(family,vector,prog):
    if vector["path_count"]>1:
        return "SERVICE_UNRESOLVED","TIED_GEOMETRY_NOT_EXPANDED"
    if prog["prior_missing_ride_edges"]:
        return "SERVICE_UNRESOLVED","PROGRESSION_PRIOR_PARTIAL"
    if vector["line_changes"]>0 or prog["transfer_edges"]>0:
        return "SERVICE_UNRESOLVED","TRANSFER_WALK_WAIT_UNRESOLVED"
    if family.startswith("L2_"):
        return "SERVICE_UNRESOLVED","L2_STREAM_HEADWAY_SHARE_UNRESOLVED"
    if family=="L11_CROSS_BRANCH":
        return "SERVICE_UNRESOLVED","L11_CROSS_BRANCH_REBOARD_SERVICE_CHAIN_UNRESOLVED"
    if family.startswith("L11_"):
        return "PHASE_DEPENDENT","L11_NOMINAL_HEADWAY_PHASE_UNOBSERVED"
    return "SERVICE_UNRESOLVED","UNCLASSIFIED_SERVICE_FAMILY"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--route-seeds",type=Path,required=True)
    ap.add_argument("--route-qualification",type=Path,required=True)
    ap.add_argument("--prior",type=Path,required=True)
    ap.add_argument("--service-worlds",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)

    rq=json.loads(a.route_qualification.read_text(encoding="utf-8"))
    if rq.get("schema")!="mppd.shanghai.first-science-structural-route-seeds.v2":
        raise SystemExit(f"unexpected route seed schema: {rq.get('schema')}")
    worlds=json.loads(a.service_worlds.read_text(encoding="utf-8"))
    if worlds.get("schema")!="mppd.shanghai.first-science-service-worlds.v1":
        raise SystemExit("unexpected service worlds schema")
    prior=prior_lookup(a.prior)

    counts=Counter(); by_family=defaultdict(Counter); by_world=defaultdict(Counter)
    od_classes=defaultdict(set); records=0; vectors=0
    with gzip.open(a.route_seeds,"rt",encoding="utf-8") as f, gzip.open(a.out_dir/"first_science_service_screening_stage1.jsonl.gz","wt",encoding="utf-8") as out:
        for line in f:
            rec=json.loads(line); records+=1
            out_vectors=[]
            for v in rec["vectors"]:
                vectors+=1
                prog=progression(v["representative_path"],prior)
                if prog["ride_edges"]!=v["ride_intervals"]:
                    raise AssertionError((rec["world"],rec["origin"],rec["destination"],v,prog))
                if prog["transfer_edges"]!=v["line_changes"]:
                    raise AssertionError((rec["world"],rec["origin"],rec["destination"],v,prog))
                cls,reason=classify_vector(rec["family"],v,prog)
                counts[cls]+=1;by_family[rec["family"]][cls]+=1;by_world[rec["world"]][cls]+=1
                od_classes[(rec["world"],rec["family"],rec["origin"],rec["destination"])].add(cls)
                out_vectors.append({**v,**prog,"stage1_class":cls,"stage1_reason":reason})
            out.write(json.dumps({**rec,"vectors":out_vectors},ensure_ascii=False,separators=(",",":"))+"\n")

    od_summary=Counter()
    for classes in od_classes.values():
        if classes=={"PHASE_DEPENDENT"}: od_summary["only_phase_dependent"]+=1
        elif classes=={"SERVICE_UNRESOLVED"}: od_summary["only_service_unresolved"]+=1
        else: od_summary["mixed"]+=1

    h=hashlib.sha256()
    with a.route_seeds.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    result={
      "schema":"mppd.shanghai.first-science-service-screening.stage1.v1",
      "route_seed_sha256":h.hexdigest(),
      "route_seed_records":records,
      "route_vectors":vectors,
      "vector_classes":dict(counts),
      "od_class_summary":dict(od_summary),
      "by_family":{k:dict(v) for k,v in sorted(by_family.items())},
      "by_world":{k:dict(v) for k,v in sorted(by_world.items())},
      "progression_prior_edges":len(prior),
      "service_worlds_schema":worlds["schema"],
      "classification_semantics":{
        "PHASE_DEPENDENT":"Single representative geometry is unique for its structural vector, all ride edges have scheduled progression prior, no structural transfer is present, and L11 service phase remains unobserved.",
        "SERVICE_UNRESOLVED":"At least one tied-geometry, progression, transfer/walking, reboard, or L2 stream-headway uncertainty prevents total-time qualification."
      },
      "safe_deletion":"NONE_AT_STAGE1",
      "next_required":[
        "Compute timing bounds over tied-path parent DAG for vectors with path_count>1 rather than expanding every tied geometry.",
        "Qualify or bound transfer walking/wait where possible.",
        "Expand structurally dominated routes only after the structural-frontier seed timing audit is stable."
      ],
      "boundaries":[
        "T_prog is a scheduled 2015 station-progression prior, not observed 2016 ATS.",
        "Stage1 never uses nominal headway as a hard maximum wait.",
        "No route receives zero probability and no passenger-route posterior is estimated.",
        "Representative geometry is never substituted for a tied-path set when path_count>1."
      ]
    }
    (a.out_dir/"qualification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
