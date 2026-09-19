#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

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
    if not out:
        raise ValueError("empty progression prior")
    return out

def read_support(path):
    by_od=defaultdict(list)
    total=0
    with path.open(encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            n=int(r["journeys"])
            if n<0:
                raise ValueError("negative journey support")
            key=(r["family"],r["origin"],r["destination"])
            rec={"day":r["day"],"period":r["period"],"journeys":n}
            by_od[key].append(rec); total+=n
    for key in by_od:
        by_od[key].sort(key=lambda x:(x["day"],x["period"]))
    if not by_od:
        raise ValueError("empty OD support")
    return by_od,total

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

def iso_day(day):
    if len(day)==8 and day.isdigit():
        return f"{day[:4]}-{day[4:6]}-{day[6:]}"
    return day

def classify_slice(family,day,period,vector,prog,worlds):
    if vector["path_count"]>1:
        return "SERVICE_UNRESOLVED","TIED_GEOMETRY_NOT_EXPANDED"
    if prog["prior_missing_ride_edges"]:
        return "SERVICE_UNRESOLVED","PROGRESSION_PRIOR_PARTIAL"
    if vector["line_changes"]>0 or prog["transfer_edges"]>0:
        return "SERVICE_UNRESOLVED","TRANSFER_WALK_WAIT_UNRESOLVED"
    if family=="L11_CROSS_BRANCH":
        return "SERVICE_UNRESOLVED","L11_CROSS_BRANCH_REBOARD_SERVICE_CHAIN_UNRESOLVED"
    if family.startswith("L11_"):
        periods=worlds.get("L11",{}).get("periods",{})
        p=periods.get(period)
        if p is None:
            return "SERVICE_UNRESOLVED","L11_PERIOD_SERVICE_WORLD_NOT_QUALIFIED"
        if not p.get("phase_worlds"):
            return "SERVICE_UNRESOLVED","L11_PHASE_WORLD_MISSING"
        return "PHASE_DEPENDENT",f"L11_{period}_PHASE_UNRESOLVED"
    if family.startswith("L2_"):
        l2=worlds.get("L02",{})
        dclass=l2.get("date_classes",{}).get(iso_day(day))
        if dclass is None:
            return "SERVICE_UNRESOLVED","L2_DATE_CLASS_NOT_QUALIFIED"
        if period=="AM_0700_0900":
            am=l2.get("AM_0700_0900",{}).get("all_dates",{})
            if not am or am.get("stream_specific_headways")!="UNRESOLVED":
                return "SERVICE_UNRESOLVED","L2_AM_SERVICE_WORLD_INCOMPLETE"
            return "SERVICE_UNRESOLVED","L2_AM_STREAM_HEADWAY_UNRESOLVED"
        if period in {"PM_DOC_1730_1900","PM_SHOULDER"}:
            pm=l2.get("PM",{}).get(dclass)
            if pm is None:
                return "SERVICE_UNRESOLVED","L2_PM_DATE_CLASS_WORLD_MISSING"
            if dclass=="POST_20160812" and pm.get("stream_share")=="UNRESOLVED":
                return "SERVICE_UNRESOLVED","L2_PM_POST_STREAM_SHARE_UNRESOLVED"
            return "SERVICE_UNRESOLVED","L2_PM_STREAM_HEADWAY_UNRESOLVED"
        return "SERVICE_UNRESOLVED","L2_PERIOD_SERVICE_WORLD_NOT_QUALIFIED"
    return "SERVICE_UNRESOLVED","UNCLASSIFIED_SERVICE_FAMILY"

def summarise_slice(classes):
    if classes=={"PHASE_DEPENDENT"}:
        return "ONLY_PHASE_DEPENDENT"
    if classes=={"SERVICE_UNRESOLVED"}:
        return "ONLY_SERVICE_UNRESOLVED"
    return "MIXED"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--route-seeds",type=Path,required=True)
    ap.add_argument("--route-qualification",type=Path,required=True)
    ap.add_argument("--od-support",type=Path,required=True)
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
    support,support_total=read_support(a.od_support)

    counts=Counter(); reasons=Counter(); by_family=defaultdict(Counter); by_world=defaultdict(Counter)
    slice_classes=defaultdict(set); records=0; vectors=0; seen_ods=set()
    with gzip.open(a.route_seeds,"rt",encoding="utf-8") as f, gzip.open(a.out_dir/"first_science_service_screening_stage1.jsonl.gz","wt",encoding="utf-8") as out:
        for line in f:
            rec=json.loads(line); records+=1
            key=(rec["family"],rec["origin"],rec["destination"])
            slices=support.get(key)
            if not slices:
                raise AssertionError(f"route seed lacks OD support: {key}")
            seen_ods.add(key)
            out_vectors=[]
            for v in rec["vectors"]:
                vectors+=1
                prog=progression(v["representative_path"],prior)
                if prog["ride_edges"]!=v["ride_intervals"]:
                    raise AssertionError((rec["world"],rec["origin"],rec["destination"],v,prog))
                if prog["transfer_edges"]!=v["line_changes"]:
                    raise AssertionError((rec["world"],rec["origin"],rec["destination"],v,prog))
                slice_results=[]
                for s in slices:
                    cls,reason=classify_slice(rec["family"],s["day"],s["period"],v,prog,worlds)
                    counts[cls]+=1; reasons[reason]+=1
                    by_family[rec["family"]][cls]+=1; by_world[rec["world"]][cls]+=1
                    sk=(rec["world"],rec["family"],rec["origin"],rec["destination"],s["day"],s["period"])
                    slice_classes[sk].add(cls)
                    slice_results.append({**s,"stage1_class":cls,"stage1_reason":reason})
                out_vectors.append({**v,**prog,"support_slices":slice_results})
            out.write(json.dumps({**rec,"vectors":out_vectors},ensure_ascii=False,separators=(",",":"))+"\n")

    missing_route_ods=sorted(set(support)-seen_ods)
    if missing_route_ods:
        raise AssertionError(f"OD support lacks route seed coverage for {len(missing_route_ods)} OD pairs")

    world_slice_counts=Counter(); world_slice_mass=Counter()
    support_mass_by_world=Counter()
    for (world,fam,o,d,day,period),classes in slice_classes.items():
        label=summarise_slice(classes)
        n=next(x["journeys"] for x in support[(fam,o,d)] if x["day"]==day and x["period"]==period)
        world_slice_counts[(world,label)]+=1
        world_slice_mass[(world,label)]+=n
        support_mass_by_world[world]+=n

    expected_worlds=sorted(rq.get("worlds",[]))
    for world in expected_worlds:
        if support_mass_by_world[world]!=support_total:
            raise AssertionError((world,support_mass_by_world[world],support_total))

    result={
      "schema":"mppd.shanghai.first-science-service-screening.stage1.v2",
      "input_sha256":{
        "route_seeds":sha256_file(a.route_seeds),
        "route_qualification":sha256_file(a.route_qualification),
        "od_support":sha256_file(a.od_support),
        "progression_prior":sha256_file(a.prior),
        "service_worlds":sha256_file(a.service_worlds)
      },
      "route_seed_records":records,
      "route_vectors":vectors,
      "unique_supported_od_pairs":len(support),
      "support_journeys_once":support_total,
      "topology_worlds":expected_worlds,
      "vector_slice_classes":dict(counts),
      "vector_slice_reasons":dict(reasons),
      "by_family":{k:dict(v) for k,v in sorted(by_family.items())},
      "by_world":{k:dict(v) for k,v in sorted(by_world.items())},
      "world_slice_counts":{w:{label:world_slice_counts[(w,label)] for label in ["ONLY_PHASE_DEPENDENT","ONLY_SERVICE_UNRESOLVED","MIXED"] if world_slice_counts[(w,label)]} for w in expected_worlds},
      "world_slice_support_mass":{w:{label:world_slice_mass[(w,label)] for label in ["ONLY_PHASE_DEPENDENT","ONLY_SERVICE_UNRESOLVED","MIXED"] if world_slice_mass[(w,label)]} for w in expected_worlds},
      "progression_prior_edges":len(prior),
      "service_worlds_schema":worlds["schema"],
      "safe_deletion":"NONE_AT_STAGE1",
      "next_required":[
        "Compute timing bounds for tied geometries without substituting the representative path.",
        "Qualify or bound transfer walking/wait where evidence permits.",
        "Add service-specific timing envelopes before any posterior probability or route deletion."
      ],
      "boundaries":[
        "T_prog is a scheduled 2015 station-progression prior, not observed 2016 ATS.",
        "Service screening is evaluated by actual OD-support day and period slices.",
        "An unsupported day/period is SERVICE_UNRESOLVED, never silently promoted to PHASE_DEPENDENT.",
        "Topology worlds are alternative admissible worlds and are not probabilistically weighted; support mass must not be summed across worlds.",
        "Stage1 never uses nominal headway as a hard maximum wait.",
        "No route receives zero probability and no passenger-route posterior is estimated.",
        "Representative geometry is never substituted for a tied-path set when path_count>1."
      ]
    }
    (a.out_dir/"qualification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
