#!/usr/bin/env python3
from __future__ import annotations
import argparse,gzip,json
from collections import Counter,defaultdict
from pathlib import Path
from shanghai_route_frontier import solve_frontier
from shanghai_first_science_service_screen_stage1 import prior_lookup,sha256_file,state_parts

Q=("p10","median","p90")

def label_progression_bounds(labels,targets,cost,prior):
    memo={}
    entries=[]
    for node,by_cost in labels.items():
        for key in by_cost:
            entries.append((sum(key),key[0],key[1],node,key))
    entries.sort()
    for _,_,_,node,key in entries:
        rec=labels[node][key]
        if not rec["parents"]:
            metric={"total_paths":1,"covered_paths":1}
            for q in Q:
                metric[f"{q}_sum_min_s"]=0.0
                metric[f"{q}_sum_max_s"]=0.0
            memo[(node,key)]=metric
            continue
        total=0; covered=0
        mins={q:None for q in Q}; maxs={q:None for q in Q}
        for parent,pcost in rec["parents"]:
            pm=memo[(parent,pcost)]
            total+=pm["total_paths"]
            dr=key[0]-pcost[0]; dt=key[1]-pcost[1]
            if (dr,dt) not in {(1,0),(0,1)}:
                raise AssertionError((parent,node,pcost,key))
            edge=None
            if dr==1:
                lp,np=state_parts(parent); ln,nn=state_parts(node)
                if lp!=ln:
                    raise AssertionError(("ride-cross-line",parent,node))
                edge=prior.get((lp,frozenset((np,nn))))
                if edge is None:
                    continue
            if pm["covered_paths"]==0:
                continue
            covered+=pm["covered_paths"]
            for q in Q:
                add=0.0 if edge is None else edge[q]
                lo=pm[f"{q}_sum_min_s"]+add
                hi=pm[f"{q}_sum_max_s"]+add
                mins[q]=lo if mins[q] is None else min(mins[q],lo)
                maxs[q]=hi if maxs[q] is None else max(maxs[q],hi)
        if total!=rec["count"]:
            raise AssertionError(("label-count-mismatch",node,key,total,rec["count"]))
        metric={"total_paths":total,"covered_paths":covered}
        for q in Q:
            metric[f"{q}_sum_min_s"]=mins[q]
            metric[f"{q}_sum_max_s"]=maxs[q]
        memo[(node,key)]=metric

    total=covered=0
    mins={q:None for q in Q}; maxs={q:None for q in Q}
    for t in sorted(set(targets)):
        if cost not in labels.get(t,{}):
            continue
        m=memo[(t,cost)]
        total+=m["total_paths"];covered+=m["covered_paths"]
        if m["covered_paths"]:
            for q in Q:
                lo=m[f"{q}_sum_min_s"]; hi=m[f"{q}_sum_max_s"]
                mins[q]=lo if mins[q] is None else min(mins[q],lo)
                maxs[q]=hi if maxs[q] is None else max(maxs[q],hi)
    out={"total_paths":total,"fully_progression_covered_paths":covered,
         "progression_coverage_share":covered/total if total else None}
    for q in Q:
        out[f"covered_path_{q}_sum_min_s"]=mins[q]
        out[f"covered_path_{q}_sum_max_s"]=maxs[q]
    out["all_paths_progression_covered"]=(total>0 and covered==total)
    out["all_tied_paths_progression_equivalent"]=(
        total>1 and covered==total and
        all(mins[q] is not None and mins[q]==maxs[q] for q in Q)
    )
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage1",type=Path,required=True)
    ap.add_argument("--stage1-qualification",type=Path,required=True)
    ap.add_argument("--station",type=Path,required=True)
    ap.add_argument("--connections",type=Path,required=True)
    ap.add_argument("--prior",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    q1=json.loads(a.stage1_qualification.read_text(encoding="utf-8"))
    if q1.get("schema")!="mppd.shanghai.first-science-service-screening.stage1.v2":
        raise SystemExit(f"unexpected Stage1 schema: {q1.get('schema')}")

    # Deferred import keeps the audit helper independently testable without DuckDB.
    from shanghai_first_science_route_seeds import build_worlds

    prior=prior_lookup(a.prior)
    worlds=build_worlds(a.station,a.connections)
    grouped=defaultdict(list)
    total_records=0; tied_vectors=0; tied_path_instances=0
    with gzip.open(a.stage1,"rt",encoding="utf-8") as f:
        for line in f:
            rec=json.loads(line); total_records+=1
            tv=[v for v in rec["vectors"] if int(v["path_count"])>1]
            if not tv:
                continue
            grouped[(rec["world"],rec["origin"])].append((rec,tv))
            tied_vectors+=len(tv)
            tied_path_instances+=sum(int(v["path_count"]) for v in tv)

    stats=Counter(); audited=[]
    for (world_name,origin),items in sorted(grouped.items()):
        world=worlds.get(world_name)
        if world is None:
            raise AssertionError(f"unknown topology world {world_name}")
        groups=world["endpoint_groups"]
        if origin not in groups:
            raise AssertionError((world_name,"missing-origin",origin))
        from shanghai_route_frontier import build_adjacency
        adj=build_adjacency(world["nodes"],world["edges"])
        labels=solve_frontier(adj,groups[origin])
        for rec,vectors in items:
            destination=rec["destination"]
            if destination not in groups:
                raise AssertionError((world_name,"missing-destination",destination))
            for v in vectors:
                cost=(int(v["ride_intervals"]),int(v["line_changes"]))
                b=label_progression_bounds(labels,groups[destination],cost,prior)
                if b["total_paths"]!=int(v["path_count"]):
                    raise AssertionError(("seed-count-mismatch",world_name,origin,destination,cost,b["total_paths"],v["path_count"]))
                stats["tied_vectors_audited"]+=1
                stats["tied_path_instances_audited"]+=b["total_paths"]
                if b["all_paths_progression_covered"]:
                    stats["all_paths_progression_covered_vectors"]+=1
                elif b["fully_progression_covered_paths"]>0:
                    stats["partial_progression_coverage_vectors"]+=1
                else:
                    stats["no_progression_coverage_vectors"]+=1
                if b["all_tied_paths_progression_equivalent"]:
                    stats["progression_equivalent_tied_vectors"]+=1
                audited.append({
                  "world":world_name,"family":rec["family"],"origin":origin,"destination":destination,
                  "ride_intervals":cost[0],"line_changes":cost[1],"seed_path_count":int(v["path_count"]),
                  "audit":b
                })

    if stats["tied_vectors_audited"]!=tied_vectors:
        raise AssertionError((stats["tied_vectors_audited"],tied_vectors))
    if stats["tied_path_instances_audited"]!=tied_path_instances:
        raise AssertionError((stats["tied_path_instances_audited"],tied_path_instances))

    out_jsonl=a.out_dir/"tied_geometry_progression_audit.jsonl.gz"
    with gzip.open(out_jsonl,"wt",encoding="utf-8") as f:
        for rec in audited:
            f.write(json.dumps(rec,ensure_ascii=False,separators=(",",":"))+"\n")
    result={
      "schema":"mppd.shanghai.first-science.tied-geometry-progression-audit.v1",
      "input_sha256":{
        "stage1":sha256_file(a.stage1),
        "stage1_qualification":sha256_file(a.stage1_qualification),
        "station":sha256_file(a.station),
        "connections":sha256_file(a.connections),
        "progression_prior":sha256_file(a.prior)
      },
      "stage1_records_seen":total_records,
      "tied_vectors_expected":tied_vectors,
      "tied_path_instances_expected":tied_path_instances,
      "stats":dict(stats),
      "scientific_status":"DIAGNOSTIC_ONLY_NO_ROUTE_DELETION",
      "boundaries":[
        "The audit operates on the exact frontier parent DAG and never substitutes the representative tied geometry.",
        "Path-count equality with Stage1 is a hard gate.",
        "Additive p10/median/p90 progression sums are scheduled-prior coordinates, not quantiles of realized 2016 travel time.",
        "Transfer walking/wait and service phase remain unresolved and are not added as zero-time operational truth.",
        "Progression equivalence alone does not authorize route deletion or posterior probability assignment."
      ]
    }
    (a.out_dir/"qualification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
