#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,hashlib,json
from collections import defaultdict
from pathlib import Path
import duckdb

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def route_clear_record(records, expected_worlds):
    if {r["world"] for r in records} != set(expected_worlds):
        return None
    if len(records)!=len(expected_worlds):
        return None
    baseline=None
    for r in records:
        vs=r.get("vectors",[])
        if len(vs)!=1:
            return None
        v=vs[0]
        if int(v.get("path_count",0))!=1 or int(v.get("line_changes",0))!=0:
            return None
        if int(v.get("prior_missing_ride_edges",0))!=0:
            return None
        if v.get("t_prog_median_s") is None:
            return None
        sig=(tuple(v.get("representative_path",[])),float(v["t_prog_median_s"]),int(v["ride_intervals"]))
        if baseline is None:
            baseline=sig
        elif sig!=baseline:
            return None
    return {
      "representative_path":list(baseline[0]),
      "t_prog_median_s":baseline[1],
      "ride_intervals":baseline[2]
    }

def family_line(family):
    if family.startswith("L11_"): return "11"
    if family.startswith("L2_"): return "02"
    raise ValueError(f"unsupported first-science family {family}")

def period_sql():
    return """
    CASE
      WHEN rc.origin_line='11' THEN
        CASE
          WHEN substr(j.entry_time,12,8)>='07:00:00' AND substr(j.entry_time,12,8)<'08:30:00' THEN 'AM_0700_0830'
          WHEN substr(j.entry_time,12,8)>='09:00:00' AND substr(j.entry_time,12,8)<'16:00:00' THEN 'OFF_0900_1600'
          WHEN substr(j.entry_time,12,8)>='17:00:00' AND substr(j.entry_time,12,8)<'19:30:00' THEN 'PM_1700_1930'
          ELSE 'OTHER' END
      WHEN rc.origin_line='02' THEN
        CASE
          WHEN substr(j.entry_time,12,8)>='07:00:00' AND substr(j.entry_time,12,8)<'09:00:00' THEN 'AM_0700_0900'
          WHEN substr(j.entry_time,12,8)>='09:00:00' AND substr(j.entry_time,12,8)<'16:00:00' THEN 'OFF_0900_1600'
          WHEN substr(j.entry_time,12,8)>='17:30:00' AND substr(j.entry_time,12,8)<'19:00:00' THEN 'PM_DOC_1730_1900'
          WHEN (substr(j.entry_time,12,8)>='17:00:00' AND substr(j.entry_time,12,8)<'17:30:00')
            OR (substr(j.entry_time,12,8)>='19:00:00' AND substr(j.entry_time,12,8)<'19:30:00') THEN 'PM_SHOULDER'
          ELSE 'OTHER' END
      ELSE 'OTHER'
    END
    """

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage1",type=Path,required=True)
    ap.add_argument("--stage1-qualification",type=Path,required=True)
    ap.add_argument("--journey-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)

    q1=json.loads(a.stage1_qualification.read_text(encoding="utf-8"))
    if q1.get("schema")!="mppd.shanghai.first-science-service-screening.stage1.v2":
        raise SystemExit(f"unexpected Stage1 schema: {q1.get('schema')}")
    worlds=q1.get("topology_worlds",[])
    if not worlds:
        raise SystemExit("Stage1 topology worlds missing")

    groups=defaultdict(list)
    with gzip.open(a.stage1,"rt",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            groups[(r["family"],r["origin"],r["destination"])].append(r)

    clear=[]
    expected_support=0
    for (fam,o,d),records in sorted(groups.items()):
        rc=route_clear_record(records,worlds)
        if rc is None:
            continue
        line=family_line(fam)
        # support_slices are identical across topology worlds; use one world only.
        slices=records[0]["vectors"][0].get("support_slices",[])
        support=sum(int(s["journeys"]) for s in slices)
        expected_support+=support
        clear.append({
          "family":fam,"origin_line":line,"origin_station":o,
          "destination_line":line,"destination_station":d,
          "t_prog_median_s":rc["t_prog_median_s"],
          "ride_intervals":rc["ride_intervals"],
          "support_journeys":support,
          "representative_path":rc["representative_path"]
        })

    if not clear:
        raise SystemExit("no topology-stable route-clear OD pairs found")

    con=duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute("""CREATE TABLE route_clear(
      family VARCHAR, origin_line VARCHAR, origin_station VARCHAR,
      destination_line VARCHAR, destination_station VARCHAR,
      t_prog_median_s DOUBLE, ride_intervals INTEGER, support_journeys BIGINT
    )""")
    con.executemany(
      "INSERT INTO route_clear VALUES (?,?,?,?,?,?,?,?)",
      [(r["family"],r["origin_line"],r["origin_station"],r["destination_line"],r["destination_station"],
        r["t_prog_median_s"],r["ride_intervals"],r["support_journeys"]) for r in clear]
    )

    rows=[]
    total_matched=0
    invalid_duration=0
    stitched=0
    parquet_hashes={}
    for day in ["20160701","20160802","20160901"]:
        p=a.journey_dir/f"SPTCC-{day}.formal_mppd_journeys.parquet"
        if not p.is_file(): raise FileNotFoundError(p)
        parquet_hashes[day]=sha256_file(p)
        ps=str(p).replace("'","''")
        invalid=con.execute(f"""
          SELECT count(*)
          FROM read_parquet('{ps}') j
          JOIN route_clear rc
            ON j.origin_line=rc.origin_line AND j.origin_station=rc.origin_station
           AND j.destination_line=rc.destination_line AND j.destination_station=rc.destination_station
          WHERE try_cast(j.duration_s AS DOUBLE) IS NULL OR try_cast(j.duration_s AS DOUBLE)<=0
        """).fetchone()[0]
        invalid_duration+=int(invalid)
        st=con.execute(f"""
          SELECT count(*)
          FROM read_parquet('{ps}') j
          JOIN route_clear rc
            ON j.origin_line=rc.origin_line AND j.origin_station=rc.origin_station
           AND j.destination_line=rc.destination_line AND j.destination_station=rc.destination_station
          WHERE try_cast(j.virtual_transfer_count AS INTEGER)>0
        """).fetchone()[0]
        stitched+=int(st)
        q=f"""
          SELECT
            rc.family,
            '{day}' AS day,
            {period_sql()} AS period,
            rc.origin_station,
            rc.destination_station,
            rc.ride_intervals,
            rc.t_prog_median_s,
            count(*)::BIGINT AS journeys,
            quantile_cont(cast(j.duration_s AS DOUBLE),0.10) AS duration_q10_s,
            quantile_cont(cast(j.duration_s AS DOUBLE),0.50) AS duration_q50_s,
            quantile_cont(cast(j.duration_s AS DOUBLE),0.90) AS duration_q90_s,
            quantile_cont(cast(j.duration_s AS DOUBLE)-rc.t_prog_median_s,0.10) AS residual_q10_s,
            quantile_cont(cast(j.duration_s AS DOUBLE)-rc.t_prog_median_s,0.50) AS residual_q50_s,
            quantile_cont(cast(j.duration_s AS DOUBLE)-rc.t_prog_median_s,0.90) AS residual_q90_s,
            sum(CASE WHEN cast(j.duration_s AS DOUBLE)-rc.t_prog_median_s<0 THEN 1 ELSE 0 END)::BIGINT AS negative_residual_journeys,
            sum(CASE WHEN try_cast(j.virtual_transfer_count AS INTEGER)>0 THEN 1 ELSE 0 END)::BIGINT AS stitched_journeys
          FROM read_parquet('{ps}') j
          JOIN route_clear rc
            ON j.origin_line=rc.origin_line AND j.origin_station=rc.origin_station
           AND j.destination_line=rc.destination_line AND j.destination_station=rc.destination_station
          WHERE cast(j.duration_s AS DOUBLE)>0
          GROUP BY 1,2,3,4,5,6,7
          ORDER BY 1,2,3,4,5
        """
        part=con.execute(q).fetchall()
        cols=[d[0] for d in con.description]
        for tup in part:
            rec=dict(zip(cols,tup))
            rows.append(rec); total_matched+=int(rec["journeys"])
    con.close()

    if invalid_duration:
        raise AssertionError(f"route-clear formal rows with invalid duration: {invalid_duration}")
    if total_matched!=expected_support:
        raise AssertionError(("route-clear support mismatch",total_matched,expected_support))

    csv_path=a.out_dir/"route_clear_afc_residual_panel.csv"
    with csv_path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)

    clear_path=a.out_dir/"route_clear_od_registry.json"
    clear_path.write_text(json.dumps({
      "schema":"mppd.shanghai.route-clear-od-registry.v1",
      "topology_worlds":worlds,
      "route_clear_od_count":len(clear),
      "route_clear_support_journeys":expected_support,
      "records":clear
    },ensure_ascii=False,indent=2),encoding="utf-8")

    neg=sum(int(r["negative_residual_journeys"]) for r in rows)
    result={
      "schema":"mppd.shanghai.route-clear-afc-residual-audit.v1",
      "input_sha256":{
        "stage1":sha256_file(a.stage1),
        "stage1_qualification":sha256_file(a.stage1_qualification),
        "parquet":parquet_hashes
      },
      "topology_worlds":worlds,
      "route_clear_od_count":len(clear),
      "route_clear_support_journeys":expected_support,
      "matched_formal_journeys":total_matched,
      "panel_rows":len(rows),
      "negative_residual_journeys":neg,
      "stitched_journeys":stitched,
      "scientific_status":"PASSENGER_FACING_DIAGNOSTIC_NOT_WAIT_TIME_TRUTH",
      "boundaries":[
        "Route-clear means one identical no-transfer path with path_count=1 and complete scheduled-progression prior in every topology world.",
        "Residual = observed AFC gate-to-gate duration minus 2015 scheduled-progression median coordinate.",
        "Residual is not pure platform waiting time; it may include access/egress, dwell variation, service-phase effects and cross-year schedule mismatch.",
        "Selection is structural and topology-stability based, not outcome-selected.",
        "This diagnostic may inform a later service likelihood but cannot by itself identify ATS or authorize route deletion.",
        "Only aggregate OD/day/period statistics are persisted; no passenger identifier or row-level output is emitted."
      ]
    }
    (a.out_dir/"qualification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
