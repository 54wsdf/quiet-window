#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import duckdb

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--strict",type=Path,required=True)
    ap.add_argument("--formal",type=Path,required=True)
    ap.add_argument("--formal-audit",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    audit=json.loads(a.formal_audit.read_text(encoding="utf-8"))

    con=duckdb.connect(":memory:")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("""
      CREATE TABLE strict AS
      SELECT
        strftime(entry_ts,'%Y-%m-%d %H:%M:%S') entry_time,
        strftime(exit_ts,'%Y-%m-%d %H:%M:%S') exit_time,
        entry_station origin_token,
        exit_station destination_token,
        duration_sec::BIGINT duration_s,
        count(*)::BIGINT n
      FROM read_parquet(?)
      GROUP BY 1,2,3,4,5
    """,[str(a.strict)])
    con.execute("""
      CREATE TABLE formal AS
      SELECT entry_time,exit_time,origin_token,destination_token,
             try_cast(duration_s AS BIGINT) duration_s,
             try_cast(virtual_transfer_count AS INTEGER) virtual_transfer_count,
             count(*)::BIGINT n
      FROM read_csv_auto(?,header=true,all_varchar=true)
      GROUP BY 1,2,3,4,5,6
    """,[str(a.formal)])

    strict_rows=con.execute("SELECT sum(n) FROM strict").fetchone()[0]
    formal_rows=con.execute("SELECT sum(n) FROM formal").fetchone()[0]
    nonstitched=con.execute("SELECT coalesce(sum(n),0) FROM formal WHERE virtual_transfer_count=0").fetchone()[0]
    stitched=con.execute("SELECT coalesce(sum(n),0) FROM formal WHERE virtual_transfer_count>0").fetchone()[0]
    stitch_ops=con.execute("SELECT coalesce(sum(n*virtual_transfer_count),0) FROM formal").fetchone()[0]

    con.execute("""
      CREATE TABLE f0 AS SELECT entry_time,exit_time,origin_token,destination_token,duration_s,sum(n)::BIGINT n
      FROM formal WHERE virtual_transfer_count=0 GROUP BY 1,2,3,4,5
    """)
    exact=con.execute("""
      SELECT coalesce(sum(least(s.n,f.n)),0)
      FROM strict s JOIN f0 f USING(entry_time,exit_time,origin_token,destination_token,duration_s)
    """).fetchone()[0]
    formal_nonstitched_unmatched=nonstitched-exact
    strict_not_exactly_retained=strict_rows-exact

    # OD-level comparison is robust to individual row multiplicity and useful for
    # locating where chronology barriers remove the most strict mass.
    od=con.execute("""
      WITH s AS (
        SELECT origin_token,destination_token,sum(n)::BIGINT strict_n FROM strict GROUP BY 1,2
      ), f AS (
        SELECT origin_token,destination_token,sum(n)::BIGINT formal_n FROM formal GROUP BY 1,2
      )
      SELECT coalesce(s.origin_token,f.origin_token) origin_token,
             coalesce(s.destination_token,f.destination_token) destination_token,
             coalesce(strict_n,0)::BIGINT strict_n,
             coalesce(formal_n,0)::BIGINT formal_n,
             (coalesce(formal_n,0)-coalesce(strict_n,0))::BIGINT delta
      FROM s FULL OUTER JOIN f USING(origin_token,destination_token)
      ORDER BY abs(delta) DESC, origin_token, destination_token
    """).fetchdf()
    od.to_csv(a.out.with_name("od_delta.csv"),index=False)

    result={
      "schema":"mppd.shanghai.formal-vs-strict-reconciliation.v1",
      "strict_rows":int(strict_rows),
      "formal_rows":int(formal_rows),
      "formal_nonstitched_rows":int(nonstitched),
      "formal_stitched_rows":int(stitched),
      "virtual_stitch_operations_from_rows":int(stitch_ops),
      "audit_gate_segments":int(audit.get("gate_segments",0)),
      "audit_journeys":int(audit.get("journeys",0)),
      "audit_virtual_transfer_stitches":int(audit.get("virtual_transfer_stitches",0)),
      "exact_retained_strict_segment_rows":int(exact),
      "strict_rows_not_exactly_retained":int(strict_not_exactly_retained),
      "formal_nonstitched_rows_not_in_strict":int(formal_nonstitched_unmatched),
      "invariants":{
        "formal_matches_audit":int(formal_rows)==int(audit.get("journeys",0)),
        "stitch_mass_matches_audit":int(stitch_ops)==int(audit.get("virtual_transfer_stitches",0)),
        "nonstitched_formal_subset_of_strict":int(formal_nonstitched_unmatched)==0,
        "journey_mass_equals_gate_minus_stitches":int(audit.get("journeys",0))==int(audit.get("gate_segments",0))-int(audit.get("virtual_transfer_stitches",0))
      },
      "barrier_counters":{k:v for k,v in audit.items() if "barrier" in k or "duplicate" in k or "over_max" in k or k in ("entry_before_exit","exit_without_entry","open_entry_at_end")},
      "interpretation_boundary":[
        "Exact signature reconciliation compares non-stitched formal journeys with pre-existing strict gate segments.",
        "Stitched formal journeys are composite rows and are expected not to match one strict segment signature.",
        "A barrier counter is an event/state-machine diagnostic, not automatically a one-to-one count of removed strict journeys.",
        "This reconciliation does not assign routes, trains, or service posterior."
      ]
    }
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not all(result["invariants"].values()):
        raise SystemExit("reconciliation invariant failed")

if __name__=="__main__":
    main()
