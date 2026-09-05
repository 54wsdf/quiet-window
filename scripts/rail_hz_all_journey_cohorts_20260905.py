from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

MAX_TRIP_S = 8 * 3600
USECOLS = ["time", "lineID", "stationID", "status", "userID"]


def build_day(input_path: Path, source_file: str, cohorts_output: Path, summary_output: Path) -> dict:
    df = pd.read_csv(
        input_path,
        compression="gzip",
        usecols=USECOLS,
        dtype={"lineID":"string", "stationID":"int16", "status":"int8", "userID":"string"},
    )
    raw_rows = int(len(df))
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    df = df.sort_values(["userID", "time"], kind="mergesort")
    g = df.groupby("userID", sort=False, observed=True)
    df["prev_time"] = g["time"].shift(1)
    df["prev_status"] = g["status"].shift(1)
    df["prev_line"] = g["lineID"].shift(1)
    df["prev_station"] = g["stationID"].shift(1)

    pair = df[(df["status"] == 0) & (df["prev_status"] == 1)].copy()
    pair["duration_s"] = (pair["time"] - pair["prev_time"]).dt.total_seconds()
    pair = pair[(pair["duration_s"] >= 0) & (pair["duration_s"] <= MAX_TRIP_S)].copy()
    pair["origin_line"] = pair["prev_line"].astype("string")
    pair["origin_station"] = pair["prev_station"].astype("int16")
    pair["destination_line"] = pair["lineID"].astype("string")
    pair["destination_station"] = pair["stationID"].astype("int16")
    pair["entry_time"] = pair["prev_time"]
    pair["exit_time"] = pair["time"]

    pair_count = int(len(pair))
    same_station_count = int((pair["origin_station"] == pair["destination_station"]).sum())
    cross_line_count = int((pair["origin_line"] != pair["destination_line"]).sum())
    unique_users_with_valid_pair = int(pair["userID"].nunique())

    keys = ["origin_line","origin_station","destination_line","destination_station","entry_time","exit_time"]
    cohorts = pair.groupby(keys, sort=True, observed=True).size().rename("passenger_mass").reset_index()
    cohort_count = int(len(cohorts))
    cohorts.to_csv(cohorts_output, index=False, compression="gzip", date_format="%Y-%m-%d %H:%M:%S")

    user_size = g.size()
    exact2_users = user_size[user_size == 2].index
    exact2 = df[df["userID"].isin(exact2_users)]
    eg = exact2.groupby("userID", sort=False, observed=True)
    first_status = eg["status"].first()
    last_status = eg["status"].last()
    exact2_10_users = int(((first_status == 1) & (last_status == 0)).sum())

    source_date = source_file.removeprefix("record_").removesuffix(".csv.gz")
    durations = pair["duration_s"]
    summary = {
        "schema": "rail.hz-all-consecutive-journey-cohorts.v1",
        "dataset_id": "CN_HZ_Tianchi_2019",
        "source_date": source_date,
        "source_file": source_file,
        "raw_afc_rows": raw_rows,
        "unique_user_count": int(df["userID"].nunique()),
        "all_consecutive_valid_1_to_0_journeys": pair_count,
        "unique_users_with_valid_journey": unique_users_with_valid_pair,
        "exact_second_cohort_count": cohort_count,
        "cohort_compression_ratio": cohort_count / pair_count if pair_count else None,
        "same_station_journey_count": same_station_count,
        "cross_line_journey_count": cross_line_count,
        "cross_line_journey_share": cross_line_count / pair_count if pair_count else None,
        "travel_time_sec": {
            "median": float(durations.quantile(0.5)) if pair_count else None,
            "p90": float(durations.quantile(0.9)) if pair_count else None,
            "p99": float(durations.quantile(0.99)) if pair_count else None,
            "max": float(durations.max()) if pair_count else None,
        },
        "exact_two_event_benchmark": {
            "exact_two_event_users": int(len(exact2_users)),
            "exact_two_event_1_to_0_users": exact2_10_users,
            "role": "HIGH_CONFIDENCE_BENCHMARK_ONLY_NOT_FORMAL_PASSENGER_DENOMINATOR",
        },
        "formal_passenger_domain": {
            "raw_events_retained_as_observation_world": raw_rows,
            "journey_factors": "ALL_CONSECUTIVE_VALID_1_TO_0_PAIRS",
            "multi_trip_users_retained": True,
            "unpaired_events_are_not_deleted_from_observation_world": True,
            "user_id_persisted": False,
        },
        "calendar_boundary_note": "This file pairs events within the source file. Full-service-day cross-midnight reassignment must be decided from the HZ25 activity-boundary audit before formal R1B certification.",
        "cohort_file": cohorts_output.name,
    }
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["source_date","raw_afc_rows","unique_user_count","all_consecutive_valid_1_to_0_journeys","exact_second_cohort_count"]}, ensure_ascii=False))
    return summary


def aggregate(summary_dir: Path, output: Path) -> dict:
    paths = sorted(summary_dir.glob("record_2019-01-*.journey_summary.json"))
    days = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    expected = [f"2019-01-{d:02d}" for d in range(1,26)]
    actual = [x["source_date"] for x in days]
    total_rows = sum(x["raw_afc_rows"] for x in days)
    total_pairs = sum(x["all_consecutive_valid_1_to_0_journeys"] for x in days)
    total_cohorts = sum(x["exact_second_cohort_count"] for x in days)
    exact2_10 = sum(x["exact_two_event_benchmark"]["exact_two_event_1_to_0_users"] for x in days)
    gates = {
        "all_25_calendar_sources": len(days)==25 and actual==expected,
        "all_raw_events_accounted": total_rows==58_637_237,
        "all_consecutive_domain_strictly_contains_exact_two_benchmark": total_pairs > exact2_10,
        "every_day_has_consecutive_journeys": all(x["all_consecutive_valid_1_to_0_journeys"]>0 for x in days),
        "multi_trip_users_not_excluded_by_design": all(x["formal_passenger_domain"]["multi_trip_users_retained"] for x in days),
        "no_user_ids_persisted": all(not x["formal_passenger_domain"]["user_id_persisted"] for x in days),
    }
    result = {
        "schema": "rail.hz-25day-all-consecutive-journey-cohorts.v1",
        "status": "QUALIFIED_CALENDAR_PAIRING_DOMAIN_PENDING_SERVICE_DAY_BOUNDARY" if all(gates.values()) else "JOURNEY_DOMAIN_GATE_FAILED",
        "days": len(days),
        "dates": actual,
        "total_raw_afc_rows": total_rows,
        "total_all_consecutive_valid_1_to_0_journeys": total_pairs,
        "total_exact_second_cohorts": total_cohorts,
        "total_exact_two_1_to_0_benchmark": exact2_10,
        "formal_denominator_expansion_over_exact_two": total_pairs - exact2_10,
        "integrity_gates": gates,
        "next_gate": "SERVICE_DAY_BOUNDARY_AND_CROSS_MIDNIGHT_REASSIGNMENT",
        "day_summaries": days,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "total_raw_afc_rows": total_rows,
        "total_all_consecutive_valid_1_to_0_journeys": total_pairs,
        "total_exact_second_cohorts": total_cohorts,
        "formal_denominator_expansion_over_exact_two": result["formal_denominator_expansion_over_exact_two"],
        "integrity_gates": gates,
    }, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)
    return result


def main() -> None:
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command", required=True)
    s=sub.add_parser("day")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--source-file", required=True)
    s.add_argument("--cohorts-output", type=Path, required=True)
    s.add_argument("--summary-output", type=Path, required=True)
    s=sub.add_parser("aggregate")
    s.add_argument("--summary-dir", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    if args.command=="day":
        build_day(args.input,args.source_file,args.cohorts_output,args.summary_output)
    else:
        aggregate(args.summary_dir,args.output)


if __name__=="__main__":
    main()
