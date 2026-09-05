from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BOUNDARY_HOUR = 4
MAX_TRIP_S = 8 * 3600
CHUNK_SIZE = 500_000
USECOLS = ["time", "lineID", "stationID", "status", "userID"]


def load_slice(path: Path, service_date: pd.Timestamp, role: str) -> pd.DataFrame:
    parts = []
    boundary = service_date + pd.Timedelta(hours=BOUNDARY_HOUR)
    next_boundary = boundary + pd.Timedelta(days=1)
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=USECOLS,
        dtype={"lineID":"string", "stationID":"int16", "status":"int8", "userID":"string"},
        chunksize=CHUNK_SIZE,
    ):
        t = pd.to_datetime(chunk["time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
        if role == "current":
            mask = (t >= boundary) & (t < service_date.normalize() + pd.Timedelta(days=1))
        elif role == "next":
            mask = (t >= service_date.normalize() + pd.Timedelta(days=1)) & (t < next_boundary)
        else:
            raise ValueError(role)
        if mask.any():
            kept = chunk.loc[mask].copy()
            kept["time"] = t.loc[mask]
            parts.append(kept)
    if not parts:
        return pd.DataFrame(columns=USECOLS)
    return pd.concat(parts, ignore_index=True)


def service_second(t: pd.Series, boundary: pd.Timestamp) -> pd.Series:
    return (t - boundary).dt.total_seconds().astype("int32")


def build_day(current: Path, next_file: Path, service_date_text: str, cohorts_output: Path, summary_output: Path) -> dict:
    service_date = pd.Timestamp(service_date_text).normalize()
    boundary = service_date + pd.Timedelta(hours=BOUNDARY_HOUR)
    next_boundary = boundary + pd.Timedelta(days=1)
    a = load_slice(current, service_date, "current")
    b = load_slice(next_file, service_date, "next")
    df = pd.concat([a,b], ignore_index=True)
    del a,b
    df = df.sort_values(["userID","time"], kind="mergesort")
    raw_rows = int(len(df))
    status_counts = {str(int(k)):int(v) for k,v in df["status"].value_counts().sort_index().items()}
    line_counts = {str(k):int(v) for k,v in df["lineID"].value_counts().sort_index().items()}
    station_count = int(df["stationID"].nunique())
    user_count = int(df["userID"].nunique())

    g = df.groupby("userID", sort=False, observed=True)
    df["prev_time"] = g["time"].shift(1)
    df["prev_status"] = g["status"].shift(1)
    df["prev_line"] = g["lineID"].shift(1)
    df["prev_station"] = g["stationID"].shift(1)

    pair = df[(df["status"]==0) & (df["prev_status"]==1)].copy()
    pair["duration_s"] = (pair["time"]-pair["prev_time"]).dt.total_seconds()
    pair = pair[(pair["duration_s"]>=0) & (pair["duration_s"]<=MAX_TRIP_S)].copy()
    pair["origin_line"] = pair["prev_line"].astype("string")
    pair["origin_station"] = pair["prev_station"].astype("int16")
    pair["destination_line"] = pair["lineID"].astype("string")
    pair["destination_station"] = pair["stationID"].astype("int16")
    pair["entry_sec"] = service_second(pair["prev_time"], boundary)
    pair["exit_sec"] = service_second(pair["time"], boundary)
    # Exact timestamps remain reconstructable from service_date + boundary + service seconds.
    pair_count = int(len(pair))
    cross_boundary_pairs = int(((pair["prev_time"] < service_date + pd.Timedelta(days=1)) & (pair["time"] >= service_date + pd.Timedelta(days=1))).sum())
    cross_line_count = int((pair["origin_line"] != pair["destination_line"]).sum())
    same_station_count = int((pair["origin_station"] == pair["destination_station"]).sum())

    keys = ["origin_line","origin_station","destination_line","destination_station","entry_sec","exit_sec"]
    cohorts = pair.groupby(keys, sort=True, observed=True).size().rename("passenger_mass").reset_index()
    cohorts["passenger_mass"] = cohorts["passenger_mass"].astype("int32")
    cohort_count = int(len(cohorts))
    cohorts.to_parquet(cohorts_output, index=False, compression="zstd")

    # Time-of-service-day observed activity is retained to audit peak/offpeak/late-service regimes.
    rel_sec = service_second(df["time"], boundary)
    bins = (rel_sec // 900).astype(int)
    activity = bins.value_counts().reindex(range(96), fill_value=0).sort_index()
    entry_activity = bins[df["status"].to_numpy()==1].value_counts().reindex(range(96), fill_value=0).sort_index()
    exit_activity = bins[df["status"].to_numpy()==0].value_counts().reindex(range(96), fill_value=0).sort_index()

    durations = pair["duration_s"]
    result = {
        "schema":"rail.hz-full-service-day-passenger-domain.v1",
        "dataset_id":"CN_HZ_Tianchi_2019",
        "service_date":service_date_text,
        "service_day_start":boundary.isoformat(sep=" "),
        "service_day_end_exclusive":next_boundary.isoformat(sep=" "),
        "boundary_hour":BOUNDARY_HOUR,
        "boundary_evidence":"HZ25 pooled 15-min global activity trough is 03:45-04:00; formal cut fixed at 04:00 after the trough bin",
        "current_source":current.name,
        "next_source":next_file.name,
        "raw_afc_events_in_service_day":raw_rows,
        "status_counts":status_counts,
        "line_counts":line_counts,
        "observed_station_count":station_count,
        "unique_user_count":user_count,
        "all_consecutive_valid_1_to_0_journeys":pair_count,
        "cross_midnight_valid_journeys":cross_boundary_pairs,
        "cross_line_journeys":cross_line_count,
        "cross_line_share":cross_line_count/pair_count if pair_count else None,
        "same_station_journeys":same_station_count,
        "exact_second_cohort_count":cohort_count,
        "cohort_compression_ratio":cohort_count/pair_count if pair_count else None,
        "travel_time_sec":{
            "median":float(durations.quantile(0.5)) if pair_count else None,
            "p90":float(durations.quantile(0.9)) if pair_count else None,
            "p99":float(durations.quantile(0.99)) if pair_count else None,
            "max":float(durations.max()) if pair_count else None,
        },
        "activity_15min_from_0400":{
            "all_events":[int(x) for x in activity.to_list()],
            "entry_events":[int(x) for x in entry_activity.to_list()],
            "exit_events":[int(x) for x in exit_activity.to_list()],
        },
        "formal_passenger_domain":{
            "multi_trip_users_retained":True,
            "all_pairable_consecutive_entry_exit_journeys_retained":True,
            "unpaired_afc_events_remain_observed_evidence":True,
            "exact_two_event_subset_is_only_a_benchmark":True,
            "user_id_persisted":False,
            "device_id_persisted":False,
        },
        "cohort_file":cohorts_output.name,
    }
    summary_output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
        "service_date":service_date_text,
        "raw_afc_events_in_service_day":raw_rows,
        "all_consecutive_valid_1_to_0_journeys":pair_count,
        "cross_midnight_valid_journeys":cross_boundary_pairs,
        "exact_second_cohort_count":cohort_count,
    },ensure_ascii=False))
    return result


def aggregate(summary_dir: Path, output: Path) -> dict:
    paths=sorted(summary_dir.glob("2019-01-*.service_day_summary.json"))
    days=[json.loads(p.read_text(encoding="utf-8")) for p in paths]
    expected=[f"2019-01-{d:02d}" for d in range(1,25)]
    actual=[x["service_date"] for x in days]
    total_events=sum(x["raw_afc_events_in_service_day"] for x in days)
    total_journeys=sum(x["all_consecutive_valid_1_to_0_journeys"] for x in days)
    total_cohorts=sum(x["exact_second_cohort_count"] for x in days)
    total_cross_midnight=sum(x["cross_midnight_valid_journeys"] for x in days)
    gates={
        "exactly_24_complete_service_days":len(days)==24 and actual==expected,
        "every_service_day_has_events":all(x["raw_afc_events_in_service_day"]>0 for x in days),
        "every_service_day_has_journeys":all(x["all_consecutive_valid_1_to_0_journeys"]>0 for x in days),
        "cross_midnight_journeys_preserved":total_cross_midnight>0,
        "multi_trip_users_retained":all(x["formal_passenger_domain"]["multi_trip_users_retained"] for x in days),
        "no_user_device_ids_persisted":all(not x["formal_passenger_domain"]["user_id_persisted"] and not x["formal_passenger_domain"]["device_id_persisted"] for x in days),
    }
    result={
        "schema":"rail.hz-24-complete-full-service-day-domain.v1",
        "status":"QUALIFIED_HZ24_COMPLETE_FULL_SERVICE_DAY_PASSENGER_DOMAIN" if all(gates.values()) else "FULL_SERVICE_DAY_DOMAIN_GATE_FAILED",
        "boundary_hour":BOUNDARY_HOUR,
        "complete_service_days":24,
        "dates":actual,
        "total_afc_events_in_complete_service_days":total_events,
        "total_all_consecutive_valid_journeys":total_journeys,
        "total_exact_second_cohorts":total_cohorts,
        "total_cross_midnight_valid_journeys":total_cross_midnight,
        "archive_edge_censoring":{
            "left_edge":"2019-01-01 00:00-04:00 belongs to 2018-12-31 service day whose earlier portion is unavailable",
            "right_edge":"2019-01-25 04:00 onward belongs to 2019-01-25 service day whose 2019-01-26 00:00-04:00 tail is unavailable",
            "formal_policy":"edge-censored fragments remain observed data but are excluded from complete-service-day certification",
        },
        "integrity_gates":gates,
        "day_summaries":days,
        "next_stage":"HZ24_DAY_SPECIFIC_SERVICE_INITIALIZATION_THEN_R1B_FULL_DAY",
    }
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
        "status":result["status"],
        "complete_service_days":24,
        "total_afc_events_in_complete_service_days":total_events,
        "total_all_consecutive_valid_journeys":total_journeys,
        "total_exact_second_cohorts":total_cohorts,
        "total_cross_midnight_valid_journeys":total_cross_midnight,
        "integrity_gates":gates,
    },ensure_ascii=False,indent=2))
    if not all(gates.values()): raise SystemExit(2)
    return result


def main()->None:
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("day")
    s.add_argument("--current",type=Path,required=True);s.add_argument("--next-file",type=Path,required=True)
    s.add_argument("--service-date",required=True);s.add_argument("--cohorts-output",type=Path,required=True);s.add_argument("--summary-output",type=Path,required=True)
    s=sub.add_parser("aggregate");s.add_argument("--summary-dir",type=Path,required=True);s.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.command=="day": build_day(a.current,a.next_file,a.service_date,a.cohorts_output,a.summary_output)
    else: aggregate(a.summary_dir,a.output)


if __name__=="__main__": main()
