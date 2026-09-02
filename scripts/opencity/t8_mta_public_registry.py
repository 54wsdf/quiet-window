#!/usr/bin/env python3
"""T8 public_data registry rehearsal for MTA subway.

Inputs are deliberately separated by provenance upstream:
- official MTA OD estimate slice (28vm-gjqr),
- official MTA hourly ridership slice (5wq4-mkjj),
- official MTA static GTFS,
- one declared capacity assumption supplied on the command line.

The baseline follows the task-card idea of schedule-proportional loading. For a
single GTFS route corridor, observed hourly station entries are split to
corridor destinations using the public OD destination shares, routed along the
route's physical station order, then distributed over scheduled runs in
proportion to declared train capacity. Capacity does not constrain demand and
there is no reassignment.

This is a public-source rehearsal. It does not manufacture organizer-held APC,
waiting-time, or crowding-label truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def norm_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def read_gtfs(path: Path):
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        def rd(name):
            if name not in names:
                return pd.DataFrame()
            return pd.read_csv(z.open(name), dtype=str)
        return {
            "stops": rd("stops.txt"),
            "routes": rd("routes.txt"),
            "trips": rd("trips.txt"),
            "stop_times": rd("stop_times.txt"),
            "calendar": rd("calendar.txt"),
            "calendar_dates": rd("calendar_dates.txt"),
            "feed_info": rd("feed_info.txt"),
        }


def top_level_stop_map(stops: pd.DataFrame):
    stops = stops.copy()
    stops["stop_id"] = stops["stop_id"].astype(str)
    parent = stops.get("parent_station", pd.Series("", index=stops.index)).fillna("").astype(str)
    stops["top_stop_id"] = np.where(parent.str.len() > 0, parent, stops["stop_id"])
    top = stops.drop_duplicates("top_stop_id").set_index("top_stop_id")
    # For parent rows, prefer their own coordinates/name when present.
    for sid, row in stops[stops["stop_id"] == stops["top_stop_id"]].iterrows():
        pass
    return dict(zip(stops["stop_id"], stops["top_stop_id"])), top


def select_route_id(routes: pd.DataFrame, requested: str) -> str:
    ids = set(routes.get("route_id", pd.Series(dtype=str)).astype(str))
    if requested in ids:
        return requested
    short = routes.get("route_short_name", pd.Series("", index=routes.index)).fillna("").astype(str)
    hit = routes.loc[short == requested, "route_id"] if "route_id" in routes else pd.Series(dtype=str)
    if len(hit):
        return str(hit.iloc[0])
    raise RuntimeError(f"route {requested!r} not found; sample route_ids={sorted(ids)[:20]}")


def canonical_route_sequence(gtfs, route_id: str, stop_to_top: dict[str, str]):
    trips = gtfs["trips"]
    st = gtfs["stop_times"].copy()
    rt = trips[trips["route_id"].astype(str) == route_id][["trip_id"]]
    st = st[st["trip_id"].isin(rt["trip_id"])].copy()
    st["stop_sequence_num"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.dropna(subset=["stop_sequence_num"]).sort_values(["trip_id", "stop_sequence_num"])
    st["top_stop_id"] = st["stop_id"].astype(str).map(stop_to_top)
    candidates = []
    for trip_id, g in st.groupby("trip_id", sort=False):
        seq = []
        for sid in g["top_stop_id"].dropna().astype(str):
            if not seq or seq[-1] != sid:
                seq.append(sid)
        if len(seq) >= 2:
            candidates.append((len(set(seq)), len(seq), str(trip_id), seq))
    if not candidates:
        raise RuntimeError(f"no usable stop sequence for route {route_id}")
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
    seq = candidates[0][3]
    # Remove any repeated top-level station while retaining first occurrence.
    seen = set(); clean = []
    for x in seq:
        if x not in seen:
            clean.append(x); seen.add(x)
    return clean, st


def map_route_stations_to_complexes(route_seq, top_stops: pd.DataFrame, hourly: pd.DataFrame, max_m=200.0):
    hs = hourly[["station_complex_id", "station_complex", "latitude", "longitude"]].copy()
    hs["station_complex_id"] = hs["station_complex_id"].map(norm_id)
    hs["latitude"] = pd.to_numeric(hs["latitude"], errors="coerce")
    hs["longitude"] = pd.to_numeric(hs["longitude"], errors="coerce")
    hs = hs.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_complex_id")
    out = []
    for pos, sid in enumerate(route_seq):
        if sid not in top_stops.index:
            raise RuntimeError(f"top-level GTFS stop {sid} missing from stop inventory")
        row = top_stops.loc[sid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        lat = float(row["stop_lat"]); lon = float(row["stop_lon"])
        d = haversine_m(hs["latitude"].to_numpy(), hs["longitude"].to_numpy(), lat, lon)
        j = int(np.argmin(d)); best = hs.iloc[j]; dist = float(d[j])
        if dist > max_m:
            raise RuntimeError(f"GTFS stop {sid} has no hourly station-complex match within {max_m} m; nearest={dist:.1f} m")
        out.append({
            "position": pos,
            "gtfs_top_stop_id": sid,
            "gtfs_stop_name": str(row.get("stop_name", "")),
            "gtfs_latitude": lat,
            "gtfs_longitude": lon,
            "station_complex_id": norm_id(best["station_complex_id"]),
            "station_complex": str(best["station_complex"]),
            "match_distance_m": dist,
        })
    m = pd.DataFrame(out)
    if m["station_complex_id"].duplicated().any():
        dup = m[m["station_complex_id"].duplicated(False)][["gtfs_stop_name", "station_complex_id", "station_complex"]]
        raise RuntimeError(f"non-unique GTFS-to-complex mapping: {dup.to_dict('records')}")
    return m


def service_ids_for_date(calendar: pd.DataFrame, calendar_dates: pd.DataFrame, d: date):
    active = set()
    ds = d.strftime("%Y%m%d")
    if not calendar.empty:
        cal = calendar.copy()
        cal["start_date"] = cal["start_date"].astype(str)
        cal["end_date"] = cal["end_date"].astype(str)
        weekday = d.strftime("%A").lower()
        q = cal[(cal["start_date"] <= ds) & (cal["end_date"] >= ds)]
        if weekday in q.columns:
            q = q[pd.to_numeric(q[weekday], errors="coerce").fillna(0) == 1]
        active.update(q["service_id"].astype(str))
    if not calendar_dates.empty:
        cd = calendar_dates[calendar_dates["date"].astype(str) == ds]
        for r in cd.to_dict("records"):
            sid = str(r["service_id"]); typ = str(r.get("exception_type", ""))
            if typ == "1": active.add(sid)
            elif typ == "2": active.discard(sid)
    return active


def parse_gtfs_seconds(x):
    try:
        a = str(x).split(":")
        return int(a[0]) * 3600 + int(a[1]) * 60 + int(float(a[2]))
    except Exception:
        return None


def build_supply(gtfs, route_id, route_seq, stop_to_top, start: date, days: int, capacity: float):
    trips = gtfs["trips"].copy()
    trips = trips[trips["route_id"].astype(str) == route_id].copy()
    st = gtfs["stop_times"].copy()
    st["stop_sequence_num"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.dropna(subset=["stop_sequence_num"]).sort_values(["trip_id", "stop_sequence_num"])
    st = st[st["trip_id"].isin(trips["trip_id"])].copy()
    st["top_stop_id"] = st["stop_id"].astype(str).map(stop_to_top)
    trip_service = dict(zip(trips["trip_id"].astype(str), trips["service_id"].astype(str)))
    pos = {s: i for i, s in enumerate(route_seq)}
    trip_rows = {str(k): g.copy() for k, g in st.groupby(st["trip_id"].astype(str), sort=False)}

    traversal = []
    departures = []
    for n in range(days):
        d = start + timedelta(days=n)
        active = service_ids_for_date(gtfs["calendar"], gtfs["calendar_dates"], d)
        for trip_id, g in trip_rows.items():
            if trip_service.get(trip_id) not in active:
                continue
            visits = []
            for r in g.to_dict("records"):
                sid = str(r.get("top_stop_id", ""))
                if sid not in pos:
                    continue
                sec = parse_gtfs_seconds(r.get("departure_time") or r.get("arrival_time"))
                if sec is None:
                    continue
                if not visits or visits[-1][0] != sid:
                    visits.append((sid, sec))
            for sid, sec in visits:
                service_date = d + timedelta(days=sec // 86400)
                hour = (sec % 86400) // 3600
                departures.append((service_date.isoformat(), int(hour), sid, trip_id, int(sec)))
            for (a, sec_a), (b, _sec_b) in zip(visits, visits[1:]):
                ia, ib = pos[a], pos[b]
                if ia == ib:
                    continue
                step = 1 if ib > ia else -1
                service_date = d + timedelta(days=sec_a // 86400)
                hour = (sec_a % 86400) // 3600
                for i in range(ia, ib, step):
                    u = route_seq[i] if step == 1 else route_seq[i]
                    v = route_seq[i + step]
                    seg_lo = min(i, i + step)
                    segment_id = f"{seg_lo:02d}:{route_seq[seg_lo]}->{route_seq[seg_lo+1]}"
                    direction = 1 if step == 1 else -1
                    traversal.append((service_date.isoformat(), int(hour), segment_id, direction, u, v, trip_id, capacity))
    t = pd.DataFrame(traversal, columns=["date", "hour", "segment_id", "direction", "from_top_stop", "to_top_stop", "trip_id", "declared_capacity"])
    dep = pd.DataFrame(departures, columns=["date", "hour", "top_stop_id", "trip_id", "departure_seconds"])
    if t.empty:
        raise RuntimeError("no scheduled route traversals in selected period")
    return t, dep


def build_demand(hourly, od, mapping, start: date, days: int):
    h = hourly.copy()
    h["station_complex_id"] = h["station_complex_id"].map(norm_id)
    h["ridership"] = pd.to_numeric(h["ridership"], errors="coerce").fillna(0.0)
    h["timestamp"] = pd.to_datetime(h["transit_timestamp"], errors="coerce")
    h = h.dropna(subset=["timestamp"])
    h["date"] = h["timestamp"].dt.date.astype(str)
    h["hour"] = h["timestamp"].dt.hour.astype(int)
    end = start + timedelta(days=days)
    h = h[(h["timestamp"] >= pd.Timestamp(start)) & (h["timestamp"] < pd.Timestamp(end))]
    # Source slices may already be aggregated, but re-aggregate defensively across fare classes/payment methods.
    h = h.groupby(["date", "hour", "station_complex_id"], as_index=False)["ridership"].sum()
    corridor = set(mapping["station_complex_id"].astype(str))
    h = h[h["station_complex_id"].isin(corridor)].copy()
    h["day_of_week"] = pd.to_datetime(h["date"]).dt.day_name()

    o = od.copy()
    for c in ["origin_station_complex_id", "destination_station_complex_id"]:
        o[c] = o[c].map(norm_id)
    o["hour_of_day"] = pd.to_numeric(o["hour_of_day"], errors="coerce")
    o["estimated_average_ridership"] = pd.to_numeric(o["estimated_average_ridership"], errors="coerce").fillna(0.0)
    o = o.dropna(subset=["hour_of_day"])
    o["hour_of_day"] = o["hour_of_day"].astype(int)
    o = o[o["origin_station_complex_id"].isin(corridor)].copy()
    denom = o.groupby(["day_of_week", "hour_of_day", "origin_station_complex_id"], as_index=False)["estimated_average_ridership"].sum().rename(columns={"estimated_average_ridership": "origin_all_destination_od_total"})
    c = o[o["destination_station_complex_id"].isin(corridor) & (o["destination_station_complex_id"] != o["origin_station_complex_id"])].copy()
    c = c.groupby(["day_of_week", "hour_of_day", "origin_station_complex_id", "destination_station_complex_id"], as_index=False)["estimated_average_ridership"].sum()
    c = c.merge(denom, on=["day_of_week", "hour_of_day", "origin_station_complex_id"], how="left")
    c["destination_share_of_all_trips"] = np.where(c["origin_all_destination_od_total"] > 0, c["estimated_average_ridership"] / c["origin_all_destination_od_total"], 0.0)

    d = h.merge(c, left_on=["day_of_week", "hour", "station_complex_id"], right_on=["day_of_week", "hour_of_day", "origin_station_complex_id"], how="left")
    d["destination_share_of_all_trips"] = d["destination_share_of_all_trips"].fillna(0.0)
    d["corridor_od_trips"] = d["ridership"] * d["destination_share_of_all_trips"]
    d = d[d["corridor_od_trips"] > 0].copy()
    return h, c, d


def assign_corridor(demand, mapping, route_seq):
    complex_to_pos = dict(zip(mapping["station_complex_id"].astype(str), mapping["position"].astype(int)))
    rows = defaultdict(float)
    origin_corridor = defaultdict(float)
    for r in demand[["date", "hour", "origin_station_complex_id", "destination_station_complex_id", "corridor_od_trips"]].itertuples(index=False):
        o = complex_to_pos.get(str(r.origin_station_complex_id)); x = complex_to_pos.get(str(r.destination_station_complex_id))
        if o is None or x is None or o == x:
            continue
        v = float(r.corridor_od_trips)
        origin_corridor[(r.date, int(r.hour), str(r.origin_station_complex_id))] += v
        step = 1 if x > o else -1
        direction = 1 if step == 1 else -1
        for i in range(o, x, step):
            lo = min(i, i + step)
            seg = f"{lo:02d}:{route_seq[lo]}->{route_seq[lo+1]}"
            rows[(r.date, int(r.hour), seg, direction)] += v
    seg = pd.DataFrame([(k[0], k[1], k[2], k[3], v) for k, v in rows.items()], columns=["date", "hour", "segment_id", "direction", "segment_passenger_demand"])
    ob = pd.DataFrame([(k[0], k[1], k[2], v) for k, v in origin_corridor.items()], columns=["date", "hour", "station_complex_id", "corridor_boardings"])
    return seg, ob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly", type=Path, required=True, help="official MTA hourly ridership slice, CSV")
    ap.add_argument("--od", type=Path, required=True, help="official MTA OD estimate slice, CSV")
    ap.add_argument("--gtfs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--route-id", default="7")
    ap.add_argument("--start-date", default="2026-06-01")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--capacity-per-train", type=float, default=1000.0)
    ap.add_argument("--mapping-tolerance-m", type=float, default=200.0)
    args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()

    hourly = pd.read_csv(args.hourly, dtype=str)
    od = pd.read_csv(args.od, dtype=str)
    gtfs = read_gtfs(args.gtfs)
    stop_to_top, top_stops = top_level_stop_map(gtfs["stops"])
    route_id = select_route_id(gtfs["routes"], str(args.route_id))
    route_seq, _ = canonical_route_sequence(gtfs, route_id, stop_to_top)
    mapping = map_route_stations_to_complexes(route_seq, top_stops, hourly, args.mapping_tolerance_m)
    mapping.to_csv(out / "route_station_complex_mapping.csv", index=False)

    supply, departures = build_supply(gtfs, route_id, route_seq, stop_to_top, start, args.days, args.capacity_per_train)
    supply.to_csv(out / "scheduled_run_segment_supply.csv", index=False)

    hourly_station, od_pattern, demand = build_demand(hourly, od, mapping, start, args.days)
    od_pattern.to_csv(out / "corridor_od_destination_shares.csv", index=False)
    seg_demand, origin_boardings = assign_corridor(demand, mapping, route_seq)
    seg_demand.to_csv(out / "segment_hour_demand.csv", index=False)

    # Schedule-proportional loading: each scheduled traversal receives demand in proportion to declared capacity.
    cap = supply.groupby(["date", "hour", "segment_id", "direction"], as_index=False)["declared_capacity"].sum().rename(columns={"declared_capacity": "scheduled_capacity_total"})
    runs = supply.merge(cap, on=["date", "hour", "segment_id", "direction"], how="left").merge(seg_demand, on=["date", "hour", "segment_id", "direction"], how="left")
    runs["segment_passenger_demand"] = runs["segment_passenger_demand"].fillna(0.0)
    runs["assigned_load"] = np.where(runs["scheduled_capacity_total"] > 0, runs["segment_passenger_demand"] * runs["declared_capacity"] / runs["scheduled_capacity_total"], np.nan)
    runs["load_capacity_ratio"] = runs["assigned_load"] / runs["declared_capacity"]
    runs["crowding_class"] = pd.cut(runs["load_capacity_ratio"], bins=[-np.inf, 0.7, 1.0, np.inf], labels=["below_0.7", "0.7_to_1.0", "above_1.0"], right=False).astype(str)
    runs.to_csv(out / "segment_load_per_run_hourly.csv", index=False)

    # Demand on segment-hours with no scheduled run is retained explicitly rather than silently dropped.
    supply_keys = set(map(tuple, supply[["date", "hour", "segment_id", "direction"]].drop_duplicates().to_numpy()))
    seg_demand["has_scheduled_supply"] = [tuple(x) in supply_keys for x in seg_demand[["date", "hour", "segment_id", "direction"]].to_numpy()]
    unsupplied = seg_demand[~seg_demand["has_scheduled_supply"]].copy()
    unsupplied.to_csv(out / "unsupplied_segment_hour_demand.csv", index=False)

    # Waiting-time proxy by station-hour from route departures; this is schedule-derived, not observed waiting truth.
    top_to_complex = dict(zip(mapping["gtfs_top_stop_id"].astype(str), mapping["station_complex_id"].astype(str)))
    dep = departures.copy(); dep["station_complex_id"] = dep["top_stop_id"].astype(str).map(top_to_complex)
    dep = dep.dropna(subset=["station_complex_id"])
    dep_count = dep.groupby(["date", "hour", "station_complex_id"], as_index=False).size().rename(columns={"size": "scheduled_departures"})
    wait = origin_boardings.merge(dep_count, on=["date", "hour", "station_complex_id"], how="outer")
    wait["corridor_boardings"] = wait["corridor_boardings"].fillna(0.0); wait["scheduled_departures"] = wait["scheduled_departures"].fillna(0).astype(int)
    wait["half_headway_wait_proxy_s"] = np.where(wait["scheduled_departures"] > 0, 1800.0 / wait["scheduled_departures"], np.nan)
    wait.to_csv(out / "waiting_time_by_stop_hour.csv", index=False)

    assigned_by_key = runs.groupby(["date", "hour", "segment_id", "direction"], as_index=False)["assigned_load"].sum()
    chk = seg_demand.merge(assigned_by_key, on=["date", "hour", "segment_id", "direction"], how="left")
    chk["assigned_load"] = chk["assigned_load"].fillna(0.0)
    supplied_chk = chk[chk["has_scheduled_supply"]]
    abs_err = float(np.abs(supplied_chk["segment_passenger_demand"] - supplied_chk["assigned_load"]).sum())
    supplied_demand = float(supplied_chk["segment_passenger_demand"].sum())
    unsupplied_demand = float(unsupplied["segment_passenger_demand"].sum()) if len(unsupplied) else 0.0
    corridor_boardings = float(origin_boardings["corridor_boardings"].sum()) if len(origin_boardings) else 0.0
    weighted_wait = wait.dropna(subset=["half_headway_wait_proxy_s"])
    mean_wait = float(np.average(weighted_wait["half_headway_wait_proxy_s"], weights=weighted_wait["corridor_boardings"])) if len(weighted_wait) and weighted_wait["corridor_boardings"].sum() > 0 else None

    feed_version = None
    if not gtfs["feed_info"].empty:
        feed_version = gtfs["feed_info"].iloc[0].to_dict()

    summary = {
        "task": "T8 transit supply-demand-capacity fusion",
        "run_class": "public_data registry / official public source rehearsal",
        "route_id": route_id,
        "period_start": start.isoformat(),
        "period_days": args.days,
        "period_end_inclusive": (start + timedelta(days=args.days - 1)).isoformat(),
        "official_inputs": {
            "mta_od_dataset_id": "28vm-gjqr",
            "mta_hourly_dataset_id": "5wq4-mkjj",
            "gtfs_sha256": sha256(args.gtfs),
            "hourly_slice_sha256": sha256(args.hourly),
            "od_slice_sha256": sha256(args.od),
            "gtfs_feed_info": feed_version,
        },
        "declared_non_official_assumptions": {
            "capacity_per_train": args.capacity_per_train,
            "capacity_provenance_class": "self-added declared assumption permitted by the T8 task card",
            "crowding_bins_load_capacity_ratio": [0.7, 1.0],
            "waiting_proxy": "1800 / scheduled departures in station-hour (half uniform headway)",
        },
        "station_mapping": {
            "route_station_count": int(len(mapping)),
            "max_match_distance_m": float(mapping["match_distance_m"].max()),
            "median_match_distance_m": float(mapping["match_distance_m"].median()),
        },
        "demand": {
            "observed_hourly_corridor_station_rows": int(len(hourly_station)),
            "corridor_od_pair_rows_after_scaling": int(len(demand)),
            "corridor_boardings_total_30d": corridor_boardings,
            "segment_passenger_demand_total": float(seg_demand["segment_passenger_demand"].sum()),
            "unsupplied_segment_passenger_demand": unsupplied_demand,
        },
        "supply": {
            "scheduled_run_segment_rows": int(len(supply)),
            "unique_trip_ids": int(supply["trip_id"].nunique()),
            "scheduled_segment_hour_keys": int(len(cap)),
        },
        "baseline": {
            "name": "schedule_proportional_loading",
            "capacity_constraint": False,
            "reassignment": False,
            "assigned_segment_passenger_total_on_supplied_keys": float(runs["assigned_load"].sum()),
            "assignment_conservation_absolute_error": abs_err,
            "assignment_conservation_relative_error": abs_err / supplied_demand if supplied_demand > 0 else None,
            "weighted_mean_half_headway_wait_proxy_s": mean_wait,
            "mean_load_capacity_ratio": float(runs["load_capacity_ratio"].mean()),
            "p95_load_capacity_ratio": float(runs["load_capacity_ratio"].quantile(0.95)),
            "max_load_capacity_ratio": float(runs["load_capacity_ratio"].max()),
            "run_segment_rows_above_declared_capacity": int((runs["load_capacity_ratio"] > 1.0).sum()),
        },
        "formal_scoring": {
            "organizer_held_truth_available": False,
            "official_E_score": None,
            "status": "PUBLIC_REHEARSAL_ONLY",
            "reason": "The public registry supplies schedule/demand sources but not organizer-held APC segment loads, independent waiting-time truth, or crowding-label truth for this instance.",
        },
        "scope_boundary": "Only trips whose observed entry is at a mapped route station and whose OD-estimated destination is another station on the selected route are loaded. OD probability denominators include all destinations, so outside-corridor demand is not renormalized onto the corridor.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()