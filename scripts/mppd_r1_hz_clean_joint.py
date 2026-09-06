from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import scripts.mppd_r1_hz_joint_full_day as model
import scripts.mppd_r1_hz_event_level_joint as event_model

SCHEMA = "mppd.r1-hz-clean-raw-cohort-stats.v1"
BEAM = 48
FINAL_TOP_K = 64
EPS = 1e-300


@dataclass
class Ride:
    root: dict[str, Any]
    board_station: str
    alight_station: str
    dep: float
    arr: float


@dataclass
class State:
    loglik: float
    route_rank: int
    rides: list[Ride]
    access: dict[str, float]
    transfers: list[tuple[str, dict[str, Any]]]


class LegIndex:
    def __init__(self, roots: dict[str, dict[str, Any]]):
        self.roots = roots
        self.cache: dict[tuple[Any, ...], tuple[list[Ride], list[float]]] = {}

    @staticmethod
    def key(leg: dict[str, Any]) -> tuple[Any, ...]:
        opts = tuple(sorted((str(x["path_id"]), str(x["direction"])) for x in leg["compatible_service_options"]))
        return (str(int(leg["from_station"])), str(int(leg["to_station"])), opts)

    def get(self, leg: dict[str, Any]) -> tuple[list[Ride], list[float]]:
        k = self.key(leg)
        if k in self.cache:
            return self.cache[k]
        b, a, opts = k
        allowed = set(opts)
        rows: list[Ride] = []
        for r in self.roots.values():
            if (str(r["path_id"]), str(r["direction"])) not in allowed:
                continue
            if b not in r["events"] or a not in r["events"]:
                continue
            dep = float(r["events"][b]["time_s"]); arr = float(r["events"][a]["time_s"])
            if arr <= dep:
                continue
            rows.append(Ride(r, b, a, dep, arr))
        rows.sort(key=lambda x: (x.dep, x.arr, x.root["root_id"]))
        times = [x.dep for x in rows]
        self.cache[k] = (rows, times)
        return rows, times

    def between(self, leg: dict[str, Any], lo: float, hi: float) -> list[Ride]:
        rows, times = self.get(leg)
        i = bisect.bisect_left(times, lo)
        j = bisect.bisect_right(times, hi)
        return rows[i:j]

    def previous_departures(self, leg: dict[str, Any], ready_lower: float, selected_dep: float) -> list[float]:
        rows, times = self.get(leg)
        i = bisect.bisect_left(times, ready_lower)
        j = bisect.bisect_left(times, selected_dep - 1e-9)
        return [x.dep for x in rows[i:j]]


def cohort_id(row: dict[str, Any]) -> str:
    text = "|".join([
        str(row["origin_line"]), str(int(row["origin_station"])),
        str(row["destination_line"]), str(int(row["destination_station"])),
        f"{float(row['entry_sec']):.6f}", f"{float(row['exit_sec']):.6f}",
        f"{float(row['passenger_mass']):.9f}",
    ])
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def route_key(row: dict[str, Any]) -> str:
    return f"{row['origin_line']}:{int(row['origin_station'])}->{row['destination_line']}:{int(row['destination_station'])}"


def movement_key(route: dict[str, Any], j: int) -> str:
    meta = route.get("inter_leg_movement_meta", [])
    if j < len(meta) and meta[j].get("movement"):
        return str(meta[j]["movement"])
    m = route.get("transfer_movements", [])
    return str(m[j]) if j < len(m) else f"TRANSFER_{j}"


def load_route_support(path: Path) -> dict[str, list[dict[str, Any]]]:
    x = json.loads(path.read_text(encoding="utf-8"))
    if x.get("status") != "QUALIFIED_LINE_AWARE_ROUTE_SUPPORT":
        raise SystemExit("route support not qualified")
    return x["route_support"]


def load_shared_roots(source: Path, params_path: Path | None):
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path else None
    with tempfile.TemporaryDirectory(prefix="mppd-clean-roots-") as td:
        adjusted = Path(td) / "roots.json"
        event_model.write_adjusted_roots(source, params, adjusted)
        raw, roots, stations = model.load_roots(adjusted)
    return raw, roots, stations


def route_prior(route: dict[str, Any], min_cost: float) -> float:
    return -(float(route.get("base_ranking_cost_s", 0.0)) - min_cost) / 1800.0


def evaluate_route(row: dict[str, Any], route: dict[str, Any], index: LegIndex, params: dict[str, Any], min_cost: float):
    legs = route.get("ride_legs", [])
    if not legs:
        return []
    entry = float(row["entry_sec"]); exit_t = float(row["exit_sec"])
    origin = str(int(row["origin_station"])); dest = str(int(row["destination_station"]))
    am = params["access"].get(origin, {"mu": math.log(60.0), "sigma": .75})
    em = params["egress"].get(dest, {"mu": math.log(60.0), "sigma": .75})

    first = legs[0]
    states: list[State] = []
    # Hard physics only: at least 5 s to enter and at least 5 s to exit eventually.
    first_candidates = index.between(first, entry + model.MIN_MOVE_S, exit_t - model.MIN_MOVE_S)
    for ride in first_candidates:
        if ride.arr > exit_t - model.MIN_MOVE_S:
            continue
        prev = index.previous_departures(first, entry + model.MIN_MOVE_S, ride.dep)
        a = model.boarding_component(entry, ride.dep, prev, float(am["mu"]), float(am["sigma"]), float(params["access_hazard"]))
        if a is None:
            continue
        states.append(State(float(a["loglik"]), int(route["rank"]), [ride], a, []))
    if not states:
        return []
    states.sort(key=lambda x: x.loglik, reverse=True)
    states = states[:BEAM]

    for j, leg in enumerate(legs[1:]):
        nxt: list[State] = []
        mk = movement_key(route, j)
        tmodel = params["transfers"].get(mk, model.transfer_default())
        for st in states:
            last = st.rides[-1]
            lo = last.arr + model.MIN_MOVE_S
            if lo > exit_t - model.MIN_MOVE_S:
                continue
            candidates = index.between(leg, lo, exit_t - model.MIN_MOVE_S)
            for ride in candidates:
                if ride.board_station != last.alight_station:
                    continue
                prev = index.previous_departures(leg, lo, ride.dep)
                tr = model.transfer_mixture(last.arr, ride.dep, prev, tmodel, float(params["transfer_hazard"]))
                if tr is None:
                    continue
                nxt.append(State(st.loglik + float(tr["loglik"]), st.route_rank, st.rides + [ride], st.access, st.transfers + [(mk, tr)]))
        if not nxt:
            return []
        nxt.sort(key=lambda x: x.loglik, reverse=True)
        states = nxt[:BEAM]

    prior = route_prior(route, min_cost)
    final: list[State] = []
    for st in states:
        e = exit_t - st.rides[-1].arr
        ell = model.truncated_logpdf(e, float(em["mu"]), float(em["sigma"]))
        if not math.isfinite(ell):
            continue
        st.loglik += ell + prior
        final.append(st)
    return final


def iter_rows(path: Path):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=50000):
        for row in batch.to_pylist():
            yield row


def run_shard(cohorts_path: Path, roots_path: Path, routes_path: Path, params_path: Path, out_stats: Path, shard_index: int, chains_out: Path | None):
    params = json.loads(params_path.read_text(encoding="utf-8"))
    root_raw, roots, _stations = load_shared_roots(roots_path, params_path)
    routes = load_route_support(routes_path); index = LegIndex(roots)

    access_stats: dict[str, Any] = {}; egress_stats: dict[str, Any] = {}; transfer_stats: dict[str, Any] = {}; root_egress: dict[str, Any] = {}
    boards_access = skips_access = boards_transfer = skips_transfer = 0.0
    passenger_mass = resolved_mass = unresolved_mass = 0.0
    cohort_count = resolved_cohorts = 0; candidate_evaluations = 0; closure_max = 0.0
    no_route_mass = no_feasible_chain_mass = 0.0

    writer = pq.ParquetWriter(chains_out, model.chain_schema(), compression="zstd") if chains_out else None
    buffer: list[dict[str, Any]] = []
    def flush():
        nonlocal buffer
        if writer is not None and buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=model.chain_schema())); buffer=[]

    try:
        for row in iter_rows(cohorts_path):
            cohort_count += 1; mass=float(row["passenger_mass"]); passenger_mass += mass
            rk=route_key(row); rcands=routes.get(rk)
            if not rcands:
                unresolved_mass += mass; no_route_mass += mass; continue
            min_cost=min(float(r.get("base_ranking_cost_s",0.0)) for r in rcands)
            states: list[State] = []
            for route in rcands:
                ss=evaluate_route(row, route, index, params, min_cost)
                candidate_evaluations += len(ss)
                states.extend(ss)
            if not states:
                unresolved_mass += mass; no_feasible_chain_mass += mass; continue
            states.sort(key=lambda x:x.loglik, reverse=True); states=states[:FINAL_TOP_K]
            z=model.logsumexp(st.loglik for st in states)
            if not math.isfinite(z):
                unresolved_mass += mass; no_feasible_chain_mass += mass; continue
            resolved_mass += mass; resolved_cohorts += 1
            post=[math.exp(st.loglik-z) for st in states]

            best_i=max(range(len(states)), key=lambda i:post[i])
            for q,st in zip(post,states):
                w=mass*q; origin=str(int(row["origin_station"])); dest=str(int(row["destination_station"]))
                a=st.access; model.add_moment(access_stats, origin, w, float(a["elog"]), float(a["elog2"]))
                boards_access += w; skips_access += w*float(a["eskip"])
                e=float(row["exit_sec"])-st.rides[-1].arr; le=math.log(e)
                model.add_moment(egress_stats,dest,w,le,le*le)
                model.add_root_egress(root_egress,str(st.rides[-1].root["root_id"]),st.rides[-1].alight_station,w,e)
                for mk,tr in st.transfers:
                    for c in tr["components"]:
                        model.add_moment(transfer_stats,f"{mk}|{int(c['component'])}",w*float(c["prob"]),float(c["elog"]),float(c["elog2"]))
                    boards_transfer += w; skips_transfer += w*float(tr["eskip"])

            if writer is not None:
                st=states[best_i]; q=post[best_i]
                acc=float(st.access["etime"]); first_dep=st.rides[0].dep; initial_wait=first_dep-float(row["entry_sec"])-acc
                trs=[]; tsum=wsum=0.0
                for j,(mk,tr) in enumerate(st.transfers):
                    comp=max(tr["components"], key=lambda x:float(x["prob"]))
                    kt=max(model.MIN_MOVE_S,float(comp["etime"])); wait=st.rides[j+1].dep-st.rides[j].arr-kt
                    tmodel=params["transfers"].get(mk,model.transfer_default()); path_id=str(tmodel["components"][int(comp["component"])]["path_id"])
                    trs.append({"movement":mk,"path_id":path_id,"transfer_time_s":kt,"wait_time_s":wait}); tsum+=kt; wsum+=wait
                rides=[]; ride_time=0.0
                for r in st.rides:
                    rides.append({"service_id":str(r.root["root_id"]),"board_station":int(r.board_station),"alight_station":int(r.alight_station),"board_time_s":r.dep,"alight_time_s":r.arr}); ride_time+=r.arr-r.dep
                egr=float(row["exit_sec"])-st.rides[-1].arr
                total=acc+initial_wait+ride_time+tsum+wsum+egr; observed=float(row["exit_sec"])-float(row["entry_sec"]); cerr=total-observed; closure_max=max(closure_max,abs(cerr))
                buffer.append({"cohort_id":cohort_id(row),"origin_station":int(row["origin_station"]),"destination_station":int(row["destination_station"]),
                    "entry_sec":float(row["entry_sec"]),"exit_sec":float(row["exit_sec"]),"passenger_mass":mass,"route_rank":int(st.route_rank),
                    "root_chain":">".join(str(r.root["root_id"]) for r in st.rides),"posterior_probability":float(q),"access_time_s":acc,
                    "initial_wait_s":initial_wait,"egress_time_s":egr,"transfer_assignments_json":json.dumps(trs,ensure_ascii=False,separators=(",",":")),
                    "ride_assignments_json":json.dumps(rides,ensure_ascii=False,separators=(",",":")),"travel_time_closure_error_s":cerr})
                if len(buffer)>=20000: flush()
        flush()
    finally:
        if writer is not None: writer.close()

    result={"schema":SCHEMA,"service_date":root_raw.get("source_date","2019-01-04"),"shard_index":int(shard_index),"parameter_iteration":int(params.get("iteration",0)),
        "passenger_mass":passenger_mass,"resolved_mass":resolved_mass,"unresolved_mass":unresolved_mass,"cohort_count":cohort_count,"resolved_cohort_count":resolved_cohorts,
        "candidate_evaluations":candidate_evaluations,"support_top_k":FINAL_TOP_K,"support_truncated_cohort_count":0,
        "access_stats":access_stats,"egress_stats":egress_stats,"transfer_stats":transfer_stats,"root_egress_stats":root_egress,
        "hazard_stats":{"access_boards":boards_access,"access_skips":skips_access,"transfer_boards":boards_transfer,"transfer_skips":skips_transfer},
        "diagnostics":{"max_abs_map_chain_time_closure_error_s":closure_max,"no_route_mass":no_route_mass,"no_feasible_chain_mass":no_feasible_chain_mass},
        "semantics":{"raw_afc_cohorts_used_directly":True,"legacy_e0_genealogy_not_used":True,"legacy_temporal_kernels_not_used":True,
            "candidate_service_chains_generated_from_current_shared_timetable":True,"five_second_physical_bounds_applied_before_behavioral_likelihood":True}}
    out_stats.write_text(json.dumps(result,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    print(json.dumps({"shard_index":shard_index,"passenger_mass":passenger_mass,"resolved_mass":resolved_mass,"resolved_share":resolved_mass/passenger_mass if passenger_mass else None,
        "cohort_count":cohort_count,"resolved_cohort_count":resolved_cohorts,"candidate_evaluations":candidate_evaluations,"no_route_mass":no_route_mass,"no_feasible_chain_mass":no_feasible_chain_mass},ensure_ascii=False,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--cohorts",type=Path,required=True); p.add_argument("--roots",type=Path,required=True); p.add_argument("--routes",type=Path,required=True); p.add_argument("--params",type=Path,required=True)
    p.add_argument("--out-stats",type=Path,required=True); p.add_argument("--shard-index",type=int,required=True); p.add_argument("--chains-out",type=Path)
    a=p.parse_args(); run_shard(a.cohorts,a.roots,a.routes,a.params,a.out_stats,a.shard_index,a.chains_out)


if __name__=="__main__": main()
