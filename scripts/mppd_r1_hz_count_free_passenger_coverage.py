from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ACCESS_MIN_S = 15.0
EGRESS_MIN_S = 15.0
TRANSFER_MIN_S = 5.0
SCHEMA_SHARD = "mppd.r1-hz-count-free-passenger-coverage-shard.v1"
SCHEMA_MERGED = "mppd.r1-hz-count-free-passenger-coverage-full-day.v1"


def route_key(row: dict[str, Any]) -> str:
    return f"{row['origin_line']}:{int(row['origin_station'])}->{row['destination_line']}:{int(row['destination_station'])}"


def load_routes(path: Path) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status") != "QUALIFIED_LINE_AWARE_ROUTE_SUPPORT":
        raise SystemExit("route support is not qualified")
    return raw["route_support"]


def load_services(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sem = raw.get("semantics", {})
    required_false = ["train_count_is_input", "planned_timetable_used", "legacy_candidate_roots_used_as_input"]
    for key in required_false:
        if sem.get(key) is not False:
            raise SystemExit(f"count-free service contract violated: {key}")
    if sem.get("service_trajectory_count_is_inferred") is not True:
        raise SystemExit("service trajectory count is not inferred")
    services: dict[str, dict[str, Any]] = {}
    for tr in raw.get("trajectories", []):
        sid = str(tr["trajectory_id"])
        events = {str(int(e["station"])): {**e, "time_s": float(e["time_s"])} for e in tr["events"]}
        options = {(str(tr["path_id"]), str(tr["direction"]))}
        # A shared AFC ridge can carry multiple path hypotheses. It may serve an
        # alternative only on stations actually materialized in this trajectory;
        # no unobserved branch station is invented here.
        for alt in tr.get("path_alternatives", []):
            direction = str(alt.get("direction", tr["direction"]))
            options.add((str(alt["path_id"]), direction))
        services[sid] = {
            "trajectory_id": sid,
            "path_id": str(tr["path_id"]),
            "direction": str(tr["direction"]),
            "options": options,
            "events": events,
            "support_weight": float(tr.get("support_weight", 0.0)),
            "evidence_score": float(tr.get("evidence_score", 0.0)),
            "path_ambiguous": bool(tr.get("path_ambiguous", False)),
        }
    return services, raw


class LegIndex:
    def __init__(self, services: dict[str, dict[str, Any]]):
        self.services = services
        self.cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    @staticmethod
    def key(leg: dict[str, Any]) -> tuple[Any, ...]:
        opts = tuple(sorted((str(x["path_id"]), str(x["direction"])) for x in leg["compatible_service_options"]))
        return (str(int(leg["from_station"])), str(int(leg["to_station"])), opts)

    def get(self, leg: dict[str, Any]) -> dict[str, Any]:
        key = self.key(leg)
        if key in self.cache:
            return self.cache[key]
        board, alight, opts = key
        allow = set(opts)
        rows: list[tuple[float, float, str]] = []
        for sid, tr in self.services.items():
            if not (tr["options"] & allow):
                continue
            if board not in tr["events"] or alight not in tr["events"]:
                continue
            dep = float(tr["events"][board]["time_s"])
            arr = float(tr["events"][alight]["time_s"])
            if arr <= dep:
                continue
            rows.append((dep, arr, sid))
        rows.sort(key=lambda x: (x[0], x[1], x[2]))
        deps = [x[0] for x in rows]
        suffix_best: list[int] = [0] * len(rows)
        if rows:
            best = len(rows) - 1
            suffix_best[-1] = best
            for i in range(len(rows) - 2, -1, -1):
                if rows[i][1] <= rows[best][1]:
                    best = i
                suffix_best[i] = best
        out = {"rows": rows, "deps": deps, "suffix_best": suffix_best}
        self.cache[key] = out
        return out

    def earliest_arrival(self, leg: dict[str, Any], ready_s: float) -> tuple[float, float, str] | None:
        data = self.get(leg)
        rows = data["rows"]
        if not rows:
            return None
        i = bisect.bisect_left(data["deps"], ready_s)
        if i >= len(rows):
            return None
        return rows[data["suffix_best"][i]]

    def local_gap(self, leg: dict[str, Any], ready_s: float) -> dict[str, float | None]:
        data = self.get(leg)
        deps = data["deps"]
        i = bisect.bisect_left(deps, ready_s)
        before = deps[i - 1] if i > 0 else None
        after = deps[i] if i < len(deps) else None
        return {
            "previous_departure_s": before,
            "next_departure_s": after,
            "wait_to_next_s": (after - ready_s) if after is not None else None,
            "missed_previous_by_s": (ready_s - before) if before is not None else None,
        }


def route_failure_pressure(route: dict[str, Any], leg_index: int, ready_s: float, reason: str, index: LegIndex) -> dict[str, Any]:
    legs = route.get("ride_legs", [])
    if not legs:
        return {"reason": "ROUTE_HAS_NO_RIDE_LEGS"}
    j = min(max(0, leg_index), len(legs) - 1)
    leg = legs[j]
    opts = sorted(f"{x['path_id']}:{x['direction']}" for x in leg["compatible_service_options"])
    return {
        "reason": reason,
        "leg_index": j,
        "from_station": int(leg["from_station"]),
        "to_station": int(leg["to_station"]),
        "compatible_service_options": opts,
        "ready_s": float(ready_s),
        "time_bin_300s": int(ready_s // 300),
        **index.local_gap(leg, ready_s),
    }


def evaluate_route(row: dict[str, Any], route: dict[str, Any], index: LegIndex) -> dict[str, Any]:
    legs = route.get("ride_legs", [])
    if not legs:
        return {"ok": False, "reason": "ROUTE_HAS_NO_RIDE_LEGS", "progress": -1, "pressure": {"reason": "ROUTE_HAS_NO_RIDE_LEGS"}}
    entry = float(row["entry_sec"])
    exit_t = float(row["exit_sec"])
    deadline = exit_t - EGRESS_MIN_S
    ready = entry + ACCESS_MIN_S
    chain = []
    for j, leg in enumerate(legs):
        ride = index.earliest_arrival(leg, ready)
        if ride is None:
            reason = "FIRST_LEG_NO_SERVICE" if j == 0 else "TRANSFER_NO_DOWNSTREAM_SERVICE"
            return {
                "ok": False,
                "reason": reason,
                "progress": j - 1,
                "pressure": route_failure_pressure(route, j, ready, reason, index),
            }
        dep, arr, sid = ride
        chain.append({"trajectory_id": sid, "from_station": int(leg["from_station"]), "to_station": int(leg["to_station"]), "departure_s": dep, "arrival_s": arr})
        if j < len(legs) - 1:
            ready = arr + TRANSFER_MIN_S
    final_arrival = chain[-1]["arrival_s"]
    if final_arrival > deadline:
        return {
            "ok": False,
            "reason": "FINAL_EGRESS_HORIZON",
            "progress": len(legs),
            "final_arrival_s": final_arrival,
            "latest_allowed_arrival_s": deadline,
            "excess_s": final_arrival - deadline,
            "pressure": {
                "reason": "FINAL_EGRESS_HORIZON",
                "destination_station": int(row["destination_station"]),
                "latest_allowed_arrival_s": deadline,
                "time_bin_300s": int(deadline // 300),
                "excess_s": final_arrival - deadline,
            },
        }
    return {
        "ok": True,
        "reason": "FEASIBLE_COMPLETE_CHAIN",
        "progress": len(legs),
        "chain": chain,
        "access_budget_s": chain[0]["departure_s"] - entry,
        "egress_budget_s": exit_t - final_arrival,
    }


def choose_failure(failures: list[dict[str, Any]]) -> dict[str, Any]:
    if not failures:
        return {"reason": "UNKNOWN_FAILURE", "progress": -99, "pressure": {"reason": "UNKNOWN_FAILURE"}}
    rank = {"FINAL_EGRESS_HORIZON": 3, "TRANSFER_NO_DOWNSTREAM_SERVICE": 2, "FIRST_LEG_NO_SERVICE": 1, "ROUTE_HAS_NO_RIDE_LEGS": 0}
    return max(failures, key=lambda x: (int(x.get("progress", -99)), rank.get(str(x.get("reason")), -1), -float(x.get("excess_s", 0.0))))


def iter_shard_rows(path: Path, shard_index: int, shard_count: int):
    pf = pq.ParquetFile(path)
    ordinal = 0
    for batch in pf.iter_batches(batch_size=50000):
        for row in batch.to_pylist():
            if ordinal % shard_count == shard_index:
                yield ordinal, row
            ordinal += 1


def pressure_key(p: dict[str, Any]) -> str:
    reason = str(p.get("reason", "UNKNOWN"))
    if reason in {"FIRST_LEG_NO_SERVICE", "TRANSFER_NO_DOWNSTREAM_SERVICE"}:
        opts = ",".join(p.get("compatible_service_options", []))
        return f"{reason}|{p.get('from_station')}->{p.get('to_station')}|{opts}|bin={p.get('time_bin_300s')}"
    if reason == "FINAL_EGRESS_HORIZON":
        return f"{reason}|dest={p.get('destination_station')}|bin={p.get('time_bin_300s')}"
    return reason


def run_shard(cohorts: Path, routes_path: Path, services_path: Path, output: Path, shard_index: int, shard_count: int, example_limit: int = 40) -> dict[str, Any]:
    routes = load_routes(routes_path)
    services, service_raw = load_services(services_path)
    index = LegIndex(services)
    passenger_mass = resolved_mass = no_route_mass = unresolved_route_exists_mass = 0.0
    cohort_count = resolved_cohort_count = 0
    failure_mass: Counter[str] = Counter()
    pressure_mass: Counter[str] = Counter()
    pressure_detail: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    trajectory_assignment_mass: Counter[str] = Counter()
    resolved_access_min = float("inf")
    resolved_egress_min = float("inf")

    for ordinal, row in iter_shard_rows(cohorts, shard_index, shard_count):
        cohort_count += 1
        mass = float(row["passenger_mass"])
        passenger_mass += mass
        rr = routes.get(route_key(row))
        if not rr:
            no_route_mass += mass
            failure_mass["NO_ROUTE_SUPPORT"] += mass
            if len(examples) < example_limit:
                examples.append({"ordinal": ordinal, "mass": mass, "route_key": route_key(row), "entry_s": float(row["entry_sec"]), "exit_s": float(row["exit_sec"]), "reason": "NO_ROUTE_SUPPORT"})
            continue

        failures: list[dict[str, Any]] = []
        success = None
        # Route support is already ranked. A feasibility probe does not need to
        # enumerate train identities from legacy roots; every candidate is tested
        # directly against the inferred count-free service field.
        for route in rr:
            result = evaluate_route(row, route, index)
            if result["ok"]:
                success = result
                break
            failures.append(result)

        if success is not None:
            resolved_mass += mass
            resolved_cohort_count += 1
            resolved_access_min = min(resolved_access_min, float(success["access_budget_s"]))
            resolved_egress_min = min(resolved_egress_min, float(success["egress_budget_s"]))
            for ride in success["chain"]:
                trajectory_assignment_mass[str(ride["trajectory_id"])] += mass
            continue

        unresolved_route_exists_mass += mass
        best = choose_failure(failures)
        reason = str(best["reason"])
        failure_mass[reason] += mass
        p = dict(best.get("pressure", {"reason": reason}))
        pk = pressure_key(p)
        pressure_mass[pk] += mass
        if pk not in pressure_detail:
            pressure_detail[pk] = p
        if len(examples) < example_limit:
            examples.append({
                "ordinal": ordinal,
                "mass": mass,
                "route_key": route_key(row),
                "entry_s": float(row["entry_sec"]),
                "exit_s": float(row["exit_sec"]),
                "reason": reason,
                "progress": best.get("progress"),
                "pressure": p,
                "route_failure_reasons": dict(Counter(str(x["reason"]) for x in failures)),
            })

    top_pressure = []
    for key, mass in pressure_mass.most_common(200):
        top_pressure.append({"key": key, "passenger_mass": mass, **pressure_detail[key]})
    used_mass = sum(1 for _sid, m in trajectory_assignment_mass.items() if m > 0)
    result = {
        "schema": SCHEMA_SHARD,
        "service_date": service_raw.get("service_date", "2019-01-04"),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "passenger_mass": passenger_mass,
        "resolved_mass": resolved_mass,
        "resolved_share": resolved_mass / passenger_mass if passenger_mass else None,
        "unresolved_mass": passenger_mass - resolved_mass,
        "no_route_mass": no_route_mass,
        "unresolved_route_exists_mass": unresolved_route_exists_mass,
        "cohort_count": cohort_count,
        "resolved_cohort_count": resolved_cohort_count,
        "failure_mass": dict(failure_mass),
        "top_residual_service_pressure": top_pressure,
        "examples": examples,
        "trajectory_assignment_mass": dict(trajectory_assignment_mass),
        "assigned_trajectory_count": used_mass,
        "physical_gate_audit": {
            "access_min_required_s": ACCESS_MIN_S,
            "egress_min_required_s": EGRESS_MIN_S,
            "transfer_min_required_s": TRANSFER_MIN_S,
            "minimum_resolved_access_budget_s": resolved_access_min if resolved_access_min < float("inf") else None,
            "minimum_resolved_egress_budget_s": resolved_egress_min if resolved_egress_min < float("inf") else None,
        },
        "service_contract": {
            "count_free_inferred_service_trajectory_count": int(service_raw["inferred_service_trajectory_count"]),
            "train_count_is_input": False,
            "planned_timetable_used": False,
            "legacy_candidate_roots_used_as_input": False,
        },
        "semantics": {
            "this_is_joint_feedback_probe_not_final_r1": True,
            "passenger_chains_regenerated_against_count_free_services": True,
            "legacy_root_chain_ids_not_reused": True,
            "route_support_used_as_network_path_hypotheses": True,
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["shard_index", "passenger_mass", "resolved_mass", "resolved_share", "no_route_mass", "unresolved_route_exists_mass", "failure_mass", "assigned_trajectory_count"]}, ensure_ascii=False, indent=2))
    return result


def merge(input_dir: Path, output: Path) -> dict[str, Any]:
    files = sorted(input_dir.glob("shard_*.json"))
    if not files:
        raise SystemExit("no shard results")
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    shard_count = int(rows[0]["shard_count"])
    if len(rows) != shard_count or {int(x["shard_index"]) for x in rows} != set(range(shard_count)):
        raise SystemExit("incomplete shard set")
    pm = sum(float(x["passenger_mass"]) for x in rows)
    rm = sum(float(x["resolved_mass"]) for x in rows)
    nr = sum(float(x["no_route_mass"]) for x in rows)
    ure = sum(float(x["unresolved_route_exists_mass"]) for x in rows)
    failures: Counter[str] = Counter()
    pressures: Counter[str] = Counter()
    pressure_detail: dict[str, dict[str, Any]] = {}
    assignments: Counter[str] = Counter()
    examples = []
    for x in rows:
        failures.update({k: float(v) for k, v in x["failure_mass"].items()})
        assignments.update({k: float(v) for k, v in x["trajectory_assignment_mass"].items()})
        for p in x["top_residual_service_pressure"]:
            key = str(p["key"])
            pressures[key] += float(p["passenger_mass"])
            pressure_detail.setdefault(key, {k: v for k, v in p.items() if k not in {"key", "passenger_mass"}})
        examples.extend(x.get("examples", []))
    top_pressure = [{"key": key, "passenger_mass": mass, **pressure_detail[key]} for key, mass in pressures.most_common(300)]
    service_count = int(rows[0]["service_contract"]["count_free_inferred_service_trajectory_count"])
    assigned_count = sum(1 for _sid, m in assignments.items() if m > 0)
    historical_total = 1206156.0
    historical_resolved = 528605.0
    historical_no_chain = 668234.0
    result = {
        "schema": SCHEMA_MERGED,
        "service_date": rows[0]["service_date"],
        "status": "COUNT_FREE_PASSENGER_FEEDBACK_PROBE_COMPLETED_REQUIRES_SERVICE_UPDATE",
        "passenger_mass": pm,
        "resolved_mass": rm,
        "resolved_share": rm / pm if pm else None,
        "unresolved_mass": pm - rm,
        "no_route_mass": nr,
        "unresolved_route_exists_mass": ure,
        "failure_mass": dict(failures),
        "count_free_service_trajectory_count": service_count,
        "passenger_assigned_trajectory_count": assigned_count,
        "passenger_unassigned_trajectory_count": service_count - assigned_count,
        "trajectory_assignment_mass": dict(assignments),
        "top_residual_service_pressure": top_pressure,
        "representative_unresolved_examples": examples[:160],
        "historical_fixed_root_comparison_only": {
            "passenger_mass": historical_total,
            "resolved_mass": historical_resolved,
            "resolved_share": historical_resolved / historical_total,
            "route_exists_but_no_feasible_chain_mass": historical_no_chain,
            "resolved_mass_change": rm - historical_resolved if abs(pm - historical_total) < 1e-6 else None,
            "resolved_share_change": (rm / pm - historical_resolved / historical_total) if pm and abs(pm - historical_total) < 1e-6 else None,
        },
        "physical_gate_audit": {
            "access_min_required_s": ACCESS_MIN_S,
            "egress_min_required_s": EGRESS_MIN_S,
            "transfer_min_required_s": TRANSFER_MIN_S,
            "minimum_resolved_access_budget_s": min(x["physical_gate_audit"]["minimum_resolved_access_budget_s"] for x in rows if x["physical_gate_audit"]["minimum_resolved_access_budget_s"] is not None),
            "minimum_resolved_egress_budget_s": min(x["physical_gate_audit"]["minimum_resolved_egress_budget_s"] for x in rows if x["physical_gate_audit"]["minimum_resolved_egress_budget_s"] is not None),
        },
        "next_joint_update": {
            "birth_or_time_shift_targets": "top_residual_service_pressure for FIRST_LEG_NO_SERVICE and TRANSFER_NO_DOWNSTREAM_SERVICE",
            "death_targets": "count-free trajectories with zero passenger assignment and weak AFC evidence",
            "split_targets": "high residual pressure overlapping broad AFC ridge support",
            "scientific_rule": "service count remains latent; passenger residuals update the same joint hidden world rather than opening a new R1 stage",
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "passenger_mass": pm,
        "resolved_mass": rm,
        "resolved_share": result["resolved_share"],
        "historical_resolved_share": result["historical_fixed_root_comparison_only"]["resolved_share"],
        "resolved_share_change": result["historical_fixed_root_comparison_only"]["resolved_share_change"],
        "unresolved_route_exists_mass": ure,
        "failure_mass": result["failure_mass"],
        "count_free_service_trajectory_count": service_count,
        "passenger_assigned_trajectory_count": assigned_count,
    }, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--cohorts", type=Path, required=True)
    s.add_argument("--routes", type=Path, required=True)
    s.add_argument("--services", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--shard-index", type=int, required=True)
    s.add_argument("--shard-count", type=int, default=8)
    s.add_argument("--example-limit", type=int, default=40)
    s = sub.add_parser("merge")
    s.add_argument("--input-dir", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.command == "shard":
        run_shard(a.cohorts, a.routes, a.services, a.output, a.shard_index, a.shard_count, a.example_limit)
    else:
        merge(a.input_dir, a.output)


if __name__ == "__main__":
    main()
