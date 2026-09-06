from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize

import scripts.mppd_r1_hz_count_free_passenger_coverage as cov

SCHEMA = "mppd.r1-hz-count-free-joint-movement-fit.v1"
ACCESS_LOWER = 15.0
EGRESS_LOWER = 15.0
TRANSFER_LOWER = 5.0
EPS = 1e-300
N = NormalDist()


def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def survival(lower: float, mu: float, sigma: float) -> float:
    z = (math.log(lower) - mu) / sigma
    return max(EPS, 1.0 - N.cdf(z))


def interval_moments(lower: float, mu: float, sigma: float, lo: float, hi: float) -> tuple[float, float, float]:
    lo = max(lower, lo)
    if hi <= lo:
        return 0.0, 0.0, 0.0
    a = (math.log(lo) - mu) / sigma
    b = (math.log(hi) - mu) / sigma
    p = max(0.0, N.cdf(b) - N.cdf(a))
    if p <= EPS:
        return 0.0, 0.0, 0.0
    ez = (normal_pdf(a) - normal_pdf(b)) / p
    ez2 = 1.0 + (a * normal_pdf(a) - b * normal_pdf(b)) / p
    ey = mu + sigma * ez
    ey2 = mu * mu + 2.0 * mu * sigma * ez + sigma * sigma * ez2
    return p, ey, ey2


def boarding_moments(
    lower: float,
    base_time: float,
    selected_time: float,
    previous_times: list[float],
    mu: float,
    sigma: float,
    hazard: float,
) -> dict[str, float] | None:
    upper = selected_time - base_time
    if upper < lower:
        return None
    gaps = sorted(t - base_time for t in previous_times if base_time + lower < t < selected_time - 1e-9)
    thresholds = [lower] + gaps + [upper]
    surv = survival(lower, mu, sigma)
    terms = []
    nprev = len(gaps)
    for i in range(len(thresholds) - 1):
        p0, ey, ey2 = interval_moments(lower, mu, sigma, thresholds[i], thresholds[i + 1])
        if p0 <= 0:
            continue
        p = p0 / surv
        skipped = nprev - i
        w = p * hazard * ((1.0 - hazard) ** skipped)
        if w > 0:
            terms.append((w, ey, ey2, skipped))
    total = sum(x[0] for x in terms)
    if total <= EPS:
        return None
    return {
        "loglik": math.log(total),
        "elog": sum(w * ey for w, ey, _ey2, _sk in terms) / total,
        "elog2": sum(w * ey2 for w, _ey, ey2, _sk in terms) / total,
        "eskip": sum(w * sk for w, _ey, _ey2, sk in terms) / total,
        "upper": upper,
    }


def fit_truncated_from_expected(lower: float, w: float, sy: float, sy2: float, mu0: float, sigma0: float) -> tuple[float, float]:
    if w <= 0:
        return mu0, sigma0

    def objective(z: np.ndarray) -> float:
        mu = float(z[0]); sigma = float(math.exp(z[1]))
        quad = sy2 - 2.0 * mu * sy + w * mu * mu
        return w * math.log(sigma) + 0.5 * quad / (sigma * sigma) + w * math.log(survival(lower, mu, sigma))

    res = minimize(
        objective,
        np.array([mu0, math.log(sigma0)], dtype=float),
        method="L-BFGS-B",
        bounds=[(math.log(lower), math.log(3600.0)), (math.log(0.15), math.log(1.8))],
    )
    if not res.success or not np.all(np.isfinite(res.x)):
        return mu0, sigma0
    return float(res.x[0]), float(math.exp(res.x[1]))


def truncated_quantile(lower: float, mu: float, sigma: float, q: float) -> float:
    f0 = N.cdf((math.log(lower) - mu) / sigma)
    u = min(1.0 - 1e-12, max(1e-12, f0 + q * (1.0 - f0)))
    return max(lower, math.exp(mu + sigma * N.inv_cdf(u)))


def movement_key(route: dict[str, Any], j: int) -> str:
    metas = route.get("inter_leg_movement_meta", [])
    if j < len(metas) and metas[j].get("movement"):
        return str(metas[j]["movement"])
    moves = route.get("transfer_movements", [])
    if j < len(moves):
        return str(moves[j])
    legs = route.get("ride_legs", [])
    if j + 1 < len(legs):
        return f"{legs[j]['path_id']}:{legs[j]['to_station']}->{legs[j+1]['path_id']}:{legs[j+1]['from_station']}"
    return f"UNKNOWN_TRANSFER_{j}"


def previous_departures(index: cov.LegIndex, leg: dict[str, Any], base: float, selected: float, lower: float) -> list[float]:
    data = index.get(leg)
    deps = data["deps"]
    a = bisect.bisect_right(deps, base + lower)
    b = bisect.bisect_left(deps, selected - 1e-9)
    return [float(x) for x in deps[a:b]]


def initial_params(stations: set[str]) -> dict[str, Any]:
    return {
        "access_hazard": 0.85,
        "transfer_hazard": 0.80,
        "access": {s: {"mu": math.log(45.0), "sigma": 0.65, "evidence_mass": 0.0} for s in stations},
        "egress": {s: {"mu": math.log(65.0), "sigma": 0.65, "evidence_mass": 0.0} for s in stations},
        "transfers": {},
    }


def transfer_default() -> dict[str, Any]:
    return {
        "components": [
            {"path_id": "latent_path_1", "weight": 0.60, "mu": math.log(30.0), "sigma": 0.55, "evidence_mass": 0.0},
            {"path_id": "latent_path_2", "weight": 0.40, "mu": math.log(100.0), "sigma": 0.55, "evidence_mass": 0.0},
        ]
    }


def add_expected(stats: dict[str, float], mass: float, r: dict[str, float]) -> None:
    stats["w"] += mass
    stats["sy"] += mass * float(r["elog"])
    stats["sy2"] += mass * float(r["elog2"])
    stats["skip"] += mass * float(r["eskip"])


def fit_pass(
    cohorts: Path,
    routes: dict[str, list[dict[str, Any]]],
    services: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    index = cov.LegIndex(services)
    access_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"w": 0.0, "sy": 0.0, "sy2": 0.0, "skip": 0.0})
    egress_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"w": 0.0, "sy": 0.0, "sy2": 0.0})
    transfer_stats: dict[str, list[dict[str, float]]] = {}
    transfer_skip = 0.0
    transfer_mass = 0.0
    access_skip = 0.0
    access_mass = 0.0
    resolved_mass = total_mass = 0.0
    failure_mass: Counter[str] = Counter()
    movement_evidence: Counter[str] = Counter()

    pf = pq.ParquetFile(cohorts)
    for batch in pf.iter_batches(batch_size=50000):
        for row in batch.to_pylist():
            mass = float(row["passenger_mass"]); total_mass += mass
            rr = routes.get(cov.route_key(row))
            if not rr:
                failure_mass["NO_ROUTE_SUPPORT"] += mass
                continue
            success = None; chosen = None; failures = []
            for route in rr:
                r = cov.evaluate_route(row, route, index)
                if r["ok"]:
                    success = r; chosen = route; break
                failures.append(r)
            if success is None or chosen is None:
                failure_mass[str(cov.choose_failure(failures)["reason"])] += mass
                continue
            resolved_mass += mass
            chain = success["chain"]
            legs = chosen["ride_legs"]

            origin = str(int(row["origin_station"]))
            ap = params["access"].setdefault(origin, {"mu": math.log(45.0), "sigma": 0.65, "evidence_mass": 0.0})
            selected_dep = float(chain[0]["departure_s"])
            prev = previous_departures(index, legs[0], float(row["entry_sec"]), selected_dep, ACCESS_LOWER)
            ar = boarding_moments(ACCESS_LOWER, float(row["entry_sec"]), selected_dep, prev, float(ap["mu"]), float(ap["sigma"]), float(params["access_hazard"]))
            if ar is not None:
                add_expected(access_stats[origin], mass, ar)
                access_skip += mass * float(ar["eskip"]); access_mass += mass

            destination = str(int(row["destination_station"]))
            e = float(row["exit_sec"]) - float(chain[-1]["arrival_s"])
            if e >= EGRESS_LOWER:
                y = math.log(e)
                es = egress_stats[destination]
                es["w"] += mass; es["sy"] += mass * y; es["sy2"] += mass * y * y

            for j in range(len(legs) - 1):
                key = movement_key(chosen, j)
                model = params["transfers"].setdefault(key, transfer_default())
                if key not in transfer_stats:
                    transfer_stats[key] = [
                        {"w": 0.0, "sy": 0.0, "sy2": 0.0, "skip": 0.0}
                        for _ in model["components"]
                    ]
                base = float(chain[j]["arrival_s"])
                selected = float(chain[j + 1]["departure_s"])
                prev_t = previous_departures(index, legs[j + 1], base, selected, TRANSFER_LOWER)
                rows = []
                for k, comp in enumerate(model["components"]):
                    br = boarding_moments(TRANSFER_LOWER, base, selected, prev_t, float(comp["mu"]), float(comp["sigma"]), float(params["transfer_hazard"]))
                    if br is None:
                        continue
                    rows.append((k, math.log(max(EPS, float(comp["weight"]))) + float(br["loglik"]), br))
                if not rows:
                    continue
                zmax = max(x[1] for x in rows)
                z = zmax + math.log(sum(math.exp(x[1] - zmax) for x in rows))
                expected_skip = 0.0
                for k, lw, br in rows:
                    q = math.exp(lw - z)
                    st = transfer_stats[key][k]
                    st["w"] += mass * q
                    st["sy"] += mass * q * float(br["elog"])
                    st["sy2"] += mass * q * float(br["elog2"])
                    st["skip"] += mass * q * float(br["eskip"])
                    expected_skip += q * float(br["eskip"])
                transfer_mass += mass; transfer_skip += mass * expected_skip
                movement_evidence[key] += mass

    newp = json.loads(json.dumps(params))
    for station, st in access_stats.items():
        old = params["access"][station]
        mu, sigma = fit_truncated_from_expected(ACCESS_LOWER, st["w"], st["sy"], st["sy2"], float(old["mu"]), float(old["sigma"]))
        newp["access"][station] = {"mu": mu, "sigma": sigma, "evidence_mass": st["w"]}
    if access_mass > 0:
        newp["access_hazard"] = min(0.995, max(0.05, access_mass / (access_mass + access_skip)))

    for station, st in egress_stats.items():
        old = params["egress"].get(station, {"mu": math.log(65.0), "sigma": 0.65})
        mu, sigma = fit_truncated_from_expected(EGRESS_LOWER, st["w"], st["sy"], st["sy2"], float(old["mu"]), float(old["sigma"]))
        newp["egress"][station] = {"mu": mu, "sigma": sigma, "evidence_mass": st["w"]}

    for key, comp_stats in transfer_stats.items():
        old_model = params["transfers"][key]
        total = sum(x["w"] for x in comp_stats)
        comps = []
        for k, st in enumerate(comp_stats):
            old = old_model["components"][k]
            mu, sigma = fit_truncated_from_expected(TRANSFER_LOWER, st["w"], st["sy"], st["sy2"], float(old["mu"]), float(old["sigma"]))
            comps.append({
                "path_id": old["path_id"],
                "weight": st["w"] / total if total > 0 else float(old["weight"]),
                "mu": mu,
                "sigma": sigma,
                "evidence_mass": st["w"],
            })
        newp["transfers"][key] = {"components": comps}
    if transfer_mass > 0:
        newp["transfer_hazard"] = min(0.995, max(0.05, transfer_mass / (transfer_mass + transfer_skip)))

    diag = {
        "passenger_mass": total_mass,
        "resolved_mass": resolved_mass,
        "resolved_share": resolved_mass / total_mass if total_mass else None,
        "failure_mass": dict(failure_mass),
        "access_evidence_mass": access_mass,
        "transfer_evidence_mass": transfer_mass,
        "transfer_movement_evidence_mass": dict(movement_evidence),
        "access_hazard": newp["access_hazard"],
        "transfer_hazard": newp["transfer_hazard"],
    }
    return newp, diag


def interval_record(lower: float, p: dict[str, Any]) -> dict[str, Any]:
    mu = float(p["mu"]); sigma = float(p["sigma"])
    return {
        "lower_physical_s": lower,
        "q05_s": truncated_quantile(lower, mu, sigma, 0.05),
        "q25_s": truncated_quantile(lower, mu, sigma, 0.25),
        "median_s": truncated_quantile(lower, mu, sigma, 0.50),
        "q75_s": truncated_quantile(lower, mu, sigma, 0.75),
        "q95_s": truncated_quantile(lower, mu, sigma, 0.95),
        "mu": mu,
        "sigma": sigma,
        "evidence_mass": float(p.get("evidence_mass", 0.0)),
        "identified_from_current_day": float(p.get("evidence_mass", 0.0)) > 0,
    }


def build_result(params: dict[str, Any], diagnostics: list[dict[str, Any]], service_raw: dict[str, Any]) -> dict[str, Any]:
    access = {s: interval_record(ACCESS_LOWER, p) for s, p in sorted(params["access"].items(), key=lambda x: int(x[0]))}
    egress = {s: interval_record(EGRESS_LOWER, p) for s, p in sorted(params["egress"].items(), key=lambda x: int(x[0]))}
    transfers = {}
    for key, model in sorted(params["transfers"].items()):
        transfers[key] = {
            "paths": [
                {"path_id": c["path_id"], "weight": float(c["weight"]), **interval_record(TRANSFER_LOWER, c)}
                for c in model["components"]
            ],
            "evidence_mass": sum(float(c.get("evidence_mass", 0.0)) for c in model["components"]),
        }
    return {
        "schema": SCHEMA,
        "status": "COUNT_FREE_JOINT_MOVEMENT_DISTRIBUTIONS_ESTIMATED_REQUIRES_FULL_POSTERIOR_REWEIGHT",
        "service_date": service_raw.get("service_date", "2019-01-04"),
        "service_trajectory_count": int(service_raw["inferred_service_trajectory_count"]),
        "service_clock_shift_s": float(service_raw.get("absolute_clock_calibration", {}).get("common_shift_s", 0.0)),
        "physical_lower_bounds_s": {"access": ACCESS_LOWER, "egress": EGRESS_LOWER, "transfer": TRANSFER_LOWER},
        "access_hazard": float(params["access_hazard"]),
        "transfer_hazard": float(params["transfer_hazard"]),
        "access_intervals": access,
        "egress_intervals": egress,
        "transfer_path_intervals": transfers,
        "iteration_diagnostics": diagnostics,
        "audit": {
            "access_station_count": len(access),
            "access_identified_station_count": sum(x["identified_from_current_day"] for x in access.values()),
            "egress_station_count": len(egress),
            "egress_identified_station_count": sum(x["identified_from_current_day"] for x in egress.values()),
            "transfer_movement_count": len(transfers),
            "transfer_latent_path_count": sum(len(x["paths"]) for x in transfers.values()),
            "access_q05_min_s": min(x["q05_s"] for x in access.values()),
            "egress_q05_min_s": min(x["q05_s"] for x in egress.values()),
            "transfer_q05_min_s": min((p["q05_s"] for x in transfers.values() for p in x["paths"]), default=None),
        },
        "semantics": {
            "r1_is_one_joint_reconstruction": True,
            "planned_timetable_used": False,
            "service_count_inferred_from_afc": True,
            "egress_is_direct_gate_minus_final_arrival_residual_under_selected_chain": True,
            "access_is_not_departure_minus_gate_entry": True,
            "access_waiting_separated_by_service_opportunity_integration": True,
            "transfer_waiting_separated_by_service_opportunity_integration": True,
            "transfer_paths_are_latent_mixture_components": True,
            "current_chain_assignment_is_feasibility_MAP_initialization_not_final_joint_posterior": True,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohorts", type=Path, required=True)
    p.add_argument("--routes", type=Path, required=True)
    p.add_argument("--services", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    routes = cov.load_routes(a.routes)
    services, service_raw = cov.load_services(a.services)
    stations = {str(s) for tr in services.values() for s in tr["events"]}
    params = initial_params(stations)
    diagnostics = []
    for it in range(a.iterations):
        params, diag = fit_pass(a.cohorts, routes, services, params)
        diag["iteration"] = it + 1
        diagnostics.append(diag)
        print(json.dumps(diag, ensure_ascii=False, indent=2))
    result = build_result(params, diagnostics, service_raw)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "service_trajectory_count": result["service_trajectory_count"],
        "service_clock_shift_s": result["service_clock_shift_s"],
        "access_hazard": result["access_hazard"],
        "transfer_hazard": result["transfer_hazard"],
        "audit": result["audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
