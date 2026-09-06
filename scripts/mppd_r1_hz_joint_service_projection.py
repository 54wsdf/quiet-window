from __future__ import annotations

"""Scientific repair layer for the unified Hangzhou R1 solver.

Pass 0 treats candidate service-root times as latent means rather than exact realized
truth.  For each passenger candidate chain, it projects the chain's root-level time
offsets to the nearest physically feasible state (weighted by root timing uncertainty)
before evaluating access/transfer/egress likelihood.  This prevents the 5-second
physical lower bound from incorrectly rejecting a chain merely because its latent
service time has not yet been updated.

After the first global service-time update (parameter iteration > 0), projection is
disabled: pass 1 evaluates passenger chains against one shared recovered timetable,
so final passenger chains and the reported service timetable are exactly consistent.
"""

import math
import statistics
from typing import Any

import scripts.mppd_r1_hz_joint_full_day as base

_ORIGINAL = base.evaluate_candidate
MAX_OFFSET_S = 180.0
PROJECTION_ITERS = 40
TOL = 1e-7


def _root_sd(root: dict[str, Any]) -> float:
    vals = [float(e.get("sd_s", 90.0)) for e in root["events"].values() if math.isfinite(float(e.get("sd_s", 90.0)))]
    return max(15.0, statistics.median(vals) if vals else 90.0)


def _project_offsets(chain: list[str], legs: list[dict[str, Any]], roots: dict[str, Any], entry: float, exit_t: float):
    unique = list(dict.fromkeys(chain))
    center = {rid: float(roots[rid].get("offset_s", 0.0)) for rid in unique}
    sigma = {rid: _root_sd(roots[rid]) for rid in unique}
    delta = dict(center)

    # constraint represented as a^T delta >= b; a is sparse {rid: coeff}
    constraints: list[tuple[dict[str, float], float, str]] = []
    first = chain[0]; first_station = str(int(legs[0]["from_station"]))
    first_base = float(roots[first]["events"][first_station]["time_s"]) - center[first]
    constraints.append(({first: 1.0}, entry + base.MIN_MOVE_S - first_base, "ACCESS_MIN"))

    last = chain[-1]; last_station = str(int(legs[-1]["to_station"]))
    last_base = float(roots[last]["events"][last_station]["time_s"]) - center[last]
    constraints.append(({last: -1.0}, -(exit_t - base.MIN_MOVE_S - last_base), "EGRESS_MIN"))

    for j in range(len(chain) - 1):
        aroot, broot = chain[j], chain[j + 1]
        station_a = str(int(legs[j]["to_station"])); station_b = str(int(legs[j + 1]["from_station"]))
        if station_a != station_b:
            return None
        arr_base = float(roots[aroot]["events"][station_a]["time_s"]) - center[aroot]
        dep_base = float(roots[broot]["events"][station_b]["time_s"]) - center[broot]
        # dep + db - arr - da >= 5
        constraints.append(({broot: 1.0, aroot: -1.0}, base.MIN_MOVE_S - (dep_base - arr_base), f"TRANSFER_{j}_MIN"))

    # Weighted projections in metric sum((delta-center)^2/sigma^2).
    for _ in range(PROJECTION_ITERS):
        changed = False
        for rid in unique:
            clipped = max(-MAX_OFFSET_S, min(MAX_OFFSET_S, delta[rid]))
            if abs(clipped - delta[rid]) > TOL:
                delta[rid] = clipped; changed = True
        for coeff, rhs, _name in constraints:
            lhs = sum(a * delta[r] for r, a in coeff.items())
            gap = rhs - lhs
            if gap <= TOL:
                continue
            denom = sum((a * a) * (sigma[r] ** 2) for r, a in coeff.items())
            if denom <= 0:
                return None
            for r, a in coeff.items():
                delta[r] += gap * (sigma[r] ** 2) * a / denom
            changed = True
        if not changed:
            break

    for rid in unique:
        if delta[rid] < -MAX_OFFSET_S - 1e-6 or delta[rid] > MAX_OFFSET_S + 1e-6:
            return None
    for coeff, rhs, _name in constraints:
        if sum(a * delta[r] for r, a in coeff.items()) < rhs - 1e-5:
            return None

    penalty = 0.5 * sum(((delta[r] - center[r]) / sigma[r]) ** 2 for r in unique)
    return delta, penalty


def evaluate_candidate_projected(row: dict[str, Any], route: dict[str, Any], roots: dict[str, Any], seq, params: dict[str, Any]):
    # Once pass 0 has produced one shared recovered service state, final inference is
    # evaluated against that state without passenger-specific service projection.
    if int(params.get("iteration", 0)) > 0:
        return _ORIGINAL(row, route, roots, seq, params)

    chain = [x for x in str(row.get("root_chain", "")).split(">") if x]
    legs = route.get("ride_legs", [])
    if not chain or len(chain) != len(legs) or any(rid not in roots for rid in chain):
        return None
    entry = float(row["entry_sec"]); exit_t = float(row["exit_sec"])
    projected = _project_offsets(chain, legs, roots, entry, exit_t)
    if projected is None:
        return None
    offsets, penalty = projected

    # Local root views with the projected latent service state.  The shared input root
    # dictionary is never mutated.
    local_roots = dict(roots)
    for rid in set(chain):
        r = roots[rid]
        current = float(r.get("offset_s", 0.0)); shift = float(offsets[rid]) - current
        events = {s: {**e, "time_s": float(e["time_s"]) + shift} for s, e in r["events"].items()}
        local_roots[rid] = {**r, "events": events, "offset_s": float(offsets[rid])}

    # Reimplement the compact candidate likelihood because service-sequence lookup for
    # non-selected trains remains global while selected chain events use local projection.
    origin = str(int(row["origin_station"])); dest = str(int(row["destination_station"]))
    rr = []
    for rid, leg in zip(chain, legs):
        r = local_roots[rid]
        b = str(int(leg["from_station"])); a = str(int(leg["to_station"]))
        if b not in r["events"] or a not in r["events"]:
            return None
        dep = float(r["events"][b]["time_s"]); arr = float(r["events"][a]["time_s"])
        if arr <= dep:
            return None
        rr.append((r, b, a, dep, arr))

    am = params["access"].get(origin, {"mu": math.log(60.0), "sigma": 0.75})
    prev = base.family_previous_times(seq, rr[0][0], rr[0][1], rr[0][3], entry)
    access = base.boarding_component(entry, rr[0][3], prev, float(am["mu"]), float(am["sigma"]), float(params["access_hazard"]))
    if access is None:
        return None

    egress_t = exit_t - rr[-1][4]
    em = params["egress"].get(dest, {"mu": math.log(60.0), "sigma": 0.75})
    ell = base.truncated_logpdf(egress_t, float(em["mu"]), float(em["sigma"]))
    if not math.isfinite(ell):
        return None

    transfers = []; tll = 0.0
    for j in range(len(rr) - 1):
        if rr[j][2] != rr[j + 1][1]:
            return None
        mk = base.movement_key(route, j)
        model = params["transfers"].get(mk, base.transfer_default())
        prev_times = base.family_previous_times(seq, rr[j + 1][0], rr[j + 1][1], rr[j + 1][3], rr[j][4])
        tr = base.transfer_mixture(rr[j][4], rr[j + 1][3], prev_times, model, float(params["transfer_hazard"]))
        if tr is None:
            return None
        tll += float(tr["loglik"]); transfers.append((mk, tr))

    route_cost = float(route.get("base_ranking_cost_s", 0.0))
    loglik = float(access["loglik"]) + ell + tll - route_cost / 1800.0 - penalty
    return {
        "loglik": loglik, "chain": chain, "route_rank": int(row["route_rank"]), "rides": rr,
        "access": access, "egress_time": egress_t, "transfers": transfers,
        "service_projection_penalty": penalty,
        "projected_root_offsets_s": {rid: float(offsets[rid]) for rid in chain},
    }


base.evaluate_candidate = evaluate_candidate_projected

if __name__ == "__main__":
    base.main()
