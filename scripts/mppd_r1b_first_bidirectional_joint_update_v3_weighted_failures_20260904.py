import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base
import scripts.mppd_r1b_first_bidirectional_joint_update_v2_20260904 as v2


def posterior_pass_v3(cohorts, routes, meta, roots, kernels, beam, max_skip, retain_details=True):
    rides_fn, _ = base.build_joint_rides(roots)
    result = {
        "processed_mass": 0.0,
        "finite_mass": 0.0,
        "status_mass": Counter(),
        "failure_mass": Counter(),
        "route_entropy_weighted": 0.0,
        "skip_mass": Counter(),
        "non_simple_mass": 0.0,
        "first_boarding_entropy_contraction_weighted": 0.0,
        "access_tail_mass_gt_15min": 0.0,
        "egress_tail_mass_gt_15min": 0.0,
        "missing_service_pressure": Counter(),
        "cohorts": {},
        "factors": [],
        "root_usage": Counter(),
        "failure_attribution": "STRUCTURAL_PRIOR_WEIGHTED_FRACTIONAL_MASS_CONSERVING",
    }
    for oc, dc, tin, tout, mass in cohorts:
        result["processed_mass"] += mass
        cands = routes.get((oc, dc), [])
        cohort_id = f"{oc}|{dc}|{tin.isoformat()}|{tout.isoformat()}"
        if not cands:
            result["status_mass"]["NO_STRUCTURAL_ROUTE"] += mass
            if retain_details:
                result["cohorts"][cohort_id] = {"status": "NO_STRUCTURAL_ROUTE", "mass": mass}
            continue

        min_cost = min(float(c.get("base_cost", 0.0)) for c in cands)
        hypotheses = []
        fail_details = []
        route_scores = []
        for ci, cand in enumerate(cands):
            chains, failure = base.route_beam_joint(cand, meta, rides_fn, tin, tout, beam, kernels, max_skip)
            fail_details.append(failure)
            if not chains:
                route_scores.append(-math.inf)
                continue
            route_lp = base.logsumexp([x["logp"] for x in chains]) - 0.30 * max(
                0.0, float(cand.get("base_cost", 0.0)) - min_cost
            )
            route_scores.append(route_lp)
            for ch in chains:
                hypotheses.append({
                    "candidate_index": ci,
                    "line_sequence": cand.get("line_sequence", ""),
                    "transfer_count": int(cand.get("transfer_count", 0)),
                    "non_simple": base.path_is_non_simple(cand),
                    "logp": ch["logp"] - 0.30 * max(0.0, float(cand.get("base_cost", 0.0)) - min_cost),
                    "chain": ch["chain"],
                    "factors": ch["factors"],
                })

        finite_routes = [(i, lp) for i, lp in enumerate(route_scores) if math.isfinite(lp)]
        if not hypotheses or not finite_routes:
            sw = base.structural_pressure_weights(cands)
            reason_weights = Counter()
            for ci, f in enumerate(fail_details):
                reason = (f or {}).get("reason", "UNKNOWN")
                w = float(sw[ci]) if ci < len(sw) else 0.0
                reason_weights[reason] += w
                if f and reason == "MISSING_SERVICE_SEGMENT":
                    key = f"{f.get('line')}|{f.get('origin')}|{f.get('destination')}"
                    result["missing_service_pressure"][key] += mass * w
            z_reason = sum(reason_weights.values())
            if z_reason <= 0:
                reason_weights = Counter({"UNKNOWN": 1.0})
                z_reason = 1.0
            reason_probs = {k: float(v) / z_reason for k, v in reason_weights.items()}
            for reason, rp in reason_probs.items():
                result["failure_mass"][reason] += mass * rp
            result["status_mass"]["NO_FINITE_POSTERIOR"] += mass
            dominant_reason = max(reason_probs.items(), key=lambda x: x[1])[0]
            if retain_details:
                result["cohorts"][cohort_id] = {
                    "status": dominant_reason,
                    "mass": mass,
                    "failure_reason_probs": reason_probs,
                }
            continue

        z_h = base.logsumexp([h["logp"] for h in hypotheses])
        for h in hypotheses:
            h["probability"] = math.exp(h["logp"] - z_h)

        route_prob = defaultdict(float)
        first_root_prob = defaultdict(float)
        route_first = defaultdict(lambda: defaultdict(float))
        for h in hypotheses:
            p = h["probability"]
            route_prob[h["line_sequence"]] += p
            first_root = h["chain"][0]["root_key"] if h["chain"] else "NONE"
            first_root_prob[first_root] += p
            route_first[first_root][h["line_sequence"]] += p
            result["non_simple_mass"] += mass * p * (1.0 if h["non_simple"] else 0.0)
            for leg in h["chain"]:
                result["skip_mass"][int(leg["n_skip"])] += mass * p
                result["root_usage"][leg["root_key"]] += mass * p
            if h["factors"]:
                af = next((f for f in h["factors"] if f["type"] == "INTERVAL" and f["kind"] == "ACCESS"), None)
                if af:
                    access_upper = (af["upper"] - af["ready"]).total_seconds()
                    if access_upper > 900:
                        result["access_tail_mass_gt_15min"] += mass * p
                ef = next((f for f in reversed(h["factors"]) if f["type"] == "EGRESS"), None)
                if ef and not ef.get("station_only_proxy") and (ef["exit_time"] - ef["arr"]).total_seconds() > 900:
                    result["egress_tail_mass_gt_15min"] += mass * p
            if retain_details:
                for factor in h["factors"]:
                    ff = dict(factor)
                    ff["weight"] = mass * p
                    result["factors"].append(ff)

        probs = list(route_prob.values())
        H_before = -sum(p * math.log(max(p, base.EPS)) for p in probs if p > 0)
        H_after = 0.0
        for fr, frp in first_root_prob.items():
            if frp <= 0:
                continue
            conditional = [v / frp for v in route_first[fr].values()]
            H_after += frp * (-sum(p * math.log(max(p, base.EPS)) for p in conditional if p > 0))
        contraction = max(0.0, H_before - H_after)

        result["finite_mass"] += mass
        result["status_mass"]["FINITE_POSTERIOR"] += mass
        result["route_entropy_weighted"] += mass * H_before
        result["first_boarding_entropy_contraction_weighted"] += mass * contraction

        if retain_details:
            top_route = max(route_prob.items(), key=lambda x: x[1])
            top_first = max(first_root_prob.items(), key=lambda x: x[1])
            result["cohorts"][cohort_id] = {
                "status": "FINITE_POSTERIOR",
                "mass": mass,
                "route_probs": dict(route_prob),
                "route_entropy": H_before,
                "top_route": top_route[0],
                "top_route_prob": top_route[1],
                "first_root_probs": dict(first_root_prob),
                "top_first_root": top_first[0],
                "top_first_prob": top_first[1],
            }

    return result


def find_out_dir(argv):
    return v2.find_out_dir(argv)


def main():
    base.route_beam_joint = v2.route_beam_joint_v2
    base.update_kernels = v2.update_kernels_v2
    base.posterior_pass = posterior_pass_v3
    base.main()

    out = find_out_dir(sys.argv[1:])
    if out is None:
        return
    path = Path(out) / "r1b_first_bidirectional_joint_update_smoke_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = "mppd.r1b-first-bidirectional-joint-update-smoke.v3-weighted-failures"
    payload["implementation_patch"] = {
        "wrapper": "scripts/mppd_r1b_first_bidirectional_joint_update_v3_weighted_failures_20260904.py",
        "inherits_v2_kernel_damping": 0.35,
        "station_only_proxy_regression_fixed": True,
        "failure_attribution": "STRUCTURAL_PRIOR_WEIGHTED_FRACTIONAL_MASS_CONSERVING",
        "candidate_multiplicity_majority_vote_removed": True,
    }
    e0 = payload.get("iteration", {}).get("E0_passenger_posterior", {})
    e1 = payload.get("iteration", {}).get("E1_passenger_posterior", {})
    for e in (e0, e1):
        nf = float(e.get("processed_passenger_mass", 0.0)) - float(e.get("finite_posterior_mass", 0.0))
        fm = sum(float(v) for v in (e.get("failure_mass") or {}).values())
        e["failure_mass_conservation_error"] = fm - nf
        e["failure_mass_conservation_pass"] = abs(fm - nf) <= 1e-6
    payload.setdefault("scientific_boundary", []).extend([
        "No-finite cohorts now distribute passenger mass fractionally across candidate failure reasons using the same structural route prior weights used for missing-service pressure; candidate-count majority vote is removed.",
        "Fractional failure attribution is diagnostic uncertainty accounting, not observed failure truth.",
        "Failure mass is required to conserve no-finite passenger mass within floating tolerance.",
    ])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
