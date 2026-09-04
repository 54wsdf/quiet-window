import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r0_cached_full_network_support_rescan_20260904 as support
import scripts.mppd_g2v2_uncertain_service_full_network_posterior_20260904 as g2
import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base
import scripts.mppd_r1b_first_bidirectional_joint_update_v2_20260904 as v2
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch


def load_residuals(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mixture_quantile(kernel, q):
    lo, hi = 0.0, 7200.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if base.kernel_cdf(mid, kernel) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def normalized_probs(logps):
    z = base.logsumexp(logps)
    if not math.isfinite(z):
        return [1.0 / len(logps)] * len(logps) if logps else []
    return [math.exp(x - z) for x in logps]


def forward_missing_ready_envelope(cand, meta, rides_fn, tin, tout, beam, kernels, max_skip):
    legs = [x for x in base.r0.path_legs(cand["path"], meta) if x[1] != x[2]]
    states = [{
        "ready": tin,
        "ready_sd": 0.0,
        "ready_root": None,
        "logp": 0.0,
        "chain": [],
    }]
    for leg_index, (line, origin, destination) in enumerate(legs):
        rides = rides_fn(line, origin, destination)
        if not rides:
            prev_leg = states[0]["chain"][-1] if states and states[0]["chain"] else None
            movement = None if prev_leg is None else f"{prev_leg['line']}:{prev_leg['destination']}->{line}:{origin}"
            kind = "ACCESS" if prev_leg is None else "TRANSFER"
            kernel = base.kernel_for(kind, movement, kernels)
            q05 = mixture_quantile(kernel, 0.05)
            q50 = mixture_quantile(kernel, 0.50)
            q95 = mixture_quantile(kernel, 0.95)
            probs = normalized_probs([s["logp"] for s in states])
            ready50_ts = sum(
                p * (s["ready"] + timedelta(seconds=q50)).timestamp()
                for p, s in zip(probs, states)
            ) if states else None
            return {
                "reason": "MISSING_SERVICE_SEGMENT",
                "line": line,
                "origin": origin,
                "destination": destination,
                "leg_index": leg_index,
                "leg_count": len(legs),
                "kind_before_missing_gate": kind,
                "movement_before_missing_gate": movement,
                "state_count_at_gate": len(states),
                "ready_q05_min": min((s["ready"] + timedelta(seconds=q05) for s in states), default=None),
                "ready_q50_weighted_mean": datetime.fromtimestamp(ready50_ts) if ready50_ts is not None else None,
                "ready_q95_max": max((s["ready"] + timedelta(seconds=q95) for s in states), default=None),
                "observed_exit_time": tout,
            }

        nxt = []
        for st in states:
            ready = st["ready"]
            ready_sd = st["ready_sd"]
            ready_root = st["ready_root"]
            prev_leg = st["chain"][-1] if st["chain"] else None
            movement = None if prev_leg is None else f"{prev_leg['line']}:{prev_leg['destination']}->{line}:{origin}"
            kind = "ACCESS" if prev_leg is None else "TRANSFER"
            kern = base.kernel_for(kind, movement, kernels)
            for i, rd in enumerate(rides):
                if rd["arr"] > tout and (rd["arr"] - tout).total_seconds() > 3 * max(1.0, rd["arr_sd"]):
                    continue
                opts = base.skip_intervals(rides, i, max_skip)
                if not opts:
                    continue
                log_uniform = -math.log(len(opts))
                for n_skip, lower, upper in opts:
                    lp = base.interval_logprob_kernel(
                        lower["dep"] if lower else None,
                        lower["dep_sd"] if lower else 0.0,
                        upper["dep"],
                        upper["dep_sd"],
                        ready,
                        ready_sd,
                        kern,
                    ) + log_uniform
                    if not math.isfinite(lp):
                        continue
                    leg = {
                        "line": line,
                        "origin": origin,
                        "destination": destination,
                        "root_key": rd["root_key"],
                        "dep": rd["dep"],
                        "arr": rd["arr"],
                        "dep_sd": rd["dep_sd"],
                        "arr_sd": rd["arr_sd"],
                        "n_skip": n_skip,
                    }
                    nxt.append({
                        "ready": rd["arr"],
                        "ready_sd": rd["arr_sd"],
                        "ready_root": rd["root_key"],
                        "logp": st["logp"] + lp,
                        "chain": st["chain"] + [leg],
                    })
        if not nxt:
            return {"reason": "TIME_INCOMPATIBLE_CHAIN_BEFORE_OR_AT_SEGMENT", "leg_index": leg_index}
        nxt.sort(key=lambda x: (x["logp"], -x["ready"].timestamp()), reverse=True)
        states = nxt[:beam]
    return {"reason": "NO_MISSING_SEGMENT_ON_REPLAY"}


def dt_iso(x):
    return x.isoformat() if isinstance(x, datetime) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--service-init", required=True)
    ap.add_argument("--r1b-summary", required=True)
    ap.add_argument("--residual-cohorts", required=True)
    ap.add_argument("--topology-patch", required=True)
    ap.add_argument("--gtxa-overlay", required=True)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--max-skip", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    apply_topology_patch(G, meta, load_patch(args.topology_patch))
    apply_gtxa_overlay(G, meta, code_to_nodes, load_overlay(args.gtxa_overlay))
    routes = support.load_routes(args.routes)
    roots, service_manifest, service_payload = g2.load_uncertain_service(args.service_init)
    summary = json.loads(Path(args.r1b_summary).read_text(encoding="utf-8"))
    residuals = load_residuals(args.residual_cohorts)

    kernels = summary["iteration"]["M_kernel_update"]["kernels_after"]
    offsets = summary["iteration"]["M_service_timing_update"].get("nonzero_offsets", {})
    base.apply_root_offsets(roots, offsets)
    rides_fn, _ = base.build_joint_rides(roots)

    segment_stats = {}
    line_pressure = Counter()
    reason_mass = Counter()
    recovered_on_replay_mass = 0.0
    residual_output = []
    reason_prob_l1_weighted = 0.0
    total_mass = 0.0

    for rr in residuals:
        oc = str(rr["origin_code"])
        dc = str(rr["destination_code"])
        tin = datetime.fromisoformat(rr["entry_time"])
        tout = datetime.fromisoformat(rr["exit_time"])
        mass = float(rr["passenger_mass"])
        total_mass += mass
        cands = routes.get((oc, dc), [])
        sw = base.structural_pressure_weights(cands) if cands else []
        reason_w = Counter()
        seg_w = Counter()
        seg_ready = defaultdict(list)
        candidate_records = []
        any_finite = False

        for ci, cand in enumerate(cands):
            chains, failure = v2.route_beam_joint_v2(cand, meta, rides_fn, tin, tout, args.beam, kernels, args.max_skip)
            w = float(sw[ci]) if ci < len(sw) else 0.0
            if chains:
                any_finite = True
                candidate_records.append({
                    "candidate_index": ci,
                    "line_sequence": cand.get("line_sequence"),
                    "transfer_count": cand.get("transfer_count"),
                    "structural_weight": w,
                    "status": "FINITE_ON_TARGETED_REPLAY",
                })
                continue
            reason = (failure or {}).get("reason", "UNKNOWN")
            reason_w[reason] += w
            rec = {
                "candidate_index": ci,
                "line_sequence": cand.get("line_sequence"),
                "transfer_count": cand.get("transfer_count"),
                "structural_weight": w,
                "status": reason,
            }
            if reason == "MISSING_SERVICE_SEGMENT":
                seg = f"{failure.get('line')}|{failure.get('origin')}|{failure.get('destination')}"
                seg_w[seg] += w
                env = forward_missing_ready_envelope(cand, meta, rides_fn, tin, tout, args.beam, kernels, args.max_skip)
                rec["missing_segment"] = seg
                rec["ready_envelope"] = {k: dt_iso(v) for k, v in env.items()}
                if env.get("reason") == "MISSING_SERVICE_SEGMENT":
                    seg_ready[seg].append((w, env))
            candidate_records.append(rec)

        if any_finite:
            recovered_on_replay_mass += mass
        z = sum(reason_w.values())
        replay_probs = {k: v / z for k, v in reason_w.items()} if z > 0 else {}
        stored_probs = rr.get("failure_reason_probs") or {}
        keys = set(replay_probs).union(stored_probs)
        l1 = sum(abs(float(replay_probs.get(k, 0.0)) - float(stored_probs.get(k, 0.0))) for k in keys)
        reason_prob_l1_weighted += mass * l1
        for reason, p in replay_probs.items():
            reason_mass[reason] += mass * p

        missing_segments = []
        for seg, wsum in seg_w.items():
            weighted_mass = mass * wsum
            parts = seg.split("|")
            line = parts[0]
            line_pressure[line] += weighted_mass
            st = segment_stats.setdefault(seg, {
                "segment": seg,
                "line": line,
                "pressure_mass": 0.0,
                "cohort_count": 0,
                "od_pairs": set(),
                "entry_times": [],
                "exit_times": [],
                "ready_q05": [],
                "ready_q50_weighted_num": 0.0,
                "ready_q50_weight": 0.0,
                "ready_q95": [],
                "max_single_cohort_pressure": 0.0,
            })
            st["pressure_mass"] += weighted_mass
            st["cohort_count"] += 1
            st["od_pairs"].add(f"{oc}->{dc}")
            st["entry_times"].append(tin)
            st["exit_times"].append(tout)
            st["max_single_cohort_pressure"] = max(st["max_single_cohort_pressure"], weighted_mass)
            for cw, env in seg_ready.get(seg, []):
                ew = mass * cw
                if env.get("ready_q05_min"):
                    st["ready_q05"].append(env["ready_q05_min"])
                if env.get("ready_q95_max"):
                    st["ready_q95"].append(env["ready_q95_max"])
                if env.get("ready_q50_weighted_mean"):
                    st["ready_q50_weighted_num"] += ew * env["ready_q50_weighted_mean"].timestamp()
                    st["ready_q50_weight"] += ew
            missing_segments.append({"segment": seg, "structural_weight": wsum, "pressure_mass": weighted_mass})

        residual_output.append({
            "cohort_id": rr.get("cohort_id"),
            "origin_code": oc,
            "destination_code": dc,
            "entry_time": tin.isoformat(),
            "exit_time": tout.isoformat(),
            "passenger_mass": mass,
            "stored_failure_reason_probs": stored_probs,
            "replay_failure_reason_probs": replay_probs,
            "failure_reason_probability_l1": l1,
            "finite_candidate_found_on_replay": any_finite,
            "missing_segments": missing_segments,
            "candidate_records": candidate_records,
        })

    segment_rows = []
    for seg, st in segment_stats.items():
        q50 = None
        if st["ready_q50_weight"] > 0:
            q50 = datetime.fromtimestamp(st["ready_q50_weighted_num"] / st["ready_q50_weight"])
        segment_rows.append({
            "segment": seg,
            "line": st["line"],
            "pressure_mass": st["pressure_mass"],
            "cohort_count": st["cohort_count"],
            "distinct_od_count": len(st["od_pairs"]),
            "entry_time_min": min(st["entry_times"]).isoformat() if st["entry_times"] else None,
            "entry_time_max": max(st["entry_times"]).isoformat() if st["entry_times"] else None,
            "exit_time_min": min(st["exit_times"]).isoformat() if st["exit_times"] else None,
            "exit_time_max": max(st["exit_times"]).isoformat() if st["exit_times"] else None,
            "missing_gate_ready_q05_min": min(st["ready_q05"]).isoformat() if st["ready_q05"] else None,
            "missing_gate_ready_q50_weighted_mean": q50.isoformat() if q50 else None,
            "missing_gate_ready_q95_max": max(st["ready_q95"]).isoformat() if st["ready_q95"] else None,
            "max_single_cohort_pressure": st["max_single_cohort_pressure"],
            "cross_od_support": len(st["od_pairs"]),
        })
    segment_rows.sort(key=lambda x: x["pressure_mass"], reverse=True)

    residual_path = out / "r1b_residual_service_support_targeted_replay_cohorts.jsonl.gz"
    with gzip.open(residual_path, "wt", encoding="utf-8") as f:
        for row in residual_output:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "schema": "mppd.r1b-residual-service-support-targeted-replay.v1",
        "date": "2026-09-04",
        "status": "R1B_RESIDUAL_SERVICE_SUPPORT_TARGETED_REPLAY_COMPLETED",
        "source": {
            "r1b_schema": summary.get("schema"),
            "service_schema": service_payload.get("schema"),
            "residual_record_count": len(residuals),
            "residual_passenger_mass": total_mass,
            "service_root_offset_count": len(offsets),
        },
        "replay_consistency": {
            "finite_candidate_recovered_mass": recovered_on_replay_mass,
            "weighted_mean_failure_reason_probability_l1": reason_prob_l1_weighted / total_mass if total_mass else None,
            "failure_mass": dict(reason_mass),
            "failure_mass_sum": sum(reason_mass.values()),
            "residual_mass": total_mass,
            "failure_mass_conservation_error": sum(reason_mass.values()) - total_mass + recovered_on_replay_mass,
        },
        "missing_service_pressure_by_line": [
            {"line": k, "pressure_mass": v} for k, v in line_pressure.most_common()
        ],
        "top_missing_segments": segment_rows[:200],
        "proposal_readiness": {
            "existing_root_completion_or_extension": "MAY_ADVANCE_ONLY_AFTER_DIRECTION_AWARE_ROOT_CLASSIFICATION_AND_TIMING_ENVELOPE_COMPATIBILITY",
            "new_latent_service_root": "FORBIDDEN_FROM_THIS_REPLAY_ALONE_REQUIRES_MULTIPLE_COHORT_OR_OD_SUPPORT_PLUS_TRAJECTORY_CONTINUITY_AND_DIRECT_ANCHOR_NONCONTRADICTION",
            "ready_time_envelope_role": "PASSENGER_SIDE_LOWER_TIMING_EVIDENCE_NOT_OBSERVED_TRAIN_DEPARTURE_TRUTH",
        },
        "scientific_boundary": [
            "The targeted replay uses only no-finite cohorts and reconstructs the exact updated R1B E1 service offsets and kernels before candidate failure evaluation.",
            "Missing-segment structural weights are route-prior responsibilities, not passenger route truth.",
            "Ready-time envelopes integrate the current access/transfer propagation kernel and preceding service chain; they are passenger-side feasibility evidence, not observed departure times.",
            "Observed exit time is retained as a loose downstream temporal bound; no new service event is created in this audit.",
            "Any service-support activation must be rerun through the passenger posterior and preserve direct-anchor evidence classes.",
        ],
        "next_gate": "Join targeted segment timing/cross-OD support with direction-aware existing-root classification; qualify bounded existing-root completion/extension proposals first and leave true no-root support for a stricter latent-root proposal test.",
        "no_email_notification_logic": True,
    }
    (out / "r1b_residual_service_support_targeted_replay_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
