import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_g1_full_network_service_bootstrap_20260904 as g1

START = g0.START
END = g0.END


def robust_global_section_runtime(trains, meta):
    vals = []
    for (_, _), sts in trains.items():
        points = []
        for n, ev in sts.items():
            m = meta.get(n)
            if not m or m.get("seq") is None:
                continue
            points.append((int(m["seq"]), ev["departure"]))
        points.sort()
        for (s0, t0), (s1, t1) in zip(points, points[1:]):
            if s1 - s0 != 1:
                continue
            dt = abs((t1 - t0).total_seconds())
            if 30 <= dt <= 300:
                vals.append(dt)
    return statistics.median(vals) if vals else 120.0


def robust_global_headway(models):
    vals = []
    for line, dirs in models.items():
        for direction, md in dirs.items():
            if md.get("observed_train_count", 0) >= 5:
                h = float(md.get("median_observed_headway_sec") or 0)
                if 120 <= h <= 1200:
                    vals.append(h)
    return statistics.median(vals) if vals else 360.0


def candidate_headways(global_headway):
    raw = [
        0.75 * global_headway,
        global_headway,
        1.25 * global_headway,
        300, 360, 420, 480, 540, 600,
    ]
    vals = sorted({int(round(max(180, min(900, x)) / 30.0) * 30) for x in raw})
    return vals


def line_station_sequence(meta, line):
    by_seq = defaultdict(list)
    for n, m in meta.items():
        if m.get("line") != line or m.get("seq") is None:
            continue
        by_seq[int(m["seq"])].append(n)
    return dict(sorted(by_seq.items()))


def score_lattice(line, direction, seq_nodes, exit_hist, exit_mass, section_sec, headway, phase_sec, egress_lag_sec=120):
    if not seq_nodes:
        return None
    seqs = sorted(seq_nodes)
    refseq = seqs[len(seqs) // 2]
    sign = 1 if direction == "INC" else -1
    station_cache = {}
    for seq, nodes in seq_nodes.items():
        for n in nodes:
            if exit_mass[n] < 10:
                continue
            histvals = [exit_hist[n].get(START + timedelta(minutes=i), 0.0) for i in range(180)]
            station_cache[(seq, n)] = histvals
    if not station_cache:
        return None

    score = 0.0
    den = 0.0
    ref = START + timedelta(seconds=phase_sec)
    while ref < END:
        for (seq, n), histvals in station_cache.items():
            pred = ref + timedelta(seconds=sign * (seq - refseq) * section_sec + egress_lag_sec)
            if not (START <= pred < END):
                continue
            minute = g1.minute_floor(pred)
            z = g1.robust_z(exit_hist[n].get(minute, 0.0), histvals)
            w = math.sqrt(max(1.0, exit_mass[n]))
            score += w * max(0.0, z)
            den += w
        ref += timedelta(seconds=headway)
    return score / den if den else None


def choose_lattice(line, direction, seq_nodes, exit_hist, exit_mass, section_sec, global_headway):
    best = None
    for headway in candidate_headways(global_headway):
        for phase in range(0, headway, 30):
            score = score_lattice(
                line, direction, seq_nodes, exit_hist, exit_mass,
                section_sec, headway, phase
            )
            if score is None:
                continue
            cand = (score, -abs(headway - global_headway), -phase, headway, phase)
            if best is None or cand > best:
                best = cand
    if best is None:
        h = int(round(global_headway / 30.0) * 30)
        return {
            "headway_sec": h,
            "phase_sec": 0,
            "score": None,
            "evidence_class": "WEAK_STRUCTURAL_SERVICE_PRIOR",
        }
    return {
        "headway_sec": int(best[3]),
        "phase_sec": int(best[4]),
        "score": float(best[0]),
        "evidence_class": "AFC_INFERRED_SERVICE_FIELD_WEAK_LATTICE_INITIALIZATION",
    }


def generate_events(line, direction, seq_nodes, section_sec, lattice):
    seqs = sorted(seq_nodes)
    if not seqs:
        return []
    refseq = seqs[len(seqs) // 2]
    sign = 1 if direction == "INC" else -1
    headway = int(lattice["headway_sec"])
    phase = int(lattice["phase_sec"])
    out = []
    ref = START + timedelta(seconds=phase - headway)
    k = 0
    while ref < END + timedelta(seconds=headway):
        station_events = []
        for seq, nodes in seq_nodes.items():
            t = ref + timedelta(seconds=sign * (seq - refseq) * section_sec)
            for n in nodes:
                station_events.append({"node": n, "time": t.isoformat()})
        if station_events:
            out.append({
                "line": line,
                "direction": direction,
                "candidate_id": f"WEAK_{line}_{direction}_{k:04d}",
                "intercept_time": ref.isoformat(),
                "headway_sec": headway,
                "phase_sec": phase,
                "timing_uncertainty_sec": max(90.0, headway / 2.0),
                "evidence_class": lattice["evidence_class"],
                "lattice_score": lattice["score"],
                "station_events": station_events,
            })
        ref += timedelta(seconds=headway)
        k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taims", required=True)
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--service", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    trains = g1.load_service_events(args.service, meta)
    entry_hist, exit_hist, entry_mass, exit_mass, afc_stats = g1.load_afc_hist(args.taims, code_to_nodes, meta)
    models, by_line_train = g1.build_line_models(trains, meta, exit_hist, exit_mass)
    inferred, line_summary = g1.infer_candidates(models, exit_hist, exit_mass)

    section_sec = robust_global_section_runtime(trains, meta)
    global_headway = robust_global_headway(models)
    weak = []
    state_rows = []

    for row in line_summary:
        if row.get("status") == "BOOTSTRAPPED":
            continue
        line = row["line"]
        direction = row["direction"]
        seq_nodes = line_station_sequence(meta, line)
        lattice = choose_lattice(
            line, direction, seq_nodes, exit_hist, exit_mass,
            section_sec, global_headway
        )
        events = generate_events(line, direction, seq_nodes, section_sec, lattice)
        weak.extend(events)
        state_rows.append({
            "line": line,
            "direction": direction,
            "station_sequence_groups": len(seq_nodes),
            "headway_sec": lattice["headway_sec"],
            "phase_sec": lattice["phase_sec"],
            "score": lattice["score"],
            "evidence_class": lattice["evidence_class"],
            "candidate_trains": len(events),
        })

    result = {
        "schema": "mppd.g1b-full-network-weak-latent-service-init.v1",
        "date": "2026-09-04",
        "status": "G1B_WEAK_LATENT_SERVICE_INITIALIZATION_COMPLETED",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "time_window": "2026-08-29 07:00-10:00",
        "scope_assertions": {
            "full_network": True,
            "line_filter_applied": False,
            "segment_filter_applied": False,
            "unresolved_line_direction_removed": False,
            "weak_initialization_is_not_observed_truth": True,
        },
        "network": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "lines": len({m["line"] for m in meta.values()}),
            "transfer_groups": len(transfer_groups),
        },
        "global_initialization_statistics": {
            "median_adjacent_section_runtime_sec": section_sec,
            "median_observed_headway_sec": global_headway,
        },
        "unresolved_state_count_before": len(state_rows),
        "weak_candidate_count": len(weak),
        "afc_informed_weak_state_count": sum(1 for x in state_rows if x["evidence_class"].startswith("AFC_INFERRED")),
        "structural_only_weak_state_count": sum(1 for x in state_rows if x["evidence_class"] == "WEAK_STRUCTURAL_SERVICE_PRIOR"),
        "line_direction_initializations": state_rows,
        "performance": {
            "total_wall_sec": time.perf_counter() - wall0,
        },
        "scientific_boundary": [
            "Weak service lattices are initialization support for latent S only; they are not reconstructed realized timetables.",
            "AFC-informed lattices use network-wide anchored section/headway statistics plus the unresolved line's own exit-pulse alignment and remain AFC_INFERRED_SERVICE_FIELD_WEAK_LATTICE_INITIALIZATION.",
            "Structural-only lattices are explicitly WEAK_STRUCTURAL_SERVICE_PRIOR and carry lower evidence status.",
            "G3 must allow weak service events to move, disappear, split or gain/lose posterior support under the complete-network objective.",
            "No unresolved line-direction is removed from the full-network state space.",
        ],
        "next_gate": "Augment R0/G2 service trajectory support with these weak latent lattices and verify that high-order finite-chain coverage increases without any spatial or transfer-count filtering.",
        "no_email_notification_logic": True,
    }

    (outdir / "g1b_weak_latent_service_field.json").write_text(
        json.dumps({"candidates": weak}, ensure_ascii=False), encoding="utf-8"
    )
    (outdir / "g1b_full_network_weak_latent_service_init_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
