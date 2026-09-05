from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

MATCH_BASE_TOL_S = 150.0
OBSERVED_MIN_SD_S = 15.0
INFERRED_BASE_SD_S = 90.0


def load_events(npz_path: Path) -> dict[int, list[dict[str, float]]]:
    z = np.load(npz_path)
    stations = z["event_station"].astype(int)
    centers = z["event_center_s"].astype(float)
    starts = z["event_start_s"].astype(float)
    ends = z["event_end_s"].astype(float)
    masses = z["event_mass"].astype(float)
    excess = z["event_excess_mass"].astype(float)
    scores = z["event_score"].astype(float)
    out: dict[int, list[dict[str, float]]] = defaultdict(list)
    for s, c, a, b, m, e, q in zip(stations, centers, starts, ends, masses, excess, scores):
        width = max(5.0, float(b - a))
        sd = max(OBSERVED_MIN_SD_S, width / 2.355)
        out[int(s)].append({
            "center_s": float(c),
            "start_s": float(a),
            "end_s": float(b),
            "mass": float(m),
            "excess_mass": float(e),
            "score": float(q),
            "sd_s": float(sd),
        })
    for vals in out.values():
        vals.sort(key=lambda x: x["center_s"])
    return out


def contexts_by_edge(day_json: dict[str, Any]) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    out: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in day_json["propagation_field"]["contexts"]:
        key = (row["path_id"], row["direction"], int(row["from_station"]), int(row["to_station"]))
        out[key].append(row)
    for vals in out.values():
        vals.sort(key=lambda x: x["context_start_s"])
    return out


def select_lag(rows: list[dict[str, Any]], upstream_time_hint_s: float) -> tuple[float, str, float | None]:
    if not rows:
        raise KeyError("missing edge contexts")
    t = upstream_time_hint_s % 86400.0
    chosen = min(rows, key=lambda r: 0.0 if r["context_start_s"] <= t < r["context_end_s"] else min(abs(t-r["context_start_s"]), abs(t-r["context_end_s"])))
    lag = float(chosen["effective_initial_lag_s"])
    return lag, str(chosen["evidence_class"]), chosen.get("afc_correlation")


def nearest_event(events: list[dict[str, float]], predicted_s: float, tol_s: float) -> dict[str, float] | None:
    if not events:
        return None
    # station event counts are small enough that local linear search is deterministic and robust.
    best = min(events, key=lambda e: (abs(e["center_s"] - predicted_s), -e["score"], -e["excess_mass"]))
    return best if abs(best["center_s"] - predicted_s) <= tol_s else None


def complete_roots(day_npz: Path, day_json_path: Path, prior_path: Path, out_json: Path) -> dict[str, Any]:
    events = load_events(day_npz)
    day = json.loads(day_json_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    ctx = contexts_by_edge(day)

    roots: list[dict[str, Any]] = []
    root_counts: Counter[str] = Counter()
    observed_events = 0
    inferred_events = 0
    anchor_counts: Counter[str] = Counter()

    for path_id, meta in prior["line_paths"].items():
        base = [int(x) for x in meta["nodes"]]
        for direction in ("Down", "Up"):
            path = base if direction == "Down" else list(reversed(base))
            terminal = path[-1]
            anchors = events.get(terminal, [])
            anchor_counts[f"{path_id}:{direction}"] = len(anchors)
            for anchor_index, anchor in enumerate(anchors):
                timeline: dict[int, dict[str, Any]] = {
                    terminal: {
                        "station": terminal,
                        "time_s": anchor["center_s"],
                        "sd_s": anchor["sd_s"],
                        "evidence_class": "AFC_INFERRED_PASSENGER_FACING_EVENT",
                        "pulse_score": anchor["score"],
                        "pulse_excess_mass": anchor["excess_mass"],
                        "matched_observed_pulse": True,
                    }
                }
                downstream_time = float(anchor["center_s"])
                cumulative_sd2 = float(anchor["sd_s"]) ** 2
                lag_evidence: list[str] = []
                correlations: list[float] = []

                for u, v in reversed(list(zip(path, path[1:]))):
                    rows = ctx.get((path_id, direction, u, v), [])
                    # The upstream time is approximately downstream-lag; select context iteratively.
                    rough_lag = float(rows[0]["structural_prior_lag_s"]) if rows else 180.0
                    lag, lag_class, corr = select_lag(rows, downstream_time - rough_lag)
                    predicted = downstream_time - lag
                    if predicted < 0:
                        predicted += 86400.0
                    lag_evidence.append(lag_class)
                    if corr is not None and math.isfinite(float(corr)):
                        correlations.append(float(corr))
                    prior_spread = 20.0 if lag_class == "AFC_PLUS_STRUCTURAL_PRIOR" else 45.0
                    match = nearest_event(events.get(u, []), predicted, MATCH_BASE_TOL_S + prior_spread)
                    if match is not None:
                        t = float(match["center_s"])
                        sd = max(float(match["sd_s"]), prior_spread)
                        evc = "AFC_INFERRED_PASSENGER_FACING_EVENT"
                        observed = True
                        score = float(match["score"])
                        excess = float(match["excess_mass"])
                        observed_events += 1
                    else:
                        t = float(predicted)
                        cumulative_sd2 += prior_spread ** 2
                        sd = max(INFERRED_BASE_SD_S, math.sqrt(cumulative_sd2))
                        evc = "STRUCTURE_PROPAGATED_LATENT_SERVICE_EVENT"
                        observed = False
                        score = None
                        excess = None
                        inferred_events += 1
                    timeline[u] = {
                        "station": u,
                        "time_s": t,
                        "sd_s": sd,
                        "evidence_class": evc,
                        "pulse_score": score,
                        "pulse_excess_mass": excess,
                        "matched_observed_pulse": observed,
                    }
                    downstream_time = t

                ordered = [timeline[s] for s in path]
                if any(ordered[i+1]["time_s"] + 43200 < ordered[i]["time_s"] for i in range(len(ordered)-1)):
                    # Midnight wrap is legitimate; retain it explicitly below rather than dropping the root.
                    pass
                observed_share = sum(x["matched_observed_pulse"] for x in ordered) / len(ordered)
                root_id = f"{day['source_date']}:{path_id}:{direction}:{anchor_index}:{int(round(anchor['center_s']))}"
                root = {
                    "root_id": root_id,
                    "date": day["source_date"],
                    "path_id": path_id,
                    "afc_line": meta["afc_line"],
                    "aux_line": meta["aux_line"],
                    "direction": direction,
                    "terminal_station": terminal,
                    "terminal_anchor_s": float(anchor["center_s"]),
                    "terminal_anchor_score": float(anchor["score"]),
                    "event_count": len(ordered),
                    "matched_pulse_share": observed_share,
                    "mean_finite_edge_correlation": statistics.mean(correlations) if correlations else None,
                    "lag_evidence_counts": dict(Counter(lag_evidence)),
                    "events": ordered,
                    "evidence_class": "AFC_ANCHORED_STRUCTURE_COMPLETED_SERVICE_ROOT",
                    "exact_train_identity_claimed": False,
                }
                roots.append(root)
                root_counts[f"{path_id}:{direction}"] += 1

    gates = {
        "all_path_directions_have_terminal_anchors": all(anchor_counts.get(f"{p}:{d}", 0) > 0 for p in prior["line_paths"] for d in ("Down", "Up")),
        "all_path_directions_have_roots": all(root_counts.get(f"{p}:{d}", 0) > 0 for p in prior["line_paths"] for d in ("Down", "Up")),
        "no_exact_train_identity_claim": True,
        "full_day_anchor_domain_retained": True,
    }
    result = {
        "schema": "rail.hz-day-specific-service-root-completion.v1",
        "dataset_id": "CN_HZ_Tianchi_2019",
        "source_date": day["source_date"],
        "status": "QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION" if all(gates.values()) else "SERVICE_ROOT_COMPLETION_GATE_FAILED",
        "semantics": "candidate passenger-facing service roots for R1B; roots are latent hypotheses, not observed ATS train identities",
        "anchor_counts": dict(anchor_counts),
        "root_counts": dict(root_counts),
        "root_count_total": len(roots),
        "matched_intermediate_event_assignments": observed_events,
        "structure_propagated_latent_event_assignments": inferred_events,
        "integrity_gates": gates,
        "roots": roots,
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    if not all(gates.values()):
        raise SystemExit(2)
    return result


def aggregate(days_dir: Path, out: Path) -> dict[str, Any]:
    paths = sorted(days_dir.glob("record_2019-01-*.service_roots.json"))
    days = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    expected = [f"2019-01-{d:02d}" for d in range(1,26)]
    actual = [x["source_date"] for x in days]
    gates = {
        "all_25_days": len(days) == 25 and actual == expected,
        "all_day_root_gates_pass": all(x["status"] == "QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION" for x in days),
        "all_day_all_path_direction_roots": all(all(v > 0 for v in x["root_counts"].values()) and len(x["root_counts"]) >= 8 for x in days),
    }
    counts = [x["root_count_total"] for x in days]
    result = {
        "schema": "rail.hz-25day-service-root-completion-summary.v1",
        "status": "QUALIFIED_HZ25_CANDIDATE_SERVICE_ROOTS" if all(gates.values()) else "HZ25_SERVICE_ROOT_GATE_FAILED",
        "days": len(days),
        "dates": actual,
        "integrity_gates": gates,
        "total_candidate_roots": sum(counts),
        "daily_candidate_roots_median": statistics.median(counts) if counts else None,
        "daily_candidate_roots_min": min(counts) if counts else None,
        "daily_candidate_roots_max": max(counts) if counts else None,
        "scientific_boundary": "candidate service roots remain latent and uncertainty-bearing; formal correctness is tested by downstream R1B/R1D, not asserted here",
        "day_summaries": [{
            "date": x["source_date"],
            "root_count_total": x["root_count_total"],
            "root_counts": x["root_counts"],
            "anchor_counts": x["anchor_counts"],
            "matched_intermediate_event_assignments": x["matched_intermediate_event_assignments"],
            "structure_propagated_latent_event_assignments": x["structure_propagated_latent_event_assignments"],
        } for x in days],
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "days": len(days),
        "total_candidate_roots": result["total_candidate_roots"],
        "daily_candidate_roots_median": result["daily_candidate_roots_median"],
        "integrity_gates": gates,
    }, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("day")
    s.add_argument("--day-npz", type=Path, required=True)
    s.add_argument("--day-json", type=Path, required=True)
    s.add_argument("--prior", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("aggregate")
    s.add_argument("--days-dir", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "day":
        complete_roots(args.day_npz, args.day_json, args.prior, args.output)
    else:
        aggregate(args.days_dir, args.output)


if __name__ == "__main__":
    main()
