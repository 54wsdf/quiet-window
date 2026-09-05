from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

SCHEMA = "rail.hz-r1b4-realized-timetable-genealogy-audit.v1"
Z95 = 1.959963984540054


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def load_routes(path: Path) -> dict[tuple[str, int, str, int], list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status") != "QUALIFIED_LINE_AWARE_ROUTE_SUPPORT":
        raise SystemExit("route support is not qualified")
    out = {}
    for key, candidates in raw["route_support"].items():
        left, right = key.split("->")
        ol, os = left.split(":")
        dl, ds = right.split(":")
        out[(ol, int(os), dl, int(ds))] = candidates
    return out


def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_event_map(paths: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for p in paths:
        for e in iter_jsonl_gz(p):
            key = (str(e["root_id"]), int(e["station"]))
            if key in out:
                raise SystemExit(f"duplicate R1B3 root/station event: {key}")
            out[key] = e
    if len(out) != 43584:
        raise SystemExit(f"expected 43584 root-event rows, found {len(out)}")
    return out


def weighted_quantile(values: list[float], weights: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = np.asarray(values, dtype=float)
    ws = np.asarray(weights, dtype=float)
    order = np.argsort(xs)
    xs, ws = xs[order], ws[order]
    total = float(ws.sum())
    if total <= 0:
        return float(np.quantile(xs, p))
    c = np.cumsum(ws) / total
    return float(xs[min(len(xs)-1, int(np.searchsorted(c, p, side="left")))])


def event(evmap, rid: str, station: int) -> dict[str, Any]:
    e = evmap.get((rid, int(station)))
    if e is None:
        raise KeyError((rid, int(station)))
    return e


def gap_prob_positive(mean: float, sd: float) -> float:
    if sd <= 1e-12:
        return 1.0 if mean >= 0 else 0.0
    return norm_cdf(mean / sd)


def audit_shard(a: argparse.Namespace) -> dict[str, Any]:
    routes = load_routes(a.routes)
    evmap = load_event_map(a.r1b3_events)
    pf = pq.ParquetFile(a.genealogy)

    total_mass = resolved_mass = unresolved_mass = audited_mass = 0.0
    station_only_mass = 0.0
    mapping_failure_mass = 0.0
    center_infeasible_mass = 0.0
    credible95_impossible_mass = 0.0
    access_center_negative_mass = egress_center_negative_mass = transfer_center_negative_mass = 0.0
    access_credible95_impossible_mass = egress_credible95_impossible_mass = transfer_credible95_impossible_mass = 0.0
    weighted_min_factor_probability = 0.0
    weighted_product_factor_probability = 0.0
    transfer_factor_mass = 0.0
    edge_count = resolved_edge_count = 0
    failure_reasons: Counter[str] = Counter()
    conflict_examples: list[dict[str, Any]] = []

    access_gaps: list[float] = []
    access_w: list[float] = []
    egress_gaps: list[float] = []
    egress_w: list[float] = []
    transfer_gaps: list[float] = []
    transfer_w: list[float] = []

    for batch in pf.iter_batches(batch_size=50000):
        for row in batch.to_pylist():
            lm = float(row["lineage_mass"])
            total_mass += lm
            edge_count += 1
            if str(row["descendant_state_type"]) == "UNRESOLVED":
                unresolved_mass += lm
                continue
            resolved_mass += lm
            roots = [x for x in str(row["root_chain"]).split(">") if x]
            if not roots:
                station_only_mass += lm
                audited_mass += lm
                weighted_min_factor_probability += lm
                weighted_product_factor_probability += lm
                continue
            resolved_edge_count += 1
            key = (
                str(row["origin_line"]), int(row["origin_station"]),
                str(row["destination_line"]), int(row["destination_station"]),
            )
            candidates = routes.get(key)
            rank = int(row["route_rank"])
            if not candidates or rank < 1 or rank > len(candidates):
                mapping_failure_mass += lm
                failure_reasons["ROUTE_RANK_NOT_FOUND"] += 1
                continue
            route = candidates[rank - 1]
            legs = list(route.get("ride_legs", []))
            if len(legs) != len(roots):
                mapping_failure_mass += lm
                failure_reasons["ROOT_CHAIN_LENGTH_DIFFERS_FROM_RIDE_LEGS"] += 1
                continue
            try:
                first_dep = event(evmap, roots[0], int(legs[0]["from_station"]))
                last_arr = event(evmap, roots[-1], int(legs[-1]["to_station"]))
                transfer_pairs = []
                for j in range(len(roots) - 1):
                    lower = event(evmap, roots[j], int(legs[j]["to_station"]))
                    upper = event(evmap, roots[j+1], int(legs[j+1]["from_station"]))
                    transfer_pairs.append((lower, upper))
            except KeyError:
                mapping_failure_mass += lm
                failure_reasons["ROOT_STATION_EVENT_NOT_FOUND"] += 1
                continue

            audited_mass += lm
            factor_probs: list[float] = []
            center_bad = False
            credible_bad = False

            access_mean = float(first_dep["realized_time_mean_s"]) - float(row["entry_sec"])
            access_sd = float(first_dep["realized_time_sd_s"])
            access_prob = gap_prob_positive(access_mean, access_sd)
            factor_probs.append(access_prob)
            access_gaps.append(access_mean); access_w.append(lm)
            if access_mean < 0:
                access_center_negative_mass += lm; center_bad = True
            # Entire 95% service-time interval precedes AFC entry => no feasible access ordering inside reported 95% interval.
            if float(first_dep["realized_time_upper95_s"]) < float(row["entry_sec"]):
                access_credible95_impossible_mass += lm; credible_bad = True

            egress_mean = float(row["exit_sec"]) - float(last_arr["realized_time_mean_s"])
            egress_sd = float(last_arr["realized_time_sd_s"])
            egress_prob = gap_prob_positive(egress_mean, egress_sd)
            factor_probs.append(egress_prob)
            egress_gaps.append(egress_mean); egress_w.append(lm)
            if egress_mean < 0:
                egress_center_negative_mass += lm; center_bad = True
            if float(last_arr["realized_time_lower95_s"]) > float(row["exit_sec"]):
                egress_credible95_impossible_mass += lm; credible_bad = True

            min_transfer_mean = None
            for lower, upper in transfer_pairs:
                gap_mean = float(upper["realized_time_mean_s"]) - float(lower["realized_time_mean_s"])
                gap_sd = math.sqrt(float(upper["realized_time_sd_s"])**2 + float(lower["realized_time_sd_s"])**2)
                prob = gap_prob_positive(gap_mean, gap_sd)
                factor_probs.append(prob)
                transfer_factor_mass += lm
                transfer_gaps.append(gap_mean); transfer_w.append(lm)
                min_transfer_mean = gap_mean if min_transfer_mean is None else min(min_transfer_mean, gap_mean)
                if gap_mean < 0:
                    transfer_center_negative_mass += lm; center_bad = True
                # Even optimistic upper departure minus optimistic early arrival is negative within marginal 95% intervals.
                if float(upper["realized_time_upper95_s"]) < float(lower["realized_time_lower95_s"]):
                    transfer_credible95_impossible_mass += lm; credible_bad = True

            if center_bad:
                center_infeasible_mass += lm
            if credible_bad:
                credible95_impossible_mass += lm
            pmin = min(factor_probs) if factor_probs else 1.0
            pprod = math.prod(factor_probs) if factor_probs else 1.0
            weighted_min_factor_probability += lm * pmin
            weighted_product_factor_probability += lm * pprod
            if (center_bad or credible_bad) and len(conflict_examples) < 100:
                conflict_examples.append({
                    "cohort_id": str(row["cohort_id"]),
                    "lineage_mass": lm,
                    "root_chain": str(row["root_chain"]),
                    "transfer_chain": str(row["transfer_chain"]),
                    "entry_sec": float(row["entry_sec"]),
                    "exit_sec": float(row["exit_sec"]),
                    "access_gap_mean_s": access_mean,
                    "egress_gap_mean_s": egress_mean,
                    "min_transfer_gap_mean_s": min_transfer_mean,
                    "center_infeasible": center_bad,
                    "credible95_impossible": credible_bad,
                })

    resolved_plus_unresolved_error = abs((resolved_mass + unresolved_mass) - total_mass)
    gates = {
        "edge_mass_partition_conserved": resolved_plus_unresolved_error <= 1e-6,
        "all_resolved_service_chain_mass_mapped_to_r1b3_events": mapping_failure_mass <= 1e-9,
        "audit_does_not_modify_service_or_genealogy": True,
        "no_planned_timetable_absolute_time_used_in_audit": True,
    }
    result = {
        "schema": SCHEMA,
        "status": "COMPLETED_R1B4_AUDIT_SHARD" if all(gates.values()) else "R1B4_AUDIT_SHARD_MAPPING_FAILURE",
        "shard_index": a.shard_index,
        "total_lineage_mass": total_mass,
        "resolved_lineage_mass": resolved_mass,
        "unresolved_lineage_mass": unresolved_mass,
        "audited_resolved_mass": audited_mass,
        "station_only_mass": station_only_mass,
        "mapping_failure_mass": mapping_failure_mass,
        "edge_count": edge_count,
        "resolved_service_chain_edge_count": resolved_edge_count,
        "center_infeasible_lineage_mass": center_infeasible_mass,
        "credible95_impossible_lineage_mass": credible95_impossible_mass,
        "access_center_negative_mass": access_center_negative_mass,
        "egress_center_negative_mass": egress_center_negative_mass,
        "transfer_center_negative_factor_mass": transfer_center_negative_mass,
        "access_credible95_impossible_mass": access_credible95_impossible_mass,
        "egress_credible95_impossible_mass": egress_credible95_impossible_mass,
        "transfer_credible95_impossible_factor_mass": transfer_credible95_impossible_mass,
        "transfer_factor_lineage_mass": transfer_factor_mass,
        "lineage_weighted_mean_min_ordering_probability": weighted_min_factor_probability / audited_mass if audited_mass else None,
        "lineage_weighted_mean_product_ordering_probability_independence_approx": weighted_product_factor_probability / audited_mass if audited_mass else None,
        "gap_mean_s_quantiles": {
            "access": {"p10": weighted_quantile(access_gaps, access_w, .1), "median": weighted_quantile(access_gaps, access_w, .5), "p90": weighted_quantile(access_gaps, access_w, .9)},
            "egress": {"p10": weighted_quantile(egress_gaps, egress_w, .1), "median": weighted_quantile(egress_gaps, egress_w, .5), "p90": weighted_quantile(egress_gaps, egress_w, .9)},
            "transfer_connection": {"p10": weighted_quantile(transfer_gaps, transfer_w, .1), "median": weighted_quantile(transfer_gaps, transfer_w, .5), "p90": weighted_quantile(transfer_gaps, transfer_w, .9)},
        },
        "mapping_failure_reasons": dict(failure_reasons),
        "qualification_gates": gates,
        "semantic_boundary": {
            "audit_only": True,
            "service_posterior_updated": False,
            "genealogy_posterior_updated": False,
            "route_or_boarding_reassigned": False,
            "credible95_impossible_definition": "reported marginal 95% service-event intervals contain no ordering-feasible combination for at least one endpoint/transfer factor",
            "center_infeasible_is_diagnostic_not_an_update_trigger": True
        }
    }
    a.out_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    with gzip.open(a.out_examples, "wt", encoding="utf-8") as f:
        for x in conflict_examples:
            f.write(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def merge(a: argparse.Namespace) -> dict[str, Any]:
    ss = [json.loads(p.read_text(encoding="utf-8")) for p in a.summary]
    if len(ss) != 8 or not all(x["status"] == "COMPLETED_R1B4_AUDIT_SHARD" for x in ss):
        raise SystemExit("expected eight completed audit shards")
    sums = lambda k: sum(float(x[k]) for x in ss)
    resolved = sums("resolved_lineage_mass")
    unresolved = sums("unresolved_lineage_mass")
    total = sums("total_lineage_mass")
    audited = sums("audited_resolved_mass")
    mapping_fail = sums("mapping_failure_mass")
    center_bad = sums("center_infeasible_lineage_mass")
    credible_bad = sums("credible95_impossible_lineage_mass")
    minprob_num = sum(float(x["lineage_weighted_mean_min_ordering_probability"] or 0) * float(x["audited_resolved_mass"]) for x in ss)
    prodprob_num = sum(float(x["lineage_weighted_mean_product_ordering_probability_independence_approx"] or 0) * float(x["audited_resolved_mass"]) for x in ss)
    gates = {
        "eight_audit_shards_complete": True,
        "full_r1b1_lineage_mass_accounted": abs(total - 1206156.0) <= 1e-5,
        "resolved_r1b1_lineage_mass_accounted": abs(resolved - 932464.0) <= 1e-5,
        "unresolved_r1b1_lineage_mass_preserved": abs(unresolved - 273692.0) <= 1e-5,
        "zero_root_station_mapping_failure_mass": mapping_fail <= 1e-9,
        "r1b3_timetable_not_modified_by_audit": True,
        "r1b1_genealogy_not_modified_by_audit": True,
        "no_service_passenger_service_scientific_loop": True,
    }
    result = {
        "schema": SCHEMA,
        "status": "QUALIFIED_R1B4_AUDIT_COMPLETE" if all(gates.values()) else "R1B4_AUDIT_INTEGRITY_FAILURE",
        "service_date": "2019-01-04",
        "scope": "FULL_SERVICE_DAY_0400_TO_NEXT_0400_FULL_NETWORK_FULL_QUALIFIED_PASSENGER_DOMAIN",
        "total_lineage_mass": total,
        "resolved_lineage_mass": resolved,
        "unresolved_lineage_mass": unresolved,
        "audited_resolved_mass": audited,
        "mapping_failure_mass": mapping_fail,
        "center_infeasible_lineage_mass": center_bad,
        "center_infeasible_share_of_resolved": center_bad / resolved if resolved else None,
        "credible95_impossible_lineage_mass": credible_bad,
        "credible95_impossible_share_of_resolved": credible_bad / resolved if resolved else None,
        "access_center_negative_mass": sums("access_center_negative_mass"),
        "egress_center_negative_mass": sums("egress_center_negative_mass"),
        "transfer_center_negative_factor_mass": sums("transfer_center_negative_factor_mass"),
        "access_credible95_impossible_mass": sums("access_credible95_impossible_mass"),
        "egress_credible95_impossible_mass": sums("egress_credible95_impossible_mass"),
        "transfer_credible95_impossible_factor_mass": sums("transfer_credible95_impossible_factor_mass"),
        "transfer_factor_lineage_mass": sums("transfer_factor_lineage_mass"),
        "lineage_weighted_mean_min_ordering_probability": minprob_num / audited if audited else None,
        "lineage_weighted_mean_product_ordering_probability_independence_approx": prodprob_num / audited if audited else None,
        "qualification_gates": gates,
        "scientific_interpretation_boundary": {
            "audit_is_read_only": True,
            "center_conflicts_are_reported_not_self_iterated_away": True,
            "credible95_conflicts_are_reported_not_self_iterated_away": True,
            "r1c_needed_for_physical_transfer_time_distribution": True,
            "r1d_needed_for_heldout_validation": True
        },
        "next_stage": "R1C_TRANSFER_TIME_INTERVAL_INVERSION_IF_R1B_PRODUCT_ACCEPTED_WITH_EXPLICIT_AUDIT_RESIDUALS"
    }
    a.out_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("audit-shard")
    s.add_argument("--genealogy", type=Path, required=True)
    s.add_argument("--routes", type=Path, required=True)
    s.add_argument("--r1b3-events", type=Path, action="append", required=True)
    s.add_argument("--shard-index", type=int, required=True)
    s.add_argument("--out-summary", type=Path, required=True)
    s.add_argument("--out-examples", type=Path, required=True)
    m = sp.add_parser("merge")
    m.add_argument("--summary", type=Path, action="append", required=True)
    m.add_argument("--out-summary", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "audit-shard": audit_shard(a)
    else: merge(a)


if __name__ == "__main__":
    main()
