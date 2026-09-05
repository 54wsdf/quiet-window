from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import scripts.rail_hz_r1b_full_service_day_joint_update_20260905 as base

SCHEMA = "rail.hz-r1b-distributed-exact.v1"


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _kernel_from_payload(x: dict[str, Any]) -> base.Kernel:
    return base.Kernel(float(x["median_sec"]), float(x["sigma"]), str(x["evidence_class"]))


def _kernels_from_payload(x: dict[str, Any]) -> dict[str, base.Kernel]:
    return {k: _kernel_from_payload(v) for k, v in x.items()}


def split_cohorts(source: Path, out_dir: Path, shards: int, manifest: Path) -> dict[str, Any]:
    if shards < 2:
        raise SystemExit("shards must be >=2")
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(source)
    schema = pf.schema_arrow
    writers = [pq.ParquetWriter(out_dir / f"part-{i:02d}.parquet", schema=schema, compression="zstd") for i in range(shards)]
    rows = [0] * shards
    mass = [0.0] * shards
    try:
        for batch_index, batch in enumerate(pf.iter_batches(batch_size=25000)):
            shard = batch_index % shards
            table = pa.Table.from_batches([batch], schema=schema)
            writers[shard].write_table(table)
            rows[shard] += table.num_rows
            if "passenger_mass" in table.column_names:
                mass[shard] += float(sum(table["passenger_mass"].to_pylist()))
    finally:
        for w in writers:
            w.close()
    total_rows = int(sum(rows))
    total_mass = float(sum(mass))
    result = {
        "schema": SCHEMA,
        "command": "split",
        "source": source.name,
        "shards": shards,
        "source_rows": int(pf.metadata.num_rows),
        "split_rows": total_rows,
        "split_passenger_mass": total_mass,
        "row_conservation_pass": total_rows == int(pf.metadata.num_rows),
        "parts": [
            {"index": i, "file": f"part-{i:02d}.parquet", "rows": rows[i], "passenger_mass": mass[i]}
            for i in range(shards)
        ],
        "partition_semantics": "DISJOINT_COMPLETE_PARTITION_NO_PASSENGER_SUBSAMPLE",
        "numeric_equivalence": "SUFFICIENT_STATISTICS_AND_POSTERIOR_METRICS_MERGED_GLOBALLY; FLOATING_SUM_ORDER_MAY_DIFFER_AT_MACHINE_PRECISION",
    }
    if not result["row_conservation_pass"]:
        raise SystemExit("split row conservation failed")
    _dump(manifest, result)
    return result


def e0_shard(cohorts: Path, routes_path: Path, roots_path: Path, sidecar: Path, output: Path, beam: int) -> dict[str, Any]:
    routes = base.load_routes(routes_path)
    roots, _root_meta = base.load_roots(roots_path)
    cache = base.LegCache(roots)
    kernels0 = base.initial_kernels()
    metrics, factor_stats, root_usage = base.pass_e0(cohorts, routes, cache, kernels0, beam, sidecar)
    result = {
        "schema": SCHEMA,
        "command": "e0-shard",
        "cohorts": cohorts.name,
        "sidecar": sidecar.name,
        "beam": beam,
        "metrics": metrics,
        "factor_stats": factor_stats,
        "root_usage": dict(root_usage),
        "kernels0": {k: base.kernel_payload(v) for k, v in kernels0.items()},
    }
    _dump(output, result)
    return result


def _sum_factor_stats(items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(lambda: {"w": 0.0, "sum_log": 0.0, "sum_log2": 0.0})
    for item in items:
        for kind, row in item.get("factor_stats", {}).items():
            for key in ("w", "sum_log", "sum_log2"):
                out[kind][key] += float(row.get(key, 0.0))
    return dict(out)


def _merge_e0_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    ms = [x["metrics"] for x in items]
    total = sum(float(x["passenger_mass"]) for x in ms)
    finite = sum(float(x["finite_posterior_mass"]) for x in ms)
    def wmean(field: str) -> float | None:
        if finite <= 0:
            return None
        return sum(float(x[field] or 0.0) * float(x["finite_posterior_mass"]) for x in ms) / finite
    out = {
        "cohort_count": sum(int(x["cohort_count"]) for x in ms),
        "finite_cohort_count": sum(int(x["finite_cohort_count"]) for x in ms),
        "passenger_mass": total,
        "finite_posterior_mass": finite,
        "finite_posterior_share": finite / total if total else None,
        "no_route_support_mass": sum(float(x["no_route_support_mass"]) for x in ms),
        "time_or_service_incompatible_mass": sum(float(x["time_or_service_incompatible_mass"]) for x in ms),
        "station_only_supported_mass": sum(float(x["station_only_supported_mass"]) for x in ms),
        "weighted_mean_route_entropy": wmean("weighted_mean_route_entropy"),
        "weighted_mean_top_route_probability": wmean("weighted_mean_top_route_probability"),
        "weighted_mean_min_local_beam_retained_fraction": wmean("weighted_mean_min_local_beam_retained_fraction"),
    }
    out["mass_conservation_pass"] = abs((out["finite_posterior_mass"] + out["no_route_support_mass"] + out["time_or_service_incompatible_mass"]) - total) <= 1e-6
    return out


def merge_e0(inputs: list[Path], output: Path) -> dict[str, Any]:
    items = [json.loads(p.read_text(encoding="utf-8")) for p in inputs]
    if not items:
        raise SystemExit("no E0 shard inputs")
    factor_stats = _sum_factor_stats(items)
    kernels0 = base.initial_kernels()
    kernels1, kernel_diag = base.kernels_after_e0(kernels0, factor_stats)
    metrics = _merge_e0_metrics(items)
    result = {
        "schema": SCHEMA,
        "command": "merge-e0",
        "shard_count": len(items),
        "metrics": metrics,
        "factor_stats": factor_stats,
        "kernels0": {k: base.kernel_payload(v) for k, v in kernels0.items()},
        "kernel_update": kernel_diag,
        "kernels1": {k: base.kernel_payload(v) for k, v in kernels1.items()},
        "mass_conservation_pass": bool(metrics["mass_conservation_pass"]),
    }
    if not result["mass_conservation_pass"]:
        raise SystemExit("merged E0 mass conservation failed")
    _dump(output, result)
    return result


def shift_shard(cohorts: Path, routes_path: Path, roots_path: Path, e0_global: Path, output: Path, beam: int) -> dict[str, Any]:
    e0 = json.loads(e0_global.read_text(encoding="utf-8"))
    kernels0 = _kernels_from_payload(e0["kernels0"])
    kernels1 = _kernels_from_payload(e0["kernels1"])
    routes = base.load_routes(routes_path)
    roots, _ = base.load_roots(roots_path)
    cache = base.LegCache(roots)
    score_sums, weights = base.accumulate_shift_scores(cohorts, routes, cache, kernels0, kernels1, beam)
    result = {
        "schema": SCHEMA,
        "command": "shift-shard",
        "cohorts": cohorts.name,
        "score_sums": {k: [float(v) for v in vals] for k, vals in score_sums.items()},
        "weights": {k: float(v) for k, v in weights.items()},
    }
    _dump(output, result)
    return result


def merge_shift(inputs: list[Path], roots_path: Path, output: Path) -> dict[str, Any]:
    score_sums: dict[str, list[float]] = {}
    weights: dict[str, float] = defaultdict(float)
    for p in inputs:
        x = json.loads(p.read_text(encoding="utf-8"))
        for rid, vals in x.get("score_sums", {}).items():
            if rid not in score_sums:
                score_sums[rid] = [0.0] * len(vals)
            if len(score_sums[rid]) != len(vals):
                raise SystemExit("shift-grid length mismatch")
            for i, v in enumerate(vals):
                score_sums[rid][i] += float(v)
        for rid, v in x.get("weights", {}).items():
            weights[rid] += float(v)
    _roots, root_meta = base.load_roots(roots_path)
    offsets, diag = base.choose_offsets(score_sums, weights, root_meta)
    result = {
        "schema": SCHEMA,
        "command": "merge-shift",
        "shard_count": len(inputs),
        "offsets": offsets,
        "service_timing_update": diag,
    }
    _dump(output, result)
    return result


def e1_shard(cohorts: Path, sidecar: Path, routes_path: Path, roots_path: Path, e0_global: Path, shift_global: Path, output: Path, beam: int) -> dict[str, Any]:
    e0 = json.loads(e0_global.read_text(encoding="utf-8"))
    shifts = json.loads(shift_global.read_text(encoding="utf-8"))
    kernels1 = _kernels_from_payload(e0["kernels1"])
    routes = base.load_routes(routes_path)
    roots1, _ = base.load_roots(roots_path, offsets={k: int(v) for k, v in shifts["offsets"].items()})
    cache1 = base.LegCache(roots1)
    metrics = base.pass_e1_compare(cohorts, sidecar, routes, cache1, kernels1, beam)
    result = {"schema": SCHEMA, "command": "e1-shard", "cohorts": cohorts.name, "metrics": metrics}
    _dump(output, result)
    return result


def _merge_e1_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    ms = [x["metrics"] for x in items]
    total = sum(float(x["passenger_mass"]) for x in ms)
    finite = sum(float(x["finite_posterior_mass"]) for x in ms)
    both = sum(float(x["finite_both_mass"]) for x in ms)
    def wmean(field: str, denom_field: str, denom: float) -> float | None:
        if denom <= 0:
            return None
        return sum(float(x[field] or 0.0) * float(x[denom_field]) for x in ms) / denom
    out = {
        "cohort_count": sum(int(x["cohort_count"]) for x in ms),
        "passenger_mass": total,
        "finite_posterior_mass": finite,
        "finite_posterior_share": finite / total if total else None,
        "no_route_support_mass": sum(float(x["no_route_support_mass"]) for x in ms),
        "time_or_service_incompatible_mass": sum(float(x["time_or_service_incompatible_mass"]) for x in ms),
        "became_finite_mass": sum(float(x["became_finite_mass"]) for x in ms),
        "lost_finite_mass": sum(float(x["lost_finite_mass"]) for x in ms),
        "finite_both_mass": both,
        "top_route_changed_mass": sum(float(x["top_route_changed_mass"]) for x in ms),
        "top_boarding_chain_changed_mass": sum(float(x["top_boarding_chain_changed_mass"]) for x in ms),
        "weighted_mean_route_entropy": wmean("weighted_mean_route_entropy", "finite_posterior_mass", finite),
        "weighted_mean_route_entropy_change_among_finite_both": wmean("weighted_mean_route_entropy_change_among_finite_both", "finite_both_mass", both),
        "weighted_mean_top_route_probability": wmean("weighted_mean_top_route_probability", "finite_posterior_mass", finite),
        "weighted_mean_top_route_probability_change_among_finite_both": wmean("weighted_mean_top_route_probability_change_among_finite_both", "finite_both_mass", both),
    }
    out["top_route_changed_share_among_finite_both"] = out["top_route_changed_mass"] / both if both else None
    out["top_boarding_chain_changed_share_among_finite_both"] = out["top_boarding_chain_changed_mass"] / both if both else None
    out["mass_conservation_pass"] = abs((finite + out["no_route_support_mass"] + out["time_or_service_incompatible_mass"]) - total) <= 1e-6
    return out


def merge_e1(inputs: list[Path], output: Path) -> dict[str, Any]:
    items = [json.loads(p.read_text(encoding="utf-8")) for p in inputs]
    if not items:
        raise SystemExit("no E1 shard inputs")
    metrics = _merge_e1_metrics(items)
    result = {"schema": SCHEMA, "command": "merge-e1", "shard_count": len(items), "metrics": metrics, "mass_conservation_pass": bool(metrics["mass_conservation_pass"])}
    if not result["mass_conservation_pass"]:
        raise SystemExit("merged E1 mass conservation failed")
    _dump(output, result)
    return result


def finalize(e0_global: Path, shift_global: Path, e1_global: Path, roots_path: Path, output: Path) -> dict[str, Any]:
    e0 = json.loads(e0_global.read_text(encoding="utf-8"))
    shift = json.loads(shift_global.read_text(encoding="utf-8"))
    e1 = json.loads(e1_global.read_text(encoding="utf-8"))
    roots = json.loads(roots_path.read_text(encoding="utf-8"))
    m0, m1 = e0["metrics"], e1["metrics"]
    service_diag = shift["service_timing_update"]
    result = {
        "schema": "rail.hz-r1b-full-service-day-joint-update.distributed-exact.v1",
        "dataset_id": "CN_HZ_Tianchi_2019",
        "service_date": roots["source_date"],
        "status": "COMPLETED_FORMAL_R1B_FULL_SERVICE_DAY_UPDATE",
        "scope": {
            "time": "FULL_SERVICE_DAY_0400_TO_NEXT_0400",
            "network": "FULL_NETWORK",
            "passenger_subsample": False,
            "transfer_count_cap": None,
            "route_support_beam_k": 32,
            "boarding_dynamic_posterior_beam": base.BEAM,
            "boarding_skip_count_cap": None,
            "distributed_partitioning": True,
            "partition_semantics": "DISJOINT_COMPLETE_PARTITION_NO_SCIENTIFIC_SCOPE_REDUCTION",
        },
        "kernels_before": e0["kernels0"],
        "kernel_update": e0["kernel_update"],
        "kernels_after": e0["kernels1"],
        "E0": m0,
        "service_timing_update": service_diag,
        "E1": m1,
        "bidirectional_evidence": {
            "passenger_posterior_updates_temporal_kernels": any(e0["kernel_update"][k].get("fitted") for k in e0["kernel_update"]),
            "passenger_posterior_updates_service_timing": int(service_diag["roots_shifted_nonzero"]) > 0,
            "updated_state_redistributes_route_posterior": float(m1["top_route_changed_mass"]) > 0,
            "updated_state_redistributes_boarding_posterior": float(m1["top_boarding_chain_changed_mass"]) > 0,
        },
        "mass_conservation": {
            "E0_pass": bool(m0["mass_conservation_pass"]),
            "E1_pass": bool(m1["mass_conservation_pass"]),
            "passenger_mass": float(m0["passenger_mass"]),
        },
        "distributed_equivalence_boundary": "All passenger rows are retained exactly once. Global kernel and service-offset decisions are made only after merging complete sufficient statistics; numerical sums may differ from monolithic order at floating-point roundoff only.",
    }
    if not result["mass_conservation"]["E0_pass"] or not result["mass_conservation"]["E1_pass"]:
        raise SystemExit("final distributed mass conservation failed")
    if abs(float(m0["passenger_mass"]) - float(m1["passenger_mass"])) > 1e-6:
        raise SystemExit("E0/E1 passenger mass mismatch")
    _dump(output, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("split")
    s.add_argument("--source", type=Path, required=True); s.add_argument("--out-dir", type=Path, required=True); s.add_argument("--shards", type=int, default=8); s.add_argument("--manifest", type=Path, required=True)

    s = sub.add_parser("e0-shard")
    s.add_argument("--cohorts", type=Path, required=True); s.add_argument("--routes", type=Path, required=True); s.add_argument("--roots", type=Path, required=True); s.add_argument("--sidecar", type=Path, required=True); s.add_argument("--output", type=Path, required=True); s.add_argument("--beam", type=int, default=base.BEAM)

    s = sub.add_parser("merge-e0")
    s.add_argument("--input", type=Path, action="append", required=True); s.add_argument("--output", type=Path, required=True)

    s = sub.add_parser("shift-shard")
    s.add_argument("--cohorts", type=Path, required=True); s.add_argument("--routes", type=Path, required=True); s.add_argument("--roots", type=Path, required=True); s.add_argument("--e0-global", type=Path, required=True); s.add_argument("--output", type=Path, required=True); s.add_argument("--beam", type=int, default=base.BEAM)

    s = sub.add_parser("merge-shift")
    s.add_argument("--input", type=Path, action="append", required=True); s.add_argument("--roots", type=Path, required=True); s.add_argument("--output", type=Path, required=True)

    s = sub.add_parser("e1-shard")
    s.add_argument("--cohorts", type=Path, required=True); s.add_argument("--sidecar", type=Path, required=True); s.add_argument("--routes", type=Path, required=True); s.add_argument("--roots", type=Path, required=True); s.add_argument("--e0-global", type=Path, required=True); s.add_argument("--shift-global", type=Path, required=True); s.add_argument("--output", type=Path, required=True); s.add_argument("--beam", type=int, default=base.BEAM)

    s = sub.add_parser("merge-e1")
    s.add_argument("--input", type=Path, action="append", required=True); s.add_argument("--output", type=Path, required=True)

    s = sub.add_parser("finalize")
    s.add_argument("--e0-global", type=Path, required=True); s.add_argument("--shift-global", type=Path, required=True); s.add_argument("--e1-global", type=Path, required=True); s.add_argument("--roots", type=Path, required=True); s.add_argument("--output", type=Path, required=True)

    a = p.parse_args()
    if a.command == "split": split_cohorts(a.source, a.out_dir, a.shards, a.manifest)
    elif a.command == "e0-shard": e0_shard(a.cohorts, a.routes, a.roots, a.sidecar, a.output, a.beam)
    elif a.command == "merge-e0": merge_e0(a.input, a.output)
    elif a.command == "shift-shard": shift_shard(a.cohorts, a.routes, a.roots, a.e0_global, a.output, a.beam)
    elif a.command == "merge-shift": merge_shift(a.input, a.roots, a.output)
    elif a.command == "e1-shard": e1_shard(a.cohorts, a.sidecar, a.routes, a.roots, a.e0_global, a.shift_global, a.output, a.beam)
    elif a.command == "merge-e1": merge_e1(a.input, a.output)
    else: finalize(a.e0_global, a.shift_global, a.e1_global, a.roots, a.output)


if __name__ == "__main__":
    main()
