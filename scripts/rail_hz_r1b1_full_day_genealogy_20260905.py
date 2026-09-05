from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import scripts.rail_hz_r1b_full_service_day_joint_update_20260905 as base

SCHEMA = "rail.hz-r1b1-full-day-genealogy.v1"
EDGE_FLUSH_ROWS = 50000


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _kernel_from_payload(x: dict[str, Any]) -> base.Kernel:
    return base.Kernel(float(x["median_sec"]), float(x["sigma"]), str(x["evidence_class"]))


def _kernels_from_e0(path: Path) -> dict[str, base.Kernel]:
    x = json.loads(path.read_text(encoding="utf-8"))
    return {k: _kernel_from_payload(v) for k, v in x["kernels0"].items()}


def _cohort_id(row: Any) -> str:
    text = "|".join([
        str(row.origin_line), str(int(row.origin_station)),
        str(row.destination_line), str(int(row.destination_station)),
        f"{float(row.entry_sec):.6f}", f"{float(row.exit_sec):.6f}",
        f"{float(row.passenger_mass):.9f}",
    ])
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _descendant_id(roots: tuple[str, ...], movements: tuple[str, ...], station_only_key: str | None = None) -> str:
    if roots:
        text = "roots=" + ">".join(roots) + "|movements=" + ">".join(movements)
    else:
        text = "station_only=" + str(station_only_key or "UNKNOWN")
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _transfer_factors(chain: base.ChainState) -> list[dict[str, Any]]:
    return [f for f in chain.factors if f.get("type") == "TRANSFER"]


def _edge_schema() -> pa.Schema:
    return pa.schema([
        ("cohort_id", pa.string()),
        ("upstream_state_type", pa.string()),
        ("origin_line", pa.string()),
        ("origin_station", pa.int32()),
        ("destination_line", pa.string()),
        ("destination_station", pa.int32()),
        ("entry_sec", pa.float64()),
        ("exit_sec", pa.float64()),
        ("passenger_mass", pa.float64()),
        ("descendant_state_id", pa.string()),
        ("descendant_state_type", pa.string()),
        ("route_rank", pa.int16()),
        ("root_chain", pa.string()),
        ("transfer_chain", pa.string()),
        ("transfer_count", pa.int8()),
        ("posterior_probability", pa.float64()),
        ("lineage_mass", pa.float64()),
        ("unresolved_reason", pa.string()),
    ])


def _write_aggregate(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


def extract_shard(
    cohorts_path: Path,
    routes_path: Path,
    roots_path: Path,
    e0_global_path: Path,
    edges_path: Path,
    descendant_path: Path,
    roots_mass_path: Path,
    transfers_path: Path,
    summary_path: Path,
    beam: int,
    shard_index: int,
) -> dict[str, Any]:
    routes = base.load_routes(routes_path)
    roots, _root_meta = base.load_roots(roots_path)
    cache = base.LegCache(roots)
    kernels = _kernels_from_e0(e0_global_path)

    edge_schema = _edge_schema()
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(edges_path, edge_schema, compression="zstd")
    edge_buffer: list[dict[str, Any]] = []

    passenger_mass = resolved_mass = unresolved_mass = 0.0
    no_route_mass = incompatible_mass = 0.0
    finite_cohorts = cohort_count = 0
    edge_count = 0
    norm_max_abs_error = 0.0
    weighted_entropy = 0.0
    weighted_top_probability = 0.0
    station_only_mass = 0.0

    root_mass: dict[str, float] = defaultdict(float)
    transfer_mass: dict[tuple[str, str, str], float] = defaultdict(float)
    descendant_mass: dict[tuple[str, str, str, int], float] = defaultdict(float)
    transfer_count_mass: dict[int, float] = defaultdict(float)
    chain_length_mass: dict[int, float] = defaultdict(float)

    def flush() -> None:
        nonlocal edge_buffer
        if edge_buffer:
            writer.write_table(pa.Table.from_pylist(edge_buffer, schema=edge_schema))
            edge_buffer = []

    try:
        for df in base.parquet_batches(cohorts_path):
            for row in df.itertuples(index=False):
                mass = float(row.passenger_mass)
                passenger_mass += mass
                cohort_count += 1
                cid = _cohort_id(row)
                result = base.evaluate_cohort(
                    str(row.origin_line), int(row.origin_station),
                    str(row.destination_line), int(row.destination_station),
                    float(row.entry_sec), float(row.exit_sec),
                    routes, cache, kernels, beam, retain_chains=True,
                )

                common = {
                    "cohort_id": cid,
                    "upstream_state_type": "AFC_EXACT_SECOND_COHORT",
                    "origin_line": str(row.origin_line),
                    "origin_station": int(row.origin_station),
                    "destination_line": str(row.destination_line),
                    "destination_station": int(row.destination_station),
                    "entry_sec": float(row.entry_sec),
                    "exit_sec": float(row.exit_sec),
                    "passenger_mass": mass,
                }

                if not result["finite"]:
                    reason = str(result["reason"])
                    unresolved_mass += mass
                    if reason == "NO_ROUTE_SUPPORT":
                        no_route_mass += mass
                    else:
                        incompatible_mass += mass
                    edge_buffer.append({
                        **common,
                        "descendant_state_id": "UNRESOLVED:" + reason,
                        "descendant_state_type": "UNRESOLVED",
                        "route_rank": -1,
                        "root_chain": "",
                        "transfer_chain": "",
                        "transfer_count": -1,
                        "posterior_probability": 1.0,
                        "lineage_mass": mass,
                        "unresolved_reason": reason,
                    })
                    edge_count += 1
                    if len(edge_buffer) >= EDGE_FLUSH_ROWS:
                        flush()
                    continue

                finite_cohorts += 1
                resolved_mass += mass
                chain_probs = result["chain_probs"]
                psum = float(sum(float(p) for _rank, _chain, p in chain_probs))
                norm_max_abs_error = max(norm_max_abs_error, abs(psum - 1.0))
                if not chain_probs or psum <= 0:
                    raise SystemExit(f"finite cohort without normalized chains: {cid}")
                entropy = -sum(float(p) * math.log(max(float(p), 1e-300)) for _rank, _chain, p in chain_probs)
                top_p = max(float(p) for _rank, _chain, p in chain_probs)
                weighted_entropy += mass * entropy
                weighted_top_probability += mass * top_p

                station_key = f"{row.origin_line}:{int(row.origin_station)}->{row.destination_line}:{int(row.destination_station)}"
                for rank, chain, p_raw in chain_probs:
                    p = float(p_raw) / psum
                    lm = mass * p
                    roots_seq = tuple(str(x) for x in chain.roots)
                    tf = _transfer_factors(chain)
                    movements = tuple(str(x.get("movement", "UNKNOWN")) for x in tf)
                    dtype = "SERVICE_CHAIN_STATE" if roots_seq else "STATION_ONLY_STATE"
                    did = _descendant_id(roots_seq, movements, station_key if not roots_seq else None)
                    root_chain = ">".join(roots_seq)
                    transfer_chain = ">".join(movements)
                    tcount = len(movements)

                    if not roots_seq:
                        station_only_mass += lm
                    for rid in roots_seq:
                        root_mass[rid] += lm
                    for f in tf:
                        key = (str(f.get("movement", "UNKNOWN")), str(f.get("lower_root", "")), str(f.get("upper_root", "")))
                        transfer_mass[key] += lm
                    descendant_mass[(did, root_chain, transfer_chain, int(tcount))] += lm
                    transfer_count_mass[int(tcount)] += lm
                    chain_length_mass[len(roots_seq)] += lm

                    edge_buffer.append({
                        **common,
                        "descendant_state_id": did,
                        "descendant_state_type": dtype,
                        "route_rank": int(rank),
                        "root_chain": root_chain,
                        "transfer_chain": transfer_chain,
                        "transfer_count": int(tcount),
                        "posterior_probability": p,
                        "lineage_mass": lm,
                        "unresolved_reason": "",
                    })
                    edge_count += 1
                    if len(edge_buffer) >= EDGE_FLUSH_ROWS:
                        flush()
        flush()
    finally:
        writer.close()

    mass_conservation_error = abs((resolved_mass + unresolved_mass) - passenger_mass)
    conservation_pass = mass_conservation_error <= 1e-6
    normalization_pass = norm_max_abs_error <= 1e-9
    if not conservation_pass:
        raise SystemExit(f"genealogy mass conservation failed: {mass_conservation_error}")
    if not normalization_pass:
        raise SystemExit(f"chain posterior normalization failed: {norm_max_abs_error}")

    _write_aggregate(
        descendant_path,
        [
            {"descendant_state_id": k[0], "root_chain": k[1], "transfer_chain": k[2], "transfer_count": k[3], "lineage_mass": v}
            for k, v in sorted(descendant_mass.items())
        ],
        pa.schema([
            ("descendant_state_id", pa.string()), ("root_chain", pa.string()), ("transfer_chain", pa.string()),
            ("transfer_count", pa.int8()), ("lineage_mass", pa.float64()),
        ]),
    )
    _write_aggregate(
        roots_mass_path,
        [{"root_id": k, "lineage_usage_mass": v} for k, v in sorted(root_mass.items())],
        pa.schema([("root_id", pa.string()), ("lineage_usage_mass", pa.float64())]),
    )
    _write_aggregate(
        transfers_path,
        [
            {"movement": k[0], "lower_root": k[1], "upper_root": k[2], "lineage_mass": v}
            for k, v in sorted(transfer_mass.items())
        ],
        pa.schema([
            ("movement", pa.string()), ("lower_root", pa.string()), ("upper_root", pa.string()), ("lineage_mass", pa.float64()),
        ]),
    )

    result = {
        "schema": SCHEMA,
        "command": "extract-shard",
        "status": "QUALIFIED_R1B1_GENEALOGY_SHARD",
        "shard_index": int(shard_index),
        "posterior_authority": "E0_INITIAL_KERNEL_POSTERIOR_RECONSTRUCTED_WITH_IDENTICAL_FROZEN_MODEL",
        "scope": "FULL_SERVICE_DAY_PARTITION_NO_SCIENTIFIC_SUBSAMPLE",
        "beam": int(beam),
        "cohort_count": int(cohort_count),
        "finite_cohort_count": int(finite_cohorts),
        "passenger_mass": passenger_mass,
        "resolved_genealogy_mass": resolved_mass,
        "unresolved_genealogy_mass": unresolved_mass,
        "no_route_support_mass": no_route_mass,
        "time_or_service_incompatible_mass": incompatible_mass,
        "station_only_lineage_mass": station_only_mass,
        "posterior_lineage_edge_count": int(edge_count),
        "unique_descendant_state_count": int(len(descendant_mass)),
        "unique_root_count_with_mass": int(len(root_mass)),
        "unique_transfer_edge_count": int(len(transfer_mass)),
        "weighted_mean_lineage_entropy_nats": weighted_entropy / resolved_mass if resolved_mass else None,
        "weighted_mean_top_descendant_probability": weighted_top_probability / resolved_mass if resolved_mass else None,
        "transfer_count_lineage_mass": {str(k): v for k, v in sorted(transfer_count_mass.items())},
        "service_chain_length_lineage_mass": {str(k): v for k, v in sorted(chain_length_mass.items())},
        "posterior_probability_normalization_max_abs_error": norm_max_abs_error,
        "mass_conservation_abs_error": mass_conservation_error,
        "posterior_probability_normalization_pass": normalization_pass,
        "genealogy_mass_conservation_pass": conservation_pass,
        "semantic_boundary": {
            "upstream_carrier": "AFC exact-second cohort; not relabeled as a detected morphological pulse",
            "downstream_state": "probabilistic service-chain descendant state under the frozen E0 approximation",
            "probability_not_hard_assignment": True,
            "unresolved_mass_explicit": True,
        },
    }
    _dump(summary_path, result)
    return result


def _merge_numeric_maps(paths: list[Path], key_cols: list[str], value_col: str) -> list[dict[str, Any]]:
    acc: dict[tuple[Any, ...], float] = defaultdict(float)
    for p in paths:
        table = pq.read_table(p)
        for row in table.to_pylist():
            key = tuple(row[k] for k in key_cols)
            acc[key] += float(row[value_col])
    out = []
    for key, value in sorted(acc.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = {k: v for k, v in zip(key_cols, key)}
        row[value_col] = value
        out.append(row)
    return out


def merge_global(
    summaries: list[Path],
    descendants: list[Path],
    roots_mass: list[Path],
    transfers: list[Path],
    e0_global_path: Path,
    out_summary: Path,
    out_descendants: Path,
    out_roots: Path,
    out_transfers: Path,
) -> dict[str, Any]:
    ss = [json.loads(p.read_text(encoding="utf-8")) for p in summaries]
    if len(ss) != 8:
        raise SystemExit(f"expected 8 shard summaries, found {len(ss)}")
    e0 = json.loads(e0_global_path.read_text(encoding="utf-8"))
    m0 = e0["metrics"]

    passenger = sum(float(x["passenger_mass"]) for x in ss)
    resolved = sum(float(x["resolved_genealogy_mass"]) for x in ss)
    unresolved = sum(float(x["unresolved_genealogy_mass"]) for x in ss)
    no_route = sum(float(x["no_route_support_mass"]) for x in ss)
    incompatible = sum(float(x["time_or_service_incompatible_mass"]) for x in ss)
    finite_cohorts = sum(int(x["finite_cohort_count"]) for x in ss)
    cohorts = sum(int(x["cohort_count"]) for x in ss)
    edges = sum(int(x["posterior_lineage_edge_count"]) for x in ss)
    entropy_num = sum(float(x["weighted_mean_lineage_entropy_nats"] or 0.0) * float(x["resolved_genealogy_mass"]) for x in ss)
    top_num = sum(float(x["weighted_mean_top_descendant_probability"] or 0.0) * float(x["resolved_genealogy_mass"]) for x in ss)
    norm_error = max(float(x["posterior_probability_normalization_max_abs_error"]) for x in ss)

    expected_passenger = float(m0["passenger_mass"])
    expected_resolved = float(m0["finite_posterior_mass"])
    expected_no_route = float(m0["no_route_support_mass"])
    expected_incompatible = float(m0["time_or_service_incompatible_mass"])

    gates = {
        "eight_shards_present": len(ss) == 8,
        "passenger_mass_matches_frozen_e0": abs(passenger - expected_passenger) <= 1e-6,
        "resolved_mass_matches_frozen_e0": abs(resolved - expected_resolved) <= 1e-6,
        "no_route_mass_matches_frozen_e0": abs(no_route - expected_no_route) <= 1e-6,
        "incompatible_mass_matches_frozen_e0": abs(incompatible - expected_incompatible) <= 1e-6,
        "global_genealogy_mass_conservation": abs((resolved + unresolved) - passenger) <= 1e-6,
        "unresolved_decomposition_matches": abs((no_route + incompatible) - unresolved) <= 1e-6,
        "posterior_probability_normalization": norm_error <= 1e-9,
    }
    if not all(gates.values()):
        raise SystemExit("global R1B.1 genealogy qualification failed: " + json.dumps(gates))

    desc_rows = _merge_numeric_maps(descendants, ["descendant_state_id", "root_chain", "transfer_chain", "transfer_count"], "lineage_mass")
    root_rows = _merge_numeric_maps(roots_mass, ["root_id"], "lineage_usage_mass")
    transfer_rows = _merge_numeric_maps(transfers, ["movement", "lower_root", "upper_root"], "lineage_mass")

    _write_aggregate(out_descendants, desc_rows, pa.schema([
        ("descendant_state_id", pa.string()), ("root_chain", pa.string()), ("transfer_chain", pa.string()),
        ("transfer_count", pa.int8()), ("lineage_mass", pa.float64()),
    ]))
    _write_aggregate(out_roots, root_rows, pa.schema([("root_id", pa.string()), ("lineage_usage_mass", pa.float64())]))
    _write_aggregate(out_transfers, transfer_rows, pa.schema([
        ("movement", pa.string()), ("lower_root", pa.string()), ("upper_root", pa.string()), ("lineage_mass", pa.float64()),
    ]))

    transfer_count_mass: dict[str, float] = defaultdict(float)
    chain_length_mass: dict[str, float] = defaultdict(float)
    for x in ss:
        for k, v in x["transfer_count_lineage_mass"].items():
            transfer_count_mass[k] += float(v)
        for k, v in x["service_chain_length_lineage_mass"].items():
            chain_length_mass[k] += float(v)

    result = {
        "schema": SCHEMA,
        "command": "merge-global",
        "status": "QUALIFIED_R1B1_FIRST_CLASS_FULL_DAY_GENEALOGY_SUBSTRATE",
        "service_date": "2019-01-04",
        "scope": "FULL_SERVICE_DAY_0400_TO_NEXT_0400_FULL_NETWORK_FULL_QUALIFIED_PASSENGER_DOMAIN",
        "source_posterior": "FROZEN_E0_INITIAL_KERNEL_POSTERIOR_RECONSTRUCTED_EXACTLY_PER_SHARD",
        "shard_count": 8,
        "cohort_count": cohorts,
        "finite_cohort_count": finite_cohorts,
        "passenger_mass": passenger,
        "resolved_genealogy_mass": resolved,
        "resolved_genealogy_share": resolved / passenger if passenger else None,
        "unresolved_genealogy_mass": unresolved,
        "unresolved_genealogy_share": unresolved / passenger if passenger else None,
        "no_route_support_mass": no_route,
        "time_or_service_incompatible_mass": incompatible,
        "posterior_lineage_edge_count": edges,
        "unique_descendant_state_count": len(desc_rows),
        "unique_root_count_with_lineage_mass": len(root_rows),
        "unique_transfer_edge_count": len(transfer_rows),
        "weighted_mean_lineage_entropy_nats": entropy_num / resolved if resolved else None,
        "weighted_mean_top_descendant_probability": top_num / resolved if resolved else None,
        "transfer_count_lineage_mass": dict(sorted(transfer_count_mass.items(), key=lambda kv: int(kv[0]))),
        "service_chain_length_lineage_mass": dict(sorted(chain_length_mass.items(), key=lambda kv: int(kv[0]))),
        "posterior_probability_normalization_max_abs_error": norm_error,
        "qualification_gates": gates,
        "genealogy_representation": {
            "partitioned_sparse_edges": "8 parquet partitions of AFC exact-second cohort -> probabilistic service-chain descendant state",
            "global_descendant_state_mass": out_descendants.name,
            "global_service_root_usage_mass": out_roots.name,
            "global_transfer_root_to_root_mass": out_transfers.name,
            "unresolved_mass_explicit": True,
            "hard_assignment_used": False,
        },
        "next_stage": "R1B_2_BUILD_EVENT_LEVEL_SERVICE_POSTERIOR",
    }
    _dump(out_summary, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("extract-shard")
    s.add_argument("--cohorts", type=Path, required=True)
    s.add_argument("--routes", type=Path, required=True)
    s.add_argument("--roots", type=Path, required=True)
    s.add_argument("--e0-global", type=Path, required=True)
    s.add_argument("--edges", type=Path, required=True)
    s.add_argument("--descendants", type=Path, required=True)
    s.add_argument("--root-mass", type=Path, required=True)
    s.add_argument("--transfers", type=Path, required=True)
    s.add_argument("--summary", type=Path, required=True)
    s.add_argument("--beam", type=int, default=16)
    s.add_argument("--shard-index", type=int, required=True)

    m = sp.add_parser("merge-global")
    m.add_argument("--summary", type=Path, action="append", required=True)
    m.add_argument("--descendants", type=Path, action="append", required=True)
    m.add_argument("--root-mass", type=Path, action="append", required=True)
    m.add_argument("--transfers", type=Path, action="append", required=True)
    m.add_argument("--e0-global", type=Path, required=True)
    m.add_argument("--out-summary", type=Path, required=True)
    m.add_argument("--out-descendants", type=Path, required=True)
    m.add_argument("--out-roots", type=Path, required=True)
    m.add_argument("--out-transfers", type=Path, required=True)

    a = ap.parse_args()
    if a.cmd == "extract-shard":
        result = extract_shard(a.cohorts, a.routes, a.roots, a.e0_global, a.edges, a.descendants, a.root_mass, a.transfers, a.summary, a.beam, a.shard_index)
    else:
        result = merge_global(a.summary, a.descendants, a.root_mass, a.transfers, a.e0_global, a.out_summary, a.out_descendants, a.out_roots, a.out_transfers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
