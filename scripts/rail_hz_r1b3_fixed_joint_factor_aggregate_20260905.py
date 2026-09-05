from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = "rail.hz-r1b3-fixed-joint-factor-aggregate.v1"


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


def write_table(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def aggregate_shard(a: argparse.Namespace) -> dict[str, Any]:
    routes = load_routes(a.routes)
    pf = pq.ParquetFile(a.genealogy)

    access: dict[tuple[str, int, float], float] = defaultdict(float)
    egress: dict[tuple[str, int, float], float] = defaultdict(float)
    transfer: dict[tuple[str, int, str, int, str], float] = defaultdict(float)

    total_mass = resolved_mass = unresolved_mass = station_only_mass = 0.0
    access_factor_mass = egress_factor_mass = transfer_factor_mass = 0.0
    service_chain_mass = 0.0
    edge_count = resolved_edge_count = 0
    mapping_failure_mass = 0.0
    reasons: Counter[str] = Counter()

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
                continue
            resolved_edge_count += 1
            service_chain_mass += lm
            key = (
                str(row["origin_line"]), int(row["origin_station"]),
                str(row["destination_line"]), int(row["destination_station"]),
            )
            candidates = routes.get(key)
            rank = int(row["route_rank"])
            if not candidates or rank < 1 or rank > len(candidates):
                mapping_failure_mass += lm
                reasons["ROUTE_RANK_NOT_FOUND"] += 1
                continue
            route = candidates[rank - 1]
            legs = list(route.get("ride_legs", []))
            movements = list(route.get("transfer_movements", []))
            if len(legs) != len(roots):
                mapping_failure_mass += lm
                reasons["ROOT_CHAIN_LENGTH_DIFFERS_FROM_RIDE_LEGS"] += 1
                continue

            first_station = int(legs[0]["from_station"])
            last_station = int(legs[-1]["to_station"])
            access[(roots[0], first_station, float(row["entry_sec"]))] += lm
            egress[(roots[-1], last_station, float(row["exit_sec"]))] += lm
            access_factor_mass += lm
            egress_factor_mass += lm

            for j in range(len(roots) - 1):
                lower_station = int(legs[j]["to_station"])
                upper_station = int(legs[j + 1]["from_station"])
                movement = str(movements[j]) if j < len(movements) else f"TRANSFER_{j+1}"
                transfer[(roots[j], lower_station, roots[j + 1], upper_station, movement)] += lm
                transfer_factor_mass += lm

    if mapping_failure_mass > 1e-9:
        raise SystemExit(f"fixed-factor genealogy mapping failure mass: {mapping_failure_mass} {dict(reasons)}")
    if abs((resolved_mass + unresolved_mass) - total_mass) > 1e-6:
        raise SystemExit("genealogy edge mass partition failed")
    if abs(access_factor_mass - service_chain_mass) > 1e-6 or abs(egress_factor_mass - service_chain_mass) > 1e-6:
        raise SystemExit("endpoint factor mass does not match service-chain mass")

    access_rows = [
        {"root_id": k[0], "station": k[1], "entry_sec": k[2], "lineage_mass": v}
        for k, v in sorted(access.items())
    ]
    egress_rows = [
        {"root_id": k[0], "station": k[1], "exit_sec": k[2], "lineage_mass": v}
        for k, v in sorted(egress.items())
    ]
    transfer_rows = [
        {"lower_root": k[0], "lower_station": k[1], "upper_root": k[2], "upper_station": k[3], "movement": k[4], "lineage_mass": v}
        for k, v in sorted(transfer.items())
    ]
    write_table(a.out_access, access_rows, pa.schema([
        ("root_id", pa.string()), ("station", pa.int32()), ("entry_sec", pa.float64()), ("lineage_mass", pa.float64())
    ]))
    write_table(a.out_egress, egress_rows, pa.schema([
        ("root_id", pa.string()), ("station", pa.int32()), ("exit_sec", pa.float64()), ("lineage_mass", pa.float64())
    ]))
    write_table(a.out_transfer, transfer_rows, pa.schema([
        ("lower_root", pa.string()), ("lower_station", pa.int32()), ("upper_root", pa.string()), ("upper_station", pa.int32()),
        ("movement", pa.string()), ("lineage_mass", pa.float64())
    ]))

    result = {
        "schema": SCHEMA,
        "status": "QUALIFIED_R1B3_FIXED_FACTOR_AGGREGATE_SHARD",
        "shard_index": int(a.shard_index),
        "total_lineage_mass": total_mass,
        "resolved_lineage_mass": resolved_mass,
        "unresolved_lineage_mass": unresolved_mass,
        "station_only_mass": station_only_mass,
        "service_chain_mass": service_chain_mass,
        "access_factor_mass": access_factor_mass,
        "egress_factor_mass": egress_factor_mass,
        "transfer_factor_mass": transfer_factor_mass,
        "access_histogram_row_count": len(access_rows),
        "egress_histogram_row_count": len(egress_rows),
        "transfer_pair_row_count": len(transfer_rows),
        "edge_count": edge_count,
        "resolved_service_chain_edge_count": resolved_edge_count,
        "mapping_failure_mass": mapping_failure_mass,
        "scientific_semantics": {
            "source": "ORIGINAL_FIXED_AFC_AND_R1B1_GENEALOGY_EDGES",
            "r1b4_audit_residuals_used_as_input": False,
            "r1b3_previous_timetable_used_as_evidence": False,
            "factor_role": "FIXED_ORDERING_LIKELIHOOD_SUFFICIENT_DATA_WITHOUT_SCIENTIFIC_SELF_ITERATION"
        }
    }
    a.out_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genealogy", type=Path, required=True)
    ap.add_argument("--routes", type=Path, required=True)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--out-access", type=Path, required=True)
    ap.add_argument("--out-egress", type=Path, required=True)
    ap.add_argument("--out-transfer", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    aggregate_shard(ap.parse_args())


if __name__ == "__main__":
    main()
