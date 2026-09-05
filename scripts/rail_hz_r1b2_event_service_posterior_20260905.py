from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = "rail.hz-r1b2-event-service-posterior.v1"


def load_routes(path: Path) -> dict[tuple[str, int, str, int], list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw["status"] != "QUALIFIED_LINE_AWARE_ROUTE_SUPPORT":
        raise SystemExit(f"route support not qualified: {raw.get('status')}")
    out: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}
    for key, candidates in raw["route_support"].items():
        left, right = key.split("->")
        ol, os = left.split(":")
        dl, ds = right.split(":")
        out[(ol, int(os), dl, int(ds))] = candidates
    return out


def load_roots(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw["status"] != "QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION":
        raise SystemExit(f"roots not qualified: {raw.get('status')}")
    roots = raw["roots"]
    return {str(r["root_id"]): r for r in roots}, roots


def event_key(root_id: str, event_index: int) -> str:
    return f"{root_id}::e{event_index:03d}"


def find_event_index(events: list[dict[str, Any]], station: int) -> int | None:
    matches = [i for i, e in enumerate(events) if int(e["station"]) == int(station)]
    if len(matches) != 1:
        return None
    return matches[0]


def read_batches(path: Path, batch_size: int = 100000):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield batch.to_pylist()


def write_event_aggregate(path: Path, acc: dict[str, dict[str, float]]) -> None:
    rows = []
    for k, v in sorted(acc.items()):
        root_id, idx = k.rsplit("::e", 1)
        rows.append({
            "event_key": k,
            "root_id": root_id,
            "event_index": int(idx),
            "traversal_lineage_mass": float(v.get("traversal", 0.0)),
            "boarding_lineage_mass": float(v.get("boarding", 0.0)),
            "alighting_lineage_mass": float(v.get("alighting", 0.0)),
            "transfer_arrival_lineage_mass": float(v.get("transfer_arrival", 0.0)),
            "transfer_departure_lineage_mass": float(v.get("transfer_departure", 0.0)),
            "lineage_edge_contribution_count": int(v.get("edge_count", 0.0)),
        })
    schema = pa.schema([
        ("event_key", pa.string()), ("root_id", pa.string()), ("event_index", pa.int32()),
        ("traversal_lineage_mass", pa.float64()), ("boarding_lineage_mass", pa.float64()),
        ("alighting_lineage_mass", pa.float64()), ("transfer_arrival_lineage_mass", pa.float64()),
        ("transfer_departure_lineage_mass", pa.float64()), ("lineage_edge_contribution_count", pa.int64()),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def extract_shard(edges_path: Path, routes_path: Path, roots_path: Path, out_events: Path, out_summary: Path, shard_index: int) -> dict[str, Any]:
    routes = load_routes(routes_path)
    root_map, _roots = load_roots(roots_path)
    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    resolved_lineage_mass = unresolved_mass = 0.0
    mapped_chain_mass = station_only_mass = 0.0
    boarding_mass = alighting_mass = 0.0
    transfer_arrival_mass = transfer_departure_mass = 0.0
    mapping_failure_mass = 0.0
    mapping_failure_count = 0
    resolved_edge_count = unresolved_edge_count = 0
    transfer_episode_mass = 0.0
    chain_length_mass: dict[int, float] = defaultdict(float)
    failure_reasons = Counter()

    for rows in read_batches(edges_path):
        for row in rows:
            lm = float(row["lineage_mass"])
            dtype = str(row["descendant_state_type"])
            if dtype == "UNRESOLVED":
                unresolved_mass += lm
                unresolved_edge_count += 1
                continue
            resolved_lineage_mass += lm
            resolved_edge_count += 1
            roots_seq = tuple(x for x in str(row["root_chain"]).split(">") if x)
            if not roots_seq:
                station_only_mass += lm
                continue

            key = (str(row["origin_line"]), int(row["origin_station"]), str(row["destination_line"]), int(row["destination_station"]))
            candidates = routes.get(key)
            rank = int(row["route_rank"])
            if not candidates or rank < 1 or rank > len(candidates):
                mapping_failure_mass += lm; mapping_failure_count += 1; failure_reasons["ROUTE_RANK_NOT_RESOLVED"] += 1
                continue
            candidate = candidates[rank - 1]
            legs = list(candidate.get("ride_legs", []))
            if len(legs) != len(roots_seq):
                mapping_failure_mass += lm; mapping_failure_count += 1; failure_reasons["ROOT_LEG_LENGTH_MISMATCH"] += 1
                continue

            mapped_chain_mass += lm
            chain_length_mass[len(roots_seq)] += lm
            transfer_episode_mass += lm * max(0, len(roots_seq) - 1)

            chain_ok = True
            leg_indices: list[tuple[str, int, int]] = []
            for rid, leg in zip(roots_seq, legs):
                root = root_map.get(rid)
                if root is None:
                    chain_ok = False; failure_reasons["ROOT_NOT_FOUND"] += 1; break
                events = root["events"]
                i0 = find_event_index(events, int(leg["from_station"]))
                i1 = find_event_index(events, int(leg["to_station"]))
                if i0 is None or i1 is None or i1 < i0:
                    chain_ok = False; failure_reasons["LEG_EVENT_RANGE_NOT_RESOLVED"] += 1; break
                leg_indices.append((rid, i0, i1))
            if not chain_ok:
                mapped_chain_mass -= lm
                chain_length_mass[len(roots_seq)] -= lm
                transfer_episode_mass -= lm * max(0, len(roots_seq) - 1)
                mapping_failure_mass += lm; mapping_failure_count += 1
                continue

            for j, (rid, i0, i1) in enumerate(leg_indices):
                for idx in range(i0, i1 + 1):
                    ek = event_key(rid, idx)
                    acc[ek]["traversal"] += lm
                    acc[ek]["edge_count"] += 1.0
                acc[event_key(rid, i0)]["boarding"] += lm
                acc[event_key(rid, i1)]["alighting"] += lm
                boarding_mass += lm if j == 0 else 0.0
                alighting_mass += lm if j == len(leg_indices) - 1 else 0.0
                if j > 0:
                    acc[event_key(rid, i0)]["transfer_departure"] += lm
                    transfer_departure_mass += lm
                if j < len(leg_indices) - 1:
                    acc[event_key(rid, i1)]["transfer_arrival"] += lm
                    transfer_arrival_mass += lm

    gates = {
        "no_route_root_leg_mapping_failure": mapping_failure_count == 0 and abs(mapping_failure_mass) <= 1e-9,
        "mapped_plus_station_only_equals_resolved": abs((mapped_chain_mass + station_only_mass) - resolved_lineage_mass) <= 1e-6,
        "first_boarding_mass_equals_mapped_chain_mass": abs(boarding_mass - mapped_chain_mass) <= 1e-6,
        "final_alighting_mass_equals_mapped_chain_mass": abs(alighting_mass - mapped_chain_mass) <= 1e-6,
        "transfer_arrival_departure_match": abs(transfer_arrival_mass - transfer_departure_mass) <= 1e-6,
        "transfer_episode_mass_matches": abs(transfer_episode_mass - transfer_arrival_mass) <= 1e-6,
    }
    if not all(gates.values()):
        raise SystemExit("R1B2 event mapping qualification failed: " + json.dumps(gates))

    write_event_aggregate(out_events, acc)
    result = {
        "schema": SCHEMA,
        "command": "extract-shard",
        "status": "QUALIFIED_R1B2_EVENT_EVIDENCE_SHARD",
        "shard_index": int(shard_index),
        "resolved_lineage_mass": resolved_lineage_mass,
        "unresolved_mass": unresolved_mass,
        "mapped_chain_mass": mapped_chain_mass,
        "station_only_mass": station_only_mass,
        "first_boarding_mass": boarding_mass,
        "final_alighting_mass": alighting_mass,
        "transfer_episode_mass": transfer_episode_mass,
        "event_keys_with_support": len(acc),
        "resolved_lineage_edge_count": resolved_edge_count,
        "unresolved_edge_count": unresolved_edge_count,
        "mapping_failure_mass": mapping_failure_mass,
        "mapping_failure_count": mapping_failure_count,
        "mapping_failure_reasons": dict(failure_reasons),
        "chain_length_lineage_mass": {str(k): v for k, v in sorted(chain_length_mass.items())},
        "qualification_gates": gates,
        "semantic_boundary": "This stage maps already-qualified genealogy mass onto service events. It does not yet convert support mass into event-existence probability or alter event timing; those backward posterior updates belong to R1B.3."
    }
    out_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "median": None, "p90": None}
    xs = sorted(values)
    def q(p: float) -> float:
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi: return float(xs[lo])
        w = pos - lo
        return float(xs[lo] * (1-w) + xs[hi] * w)
    return {"p10": q(0.10), "median": q(0.50), "p90": q(0.90)}


def merge_global(event_files: list[Path], summaries: list[Path], roots_path: Path, genealogy_global_path: Path, out_events: Path, out_summary: Path) -> dict[str, Any]:
    root_map, roots = load_roots(roots_path)
    shard_summaries = [json.loads(p.read_text(encoding="utf-8")) for p in summaries]
    if len(shard_summaries) != 8 or len(event_files) != 8:
        raise SystemExit("expected exactly 8 R1B2 shard products")
    g = json.loads(genealogy_global_path.read_text(encoding="utf-8"))

    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for p in event_files:
        for row in pq.read_table(p).to_pylist():
            ek = str(row["event_key"])
            for src, dst in [
                ("traversal_lineage_mass", "traversal"), ("boarding_lineage_mass", "boarding"),
                ("alighting_lineage_mass", "alighting"), ("transfer_arrival_lineage_mass", "transfer_arrival"),
                ("transfer_departure_lineage_mass", "transfer_departure"),
            ]:
                acc[ek][dst] += float(row[src])
            acc[ek]["edge_count"] += int(row["lineage_edge_contribution_count"])

    inventory = []
    root_support: dict[str, float] = defaultdict(float)
    evidence_counts = Counter()
    supported_event_masses = []
    zero_support_events = 0
    total_events = 0
    for root in roots:
        rid = str(root["root_id"])
        evidence_class = str(root.get("evidence_class", "UNKNOWN"))
        evidence_counts[evidence_class] += len(root["events"])
        for idx, ev in enumerate(root["events"]):
            total_events += 1
            ek = event_key(rid, idx)
            a = acc.get(ek, {})
            traversal = float(a.get("traversal", 0.0))
            root_support[rid] = max(root_support[rid], traversal)
            if traversal > 0:
                supported_event_masses.append(traversal)
                support_state = "GENEALOGY_SUPPORTED_EVENT_SEED"
            else:
                zero_support_events += 1
                support_state = "NO_CURRENT_GENEALOGY_TRAVERSAL_SUPPORT"
            t = float(ev["time_s"]); sd = float(ev["sd_s"])
            inventory.append({
                "event_key": ek,
                "root_id": rid,
                "path_id": str(root["path_id"]),
                "direction": str(root["direction"]),
                "event_index": idx,
                "station": int(ev["station"]),
                "prior_time_s": t,
                "prior_sd_s": sd,
                "prior_lower_95_s": t - 1.96 * sd,
                "prior_upper_95_s": t + 1.96 * sd,
                "current_timing_center_s": t,
                "current_timing_sd_s": sd,
                "timing_stage": "PRE_BACKWARD_CONSTRAINT_UNCHANGED_FROM_QUALIFIED_SERVICE_INITIALIZATION",
                "evidence_class": evidence_class,
                "support_state": support_state,
                "traversal_lineage_mass": traversal,
                "boarding_lineage_mass": float(a.get("boarding", 0.0)),
                "alighting_lineage_mass": float(a.get("alighting", 0.0)),
                "transfer_arrival_lineage_mass": float(a.get("transfer_arrival", 0.0)),
                "transfer_departure_lineage_mass": float(a.get("transfer_departure", 0.0)),
                "lineage_edge_contribution_count": int(a.get("edge_count", 0)),
            })

    schema = pa.schema([
        ("event_key", pa.string()), ("root_id", pa.string()), ("path_id", pa.string()), ("direction", pa.string()),
        ("event_index", pa.int32()), ("station", pa.int32()), ("prior_time_s", pa.float64()), ("prior_sd_s", pa.float64()),
        ("prior_lower_95_s", pa.float64()), ("prior_upper_95_s", pa.float64()), ("current_timing_center_s", pa.float64()),
        ("current_timing_sd_s", pa.float64()), ("timing_stage", pa.string()), ("evidence_class", pa.string()),
        ("support_state", pa.string()), ("traversal_lineage_mass", pa.float64()), ("boarding_lineage_mass", pa.float64()),
        ("alighting_lineage_mass", pa.float64()), ("transfer_arrival_lineage_mass", pa.float64()),
        ("transfer_departure_lineage_mass", pa.float64()), ("lineage_edge_contribution_count", pa.int64()),
    ])
    pq.write_table(pa.Table.from_pylist(inventory, schema=schema), out_events, compression="zstd")

    mapped = sum(float(x["mapped_chain_mass"]) for x in shard_summaries)
    unresolved = sum(float(x["unresolved_mass"]) for x in shard_summaries)
    board = sum(float(x["first_boarding_mass"]) for x in shard_summaries)
    alight = sum(float(x["final_alighting_mass"]) for x in shard_summaries)
    transfer = sum(float(x["transfer_episode_mass"]) for x in shard_summaries)
    failure_mass = sum(float(x["mapping_failure_mass"]) for x in shard_summaries)
    roots_with_support = sum(1 for r in roots if root_support.get(str(r["root_id"]), 0.0) > 0)
    roots_without_support = [str(r["root_id"]) for r in roots if root_support.get(str(r["root_id"]), 0.0) <= 0]

    gates = {
        "all_19236_service_events_present": total_events == 19236,
        "all_8_shards_qualified": all(x["status"] == "QUALIFIED_R1B2_EVENT_EVIDENCE_SHARD" and all(x["qualification_gates"].values()) for x in shard_summaries),
        "resolved_genealogy_mass_matches_r1b1": abs(mapped - float(g["resolved_genealogy_mass"])) <= 1e-6,
        "unresolved_mass_matches_r1b1": abs(unresolved - float(g["unresolved_genealogy_mass"])) <= 1e-6,
        "first_boarding_mass_conserved": abs(board - mapped) <= 1e-6,
        "final_alighting_mass_conserved": abs(alight - mapped) <= 1e-6,
        "no_event_mapping_failure_mass": abs(failure_mass) <= 1e-9,
        "timing_not_silently_changed_in_r1b2": all(abs(float(x["current_timing_center_s"]) - float(x["prior_time_s"])) <= 1e-12 and abs(float(x["current_timing_sd_s"]) - float(x["prior_sd_s"])) <= 1e-12 for x in inventory),
    }
    if not all(gates.values()):
        raise SystemExit("R1B2 global event-posterior substrate qualification failed: " + json.dumps(gates))

    result = {
        "schema": SCHEMA,
        "command": "merge-global",
        "status": "QUALIFIED_R1B2_EVENT_LEVEL_SERVICE_POSTERIOR_SUBSTRATE",
        "service_date": "2019-01-04",
        "scope": "FULL_SERVICE_DAY_0400_TO_NEXT_0400_FULL_NETWORK_FULL_QUALIFIED_PASSENGER_DOMAIN",
        "event_count": total_events,
        "roots_total": len(roots),
        "roots_with_genealogy_support": roots_with_support,
        "roots_without_genealogy_support": roots_without_support,
        "events_with_genealogy_traversal_support": total_events - zero_support_events,
        "events_without_current_genealogy_traversal_support": zero_support_events,
        "genealogy_traversal_support_mass_quantiles": quantiles(supported_event_masses),
        "mapped_resolved_genealogy_mass": mapped,
        "unresolved_genealogy_mass": unresolved,
        "first_boarding_mass": board,
        "final_alighting_mass": alight,
        "transfer_episode_mass": transfer,
        "mapping_failure_mass": failure_mass,
        "evidence_class_event_counts": dict(evidence_counts),
        "qualification_gates": gates,
        "timing_semantics": "R1B.2 establishes an event-level posterior state and attaches genealogy evidence mass. It intentionally leaves timing center and timing uncertainty unchanged until R1B.3 applies explicit backward passenger evidence factors.",
        "next_stage": "R1B_3_APPLY_GENEALOGY_BASED_BACKWARD_SERVICE_CONSTRAINTS"
    }
    out_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("extract-shard")
    s.add_argument("--edges", type=Path, required=True)
    s.add_argument("--routes", type=Path, required=True)
    s.add_argument("--roots", type=Path, required=True)
    s.add_argument("--out-events", type=Path, required=True)
    s.add_argument("--out-summary", type=Path, required=True)
    s.add_argument("--shard-index", type=int, required=True)
    m = sp.add_parser("merge-global")
    m.add_argument("--event-file", type=Path, action="append", required=True)
    m.add_argument("--summary", type=Path, action="append", required=True)
    m.add_argument("--roots", type=Path, required=True)
    m.add_argument("--genealogy-global", type=Path, required=True)
    m.add_argument("--out-events", type=Path, required=True)
    m.add_argument("--out-summary", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "extract-shard":
        result = extract_shard(a.edges, a.routes, a.roots, a.out_events, a.out_summary, a.shard_index)
    else:
        result = merge_global(a.event_file, a.summary, a.roots, a.genealogy_global, a.out_events, a.out_summary)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
