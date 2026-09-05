from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.rail_hz_r1b2_event_service_posterior_20260905 import event_key, load_roots, quantiles

SCHEMA = "rail.hz-r1b2-event-service-posterior.v2"


def merge_global(event_files: list[Path], summaries: list[Path], roots_path: Path, genealogy_global_path: Path, out_events: Path, out_summary: Path) -> dict[str, Any]:
    _root_map, roots = load_roots(roots_path)
    shard_summaries = [json.loads(p.read_text(encoding="utf-8")) for p in summaries]
    if len(shard_summaries) != 8 or len(event_files) != 8:
        raise SystemExit("expected exactly 8 R1B2 shard products")
    g = json.loads(genealogy_global_path.read_text(encoding="utf-8"))

    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for p in event_files:
        for row in pq.read_table(p).to_pylist():
            ek = str(row["event_key"])
            for src, dst in [
                ("traversal_lineage_mass", "traversal"),
                ("boarding_lineage_mass", "boarding"),
                ("alighting_lineage_mass", "alighting"),
                ("transfer_arrival_lineage_mass", "transfer_arrival"),
                ("transfer_departure_lineage_mass", "transfer_departure"),
            ]:
                acc[ek][dst] += float(row[src])
            acc[ek]["edge_count"] += int(row["lineage_edge_contribution_count"])

    source_event_assignment_count = sum(len(r["events"]) for r in roots)
    inventory: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    root_support: dict[str, float] = defaultdict(float)
    evidence_counts = Counter()
    supported_event_masses: list[float] = []
    zero_support_events = 0

    for root in roots:
        rid = str(root["root_id"])
        evidence_class = str(root.get("evidence_class", "UNKNOWN"))
        evidence_counts[evidence_class] += len(root["events"])
        for idx, ev in enumerate(root["events"]):
            ek = event_key(rid, idx)
            if ek in seen_keys:
                raise SystemExit(f"duplicate root-event identity: {ek}")
            seen_keys.add(ek)
            a = acc.get(ek, {})
            traversal = float(a.get("traversal", 0.0))
            root_support[rid] = max(root_support[rid], traversal)
            if traversal > 0:
                supported_event_masses.append(traversal)
                support_state = "GENEALOGY_SUPPORTED_EVENT_SEED"
            else:
                zero_support_events += 1
                support_state = "NO_CURRENT_GENEALOGY_TRAVERSAL_SUPPORT"
            t = float(ev["time_s"])
            sd = float(ev["sd_s"])
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
        "all_candidate_root_event_states_enumerated_once": len(inventory) == source_event_assignment_count and len(seen_keys) == source_event_assignment_count,
        "all_8_shards_qualified": all(x["status"] == "QUALIFIED_R1B2_EVENT_EVIDENCE_SHARD" and all(x["qualification_gates"].values()) for x in shard_summaries),
        "resolved_genealogy_mass_matches_r1b1": abs(mapped - float(g["resolved_genealogy_mass"])) <= 1e-6,
        "unresolved_mass_matches_r1b1": abs(unresolved - float(g["unresolved_genealogy_mass"])) <= 1e-6,
        "first_boarding_mass_conserved": abs(board - mapped) <= 1e-6,
        "final_alighting_mass_conserved": abs(alight - mapped) <= 1e-6,
        "no_event_mapping_failure_mass": abs(failure_mass) <= 1e-9,
        "timing_not_silently_changed_in_r1b2": all(abs(float(x["current_timing_center_s"]) - float(x["prior_time_s"])) <= 1e-12 and abs(float(x["current_timing_sd_s"]) - float(x["prior_sd_s"])) <= 1e-12 for x in inventory),
    }
    if not all(gates.values()):
        raise SystemExit("R1B2 v2 global qualification failed: " + json.dumps(gates))

    result = {
        "schema": SCHEMA,
        "command": "merge-global-v2",
        "status": "QUALIFIED_R1B2_EVENT_LEVEL_SERVICE_POSTERIOR_SUBSTRATE",
        "service_date": "2019-01-04",
        "scope": "FULL_SERVICE_DAY_0400_TO_NEXT_0400_FULL_NETWORK_FULL_QUALIFIED_PASSENGER_DOMAIN",
        "service_initialization_anchor_event_count": 19236,
        "candidate_root_event_state_count": source_event_assignment_count,
        "semantic_note_on_event_counts": "19236 is the qualified AFC passenger-facing initialization anchor count. candidate_root_event_state_count is the latent root-event inventory used by the event-level service posterior. They are different scientific objects and must not be equated.",
        "roots_total": len(roots),
        "roots_with_genealogy_support": roots_with_support,
        "roots_without_genealogy_support": roots_without_support,
        "root_support_share": roots_with_support / len(roots) if roots else None,
        "events_with_genealogy_traversal_support": source_event_assignment_count - zero_support_events,
        "events_without_current_genealogy_traversal_support": zero_support_events,
        "event_support_share": (source_event_assignment_count - zero_support_events) / source_event_assignment_count if source_event_assignment_count else None,
        "genealogy_traversal_support_mass_quantiles": quantiles(supported_event_masses),
        "mapped_resolved_genealogy_mass": mapped,
        "unresolved_genealogy_mass": unresolved,
        "first_boarding_mass": board,
        "final_alighting_mass": alight,
        "transfer_episode_mass": transfer,
        "mapping_failure_mass": failure_mass,
        "evidence_class_event_counts": dict(evidence_counts),
        "qualification_gates": gates,
        "timing_semantics": "R1B.2 establishes the complete root-event posterior substrate and attaches genealogy evidence mass. It intentionally does not change timing center, uncertainty, or existence probability; R1B.3 performs those backward passenger-evidence updates.",
        "next_stage": "R1B_3_APPLY_GENEALOGY_BASED_BACKWARD_SERVICE_CONSTRAINTS"
    }
    out_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-file", type=Path, action="append", required=True)
    ap.add_argument("--summary", type=Path, action="append", required=True)
    ap.add_argument("--roots", type=Path, required=True)
    ap.add_argument("--genealogy-global", type=Path, required=True)
    ap.add_argument("--out-events", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    a = ap.parse_args()
    result = merge_global(a.event_file, a.summary, a.roots, a.genealogy_global, a.out_events, a.out_summary)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
