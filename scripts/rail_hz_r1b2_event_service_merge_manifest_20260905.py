from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "rail.hz-r1b2-event-service-posterior.v4-manifest"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def event_key(root_id: str, event_index: int) -> str:
    return f"{root_id}::e{event_index:03d}"


def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "median": None, "p90": None}
    xs = sorted(values)
    def q(p: float) -> float:
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return float(xs[lo])
        w = pos - lo
        return float(xs[lo] * (1 - w) + xs[hi] * w)
    return {"p10": q(0.10), "median": q(0.50), "p90": q(0.90)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-jsonl", type=Path, action="append", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--roots", type=Path, required=True)
    ap.add_argument("--genealogy-global", type=Path, required=True)
    ap.add_argument("--out-events", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    a = ap.parse_args()

    manifest = load_json(a.manifest)
    roots_raw = load_json(a.roots)
    genealogy = load_json(a.genealogy_global)
    if manifest.get("status") != "QUALIFIED_EIGHT_R1B2_EVENT_EVIDENCE_SHARDS":
        raise SystemExit("R1B2 shard manifest not qualified")
    if len(manifest.get("shards", [])) != 8 or len(a.event_jsonl) != 8:
        raise SystemExit("expected exactly 8 qualified shard records and 8 event JSONL files")
    if roots_raw.get("status") != "QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION":
        raise SystemExit("service roots not qualified")
    roots = roots_raw["roots"]

    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    input_event_rows = 0
    for path in a.event_jsonl:
        for row in iter_jsonl_gz(path):
            input_event_rows += 1
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

    expected_event_states = sum(len(r["events"]) for r in roots)
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    supported_masses: list[float] = []
    zero_support = 0
    root_max_support: dict[str, float] = defaultdict(float)
    evidence_counts = Counter()

    a.out_events.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out_events, "wt", encoding="utf-8") as out:
        for root in roots:
            rid = str(root["root_id"])
            eclass = str(root.get("evidence_class", "UNKNOWN"))
            evidence_counts[eclass] += len(root["events"])
            for idx, ev in enumerate(root["events"]):
                ek = event_key(rid, idx)
                if ek in seen:
                    raise SystemExit(f"duplicate root-event identity: {ek}")
                seen.add(ek)
                x = acc.get(ek, {})
                traversal = float(x.get("traversal", 0.0))
                root_max_support[rid] = max(root_max_support[rid], traversal)
                if traversal > 0:
                    supported_masses.append(traversal)
                    support_state = "GENEALOGY_SUPPORTED_EVENT_SEED"
                else:
                    zero_support += 1
                    support_state = "NO_CURRENT_GENEALOGY_TRAVERSAL_SUPPORT"
                t = float(ev["time_s"]); sd = float(ev["sd_s"])
                row = {
                    "event_key": ek,
                    "root_id": rid,
                    "path_id": str(root["path_id"]),
                    "direction": str(root["direction"]),
                    "event_index": int(idx),
                    "station": int(ev["station"]),
                    "prior_time_s": t,
                    "prior_sd_s": sd,
                    "prior_lower_95_s": t - 1.96 * sd,
                    "prior_upper_95_s": t + 1.96 * sd,
                    "current_timing_center_s": t,
                    "current_timing_sd_s": sd,
                    "timing_stage": "PRE_BACKWARD_CONSTRAINT_UNCHANGED_FROM_QUALIFIED_SERVICE_INITIALIZATION",
                    "evidence_class": eclass,
                    "support_state": support_state,
                    "traversal_lineage_mass": traversal,
                    "boarding_lineage_mass": float(x.get("boarding", 0.0)),
                    "alighting_lineage_mass": float(x.get("alighting", 0.0)),
                    "transfer_arrival_lineage_mass": float(x.get("transfer_arrival", 0.0)),
                    "transfer_departure_lineage_mass": float(x.get("transfer_departure", 0.0)),
                    "lineage_edge_contribution_count": int(x.get("edge_count", 0)),
                }
                inventory.append(row)
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    gc = manifest["global_checks"]
    resolved = float(gc["resolved_lineage_mass"])
    unresolved = float(gc["unresolved_mass"])
    mapped = float(gc["mapped_chain_mass"])
    station_only = float(gc["station_only_mass"])
    first_board = float(gc["first_boarding_mass"])
    final_alight = float(gc["final_alighting_mass"])
    transfer_episode = float(gc["transfer_episode_mass"])
    mapping_failure = float(gc["mapping_failure_mass"])

    roots_with_support = sum(1 for r in roots if root_max_support.get(str(r["root_id"]), 0.0) > 0)
    roots_without_support = [str(r["root_id"]) for r in roots if root_max_support.get(str(r["root_id"]), 0.0) <= 0]
    events_supported = expected_event_states - zero_support

    gates = {
        "eight_event_files_present": len(a.event_jsonl) == 8,
        "eight_shards_scientifically_qualified": bool(gc.get("all_shards_scientifically_qualified")),
        "all_candidate_root_event_states_enumerated_once": len(inventory) == expected_event_states and len(seen) == expected_event_states,
        "resolved_genealogy_mass_matches_r1b1": abs(resolved - float(genealogy["resolved_genealogy_mass"])) <= 1e-6,
        "unresolved_genealogy_mass_matches_r1b1": abs(unresolved - float(genealogy["unresolved_genealogy_mass"])) <= 1e-6,
        "resolved_decomposition_matches": abs((mapped + station_only) - resolved) <= 1e-6,
        "first_boarding_mass_conserved_over_service_chains": abs(first_board - mapped) <= 1e-6,
        "final_alighting_mass_conserved_over_service_chains": abs(final_alight - mapped) <= 1e-6,
        "no_event_mapping_failure_mass": abs(mapping_failure) <= 1e-9,
        "timing_not_silently_changed_in_r1b2": all(
            abs(float(x["current_timing_center_s"]) - float(x["prior_time_s"])) <= 1e-12
            and abs(float(x["current_timing_sd_s"]) - float(x["prior_sd_s"])) <= 1e-12
            for x in inventory
        ),
    }
    if not all(gates.values()):
        raise SystemExit("R1B2 manifest-driven global qualification failed: " + json.dumps(gates))

    result = {
        "schema": SCHEMA,
        "status": "QUALIFIED_R1B2_EVENT_LEVEL_SERVICE_POSTERIOR_SUBSTRATE",
        "service_date": "2019-01-04",
        "scope": "FULL_SERVICE_DAY_0400_TO_NEXT_0400_FULL_NETWORK_FULL_QUALIFIED_PASSENGER_DOMAIN",
        "service_initialization_anchor_event_count": 19236,
        "candidate_root_event_state_count": expected_event_states,
        "semantic_note_on_event_counts": "19236 is the qualified AFC passenger-facing initialization anchor count. candidate_root_event_state_count is the complete latent root-event inventory. They are different scientific objects.",
        "input_event_aggregate_rows": input_event_rows,
        "roots_total": len(roots),
        "roots_with_genealogy_support": roots_with_support,
        "roots_without_genealogy_support": roots_without_support,
        "root_support_share": roots_with_support / len(roots) if roots else None,
        "events_with_genealogy_traversal_support": events_supported,
        "events_without_current_genealogy_traversal_support": zero_support,
        "event_support_share": events_supported / expected_event_states if expected_event_states else None,
        "genealogy_traversal_support_mass_quantiles": quantiles(supported_masses),
        "resolved_genealogy_mass": resolved,
        "mapped_service_chain_mass": mapped,
        "station_only_mass": station_only,
        "unresolved_genealogy_mass": unresolved,
        "first_boarding_mass": first_board,
        "final_alighting_mass": final_alight,
        "transfer_episode_mass": transfer_episode,
        "mapping_failure_mass": mapping_failure,
        "evidence_class_event_counts": dict(evidence_counts),
        "qualification_gates": gates,
        "timing_semantics": "R1B.2 establishes the event-level service posterior substrate and attaches genealogy evidence mass. It intentionally does not alter event timing, event uncertainty, or event-existence probability; R1B.3 performs backward passenger-evidence updates.",
        "next_stage": "R1B_3_APPLY_GENEALOGY_BASED_BACKWARD_SERVICE_CONSTRAINTS"
    }
    a.out_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
