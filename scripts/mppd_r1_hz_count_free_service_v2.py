from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import scripts.mppd_r1_hz_count_free_service as base

SCHEMA = "mppd.r1-hz-count-free-service-discovery.v2"


def event_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    aa = set(a["event_ids"])
    bb = set(b["event_ids"])
    return len(aa & bb) / max(1, min(len(aa), len(bb))) if aa and bb else 0.0


def deduplicate_shared_event_candidates(
    candidates: list[dict[str, Any]],
    overlap_threshold: float = 0.60,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Collapse rival service hypotheses using shared AFC station-pulse evidence.

    Path-relative reference times are deliberately ignored here. B_main:Up and
    B_branch:Up have different reference terminals, so comparing their reference
    timestamps can leave two hypotheses for the same passenger-facing train.
    Event identity (station + pulse time) is invariant across path and direction
    hypotheses and is therefore used for competition/merge.
    """
    operations: Counter[str] = Counter()
    by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_line[str(c["afc_line"])].append(c)

    final: list[dict[str, Any]] = []
    for _line, vals in by_line.items():
        vals = sorted(vals, key=lambda z: (-float(z["evidence_score"]), -int(z["station_count"]), -float(z["support_weight"])))
        used: set[int] = set()
        for i, seed in enumerate(vals):
            if i in used:
                continue
            family = [seed]
            used.add(i)
            changed = True
            while changed:
                changed = False
                for j, candidate in enumerate(vals):
                    if j in used:
                        continue
                    if max(event_overlap(member, candidate) for member in family) >= overlap_threshold:
                        family.append(candidate)
                        used.add(j)
                        changed = True

            best = max(
                family,
                key=lambda z: (int(z["station_count"]), float(z["evidence_score"]), float(z["support_weight"])),
            )
            merged = dict(best)
            merged["path_alternatives"] = sorted(
                [
                    {
                        "path_id": str(x["path_id"]),
                        "direction": str(x["direction"]),
                        "support_weight": float(x["support_weight"]),
                        "station_count": int(x["station_count"]),
                        "evidence_score": float(x["evidence_score"]),
                    }
                    for x in family
                ],
                key=lambda z: -z["evidence_score"],
            )
            merged["path_ambiguous"] = len({str(x["path_id"]) for x in family}) > 1
            merged["direction_alternatives"] = sorted({str(x["direction"]) for x in family})
            merged["direction_ambiguous"] = len(merged["direction_alternatives"]) > 1
            if len(family) > 1:
                operations["merge_shared_event_duplicate"] += len(family) - 1
                if merged["path_ambiguous"]:
                    operations["path_reassignment_competition"] += 1
                if merged["direction_ambiguous"]:
                    operations["direction_reassignment_competition"] += 1
            final.append(merged)

    final.sort(key=lambda z: (str(z["afc_line"]), str(z["direction"]), float(z["reference_time_s"])))
    return final, operations


def materialize_with_ambiguity(
    candidates: list[dict[str, Any]],
    offsets_by_pd: dict[tuple[str, str], dict[int, float]],
    station_phase: dict[int, float],
) -> list[dict[str, Any]]:
    out = []
    for idx, c in enumerate(candidates):
        offsets = offsets_by_pd[(str(c["path_id"]), str(c["direction"]))]
        t0 = float(c["reference_time_s"])
        events = []
        for seq, (station, off) in enumerate(offsets.items()):
            events.append(
                {
                    "station": int(station),
                    "sequence_index": int(seq),
                    "time_s": t0 + float(off),
                    "station_phase_nuisance_s": float(station_phase.get(station, 0.0)),
                }
            )
        out.append(
            {
                "trajectory_id": f"cf2:{c['afc_line']}:{c['direction']}:{idx}:{int(round(t0))}",
                "afc_line": str(c["afc_line"]),
                "path_id": str(c["path_id"]),
                "path_alternatives": c.get("path_alternatives", []),
                "path_ambiguous": bool(c.get("path_ambiguous", False)),
                "direction": str(c["direction"]),
                "direction_alternatives": c.get("direction_alternatives", [str(c["direction"])]),
                "direction_ambiguous": bool(c.get("direction_ambiguous", False)),
                "reference_time_s": t0,
                "support_station_count": int(c["station_count"]),
                "support_event_count": int(c["event_count"]),
                "support_event_ids": sorted(str(x) for x in c.get("event_ids", [])),
                "support_weight": float(c["support_weight"]),
                "evidence_score": float(c["evidence_score"]),
                "objective_gain": float(c["objective_gain"]),
                "residual_median_abs_s": c["residual_median_abs_s"],
                "residual_p90_abs_s": c["residual_p90_abs_s"],
                "events": events,
            }
        )
    return out


# Patch only the representation-competition layer. The AFC event detector,
# AFC-only edge-lag inference, birth/death/split logic and complexity objective
# remain exactly the v1 engine used in the first full-day probe.
base.SCHEMA = SCHEMA
base.deduplicate_path_candidates = deduplicate_shared_event_candidates
base.materialize_trajectories = materialize_with_ambiguity


if __name__ == "__main__":
    base.main()
