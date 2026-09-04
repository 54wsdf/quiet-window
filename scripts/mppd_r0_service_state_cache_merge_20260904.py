import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", required=True)
    ap.add_argument("--inferred", required=True)
    ap.add_argument("--weak", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    direct = json.loads(Path(args.direct).read_text(encoding="utf-8"))
    inferred = json.loads(Path(args.inferred).read_text(encoding="utf-8"))
    weak = json.loads(Path(args.weak).read_text(encoding="utf-8"))

    states = []
    class_counts = Counter()
    line_counts = defaultdict(Counter)

    for tr in direct.get("trains", []):
        state = {
            "service_id": f"OBS::{tr['line']}::{tr['train']}",
            "line": tr["line"],
            "direction": tr.get("direction"),
            "evidence_class": "PARTIAL_DIRECT_SERVICE_ANCHOR",
            "timing_uncertainty_sec": 0.0,
            "station_events": tr.get("station_events", []),
        }
        states.append(state)
        class_counts[state["evidence_class"]] += 1
        line_counts[state["line"]][state["evidence_class"]] += 1

    for c in inferred.get("candidates", []):
        events = [
            {
                "node": e["node"],
                "arrival": e["time"],
                "departure": e["time"],
            }
            for e in c.get("station_events", [])
        ]
        state = {
            "service_id": c["candidate_id"],
            "line": c["line"],
            "direction": c.get("direction"),
            "evidence_class": "AFC_INFERRED_SERVICE_FIELD",
            "timing_uncertainty_sec": 60.0,
            "ridge_score": c.get("ridge_score"),
            "station_events": events,
        }
        states.append(state)
        class_counts[state["evidence_class"]] += 1
        line_counts[state["line"]][state["evidence_class"]] += 1

    for c in weak.get("candidates", []):
        events = [
            {
                "node": e["node"],
                "arrival": e["time"],
                "departure": e["time"],
            }
            for e in c.get("station_events", [])
        ]
        state = {
            "service_id": c["candidate_id"],
            "line": c["line"],
            "direction": c.get("direction"),
            "evidence_class": c.get("evidence_class", "WEAK_STRUCTURAL_SERVICE_PRIOR"),
            "timing_uncertainty_sec": c.get("timing_uncertainty_sec"),
            "headway_sec": c.get("headway_sec"),
            "phase_sec": c.get("phase_sec"),
            "lattice_score": c.get("lattice_score"),
            "station_events": events,
        }
        states.append(state)
        class_counts[state["evidence_class"]] += 1
        line_counts[state["line"]][state["evidence_class"]] += 1

    states.sort(key=lambda x: (x["line"], str(x.get("direction")), x["service_id"]))
    payload = {
        "schema": "mppd.city-day-service-state-initialization.v1",
        "date": "2026-09-04",
        "status": "FULL_NETWORK_SERVICE_STATE_INITIALIZATION_CACHE",
        "authority": "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "city": "Seoul",
        "business_date": "2026-08-29",
        "time_window": "07:00-10:00",
        "states": states,
        "manifest": {
            "total_state_count": len(states),
            "evidence_class_counts": dict(class_counts),
            "line_evidence_counts": {k: dict(v) for k, v in sorted(line_counts.items())},
        },
        "hard_boundaries": [
            "This file is an initialization cache for the V6 joint state, not reconstructed ATS truth.",
            "PARTIAL_DIRECT_SERVICE_ANCHOR, AFC_INFERRED_SERVICE_FIELD and weak latent priors remain distinct evidence classes.",
            "Weak states must be movable/removable/reweighted in G3.",
            "No line is removed for missing direct service evidence."
        ],
        "no_email_notification_logic": True,
    }
    (outdir / "seoul_20260829_0700_1000_service_state_initialization.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    (outdir / "r0_service_state_cache_merge_summary.json").write_text(
        json.dumps({
            "schema": "mppd.r0-service-state-cache-merge-summary.v1",
            "status": "R0_SERVICE_STATE_CACHE_MERGE_COMPLETED",
            "total_state_count": len(states),
            "evidence_class_counts": dict(class_counts),
            "line_count": len(line_counts),
            "no_email_notification_logic": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
