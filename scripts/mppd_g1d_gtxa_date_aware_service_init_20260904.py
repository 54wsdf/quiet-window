import argparse
import json
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_g1_full_network_service_bootstrap_20260904 as g1
import scripts.mppd_g1b_full_network_weak_latent_service_init_20260904 as g1b
from scripts.mppd_gtxa_crosswalk_overlay_20260904 import apply_gtxa_overlay, load_overlay
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch

QUALIFIED_SECTION_SEC = 121.0
QUALIFIED_GLOBAL_HEADWAY_SEC = 407.25


def component_seq_nodes(code_to_nodes, stations):
    out = {}
    for code, seq in stations:
        nodes = list(code_to_nodes.get(code, []))
        if len(nodes) != 1:
            raise ValueError(f"GTX-A component code must resolve to exactly one node: {code}: {nodes}")
        out[int(seq)] = nodes
    return out


def candidate_to_state(cand, component_id):
    direction = cand["direction"]
    old_id = str(cand["candidate_id"])
    suffix = old_id.rsplit("_", 1)[-1]
    sid = f"WEAK_1032_{component_id}_{direction}_{suffix}"
    return {
        "service_id": sid,
        "root_service_id": sid,
        "line": "1032",
        "direction": direction,
        "service_component_id": component_id,
        "evidence_class": "AFC_INFERRED_SERVICE_FIELD_WEAK_LATTICE_INITIALIZATION",
        "timing_uncertainty_sec": float(cand["timing_uncertainty_sec"]),
        "headway_sec": int(cand["headway_sec"]),
        "phase_sec": int(cand["phase_sec"]),
        "lattice_score": cand.get("lattice_score"),
        "station_events": [
            {"node": e["node"], "arrival": e["time"], "departure": e["time"]}
            for e in cand.get("station_events", [])
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taims", required=True)
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--topology-patch", required=True)
    ap.add_argument("--gtxa-overlay", required=True)
    ap.add_argument("--service-init-v2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    predecessor = json.loads(Path(args.service_init_v2).read_text(encoding="utf-8"))

    G, meta, code_to_nodes, _, _, _, _ = g0.build_network(args.p1c)
    apply_topology_patch(G, meta, load_patch(args.topology_patch))
    overlay = load_overlay(args.gtxa_overlay)
    overlay_result = apply_gtxa_overlay(G, meta, code_to_nodes, overlay)

    old_1032 = [s for s in predecessor.get("states", []) if str(s.get("line")) == "1032"]
    forbidden_classes = [
        s.get("evidence_class") for s in old_1032
        if s.get("evidence_class") != "AFC_INFERRED_SERVICE_FIELD_WEAK_LATTICE_INITIALIZATION"
    ]
    if forbidden_classes:
        raise RuntimeError(f"refusing to replace non-weak GTX-A service evidence: {forbidden_classes}")
    if len(old_1032) != 49:
        raise RuntimeError(f"expected 49 predecessor weak line-1032 states, found {len(old_1032)}")

    entry_hist, exit_hist, entry_mass, exit_mass, afc_stats = g1.load_afc_hist(args.taims, code_to_nodes, meta)
    component_specs = {
        "GTXA_NORTH_20260829": [("9000", 1), ("9001", 2), ("9002", 3), ("9004", 5), ("9005", 6)],
        "GTXA_SOUTH_20260829": [("9007", 8), ("9008", 9), ("9009", 10), ("9010", 11)],
    }

    replacements = []
    component_summary = []
    for component_id, station_spec in component_specs.items():
        seq_nodes = component_seq_nodes(code_to_nodes, station_spec)
        component_exit_mass = sum(exit_mass[n] for nodes in seq_nodes.values() for n in nodes)
        component_entry_mass = sum(entry_mass[n] for nodes in seq_nodes.values() for n in nodes)
        for direction in ("INC", "DEC"):
            lattice = g1b.choose_lattice(
                "1032", direction, seq_nodes, exit_hist, exit_mass,
                QUALIFIED_SECTION_SEC, QUALIFIED_GLOBAL_HEADWAY_SEC
            )
            candidates = g1b.generate_events(
                "1032", direction, seq_nodes, QUALIFIED_SECTION_SEC, lattice
            )
            states = [candidate_to_state(c, component_id) for c in candidates]
            replacements.extend(states)
            component_summary.append({
                "component_id": component_id,
                "direction": direction,
                "station_codes": [c for c, _ in station_spec],
                "station_nodes": [n for _, nodes in sorted(seq_nodes.items()) for n in nodes],
                "entry_mass": component_entry_mass,
                "exit_mass": component_exit_mass,
                "headway_sec": lattice["headway_sec"],
                "phase_sec": lattice["phase_sec"],
                "lattice_score": lattice["score"],
                "candidate_state_count": len(states),
                "evidence_class": lattice["evidence_class"],
            })

    retained = [s for s in predecessor.get("states", []) if str(s.get("line")) != "1032"]
    final_states = retained + replacements
    manifest = dict(predecessor.get("manifest") or {})
    manifest["pre_g1d_state_count"] = len(predecessor.get("states", []))
    manifest["pre_g1d_line1032_state_count"] = len(old_1032)
    manifest["g1d_removed_invalid_cross_component_line1032_weak_states"] = len(old_1032)
    manifest["g1d_date_aware_line1032_replacement_state_count"] = len(replacements)
    manifest["g1d_final_state_count"] = len(final_states)
    manifest["g1d_line1032_operational_components"] = component_summary
    manifest["g1d_gtxa_overlay_schema"] = overlay.get("schema")
    manifest["g1d_afc_hist"] = afc_stats
    manifest["g1d_global_section_runtime_sec"] = QUALIFIED_SECTION_SEC
    manifest["g1d_global_headway_sec"] = QUALIFIED_GLOBAL_HEADWAY_SEC

    output = {
        "schema": "mppd.city-day-service-state-initialization.v3-gtxa-date-aware",
        "date": "2026-09-04",
        "status": "FULL_NETWORK_SERVICE_STATE_INITIALIZATION_WITH_DATE_AWARE_GTXA_COMPONENTS",
        "authority": predecessor.get("authority") or "00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md",
        "city": predecessor.get("city", "Seoul"),
        "business_date": predecessor.get("business_date", "2026-08-29"),
        "time_window": predecessor.get("time_window", "2026-08-29 07:00-10:00"),
        "states": final_states,
        "manifest": manifest,
        "hard_boundaries": list(predecessor.get("hard_boundaries") or []) + [
            "GTX-A line 1032 is represented as two date-aware service components on 2026-08-29: north 운정중앙-서울 and south 수서-동탄.",
            "No weak service root or station-event chain may span GTX-A north and south components on 2026-08-29.",
            "G1D replaces only predecessor weak line-1032 states; no direct or stronger service evidence is overwritten."
        ],
        "gtxa_date_aware_reconciliation": {
            "predecessor_schema": predecessor.get("schema"),
            "predecessor_state_count": len(predecessor.get("states", [])),
            "removed_line1032_weak_state_count": len(old_1032),
            "replacement_line1032_state_count": len(replacements),
            "final_state_count": len(final_states),
            "components": component_summary,
            "overlay_application": overlay_result,
            "scientific_boundary": [
                "Replacement lattices are AFC-informed latent initialization only and are not observed ATS timetables.",
                "The correction prevents a date-incompatible weak service hypothesis from connecting the operationally separate GTX-A north and south sections.",
                "G3 must still move, split, remove or reweight these weak service events under the full-network objective."
            ]
        },
        "no_email_notification_logic": True
    }
    (outdir / "seoul_20260829_0700_1000_service_state_initialization_v3_gtxa_date_aware.json").write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    (outdir / "g1d_gtxa_date_aware_service_init_summary.json").write_text(json.dumps({
        "schema": "mppd.g1d-gtxa-date-aware-service-init.v1",
        "date": "2026-09-04",
        "status": "G1D_GTXA_DATE_AWARE_SERVICE_INITIALIZATION_COMPLETED",
        "predecessor_state_count": len(predecessor.get("states", [])),
        "removed_invalid_line1032_weak_states": len(old_1032),
        "replacement_line1032_states": len(replacements),
        "final_state_count": len(final_states),
        "components": component_summary,
        "afc_hist": afc_stats,
        "qualified_global_section_runtime_sec": QUALIFIED_SECTION_SEC,
        "qualified_global_headway_sec": QUALIFIED_GLOBAL_HEADWAY_SEC,
        "no_email_notification_logic": True
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["gtxa_date_aware_reconciliation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
