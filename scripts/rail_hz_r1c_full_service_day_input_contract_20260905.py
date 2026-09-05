from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(r1b_path: Path, qualification_path: Path, cohorts_path: Path, routes_path: Path, roots_path: Path, output: Path) -> dict[str, Any]:
    r1b = load(r1b_path)
    q = load(qualification_path)
    routes = load(routes_path)
    roots = load(roots_path)
    pf = pq.ParquetFile(cohorts_path)

    service_date = str(r1b.get("service_date"))
    q_date = str(q.get("date"))
    gates = {
        "r1b_execution_completed": r1b.get("status") == "COMPLETED_FORMAL_R1B_FULL_SERVICE_DAY_UPDATE",
        "r1b_scientifically_qualified": q.get("status") == "QUALIFIED_SINGLE_FULL_SERVICE_DAY_R1B",
        "date_agreement": service_date == q_date == str(roots.get("source_date")),
        "full_service_day": r1b.get("scope", {}).get("time") == "FULL_SERVICE_DAY_0400_TO_NEXT_0400",
        "full_network": r1b.get("scope", {}).get("network") == "FULL_NETWORK",
        "no_passenger_subsample": r1b.get("scope", {}).get("passenger_subsample") is False,
        "no_transfer_count_cap": r1b.get("scope", {}).get("transfer_count_cap") is None,
        "no_boarding_skip_cap": r1b.get("scope", {}).get("boarding_skip_count_cap") is None,
        "e0_mass_conservation": bool(r1b.get("mass_conservation", {}).get("E0_pass")),
        "e1_mass_conservation": bool(r1b.get("mass_conservation", {}).get("E1_pass")),
        "positive_full_passenger_mass": float(r1b.get("mass_conservation", {}).get("passenger_mass", 0.0)) > 0,
        "route_support_qualified": routes.get("status") == "QUALIFIED_LINE_AWARE_ROUTE_SUPPORT",
        "service_roots_qualified": roots.get("status") == "QUALIFIED_CANDIDATE_SERVICE_ROOT_COMPLETION",
        "cohort_file_nonempty": int(pf.metadata.num_rows) > 0,
        "qualification_all_hard_gates": all(bool(v) for v in q.get("hard_integrity_gates", {}).values()),
        "qualification_bidirectional_mechanism": bool(q.get("scientific_bidirectional_gates", {}).get("passenger_updates_temporal_kernels")) and bool(q.get("scientific_bidirectional_gates", {}).get("passenger_updates_service_timing")) and bool(q.get("scientific_bidirectional_gates", {}).get("route_or_boarding_redistribution")),
    }

    result = {
        "schema": "rail.hz-r1c-full-service-day-input-contract.v1",
        "service_date": service_date,
        "status": "QUALIFIED_FOR_SINGLE_DAY_FULL_SERVICE_DAY_R1C" if all(gates.values()) else "R1C_INPUT_CONTRACT_NOT_SATISFIED",
        "integrity_gates": gates,
        "inputs": {
            "r1b": r1b_path.name,
            "r1b_qualification": qualification_path.name,
            "cohorts": cohorts_path.name,
            "route_support": routes_path.name,
            "service_roots": roots_path.name,
            "cohort_rows": int(pf.metadata.num_rows),
            "passenger_mass": float(r1b.get("mass_conservation", {}).get("passenger_mass", 0.0)),
        },
        "r1c_required_latent_updates": {
            "theta_a": "GLOBAL_TO_STATION_LINE_DIRECTION_TO_TIME_WITH_HIERARCHICAL_SHRINKAGE",
            "theta_k": "GLOBAL_TO_PHYSICAL_TRANSFER_OR_SERVICE_CHANGE_MOVEMENT_TO_TIME_WITH_HIERARCHICAL_SHRINKAGE",
            "theta_e": "GLOBAL_TO_STATION_LINE_DIRECTION_TO_TIME_WITH_HIERARCHICAL_SHRINKAGE",
            "mixture_capable": True,
            "mixture_forced": False,
            "formal_time_scope": "FULL_SERVICE_DAY_0400_TO_NEXT_0400",
        },
        "launch_policy": "FORMAL_R1C_MUST_NOT_START_UNLESS_STATUS_IS_QUALIFIED_FOR_SINGLE_DAY_FULL_SERVICE_DAY_R1C",
        "scientific_boundary": [
            "R1C consumes the R1B-updated passenger/service world; S^(0) is not an acceptable substitute for the formal R1C launch.",
            "Hourly or adaptive time contexts are conditioning variables inside the full service day, not a peak-window reduction.",
            "Same-line branch service changes remain real inter-leg boarding connections and must be available to Theta_K rather than hidden as through service.",
            "Interchange endpoint surface alignment belongs to access/egress semantics and is not fabricated as an in-journey transfer.",
        ],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(gates.values()):
        raise SystemExit(2)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--r1b", type=Path, required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--cohorts", type=Path, required=True)
    p.add_argument("--routes", type=Path, required=True)
    p.add_argument("--roots", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    validate(a.r1b, a.qualification, a.cohorts, a.routes, a.roots, a.output)


if __name__ == "__main__":
    main()
