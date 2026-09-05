from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(r1b_path: Path, qualification_path: Path, output: Path) -> dict[str, Any]:
    r1b = load(r1b_path)
    q = load(qualification_path)
    e0 = r1b["E0"]
    e1 = r1b["E1"]
    service = r1b["service_timing_update"]
    before = r1b["kernels_before"]
    after = r1b["kernels_after"]
    mass = float(r1b["mass_conservation"]["passenger_mass"])

    def delta_kernel(name: str) -> dict[str, Any]:
        b, a = before[name], after[name]
        return {
            "median_before_sec": float(b["median_sec"]),
            "median_after_sec": float(a["median_sec"]),
            "median_change_sec": float(a["median_sec"]) - float(b["median_sec"]),
            "sigma_before": float(b["sigma"]),
            "sigma_after": float(a["sigma"]),
            "sigma_change": float(a["sigma"]) - float(b["sigma"]),
        }

    result = {
        "schema": "rail.hz-single-full-service-day-r1b-science-report.v1",
        "service_date": r1b["service_date"],
        "qualification_status": q["status"],
        "execution_status": r1b["status"],
        "passenger_mass": mass,
        "posterior_coverage": {
            "E0_finite_mass": float(e0["finite_posterior_mass"]),
            "E0_finite_share": float(e0["finite_posterior_share"]),
            "E1_finite_mass": float(e1["finite_posterior_mass"]),
            "E1_finite_share": float(e1["finite_posterior_share"]),
            "finite_share_change": float(e1["finite_posterior_share"]) - float(e0["finite_posterior_share"]),
            "E0_no_route_support_mass": float(e0["no_route_support_mass"]),
            "E0_time_or_service_incompatible_mass": float(e0["time_or_service_incompatible_mass"]),
            "E1_no_route_support_mass": float(e1["no_route_support_mass"]),
            "E1_time_or_service_incompatible_mass": float(e1["time_or_service_incompatible_mass"]),
            "became_finite_mass": float(e1["became_finite_mass"]),
            "lost_finite_mass": float(e1["lost_finite_mass"]),
        },
        "passenger_posterior_redistribution": {
            "top_route_changed_mass": float(e1["top_route_changed_mass"]),
            "top_route_changed_share_of_all_mass": float(e1["top_route_changed_mass"]) / mass if mass else None,
            "top_route_changed_share_among_finite_both": e1.get("top_route_changed_share_among_finite_both"),
            "top_boarding_chain_changed_mass": float(e1["top_boarding_chain_changed_mass"]),
            "top_boarding_chain_changed_share_of_all_mass": float(e1["top_boarding_chain_changed_mass"]) / mass if mass else None,
            "top_boarding_chain_changed_share_among_finite_both": e1.get("top_boarding_chain_changed_share_among_finite_both"),
            "route_entropy_E0": e0.get("weighted_mean_route_entropy"),
            "route_entropy_E1": e1.get("weighted_mean_route_entropy"),
            "route_entropy_change_among_finite_both": e1.get("weighted_mean_route_entropy_change_among_finite_both"),
        },
        "service_world_update": {
            "root_count": int(service["root_count"]),
            "roots_shifted_nonzero": int(service["roots_shifted_nonzero"]),
            "root_shifted_share": float(service["root_shifted_share"]),
            "usage_weighted_mean_abs_shift_sec": service.get("usage_weighted_mean_abs_shift_sec"),
            "offset_histogram": service["offset_histogram"],
        },
        "temporal_kernel_update": {
            "access": delta_kernel("access"),
            "transfer": delta_kernel("transfer"),
            "egress": delta_kernel("egress"),
        },
        "mechanism_evidence": r1b["bidirectional_evidence"],
        "mass_conservation": r1b["mass_conservation"],
        "interpretation_flags": {
            "coverage_is_not_accuracy": True,
            "service_roots_are_latent_not_observed_ats": True,
            "small_update_magnitude_is_not_automatic_failure": True,
            "single_day_r1b_is_engineering_and_mechanism_qualification_before_r1c_and_r1bc": True,
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--r1b", type=Path, required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    x = build(a.r1b, a.qualification, a.output)
    print(json.dumps({
        "service_date": x["service_date"],
        "qualification_status": x["qualification_status"],
        "E0_finite_share": x["posterior_coverage"]["E0_finite_share"],
        "E1_finite_share": x["posterior_coverage"]["E1_finite_share"],
        "roots_shifted_nonzero": x["service_world_update"]["roots_shifted_nonzero"],
        "top_route_changed_mass": x["passenger_posterior_redistribution"]["top_route_changed_mass"],
        "top_boarding_chain_changed_mass": x["passenger_posterior_redistribution"]["top_boarding_chain_changed_mass"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
