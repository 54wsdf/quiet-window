import json
import math
import sys
from pathlib import Path

import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base

KERNEL_DAMPING = 0.35

_orig_route_beam_joint = base.route_beam_joint
_orig_update_kernels = base.update_kernels


def blend_kernel(before, raw_after, eta, evidence_class):
    b = before["components"][0]
    a = raw_after["components"][0]
    median = math.exp(
        (1.0 - eta) * math.log(max(1e-6, float(b["median_sec"])))
        + eta * math.log(max(1e-6, float(a["median_sec"])))
    )
    sigma = math.exp(
        (1.0 - eta) * math.log(max(1e-6, float(b["sigma"])))
        + eta * math.log(max(1e-6, float(a["sigma"])))
    )
    return base.kernel_from_median_sigma(median, sigma, evidence_class)


def route_beam_joint_v2(cand, meta, rides_fn, tin, tout, beam, kernels, max_skip):
    legs = [x for x in base.r0.path_legs(cand["path"], meta) if x[1] != x[2]]
    if legs:
        return _orig_route_beam_joint(cand, meta, rides_fn, tin, tout, beam, kernels, max_skip)

    le = base.egress_logdensity_kernel(tout, tin, 0.0, kernels["egress"])
    if not math.isfinite(le):
        return [], {"reason": "STATION_ONLY_PROXY_INCOMPATIBLE"}
    return [{
        "logp": le,
        "chain": [],
        "factors": [{
            "type": "EGRESS",
            "exit_time": tout,
            "arr": tout,
            "arr_sd": 0.0,
            "arr_root": None,
            "fit_eligible": False,
            "station_only_proxy": True,
            "proxy_ready_time": tin,
        }],
    }], None


def update_kernels_v2(factors, kernels):
    proxy_egress = [
        f for f in factors
        if f.get("type") == "EGRESS"
        and (f.get("station_only_proxy") or f.get("fit_eligible") is False)
    ]
    fit_factors = [
        f for f in factors
        if not (
            f.get("type") == "EGRESS"
            and (f.get("station_only_proxy") or f.get("fit_eligible") is False)
        )
    ]
    raw, diag = _orig_update_kernels(fit_factors, kernels)
    updated = {
        "schema": "mppd.r1b-kernels.v2-damped-mixture-capable-k1-first-pass",
        "access": blend_kernel(kernels["access"], raw["access"], KERNEL_DAMPING, "R1B_DAMPED_POSTERIOR_UPDATE"),
        "egress": blend_kernel(kernels["egress"], raw["egress"], KERNEL_DAMPING, "R1B_DAMPED_POSTERIOR_UPDATE"),
        "transfer_global": blend_kernel(kernels["transfer_global"], raw["transfer_global"], KERNEL_DAMPING, "R1B_DAMPED_POSTERIOR_UPDATE"),
        "transfer_by_movement": {},
    }
    for movement, raw_kernel in raw.get("transfer_by_movement", {}).items():
        updated["transfer_by_movement"][movement] = blend_kernel(
            updated["transfer_global"], raw_kernel, KERNEL_DAMPING, "R1B_DAMPED_MOVEMENT_UPDATE"
        )

    for key, kernel in (
        ("access", updated["access"]),
        ("egress", updated["egress"]),
        ("transfer_global", updated["transfer_global"]),
    ):
        d = diag.get(key, {})
        d["damping"] = KERNEL_DAMPING
        d["raw_mle_median_sec"] = d.get("median_sec")
        d["raw_mle_sigma"] = d.get("sigma")
        d["damped_median_sec"] = kernel["components"][0]["median_sec"]
        d["damped_sigma"] = kernel["components"][0]["sigma"]

    for item in diag.get("top_transfer_movements", []):
        movement = item.get("movement")
        if movement in updated["transfer_by_movement"]:
            k = updated["transfer_by_movement"][movement]["components"][0]
            item["raw_mle_median_sec"] = item.get("median_sec")
            item["raw_mle_sigma"] = item.get("sigma")
            item["damped_median_sec"] = k["median_sec"]
            item["damped_sigma"] = k["sigma"]
    diag["kernel_damping"] = KERNEL_DAMPING
    diag["station_only_egress_proxy_filter"] = {
        "excluded_factor_count": len(proxy_egress),
        "excluded_posterior_weight": sum(float(f.get("weight", 0.0)) for f in proxy_egress),
        "policy": "EXCLUDE_STATION_ONLY_OR_FIT_INELIGIBLE_EGRESS_FROM_THETA_E_MSTEP",
    }
    diag["update_role"] = "DAMPED_ONE_STEP_R1B_SMOKE_NOT_R1C_FINAL_KERNEL"
    return updated, diag


def find_out_dir(argv):
    for i, arg in enumerate(argv):
        if arg == "--out" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--out="):
            return Path(arg.split("=", 1)[1])
    return None


def main():
    base.route_beam_joint = route_beam_joint_v2
    base.update_kernels = update_kernels_v2
    base.main()

    out = find_out_dir(sys.argv[1:])
    if out is None:
        return
    path = out / "r1b_first_bidirectional_joint_update_smoke_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = "mppd.r1b-first-bidirectional-joint-update-smoke.v2"
    payload["implementation_patch"] = {
        "wrapper": "scripts/mppd_r1b_first_bidirectional_joint_update_v2_20260904.py",
        "kernel_damping": KERNEL_DAMPING,
        "station_only_proxy_regression_fixed": True,
        "station_only_proxy_excluded_from_egress_kernel_fit": True,
    }
    payload.setdefault("scope", {})["kernel_damping"] = KERNEL_DAMPING
    payload.setdefault("scientific_boundary", []).extend([
        "Kernel M-step changes are damped in log-parameter space so one approximate E-step cannot collapse station/transfer propagation times into service-timing error.",
        "Route candidates with no in-vehicle leg after physical-station mapping preserve the predecessor G2 station-only proxy likelihood and do not create a new egress failure class.",
        "Station-only proxy factors are excluded from egress-kernel fitting and egress-tail diagnostics.",
    ])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
