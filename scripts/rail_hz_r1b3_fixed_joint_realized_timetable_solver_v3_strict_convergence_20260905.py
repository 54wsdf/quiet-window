from __future__ import annotations

from scipy.optimize import minimize as scipy_minimize

import scripts.rail_hz_r1b3_fixed_joint_realized_timetable_solver_20260905 as base
import scripts.rail_hz_r1b3_fixed_joint_realized_timetable_solver_v2_dtypefix_20260905 as dtypefix


def minimize_strict(fun, x0, *args, **kwargs):
    options = dict(kwargs.pop("options", {}) or {})
    options.update({
        "maxiter": 1200,
        "ftol": 1e-13,
        "gtol": 1e-6,
        "maxls": 60,
        "maxcor": 30,
    })
    return scipy_minimize(fun, x0, *args, options=options, **kwargs)


def main():
    base.structural_factors = dtypefix.structural_factors_fixed
    base.minimize = minimize_strict
    base.main()


if __name__ == "__main__":
    main()
