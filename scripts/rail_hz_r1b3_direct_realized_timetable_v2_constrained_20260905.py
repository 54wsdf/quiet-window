from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import scripts.rail_hz_r1b3_direct_realized_timetable_20260905 as base

CONSTRAINED_ROOTS: dict[str, list[int]] = {}


def solve_root_constrained(root, ctx):
    events = list(root["events"])
    n = len(events)
    Q = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    observations = 0
    transitions = []

    for j, ev in enumerate(events):
        if bool(ev.get("matched_observed_pulse", False)):
            y = float(ev["time_s"])
            sd = max(1e-6, float(ev["sd_s"]))
            w = 1.0 / (sd * sd)
            Q[j, j] += w
            b[j] += w * y
            observations += 1

    path_id = str(root["path_id"])
    direction = str(root["direction"])
    for j in range(n - 1):
        lag, sd, eclass = base.transition_factor(path_id, direction, events[j], events[j + 1], ctx)
        w = 1.0 / (sd * sd)
        Q[j, j] += w
        Q[j + 1, j + 1] += w
        Q[j, j + 1] -= w
        Q[j + 1, j] -= w
        b[j] -= w * lag
        b[j + 1] += w * lag
        transitions.append({
            "from_station": int(events[j]["station"]),
            "to_station": int(events[j + 1]["station"]),
            "lag_mean_s": lag,
            "lag_sd_s": sd,
            "evidence_class": eclass,
        })

    if observations <= 0:
        raise SystemExit(f"root has no AFC absolute-time anchor: {root['root_id']}")
    Q += np.eye(n) * 1e-12
    mean0 = np.linalg.solve(Q, b)
    cov0 = np.linalg.inv(Q)
    if np.all(np.diff(mean0) >= base.MIN_PROGRESS_S - 1e-7):
        return mean0, np.sqrt(np.maximum(0.0, np.diag(cov0))), transitions, True

    # The scientific factors remain unchanged. This is only the exact physical-domain
    # MAP of the same quadratic posterior under train-forward-time inequalities.
    def objective(x):
        return 0.5 * float(x @ Q @ x) - float(b @ x)

    def gradient(x):
        return Q @ x - b

    def cfun(x):
        return np.diff(x) - base.MIN_PROGRESS_S

    def cjac(_x):
        A = np.zeros((n - 1, n), dtype=float)
        for j in range(n - 1):
            A[j, j] = -1.0
            A[j, j + 1] = 1.0
        return A

    x0 = np.asarray([float(e["time_s"]) for e in events], dtype=float)
    result = minimize(
        objective,
        x0,
        jac=gradient,
        constraints={"type": "ineq", "fun": cfun, "jac": cjac},
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
    )
    if not result.success:
        raise SystemExit(f"constrained realized-timetable MAP failed for {root['root_id']}: {result.message}")
    mean = np.asarray(result.x, dtype=float)
    if not np.all(np.diff(mean) >= base.MIN_PROGRESS_S - 1e-6):
        raise SystemExit(f"constrained MAP still violates monotonicity: {root['root_id']}")

    active = [j for j, gap in enumerate(np.diff(mean)) if gap <= base.MIN_PROGRESS_S + 1e-4]
    cov = cov0
    if active:
        A = np.zeros((len(active), n), dtype=float)
        for row, j in enumerate(active):
            A[row, j] = -1.0
            A[row, j + 1] = 1.0
        middle = A @ cov0 @ A.T
        cov = cov0 - cov0 @ A.T @ np.linalg.pinv(middle) @ A @ cov0
        cov = 0.5 * (cov + cov.T)
    CONSTRAINED_ROOTS[str(root["root_id"])] = active
    return mean, np.sqrt(np.maximum(0.0, np.diag(cov))), transitions, True


def _arg_value(flag: str) -> Path | None:
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv):
        return None
    return Path(sys.argv[i + 1])


def main():
    base.solve_root = solve_root_constrained
    base.main()
    summary_path = _arg_value("--out-summary")
    if summary_path and summary_path.exists() and CONSTRAINED_ROOTS:
        x = json.loads(summary_path.read_text(encoding="utf-8"))
        x["physical_monotonicity_constraint_active_root_count"] = len(CONSTRAINED_ROOTS)
        x["physical_monotonicity_constraint_active_roots"] = [
            {"root_id": rid, "active_edge_indices": edges} for rid, edges in sorted(CONSTRAINED_ROOTS.items())
        ]
        x.setdefault("scientific_semantics", {})["physical_monotonicity_constraint"] = (
            "HARD_FORWARD_TIME_DOMAIN_CONSTRAINT_ON_SAME_FIXED_POSTERIOR_NOT_NEW_EVIDENCE"
        )
        summary_path.write_text(json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({
            "constrained_root_count": len(CONSTRAINED_ROOTS),
            "constrained_roots": sorted(CONSTRAINED_ROOTS),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
