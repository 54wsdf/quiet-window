from __future__ import annotations

import numpy as np

import rail_hz_r1d_train_realized_timetable_solver_20260906 as base


def fixed_train_structural_factors(roots, service_init, model):
    pu_i, pu_y, pu_c, tr_i, tr_j, tr_lag, tr_c = base.b.structural_factors(roots, service_init, model)
    return (
        np.asarray([], dtype=np.int32),
        np.asarray([], dtype=float),
        np.asarray([], dtype=float),
        np.asarray(tr_i, dtype=np.int32),
        np.asarray(tr_j, dtype=np.int32),
        np.asarray(tr_lag, dtype=float),
        np.asarray(tr_c, dtype=float),
    )


base.train_structural_factors = fixed_train_structural_factors


if __name__ == "__main__":
    base.main()
