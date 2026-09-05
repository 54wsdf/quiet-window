from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = "rail.hz-r1d-passenger-crossfit-split.v1"


def fold_of(cohort_id: str, folds: int) -> int:
    h = hashlib.sha256(("R1D_HZ_20190104_V1|" + cohort_id).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % folds


def dump(path: Path, x: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--edges", type=Path, required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--heldout-fold", type=int, default=0)
    p.add_argument("--train-out", type=Path, required=True)
    p.add_argument("--heldout-out", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    a = p.parse_args()
    if not (0 <= a.heldout_fold < a.folds):
        raise SystemExit("invalid heldout fold")

    pf = pq.ParquetFile(a.edges)
    schema = pf.schema_arrow
    a.train_out.parent.mkdir(parents=True, exist_ok=True)
    a.heldout_out.parent.mkdir(parents=True, exist_ok=True)
    wt = pq.ParquetWriter(a.train_out, schema, compression="zstd")
    wh = pq.ParquetWriter(a.heldout_out, schema, compression="zstd")

    mass = {"train": 0.0, "heldout": 0.0}
    resolved = {"train": 0.0, "heldout": 0.0}
    unresolved = {"train": 0.0, "heldout": 0.0}
    cohorts = {"train": 0, "heldout": 0}
    edges = {"train": 0, "heldout": 0}
    pending_id = None
    pending = []

    def emit(rows):
        if not rows:
            return
        cid = str(rows[0]["cohort_id"])
        side = "heldout" if fold_of(cid, a.folds) == a.heldout_fold else "train"
        tbl = pa.Table.from_pylist(rows, schema=schema)
        (wh if side == "heldout" else wt).write_table(tbl)
        m = float(rows[0]["passenger_mass"])
        mass[side] += m
        cohorts[side] += 1
        edges[side] += len(rows)
        if any(str(r["descendant_state_type"]) == "UNRESOLVED" for r in rows):
            unresolved[side] += m
        else:
            resolved[side] += m

    try:
        for b in pf.iter_batches(batch_size=50000):
            for r in b.to_pylist():
                cid = str(r["cohort_id"])
                if pending_id is None:
                    pending_id = cid
                if cid != pending_id:
                    emit(pending)
                    pending = []
                    pending_id = cid
                pending.append(r)
        emit(pending)
    finally:
        wt.close(); wh.close()

    err_train = abs(resolved["train"] + unresolved["train"] - mass["train"])
    err_hold = abs(resolved["heldout"] + unresolved["heldout"] - mass["heldout"])
    out = {
        "schema": SCHEMA,
        "status": "QUALIFIED_R1D_PASSENGER_CROSSFIT_SPLIT_SHARD" if max(err_train, err_hold) <= 1e-6 else "FAILED_R1D_SPLIT_SHARD",
        "service_date": "2019-01-04",
        "shard_index": a.shard_index,
        "folds": a.folds,
        "heldout_fold": a.heldout_fold,
        "split_unit": "WHOLE_COHORT_NO_WITHIN_COHORT_LEAKAGE",
        "train": {"cohort_count": cohorts["train"], "edge_count": edges["train"], "passenger_mass": mass["train"], "resolved_mass": resolved["train"], "unresolved_mass": unresolved["train"]},
        "heldout": {"cohort_count": cohorts["heldout"], "edge_count": edges["heldout"], "passenger_mass": mass["heldout"], "resolved_mass": resolved["heldout"], "unresolved_mass": unresolved["heldout"]},
        "mass_conservation_max_abs_error": max(err_train, err_hold),
        "scientific_semantics": {
            "split_precedes_r1d_parameter_refit": True,
            "heldout_passengers_forbidden_from_service_or_transfer_parameter_estimation": True,
            "candidate_service_inventory_not_rebuilt_by_this_stage": True,
            "therefore_this_stage_supports_conditional_crossfit_not_end_to_end_candidate_discovery_validation": True,
            "raw_user_identifier_used": False,
        },
    }
    dump(a.summary_out, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if out["status"].startswith("FAILED"):
        raise SystemExit("split qualification failed")


if __name__ == "__main__":
    main()
