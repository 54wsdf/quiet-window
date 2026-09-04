import gzip
import json
import sys
from pathlib import Path

import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base
import scripts.mppd_r1b_first_bidirectional_joint_update_v3_weighted_failures_20260904 as v3

PASS_COHORT_CAPTURES = []
_ORIG_SUMMARIZE_PASS = base.summarize_pass


def summarize_pass_capture(pass_result):
    PASS_COHORT_CAPTURES.append(pass_result.get("cohorts", {}))
    return _ORIG_SUMMARIZE_PASS(pass_result)


def find_out_dir(argv):
    for i, arg in enumerate(argv):
        if arg == "--out" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--out="):
            return Path(arg.split("=", 1)[1])
    return None


def parse_cohort_id(cohort_id):
    parts = cohort_id.split("|")
    if len(parts) < 4:
        return {
            "cohort_id": cohort_id,
            "origin_code": None,
            "destination_code": None,
            "entry_time": None,
            "exit_time": None,
        }
    return {
        "cohort_id": cohort_id,
        "origin_code": parts[0],
        "destination_code": parts[1],
        "entry_time": parts[2],
        "exit_time": "|".join(parts[3:]),
    }


def write_residual_file(path, cohorts, pass_name):
    count = 0
    mass = 0.0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for cohort_id, row in cohorts.items():
            if row.get("status") == "FINITE_POSTERIOR":
                continue
            rec = parse_cohort_id(cohort_id)
            rec.update({
                "pass": pass_name,
                "status": row.get("status"),
                "passenger_mass": row.get("mass"),
                "failure_reason_probs": row.get("failure_reason_probs"),
            })
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
            mass += float(row.get("mass", 0.0))
    return {"cohort_count": count, "passenger_mass": mass, "file": path.name}


def main():
    base.summarize_pass = summarize_pass_capture
    v3.main()

    out = find_out_dir(sys.argv[1:])
    if out is None:
        return
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "r1b_first_bidirectional_joint_update_smoke_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "mppd.r1b-first-bidirectional-joint-update-smoke.v3-weighted-failures":
        raise RuntimeError(f"unexpected inherited schema: {payload.get('schema')}")
    if len(PASS_COHORT_CAPTURES) != 2:
        raise RuntimeError(f"expected 2 passenger-pass captures, got {len(PASS_COHORT_CAPTURES)}")

    e0_meta = write_residual_file(
        out / "r1b_weighted_failure_residual_cohorts_e0.jsonl.gz",
        PASS_COHORT_CAPTURES[0],
        "E0",
    )
    e1_meta = write_residual_file(
        out / "r1b_weighted_failure_residual_cohorts_e1.jsonl.gz",
        PASS_COHORT_CAPTURES[1],
        "E1",
    )

    payload["schema"] = "mppd.r1b-first-bidirectional-joint-update-smoke.v4-residual-records"
    payload["implementation_patch"] = {
        "wrapper": "scripts/mppd_r1b_first_bidirectional_joint_update_v4_residual_records_20260904.py",
        "inherits_v3_weighted_failure_attribution": True,
        "inherits_v2_kernel_damping": 0.35,
        "station_only_proxy_regression_fixed": True,
        "residual_cohort_records_persisted": True,
        "candidate_multiplicity_majority_vote_removed": True,
    }
    payload["residual_record_outputs"] = {
        "E0": e0_meta,
        "E1": e1_meta,
        "record_fields": [
            "cohort_id",
            "origin_code",
            "destination_code",
            "entry_time",
            "exit_time",
            "passenger_mass",
            "failure_reason_probs",
        ],
        "purpose": "TARGETED_TEMPORAL_AND_CROSS_OD_REPLAY_FOR_SERVICE_SUPPORT_QUALIFICATION",
    }
    payload.setdefault("scientific_boundary", []).extend([
        "Residual cohort records preserve model-based fractional failure probabilities and observed AFC time windows; they do not assign an observed behavioral or operational cause.",
        "Missing-segment route responsibilities are intentionally not fabricated here; a targeted replay on only these residual cohorts must recover candidate-specific missing-service support with the same structural prior weights.",
        "Persisting residual cohorts is an auditability/provenance enhancement and does not change the R1B posterior itself.",
    ])
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
