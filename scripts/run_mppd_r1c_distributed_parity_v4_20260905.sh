#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG_FILE:?CONFIG_FILE required}"
: "${REMOTE_NAME:?REMOTE_NAME required}"
: "${RUNNER_TEMP:?RUNNER_TEMP required}"
: "${SEOUL_ROOT:?SEOUL_ROOT required}"
: "${BASE_DRIVE:?BASE_DRIVE required}"
: "${ROUTE_DRIVE:?ROUTE_DRIVE required}"
: "${G1F_DRIVE:?G1F_DRIVE required}"
: "${MONO_DRIVE:?MONO_DRIVE required}"
: "${OUT_DRIVE:?OUT_DRIVE required}"

ROOT="$RUNNER_TEMP/parity_v4"
mkdir -p "$ROOT/e0/shard_00" "$ROOT/kernels" "$ROOT/scores/shard_00" "$ROOT/offsets" "$ROOT/e1/shard_00" "$ROOT/out"

rclone copyto "$REMOTE_NAME:$SEOUL_ROOT/99_manifests/seoul_p1c_line_aware_crosswalk_20260903.json" "$ROOT/p1c.json" --config "$CONFIG_FILE" --retries 6 --low-level-retries 15 --transfers 1
rclone copyto "$REMOTE_NAME:$BASE_DRIVE/g0h_gtxa_corrected_cohort_rebuild/seoul_20260829_0700_1000_afc_time_cohorts_g0h.csv.gz" "$ROOT/cohorts.csv.gz" --config "$CONFIG_FILE" --retries 6 --low-level-retries 15 --transfers 1
rclone copyto "$REMOTE_NAME:$ROUTE_DRIVE/r0_full_network_route_ensemble_candidates.jsonl.gz" "$ROOT/routes.jsonl.gz" --config "$CONFIG_FILE" --retries 6 --low-level-retries 15 --transfers 1
rclone copyto "$REMOTE_NAME:$G1F_DRIVE/seoul_20260829_0700_1000_service_state_initialization_v5_observed_event_preserving_recompletion.json" "$ROOT/service.json" --config "$CONFIG_FILE" --retries 6 --low-level-retries 15 --transfers 1
rclone copyto "$REMOTE_NAME:$MONO_DRIVE/r1b_first_bidirectional_joint_update_smoke_summary.json" "$ROOT/monolithic.json" --config "$CONFIG_FILE" --retries 6 --low-level-retries 15 --transfers 1

python -m py_compile \
  scripts/mppd_r1c_distributed_posterior_shard_v3_raw_movement_20260905.py \
  scripts/mppd_r1c_distributed_kernel_aggregator_v2_raw_movement_20260905.py \
  scripts/mppd_r1c_distributed_offset_score_shard_20260905.py \
  scripts/mppd_r1c_distributed_offset_aggregator_20260905.py \
  scripts/mppd_r1c_distributed_parity_compare_20260905.py \
  scripts/mppd_r1c_distributed_parity_compare_v4_authority_alias_20260905.py

python -m scripts.mppd_r1c_distributed_posterior_shard_v3_raw_movement_20260905 \
  --phase E0 --shard-count 50 --shard-id 0 \
  --p1c "$ROOT/p1c.json" --cohorts "$ROOT/cohorts.csv.gz" --routes "$ROOT/routes.jsonl.gz" --service-init "$ROOT/service.json" \
  --topology-patch manifests/mppd_seoul_topology_patch_g0e_20260904.json \
  --gtxa-overlay manifests/mppd_seoul_gtxa_crosswalk_overlay_g0h_20260904.json \
  --beam 8 --max-skip 2 --out "$ROOT/e0/shard_00"

python - <<'PY'
import gzip,json,os,pathlib
root=pathlib.Path(os.environ['RUNNER_TEMP'])/'parity_v4'
e=json.loads((root/'e0/shard_00/shard_summary.json').read_text())
m=json.loads((root/'monolithic.json').read_text())
assert e['scope']['shard_cohort_count']==11061 and e['scope']['shard_passenger_mass']==12457
assert e['posterior']['finite_posterior_mass']==m['iteration']['E0_passenger_posterior']['finite_posterior_mass']
assert e['posterior']['failure_mass']==m['iteration']['E0_passenger_posterior']['failure_mass']
assert e['sufficient_statistics']['relative_time_round_digits']==9
raw={}
with gzip.open(root/'e0/shard_00/e0_kernel_sufficient_stats.jsonl.gz','rt',encoding='utf-8') as f:
    for line in f:
        if line.strip():
            x=json.loads(line)
            if x.get('scope')=='TRANSFER_MOVEMENT_RAW_WEIGHT': raw[x['group']]=float(x['weight'])
assert sum(1 for v in raw.values() if v>=50)==89
PY

python -m scripts.mppd_r1c_distributed_kernel_aggregator_v2_raw_movement_20260905 --stats-root "$ROOT/e0" --out "$ROOT/kernels"
python -m scripts.mppd_r1c_distributed_offset_score_shard_20260905 --offset-stats "$ROOT/e0/shard_00/e0_offset_sufficient_stats.jsonl.gz" --kernels "$ROOT/kernels/global_kernels.json" --out "$ROOT/scores/shard_00"
python -m scripts.mppd_r1c_distributed_offset_aggregator_20260905 --score-root "$ROOT/scores" --e0-root "$ROOT/e0" --service-init "$ROOT/service.json" --min-service-usage 25 --out "$ROOT/offsets"

python -m scripts.mppd_r1c_distributed_posterior_shard_v3_raw_movement_20260905 \
  --phase E1 --shard-count 50 --shard-id 0 \
  --p1c "$ROOT/p1c.json" --cohorts "$ROOT/cohorts.csv.gz" --routes "$ROOT/routes.jsonl.gz" --service-init "$ROOT/service.json" \
  --topology-patch manifests/mppd_seoul_topology_patch_g0e_20260904.json \
  --gtxa-overlay manifests/mppd_seoul_gtxa_crosswalk_overlay_g0h_20260904.json \
  --kernels "$ROOT/kernels/global_kernels.json" --offsets "$ROOT/offsets/global_offsets.json" \
  --beam 8 --max-skip 2 --out "$ROOT/e1/shard_00"

set +e
python -m scripts.mppd_r1c_distributed_parity_compare_v4_authority_alias_20260905 \
  --monolithic "$ROOT/monolithic.json" \
  --kernels "$ROOT/kernels/global_kernels.json" --offsets "$ROOT/offsets/global_offsets.json" \
  --e0-summary "$ROOT/e0/shard_00/shard_summary.json" --e1-summary "$ROOT/e1/shard_00/shard_summary.json" \
  --e0-cohorts "$ROOT/e0/shard_00/e0_cohort_posterior.jsonl.gz" --e1-cohorts "$ROOT/e1/shard_00/e1_cohort_posterior.jsonl.gz" \
  --out "$ROOT/out"
RC=$?
set -e

# Persist evidence even on a genuine parity failure so the next correction is evidence-driven.
test -s "$ROOT/out/parity_result.json" && rclone copyto "$ROOT/out/parity_result.json" "$REMOTE_NAME:$OUT_DRIVE/parity_result.json" --config "$CONFIG_FILE" --retries 6 --low-level-retries 15 --transfers 1 || true
rclone copyto "$ROOT/kernels/global_kernels.json" "$REMOTE_NAME:$OUT_DRIVE/global_kernels.json" --config "$CONFIG_FILE" --retries 6 --low-level-retries 15 --transfers 1
rclone copyto "$ROOT/kernels/kernel_aggregation_diagnostics.json" "$REMOTE_NAME:$OUT_DRIVE/kernel_aggregation_diagnostics.json" --config "$CONFIG_FILE" --retries 6 --low-level-retries 15 --transfers 1
rclone copyto "$ROOT/offsets/global_offsets.json" "$REMOTE_NAME:$OUT_DRIVE/global_offsets.json" --config "$CONFIG_FILE" --retries 6 --low-level-retries 15 --transfers 1
exit "$RC"
