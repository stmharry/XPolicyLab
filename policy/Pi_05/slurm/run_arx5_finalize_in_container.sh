#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${PI05_CONTAINER_WORK_ROOT:-/workspace/pi05-arx5}
POLICY_ROOT="${WORK_ROOT}/RoboDojo/XPolicyLab/policy/Pi_05"
OPENPI_ROOT="${POLICY_ROOT}/openpi"
RUN_NAME="RoboDojo-real_arx_x5_6task-bimanual_arx_x5-joint-0"

export PI05_RUNTIME_ROOT="${WORK_ROOT}/runtime"
export HF_LEROBOT_HOME="${WORK_ROOT}/data/lerobot"
export HF_HOME="${WORK_ROOT}/cache/huggingface"
export XDG_CACHE_HOME="${WORK_ROOT}/cache/xdg"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export CUDA_VISIBLE_DEVICES=0

source "${POLICY_ROOT}/slurm/bootstrap_runtime.sh" "${OPENPI_ROOT}"

cd "${OPENPI_ROOT}"
uv run python ../slurm/validate_tensorboard.py \
  --log-dir "${WORK_ROOT}/tensorboard/${RUN_NAME}"
uv run python ../slurm/validate_final_checkpoint.py \
  --config pi05_base_aloha_full_real_arx-x5_seed_0 \
  --repo-id RoboDojo-real_arx_x5_6task-bimanual_arx_x5-joint \
  --dataset-root "${HF_LEROBOT_HOME}/RoboDojo-real_arx_x5_6task-bimanual_arx_x5-joint" \
  --checkpoint-root "${WORK_ROOT}/checkpoints/${RUN_NAME}" \
  --assets-base-dir "${WORK_ROOT}/assets" \
  --expected-checkpoint-step 30000 \
  --model-action-dim 32 \
  --output "${WORK_ROOT}/manifests/${RUN_NAME}-offline-validation.json"
uv run python ../slurm/finalize_run_manifest.py \
  --manifest "${WORK_ROOT}/manifests/${RUN_NAME}.json" \
  --pipeline-jobs "${WORK_ROOT}/pipeline_jobs.json" \
  --offline-validation "${WORK_ROOT}/manifests/${RUN_NAME}-offline-validation.json" \
  --checkpoint-root "${WORK_ROOT}/checkpoints/${RUN_NAME}" \
  --checkpoint-step 30000 \
  --tensorboard-dir "${WORK_ROOT}/tensorboard/${RUN_NAME}"
