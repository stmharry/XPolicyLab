#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${PI05_CONTAINER_WORK_ROOT:-/workspace/pi05-piper}
POLICY_ROOT="${WORK_ROOT}/RoboDojo/XPolicyLab/policy/Pi_05"
OPENPI_ROOT="${POLICY_ROOT}/openpi"
RUN_NAME="RoboDojo-real_piper_6task-bimanual_piper-joint-0"

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
  --dataset-root "${HF_LEROBOT_HOME}/RoboDojo-real_piper_6task-bimanual_piper-joint" \
  --checkpoint-root "${WORK_ROOT}/checkpoints/${RUN_NAME}" \
  --assets-base-dir "${WORK_ROOT}/assets" \
  --output "${WORK_ROOT}/manifests/${RUN_NAME}-offline-validation.json"
