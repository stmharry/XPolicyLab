#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${PI05_CONTAINER_WORK_ROOT:-/workspace/pi05-piper}
POLICY_ROOT="${WORK_ROOT}/RoboDojo/XPolicyLab/policy/Pi_05"
OPENPI_ROOT="${POLICY_ROOT}/openpi"
CKPT_NAME=${PI05_CKPT_NAME:-real_piper_6task}
RUN_NAME="RoboDojo-${CKPT_NAME}-bimanual_piper-joint-0"
RUN_DIR="${WORK_ROOT}/checkpoints/${RUN_NAME}"
TENSORBOARD_DIR="${WORK_ROOT}/tensorboard/${RUN_NAME}"
MANIFEST_PATH="${WORK_ROOT}/manifests/${RUN_NAME}.json"

export PI05_RUNTIME_ROOT="${WORK_ROOT}/runtime"
export PI05_PIPER_WORK_ROOT="${WORK_ROOT}"
export HF_LEROBOT_HOME="${WORK_ROOT}/data/lerobot"
export HF_HOME="${WORK_ROOT}/cache/huggingface"
export XDG_CACHE_HOME="${WORK_ROOT}/cache/xdg"
export OPENPI_ASSETS_BASE_DIR="${WORK_ROOT}/assets"
export OPENPI_CHECKPOINT_ROOT="${WORK_ROOT}/checkpoints"
export OPENPI_TENSORBOARD_DIR="${TENSORBOARD_DIR}"
export OPENPI_TRAIN_CONFIG_NAME="pi05_base_aloha_full_real_piper_seed_0"
export OPENPI_LEROBOT_REPO_ID="RoboDojo-real_piper_6task-bimanual_piper-joint"
export OPENPI_FSDP_DEVICES=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COMPILATION_CACHE_DIR="/tmp/openpi-jax-${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source "${POLICY_ROOT}/slurm/bootstrap_runtime.sh" "${OPENPI_ROOT}"
mkdir -p "${TENSORBOARD_DIR}" "${WORK_ROOT}/manifests" "${WORK_ROOT}/logs"

(
  cd "${OPENPI_ROOT}"
  uv run tensorboard --logdir "${TENSORBOARD_DIR}" --host 0.0.0.0 --port 6006 \
    >"${WORK_ROOT}/logs/tensorboard-${SLURM_JOB_ID}.log" 2>&1
) &
tensorboard_pid=$!
trap 'kill "${tensorboard_pid}" 2>/dev/null || true' EXIT

echo "[Pi_05] TensorBoard node: ${SLURMD_NODENAME}"
echo "[Pi_05] TensorBoard URL: http://localhost:6006"
echo "[Pi_05] Tunnel: ssh -N -L 6006:${SLURMD_NODENAME}:6006 gmicloud-loki-g1-cpu-001"

cd "${OPENPI_ROOT}"
uv run python ../slurm/write_run_manifest.py \
  --output "${MANIFEST_PATH}" \
  --checkpoint-dir "${RUN_DIR}" \
  --phase starting \
  --run-name "${RUN_NAME}" \
  --job-id "${SLURM_JOB_ID}" \
  --node "${SLURMD_NODENAME}" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --dataset-manifest "${HF_LEROBOT_HOME}/${OPENPI_LEROBOT_REPO_ID}/robodojo_real_piper_manifest.json"

bash "${POLICY_ROOT}/train.sh" RoboDojo "${CKPT_NAME}" bimanual_piper joint 0 0,1,2,3,4,5,6,7

uv run python ../slurm/validate_tensorboard.py --log-dir "${TENSORBOARD_DIR}"
uv run python ../slurm/write_run_manifest.py \
  --output "${MANIFEST_PATH}" \
  --checkpoint-dir "${RUN_DIR}" \
  --phase complete \
  --run-name "${RUN_NAME}" \
  --job-id "${SLURM_JOB_ID}" \
  --node "${SLURMD_NODENAME}" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --dataset-manifest "${HF_LEROBOT_HOME}/${OPENPI_LEROBOT_REPO_ID}/robodojo_real_piper_manifest.json"
