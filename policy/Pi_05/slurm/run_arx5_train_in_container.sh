#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${PI05_CONTAINER_WORK_ROOT:-/workspace/pi05-arx5}
POLICY_ROOT="${WORK_ROOT}/RoboDojo/XPolicyLab/policy/Pi_05"
OPENPI_ROOT="${POLICY_ROOT}/openpi"
CKPT_NAME=${PI05_CKPT_NAME:-real_arx_x5_6task}
RUN_NAME="RoboDojo-${CKPT_NAME}-bimanual_arx_x5-joint-0"
RUN_DIR="${WORK_ROOT}/checkpoints/${RUN_NAME}"
TENSORBOARD_DIR="${WORK_ROOT}/tensorboard/${RUN_NAME}"
MANIFEST_PATH="${WORK_ROOT}/manifests/${RUN_NAME}.json"

export PI05_RUNTIME_ROOT="${WORK_ROOT}/runtime"
export PI05_ARX5_WORK_ROOT="${WORK_ROOT}"
export HF_LEROBOT_HOME="${WORK_ROOT}/data/lerobot"
export HF_HOME="${WORK_ROOT}/cache/huggingface"
export XDG_CACHE_HOME="${WORK_ROOT}/cache/xdg"
export OPENPI_ASSETS_BASE_DIR="${WORK_ROOT}/assets"
export OPENPI_CHECKPOINT_ROOT="${WORK_ROOT}/checkpoints"
export OPENPI_TENSORBOARD_DIR="${TENSORBOARD_DIR}"
export OPENPI_TRAIN_CONFIG_NAME="pi05_base_aloha_full_real_arx-x5_seed_0"
export OPENPI_LEROBOT_REPO_ID="RoboDojo-real_arx_x5_6task-bimanual_arx_x5-joint"
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

echo "[Pi_05 ARX X5] TensorBoard node: ${SLURMD_NODENAME}"
echo "[Pi_05 ARX X5] TensorBoard URL on harry-dev: http://127.0.0.1:6007"
echo "[Pi_05 ARX X5] harry-dev tunnel: ssh -N -L 6007:${SLURMD_NODENAME}:6006 gmicloud-loki-g1-cpu-001"
echo "[Pi_05 ARX X5] Mac tunnel: ssh -N -L 6007:127.0.0.1:6007 harry-dev"

cd "${OPENPI_ROOT}"
uv run python ../slurm/write_run_manifest.py \
  --output "${MANIFEST_PATH}" \
  --checkpoint-dir "${RUN_DIR}" \
  --phase starting \
  --run-name "${RUN_NAME}" \
  --job-id "${SLURM_JOB_ID}" \
  --node "${SLURMD_NODENAME}" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --config "${OPENPI_TRAIN_CONFIG_NAME}" \
  --dataset-manifest "${HF_LEROBOT_HOME}/${OPENPI_LEROBOT_REPO_ID}/robodojo_real_arx_x5_manifest.json"

bash "${POLICY_ROOT}/train.sh" RoboDojo "${CKPT_NAME}" bimanual_arx_x5 joint 0 0,1,2,3,4,5,6,7

job_log="${WORK_ROOT}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
job_error_log="${WORK_ROOT}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
if ! grep -q "data_replicas=4, fsdp_group_size=2, total_devices=8" \
  "${job_log}" "${job_error_log}"; then
  echo "[Pi_05 ARX X5] Expected four-replica/two-device FSDP layout was not logged." >&2
  exit 1
fi
uv run python ../slurm/validate_tensorboard.py --log-dir "${TENSORBOARD_DIR}"

num_steps=${OPENPI_NUM_TRAIN_STEPS:-30000}
if [[ "${CKPT_NAME}" == "real_arx_x5_6task" && "${num_steps}" == "30000" ]]; then
  uv run python ../slurm/promote_terminal_checkpoint.py "${RUN_DIR}" \
    --source-step 29999 --target-step 30000
fi

uv run python ../slurm/write_run_manifest.py \
  --output "${MANIFEST_PATH}" \
  --checkpoint-dir "${RUN_DIR}" \
  --phase complete \
  --run-name "${RUN_NAME}" \
  --job-id "${SLURM_JOB_ID}" \
  --node "${SLURMD_NODENAME}" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --config "${OPENPI_TRAIN_CONFIG_NAME}" \
  --dataset-manifest "${HF_LEROBOT_HOME}/${OPENPI_LEROBOT_REPO_ID}/robodojo_real_arx_x5_manifest.json"
