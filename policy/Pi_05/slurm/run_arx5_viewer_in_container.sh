#!/usr/bin/env bash
set -euo pipefail

TENSORBOARD_DIR=${1:?TensorBoard event directory is required}
WORK_ROOT=${PI05_CONTAINER_WORK_ROOT:-/workspace/pi05-arx5}
OPENPI_ROOT="${WORK_ROOT}/RoboDojo/XPolicyLab/policy/Pi_05/openpi"
POLICY_ROOT="${WORK_ROOT}/RoboDojo/XPolicyLab/policy/Pi_05"

export PI05_RUNTIME_ROOT="${WORK_ROOT}/runtime"
source "${POLICY_ROOT}/slurm/bootstrap_runtime.sh" "${OPENPI_ROOT}"
echo "[Pi_05 ARX X5] TensorBoard node: ${SLURMD_NODENAME}"
echo "[Pi_05 ARX X5] harry-dev tunnel: ssh -N -L 6007:${SLURMD_NODENAME}:6006 gmicloud-loki-g1-cpu-001"
echo "[Pi_05 ARX X5] Mac tunnel: ssh -N -L 6007:127.0.0.1:6007 harry-dev"
cd "${OPENPI_ROOT}"
uv run tensorboard --logdir "${TENSORBOARD_DIR}" --host 0.0.0.0 --port 6006
