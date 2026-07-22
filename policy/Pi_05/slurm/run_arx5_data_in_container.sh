#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${PI05_CONTAINER_WORK_ROOT:-/workspace/pi05-arx5}
POLICY_ROOT="${WORK_ROOT}/RoboDojo/XPolicyLab/policy/Pi_05"
OPENPI_ROOT="${POLICY_ROOT}/openpi"

export PI05_RUNTIME_ROOT="${WORK_ROOT}/runtime"
export PI05_ARX5_WORK_ROOT="${WORK_ROOT}"
export HF_LEROBOT_HOME="${WORK_ROOT}/data/lerobot"
export OPENPI_ASSETS_BASE_DIR="${WORK_ROOT}/assets"
export HF_HOME="${WORK_ROOT}/cache/huggingface"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export XDG_CACHE_HOME="${WORK_ROOT}/cache/xdg"
export JAX_PLATFORMS=cpu

source "${POLICY_ROOT}/slurm/bootstrap_runtime.sh" "${OPENPI_ROOT}"
dataset_root="${HF_LEROBOT_HOME}/RoboDojo-real_arx_x5_6task-bimanual_arx_x5-joint"
if [[ -f "${dataset_root}/meta/info.json" ]]; then
  echo "[Pi_05 ARX X5] Existing finalized dataset found; validating without overwrite."
  bash "${POLICY_ROOT}/process_robodojo_real_arx_x5.sh" --skip-download --validate-only
else
  bash "${POLICY_ROOT}/process_robodojo_real_arx_x5.sh" --overwrite
fi
