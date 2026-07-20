#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${PI05_CONTAINER_WORK_ROOT:-/workspace/pi05-piper}
POLICY_ROOT="${WORK_ROOT}/RoboDojo/XPolicyLab/policy/Pi_05"
OPENPI_ROOT="${POLICY_ROOT}/openpi"

export PI05_RUNTIME_ROOT="${WORK_ROOT}/runtime"
export PI05_PIPER_WORK_ROOT="${WORK_ROOT}"
export HF_LEROBOT_HOME="${WORK_ROOT}/data/lerobot"
export OPENPI_ASSETS_BASE_DIR="${WORK_ROOT}/assets"
export HF_HOME="${WORK_ROOT}/cache/huggingface"
export XDG_CACHE_HOME="${WORK_ROOT}/cache/xdg"

source "${POLICY_ROOT}/slurm/bootstrap_runtime.sh" "${OPENPI_ROOT}"
bash "${POLICY_ROOT}/process_robodojo_real_piper.sh" --overwrite
