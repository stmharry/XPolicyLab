#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${PI05_ARX5_WORK_ROOT:-/home/harry/pi05-arx5}"
RAW_ROOT="${PI05_ARX5_RAW_ROOT:-${WORK_ROOT}/data/raw}"
LEROBOT_ROOT="${HF_LEROBOT_HOME:-${WORK_ROOT}/data/lerobot}"
ASSETS_ROOT="${OPENPI_ASSETS_BASE_DIR:-${WORK_ROOT}/assets}"
REPO_ID="RoboDojo-real_arx_x5_6task-bimanual_arx_x5-joint"

mkdir -p "${RAW_ROOT}" "${LEROBOT_ROOT}" "${ASSETS_ROOT}"
export HF_LEROBOT_HOME="${LEROBOT_ROOT}"

cd "${POLICY_DIR}/openpi"
uv run --group lerobot python scripts/convert_robodojo_real_arx_x5.py \
  --raw-root "${RAW_ROOT}" \
  --output-root "${LEROBOT_ROOT}" \
  --repo-id "${REPO_ID}" \
  "$@"
uv run --group lerobot python scripts/compute_norm_stats.py \
  --config-name pi05_base_aloha_full_real_arx-x5_seed_0 \
  --assets-base-dir "${ASSETS_ROOT}"
