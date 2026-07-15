# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${POLICY_DIR}/openpi"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

echo "[Pi_05] OPENPI_ROOT=${OPENPI_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

cd "${OPENPI_ROOT}"
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot
POLICY_PYTHON="${OPENPI_ROOT}/.venv/bin/python"
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "${POLICY_PYTHON}" -e .

uv pip install --python "${POLICY_PYTHON}" -e "${XPOLICYLAB_ROOT}"
"${POLICY_PYTHON}" -c "import XPolicyLab; print('XPolicyLab ok')"

echo "[Pi_05] Installation finished."
echo "[Pi_05] Activate: source ${OPENPI_ROOT}/.venv/bin/activate"
