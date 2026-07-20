#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=${1:?OpenPI root is required}
RUNTIME_ROOT=${PI05_RUNTIME_ROOT:-/workspace/pi05-piper/runtime}
UV_VERSION=${PI05_UV_VERSION:-0.11.21}

user_base=$(python -m site --user-base)
export PATH="${user_base}/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --user "uv==${UV_VERSION}"
fi

export UV_PROJECT_ENVIRONMENT="${RUNTIME_ROOT}/openpi-venv"
mkdir -p "${RUNTIME_ROOT}"
(
  cd "${OPENPI_ROOT}"
  uv sync --frozen --group lerobot
)

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg is required by the LeRobot video pipeline." >&2
  exit 1
}
