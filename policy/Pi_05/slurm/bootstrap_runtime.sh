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

mkdir -p "${RUNTIME_ROOT}/bin"
ffmpeg_exe=$(
  cd "${OPENPI_ROOT}"
  uv run python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'
)
ln -sfn "${ffmpeg_exe}" "${RUNTIME_ROOT}/bin/ffmpeg"
export PATH="${RUNTIME_ROOT}/bin:${PATH}"
ffmpeg -version >/dev/null
