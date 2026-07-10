#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-lerobot-pi05}"
LEROBOT_REV="1396b9fab7aecddd10006c33c47a487ffdcb54b4"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.12 pip
fi
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision
python -m pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_REV}"
python -m pip install pytest -e "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
