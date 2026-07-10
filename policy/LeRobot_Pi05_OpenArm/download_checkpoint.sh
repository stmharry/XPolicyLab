#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${1:-lerobot-pi05}"
hf download lerobot-data-collection/folding_final \
  --revision 695abe40dbf3aac04efda59c1501d748681fa0fb \
  --local-dir "${SCRIPT_DIR}/checkpoints/folding_final"
