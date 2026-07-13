# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
# MolmoAct2 one-command install, corresponding to INSTALLATION.md
#
# Usage:
# bash install.sh # lerobot training environment + XPolicyLab(RoboDojo default, and eval use)
# bash install.sh train # same as above
# bash install.sh infer # original-HF inference environment + XPolicyLab
# bash install.sh all # LeRobot training and original-HF inference environments
#
# Optional environment variables:
# MOLMOACT2_REPO default https://github.com/allenai/molmoact2.git
# MOLMOACT2_REVISION pinned source contract used by the public YAM profile
# LEROBOT_REPO default https://github.com/allenai/lerobot
# LEROBOT_BRANCH default molmoact2-policy
# SKIP_XPOLICYLAB=1 skip XPolicyLab installation

set -euo pipefail

MODE="${1:-train}"
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOLMOACT2_DIR="${POLICY_DIR}/molmoact2"
LEROBOT_DIR="${MOLMOACT2_DIR}/lerobot"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

MOLMOACT2_REPO="${MOLMOACT2_REPO:-https://github.com/allenai/molmoact2.git}"
MOLMOACT2_REVISION="${MOLMOACT2_REVISION:-c2282820f9b188b60e66ea1636b3efd81c45cbb4}"
LEROBOT_REPO="${LEROBOT_REPO:-https://github.com/allenai/lerobot}"
LEROBOT_BRANCH="${LEROBOT_BRANCH:-molmoact2-policy}"

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "错误: 未找到 uv，请先安装:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
  echo "uv: $(uv --version)"
}

init_molmoact2_source() {
  echo ""
  echo "=== 1. 初始化上游源码 ==="

  if [[ -d "${MOLMOACT2_DIR}/.git" ]]; then
    echo "molmoact2 已存在: ${MOLMOACT2_DIR}"
    git -C "${MOLMOACT2_DIR}" fetch --depth 1 origin "${MOLMOACT2_REVISION}"
  else
    echo "clone ${MOLMOACT2_REPO} -> ${MOLMOACT2_DIR}"
    GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout "${MOLMOACT2_REPO}" "${MOLMOACT2_DIR}"
    git -C "${MOLMOACT2_DIR}" fetch --depth 1 origin "${MOLMOACT2_REVISION}"
  fi
  GIT_LFS_SKIP_SMUDGE=1 git -C "${MOLMOACT2_DIR}" checkout --detach "${MOLMOACT2_REVISION}"

  # Original-HF inference uses only the top-level source and pyproject. Training
  # additionally needs LeRobot; unrelated hardware submodules contain a broken
  # nested declaration at this pinned revision and must not be initialized.
  if [[ "${MODE}" == "infer" ]]; then
    return
  fi
  GIT_LFS_SKIP_SMUDGE=1 git -C "${MOLMOACT2_DIR}" submodule update --init lerobot

  if [[ ! -f "${LEROBOT_DIR}/pyproject.toml" ]]; then
    if [[ -d "${LEROBOT_DIR}" ]] && [[ -n "$(ls -A "${LEROBOT_DIR}" 2>/dev/null)" ]]; then
      echo "错误: ${LEROBOT_DIR} 已存在但缺少 pyproject.toml，请删除后重试 install.sh" >&2
      exit 1
    fi
    echo "lerobot submodule 为空，手动 clone 分支 ${LEROBOT_BRANCH}"
    git clone -b "${LEROBOT_BRANCH}" "${LEROBOT_REPO}" "${LEROBOT_DIR}"
  elif [[ ! -d "${LEROBOT_DIR}/.git" ]]; then
    echo "lerobot 源码已存在（无 .git），跳过 clone: ${LEROBOT_DIR}"
  fi

  if [[ ! -f "${LEROBOT_DIR}/pyproject.toml" ]]; then
    echo "错误: LeRobot 源码不完整: ${LEROBOT_DIR}" >&2
    exit 1
  fi
}

install_infer_env() {
  echo ""
  echo "=== 2. 推理环境 (molmoact2/.venv) ==="
  cd "${MOLMOACT2_DIR}"
  UV_LINK_MODE=copy uv sync

  # The pinned upstream lock uses cu121 PyTorch 2.5.1, which has no sm_120
  # kernels. Keep upstream dependencies on older GPUs and install a tested
  # cu128 pair when Blackwell is detected (or explicitly requested).
  local compute_capability=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    compute_capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
  fi
  if [[ "${MOLMOACT2_USE_CU128:-auto}" == "1" ]] || \
     { [[ "${MOLMOACT2_USE_CU128:-auto}" == "auto" ]] && [[ "${compute_capability%%.*}" =~ ^[0-9]+$ ]] && (( ${compute_capability%%.*} >= 12 )); }; then
    echo "Detected Blackwell compute capability ${compute_capability}; installing cu128 PyTorch."
    UV_LINK_MODE=copy uv pip install \
      --python "${MOLMOACT2_DIR}/.venv/bin/python" \
      "torch==${MOLMOACT2_TORCH_VERSION:-2.10.0}" \
      "torchvision==${MOLMOACT2_TORCHVISION_VERSION:-0.25.0}" \
      --index-url "${MOLMOACT2_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}" \
      --index-strategy unsafe-best-match
  fi
  UV_LINK_MODE=copy uv pip install -e "${XPOLICYLAB_ROOT}"
  "${MOLMOACT2_DIR}/.venv/bin/python" -c "import torch; x=torch.ones(1, device='cuda'); print('cuda tensor:', x, 'torch:', torch.__version__)"
  "${MOLMOACT2_DIR}/.venv/bin/python" -c "import XPolicyLab; print('XPolicyLab original-HF inference adapter ok')"
  echo "推理环境就绪: ${MOLMOACT2_DIR}/.venv"
}

install_train_env() {
  echo ""
  echo "=== 3. 训练环境 (lerobot/.venv) ==="
  cd "${LEROBOT_DIR}"
  UV_LINK_MODE=copy uv pip install -e ".[molmoact2,training,scipy-dep]" --index-strategy unsafe-best-match
  uv run python -c "from lerobot.policies.factory import get_policy_class; print('molmoact2 policy:', get_policy_class('molmoact2'))"
  echo "训练环境就绪: ${LEROBOT_DIR}/.venv"
}

install_xpolicylab_train() {
  if [[ "${SKIP_XPOLICYLAB:-0}" == "1" ]]; then
    echo "跳过 XPolicyLab 安装 (SKIP_XPOLICYLAB=1)"
    return
  fi

  echo ""
  echo "=== 4. 安装 XPolicyLab 到 lerobot/.venv（训练与 eval 共用） ==="
  # shellcheck disable=SC1091
  source "${LEROBOT_DIR}/.venv/bin/activate"
  cd "${XPOLICYLAB_ROOT}"
  if ! python -m pip --version >/dev/null 2>&1; then
    python -m ensurepip --upgrade
  fi
  uv pip install -e .
  uv pip install h5py opencv-python
  python -c "import XPolicyLab; import cv2, h5py; print('XPolicyLab ok')"
}

verify_all() {
  echo ""
  echo "=== 安装完成 ==="
  if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
    echo "LeRobot venv:    ${LEROBOT_DIR}/.venv  (train + local eval)"
    echo "训练入口:        bash ${POLICY_DIR}/train.sh ..."
  fi
  if [[ "${MODE}" == "infer" || "${MODE}" == "all" ]]; then
    echo "Original-HF venv: ${MOLMOACT2_DIR}/.venv  (public YAM eval)"
  fi
}

main() {
  case "${MODE}" in
    train|infer|all) ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "未知模式: ${MODE}（可选: train | infer | all）" >&2
      exit 1
      ;;
  esac

  ensure_uv
  init_molmoact2_source

  case "${MODE}" in
    infer)
      install_infer_env
      ;;
    train)
      install_train_env
      install_xpolicylab_train
      ;;
    all)
      install_infer_env
      install_train_env
      install_xpolicylab_train
      ;;
  esac

  verify_all
}

main "$@"
