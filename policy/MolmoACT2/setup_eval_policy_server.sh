#!/bin/bash
set -euo pipefail
bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"

# Proxy configuration is opt-in. Existing standard proxy variables are kept.
if [[ -n "${DEPLOY_PROXY_URL:-}" ]]; then
    export http_proxy="${DEPLOY_PROXY_URL}"
    export https_proxy="${DEPLOY_PROXY_URL}"
    export HTTP_PROXY="${DEPLOY_PROXY_URL}"
    export HTTPS_PROXY="${DEPLOY_PROXY_URL}"
elif [[ -n "${DEPLOY_PROXY_HOST:-}" ]]; then
    _DEPLOY_PROXY_PORT="${DEPLOY_PROXY_PORT:-18000}"
    export http_proxy="http://${DEPLOY_PROXY_HOST}:${_DEPLOY_PROXY_PORT}"
    export https_proxy="${http_proxy}"
    export HTTP_PROXY="${http_proxy}"
    export HTTPS_PROXY="${https_proxy}"
fi
if [[ -n "${https_proxy:-${HTTPS_PROXY:-}}" ]]; then
    echo "[SERVER] using configured HTTPS proxy"
else
    echo "[SERVER] proxy disabled"
fi

if type deactivate >/dev/null 2>&1 && [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate || true
fi
unset VIRTUAL_ENV
if [[ "${ckpt_name}" == "molmoact2_bimanual_yam" && "${policy_conda_env}" == "molmoact2" ]]; then
    policy_uv_env_path="${SCRIPT_DIR}/molmoact2"
elif [[ "${policy_conda_env}" == "uv" || "${policy_conda_env}" == */* ]]; then
    if [[ "${policy_conda_env}" == "uv" ]]; then
        policy_uv_env_path=$(sed -nE 's/^policy_uv_env_path:[[:space:]]*([^[:space:]]+)[[:space:]]*$/\1/p' "${yaml_file}" | head -n 1)
        policy_uv_env_path="${policy_uv_env_path%\"}"
        policy_uv_env_path="${policy_uv_env_path#\"}"
        policy_uv_env_path="${policy_uv_env_path%\'}"
        policy_uv_env_path="${policy_uv_env_path#\'}"
        if [[ -z "${policy_uv_env_path}" ]]; then
            echo "[SERVER][ERROR] policy_uv_env_path is missing from ${yaml_file}" >&2
            exit 1
        fi
    else
        policy_uv_env_path="${policy_conda_env}"
    fi
    policy_uv_env_path="${policy_uv_env_path/#\~/${HOME}}"
    if [[ "${policy_uv_env_path}" != /* ]]; then
        policy_uv_env_path="$(realpath -m "${SCRIPT_DIR}/${policy_uv_env_path}")"
    fi
fi

if [[ -n "${policy_uv_env_path:-}" ]]; then
    if [[ ! -f "${policy_uv_env_path}/.venv/bin/activate" ]]; then
        echo "[SERVER][ERROR] uv venv not found: ${policy_uv_env_path}/.venv" >&2
        echo "[SERVER][ERROR] Run: bash ${SCRIPT_DIR}/install.sh infer" >&2
        exit 1
    fi
    echo "[SERVER] Activating uv environment: ${policy_uv_env_path}/.venv"
    source "${policy_uv_env_path}/.venv/bin/activate"
    PYTHON_BIN="$(command -v python)"
else
    if ! command -v conda >/dev/null 2>&1; then
        echo "[SERVER][ERROR] conda is required for environment ${policy_conda_env}" >&2
        exit 1
    fi
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    echo "[SERVER] Activating Conda environment: ${policy_conda_env}"
    conda activate "${policy_conda_env}"
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
echo "[SERVER] Using python: ${PYTHON_BIN}"

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}"
