#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
    echo "Usage: $0 <dataset> <task> <ckpt> <env> <action> <seed> <gpu> <policy-env>" >&2
    exit 2
fi

dataset=$1
task=$2
checkpoint=$3
environment=$4
action=$5
seed=$6
gpu=$7
policy_env=$8
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
ROBODOJO_ROOT_RESOLVED="${ROBODOJO_ROOT:-$(cd "${XPOLICYLAB_ROOT}/.." && pwd)}"
DEPLOY_YAML="${POLICY_DIR}/deploy.yml"
warning=false

pass() { echo "PASS $1: $2"; }
warn() { echo "WARN $1: $2"; warning=true; }
fail() { echo "FAIL $1: $2" >&2; echo "REMEDIATION: make policy-setup" >&2; exit 1; }

[[ "${seed}" =~ ^[0-9]+$ ]] || fail "arguments" "seed must be a nonnegative integer, got ${seed}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || fail "gpu" "GPU index must be a nonnegative integer, got ${gpu}"
command -v nvidia-smi >/dev/null 2>&1 || fail "gpu" "nvidia-smi is not available"
mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
(( gpu < ${#gpus[@]} )) || fail "gpu" "GPU ${gpu} is unavailable; detected ${#gpus[@]} GPU(s)"
pass "gpu" "physical GPU ${gpu} is available"

resolve_project() {
    local project=${policy_env}
    if [[ "${project}" == "uv" ]]; then
        project=$(sed -nE 's/^policy_uv_env_path:[[:space:]]*([^[:space:]]+)[[:space:]]*$/\1/p' "${DEPLOY_YAML}" | head -n 1)
        project="${project%\"}"; project="${project#\"}"
        project="${project%\'}"; project="${project#\'}"
    fi
    project="${project/#\~/${HOME}}"
    if [[ "${project}" != /* ]]; then
        project="${POLICY_DIR}/${project}"
    fi
    realpath -m "${project}"
}

PROJECT="$(resolve_project)"
PYTHON_BIN="${PROJECT}/.venv/bin/python"
[[ -f "${PROJECT}/pyproject.toml" ]] || fail "runtime" "uv project is missing pyproject.toml: ${PROJECT}"
[[ -f "${PROJECT}/uv.lock" ]] || fail "runtime" "uv project is missing uv.lock: ${PROJECT}"
[[ -x "${PYTHON_BIN}" ]] || fail "runtime" "uv environment is missing Python: ${PYTHON_BIN}"
command -v uv >/dev/null 2>&1 || fail "runtime" "uv is not installed"
if ! (cd "${PROJECT}" && uv lock --check --offline >/dev/null 2>&1); then
    fail "runtime" "uv.lock is stale for ${PROJECT}"
fi
pass "runtime" "locked OpenPI environment ${PROJECT}"

if ! env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROBODOJO_ROOT_RESOLVED}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -c 'import XPolicyLab, jax, numpy, openpi, orbax.checkpoint, websockets' >/dev/null 2>&1; then
    fail "imports" "Pi_05 runtime cannot import XPolicyLab, OpenPI, JAX, Orbax, NumPy, and websockets"
fi
pass "imports" "XPolicyLab and OpenPI runtime imports succeeded"

STORAGE_ROOT="${ROBODOJO_STORAGE_ROOT:-${ROBODOJO_ROOT_RESOLVED}/.robodojo}"
case "${checkpoint}" in
    pi05_arx5_multitask_v1)
        [[ "${environment}" == "arx_x5" ]] || fail "contract" "${checkpoint} requires env arx_x5, got ${environment}"
        [[ "${action}" == "joint" ]] || fail "contract" "${checkpoint} requires joint actions, got ${action}"
        SNAPSHOT="${STORAGE_ROOT}/model_weights/Pi_05/${checkpoint}/880fa61406540d80b1c3b9824f12c19b903a233f"
        PARAMS="${SNAPSHOT}/checkpoints/55000/params"
        NORM="${SNAPSHOT}/assets/norm_stats.json"
        NORM_SHA256="c57c763df05f632b6912b9d1aefbd537d7f05f9c67360ac106451d5d6b9fa32c"
        ;;
    pi05_yam_molmoact2)
        [[ "${environment}" == "bimanual_yam" ]] || fail "contract" "${checkpoint} requires env bimanual_yam, got ${environment}"
        [[ "${action}" == "joint" ]] || fail "contract" "${checkpoint} requires joint actions, got ${action}"
        SNAPSHOT="${STORAGE_ROOT}/model_weights/Pi_05/${checkpoint}/df991e11e8f6540098338c56342b1143fac5b952"
        PARAMS="${SNAPSHOT}/params"
        NORM="${SNAPSHOT}/assets/yam-bimanual-merged/norm_stats.json"
        NORM_SHA256="16daf28cec63d4829f01d7858bfed079ad18e183ce826a268f66c6669f323863"
        METADATA_SHA256="303a4e354814928e1d29b75e310f2c1ac7e7e29b62f48395b631045ca1cffc73"
        ;;
    *)
        candidate="${checkpoint/#\~/${HOME}}"
        if [[ "${candidate}" != /* ]]; then candidate="${POLICY_DIR}/${candidate}"; fi
        if [[ "${checkpoint}" == */* || "${checkpoint}" == .* ]]; then
            [[ -e "${candidate}" ]] || fail "checkpoint" "explicit checkpoint does not exist: ${candidate}"
            warn "checkpoint" "local checkpoint exists, but has no pinned Pi_05 integrity profile: ${candidate}"
        else
            warn "checkpoint" "opaque checkpoint alias ${checkpoint} has no pinned Pi_05 integrity profile"
        fi
        SNAPSHOT=""
        ;;
esac

if [[ -n "${SNAPSHOT}" ]]; then
    [[ -d "${PARAMS}" ]] || fail "checkpoint" "Orbax params directory is missing: ${PARAMS}"
    [[ -f "${NORM}" ]] || fail "checkpoint" "quantile statistics are missing: ${NORM}"
    printf '%s  %s\n' "${NORM_SHA256}" "${NORM}" | sha256sum --check --strict >/dev/null 2>&1 \
        || fail "checkpoint" "quantile statistics hash does not match the pinned release"
    if [[ -n "${METADATA_SHA256:-}" ]]; then
        printf '%s  %s\n' "${METADATA_SHA256}" "${PARAMS}/_METADATA" | sha256sum --check --strict >/dev/null 2>&1 \
            || fail "checkpoint" "Orbax metadata hash does not match the pinned release"
    fi
    if ! "${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); n=d["norm_stats"]; assert {"state","actions"} <= set(n); assert {"q01","q99"} <= set(n["state"]); assert {"q01","q99"} <= set(n["actions"])' "${NORM}" >/dev/null 2>&1; then
        fail "checkpoint" "quantile statistics do not contain state/action q01 and q99 fields"
    fi
    pass "checkpoint" "pinned Orbax params and quantile-stat integrity checks passed"
fi
if [[ "${checkpoint}" == "pi05_yam_molmoact2" ]]; then
    if ! timing_detail=$(env \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        PYTHONPATH="${ROBODOJO_ROOT_RESOLVED}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" -m XPolicyLab.policy.Pi_05.preflight_timing \
        --root "${ROBODOJO_ROOT_RESOLVED}" 2>&1); then
        fail "timing" "${timing_detail}"
    fi
    pass "timing" "${timing_detail}"
fi
pass "contract" "dataset=${dataset} task=${task} env=${environment} action=${action}"

if [[ "${warning}" == true ]]; then exit 3; fi
