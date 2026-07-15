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
fail() { echo "FAIL $1: $2" >&2; echo "REMEDIATION: make setup" >&2; exit 1; }

[[ "${seed}" =~ ^[0-9]+$ ]] || fail "arguments" "seed must be a nonnegative integer, got ${seed}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || fail "gpu" "GPU index must be a nonnegative integer, got ${gpu}"
command -v nvidia-smi >/dev/null 2>&1 || fail "gpu" "nvidia-smi is not available"
mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
(( gpu < ${#gpus[@]} )) || fail "gpu" "GPU ${gpu} is unavailable; detected ${#gpus[@]} GPU(s)"
pass "gpu" "physical GPU ${gpu} is available"

PYTHON_BIN=""
PROJECT=""
if [[ "${checkpoint}" == "molmoact2_bimanual_yam" && ( "${policy_env}" == "molmoact2" || "${policy_env}" == "uv" ) ]]; then
    PROJECT="${POLICY_DIR}/molmoact2"
elif [[ "${policy_env}" == "uv" || "${policy_env}" == */* || "${policy_env}" == .* ]]; then
    PROJECT="${policy_env}"
    if [[ "${PROJECT}" == "uv" ]]; then
        PROJECT=$(sed -nE 's/^policy_uv_env_path:[[:space:]]*([^[:space:]]+)[[:space:]]*$/\1/p' "${DEPLOY_YAML}" | head -n 1)
        PROJECT="${PROJECT%\"}"; PROJECT="${PROJECT#\"}"
        PROJECT="${PROJECT%\'}"; PROJECT="${PROJECT#\'}"
    fi
    PROJECT="${PROJECT/#\~/${HOME}}"
    if [[ "${PROJECT}" != /* ]]; then PROJECT="${POLICY_DIR}/${PROJECT}"; fi
else
    command -v conda >/dev/null 2>&1 || fail "runtime" "Conda environment ${policy_env} cannot be resolved because conda is unavailable"
    CONDA_JSON="$(conda env list --json 2>/dev/null)" || fail "runtime" "could not inspect Conda environments"
    CONDA_PREFIX_RESOLVED="$(python3 -c 'import json,sys; name=sys.argv[1]; envs=json.load(sys.stdin).get("envs", []); matches=[p for p in envs if p.rsplit("/",1)[-1]==name]; print(matches[0] if matches else "")' "${policy_env}" <<<"${CONDA_JSON}")"
    [[ -n "${CONDA_PREFIX_RESOLVED}" ]] || fail "runtime" "Conda environment does not exist: ${policy_env}"
    PYTHON_BIN="${CONDA_PREFIX_RESOLVED}/bin/python"
fi

if [[ -n "${PROJECT}" ]]; then
    PROJECT="$(realpath -m "${PROJECT}")"
    PYTHON_BIN="${PROJECT}/.venv/bin/python"
    [[ -f "${PROJECT}/pyproject.toml" ]] || fail "runtime" "uv project is missing pyproject.toml: ${PROJECT}"
    [[ -f "${PROJECT}/uv.lock" ]] || fail "runtime" "uv project is missing uv.lock: ${PROJECT}"
    [[ -x "${PYTHON_BIN}" ]] || fail "runtime" "uv environment is missing Python: ${PYTHON_BIN}"
    command -v uv >/dev/null 2>&1 || fail "runtime" "uv is not installed"
    if ! (cd "${PROJECT}" && uv lock --check --offline >/dev/null 2>&1); then
        fail "runtime" "uv.lock is stale for ${PROJECT}"
    fi
    pass "runtime" "locked MolmoACT2 environment ${PROJECT}"
else
    [[ -x "${PYTHON_BIN}" ]] || fail "runtime" "Conda environment Python is missing: ${PYTHON_BIN}"
    pass "runtime" "Conda environment ${policy_env}"
fi

if ! env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROBODOJO_ROOT_RESOLVED}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -c 'import XPolicyLab, molmo, torch, transformers, websockets; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 1' >/dev/null 2>&1; then
    fail "imports" "MolmoACT2 runtime imports or Torch CUDA probe failed"
fi
pass "imports" "XPolicyLab, MolmoACT2, Transformers, Torch, and CUDA checks succeeded"

STORAGE_ROOT="${ROBODOJO_STORAGE_ROOT:-${ROBODOJO_ROOT_RESOLVED}/.robodojo}"
if [[ "${checkpoint}" == "molmoact2_bimanual_yam" ]]; then
    [[ "${environment}" == "bimanual_yam" ]] || fail "contract" "${checkpoint} requires env bimanual_yam, got ${environment}"
    [[ "${action}" == "joint" ]] || fail "contract" "${checkpoint} requires joint actions, got ${action}"
    [[ -n "${PROJECT}" && -d "${PROJECT}/.git" ]] || fail "source" "pinned MolmoACT2 source checkout is missing: ${PROJECT}"
    SOURCE_REVISION="$(git -C "${PROJECT}" rev-parse HEAD 2>/dev/null || true)"
    [[ "${SOURCE_REVISION}" == "c2282820f9b188b60e66ea1636b3efd81c45cbb4" ]] \
        || fail "source" "MolmoACT2 source revision is ${SOURCE_REVISION:-missing}, expected c2282820f9b188b60e66ea1636b3efd81c45cbb4"
    SNAPSHOT="${STORAGE_ROOT}/model_weights/MolmoACT2/${checkpoint}/8dcbed66f2380e4393189c303ea72488eb9e63c2"
    required=(config.json model.safetensors.index.json processor_config.json norm_stats.json)
    for relative in "${required[@]}"; do
        [[ -f "${SNAPSHOT}/${relative}" ]] || fail "checkpoint" "checkpoint file is missing: ${SNAPSHOT}/${relative}"
    done
    for shard in {1..5}; do
        printf -v shard_name 'model-%05d-of-00005.safetensors' "${shard}"
        [[ -s "${SNAPSHOT}/${shard_name}" ]] || fail "checkpoint" "checkpoint shard is missing or empty: ${shard_name}"
    done
    printf '%s  %s\n' '12bb27d584b3b91ebf86b6f85327a3f934f239b5fcdf5f89298e91d7c9516112' "${SNAPSHOT}/config.json" \
        '273799bed0cbdc2c4a7a524c294652a0538c482e5e1148cf403951e836722c06' "${SNAPSHOT}/model.safetensors.index.json" \
        '6357a264ac820c5b775c884bd303ed13ffa18a36def28409312c842139879f4b' "${SNAPSHOT}/norm_stats.json" \
        | sha256sum --check --strict >/dev/null 2>&1 || fail "checkpoint" "pinned lightweight checkpoint hashes do not match"
    if ! "${PYTHON_BIN}" -c 'import json,sys,pathlib; root=pathlib.Path(sys.argv[1]); cfg=json.load(open(root/"config.json")); assert cfg["model_type"]=="molmoact2"; assert cfg["architectures"]==["MolmoAct2ForConditionalGeneration"]; idx=json.load(open(root/"model.safetensors.index.json")); assert len(idx["weight_map"])==1295; assert idx["metadata"]["total_parameters"]==5485309424; norm=json.load(open(root/"norm_stats.json")); assert norm["format"]=="molmoact2_norm_stats.v1"; assert "yam_dual_molmoact2" in norm["metadata_by_tag"]' "${SNAPSHOT}" >/dev/null 2>&1; then
        fail "checkpoint" "pinned checkpoint metadata fields are invalid"
    fi
    pass "source" "pinned MolmoACT2 source revision ${SOURCE_REVISION}"
    pass "checkpoint" "pinned structure, shards, hashes, and metadata fields passed"
else
    candidate="${checkpoint/#\~/${HOME}}"
    if [[ "${candidate}" != /* ]]; then candidate="${POLICY_DIR}/${candidate}"; fi
    if [[ "${checkpoint}" == */* || "${checkpoint}" == .* ]]; then
        [[ -e "${candidate}" ]] || fail "checkpoint" "explicit checkpoint does not exist: ${candidate}"
        warn "checkpoint" "local checkpoint exists, but has no pinned MolmoACT2 integrity profile: ${candidate}"
    else
        warn "checkpoint" "opaque checkpoint alias ${checkpoint} has no pinned MolmoACT2 integrity profile"
    fi
fi
pass "contract" "dataset=${dataset} task=${task} env=${environment} action=${action}"

if [[ "${warning}" == true ]]; then exit 3; fi
