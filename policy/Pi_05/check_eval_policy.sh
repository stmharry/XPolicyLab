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

if [[ -n "${ROBODOJO_STORAGE_ROOT:-}" ]]; then
    STORAGE_ROOT="${ROBODOJO_STORAGE_ROOT}"
else
    STORAGE_ROOT="$(
        env PYTHONPATH="${ROBODOJO_ROOT_RESOLVED}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
            "${PYTHON_BIN}" -c 'from XPolicyLab.utils.robodojo_paths import storage_root; print(storage_root())'
    )"
fi
LEROBOT_SNAPSHOT=""
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
    pi05_yam_abc_pickplace)
        [[ "${environment}" == "bimanual_yam" ]] || fail "contract" "${checkpoint} requires env bimanual_yam, got ${environment}"
        [[ "${action}" == "joint" ]] || fail "contract" "${checkpoint} requires joint actions, got ${action}"
        LEROBOT_SNAPSHOT="${STORAGE_ROOT}/model_weights/Pi_05/${checkpoint}/44cc2cd8d7edf9be332bc3cfa7475484897c61e9"
        LEROBOT_NORM_NAME="policy_preprocessor_step_3_normalizer_processor.safetensors"
        LEROBOT_MODEL_SIZE="9354050752"
        LEROBOT_MODEL_SHA256="0c697969f4cefbfe781b83389212b40493ce5ed51dc5c31f15a1d2b31233eebc"
        LEROBOT_CONFIG_SHA256="33348185438514a51dcecd003fc26f19c32be5ca685b89a9089018854ad18161"
        LEROBOT_NORM_SHA256="1bddab6693cd52b5da72ca33e0b8f704ebc48bbeee1f48a3532c4746248cd2b6"
        SNAPSHOT=""
        ;;
    pi05_piper_bimanual_v1)
        [[ "${environment}" == "bimanual_piper" ]] \
            || fail "contract" "${checkpoint} requires env bimanual_piper, got ${environment}"
        [[ "${action}" == "joint" ]] || fail "contract" "${checkpoint} requires joint actions, got ${action}"
        LEROBOT_SNAPSHOT="${STORAGE_ROOT}/model_weights/Pi_05/${checkpoint}/3701b435a9730069b56979383c6a31d77cf7f61f"
        LEROBOT_NORM_NAME="policy_preprocessor_step_2_normalizer_processor.safetensors"
        LEROBOT_MODEL_SIZE="9354045072"
        LEROBOT_MODEL_SHA256="09d36543a0524a3227f478beffc7053f6f463e728b6030b54c0fc4ce1f9c06d4"
        LEROBOT_CONFIG_SHA256="d4b024f9d1e92db6c12dba023556f346459c1447a900386f01ffc099603bd67d"
        LEROBOT_NORM_SHA256="32789a34133894427cf748bc45d9be9b9bb13fa78c14bab289427e5b86622b8b"
        SNAPSHOT=""
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

if [[ -n "${LEROBOT_SNAPSHOT}" ]]; then
    MODEL="${LEROBOT_SNAPSHOT}/model.safetensors"
    CONFIG="${LEROBOT_SNAPSHOT}/config.json"
    NORM="${LEROBOT_SNAPSHOT}/${LEROBOT_NORM_NAME}"
    [[ -f "${MODEL}" ]] || fail "checkpoint" "LeRobot model is missing: ${MODEL}"
    [[ "$(stat -c '%s' "${MODEL}")" == "${LEROBOT_MODEL_SIZE}" ]] \
        || fail "checkpoint" "LeRobot model size does not match the pinned release"
    printf '%s  %s\n' "${LEROBOT_MODEL_SHA256}" "${MODEL}" \
        | sha256sum --check --strict >/dev/null 2>&1 \
        || fail "checkpoint" "LeRobot model hash does not match the pinned release"
    printf '%s  %s\n' "${LEROBOT_CONFIG_SHA256}" "${CONFIG}" \
        | sha256sum --check --strict >/dev/null 2>&1 \
        || fail "checkpoint" "LeRobot config hash does not match the pinned release"
    printf '%s  %s\n' "${LEROBOT_NORM_SHA256}" "${NORM}" \
        | sha256sum --check --strict >/dev/null 2>&1 \
        || fail "checkpoint" "LeRobot normalization hash does not match the pinned release"
    if ! "${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["type"] == "pi05"; assert d["chunk_size"] == d["n_action_steps"] == 50; assert d["use_relative_actions"] is False; assert d["input_features"]["observation.state"]["shape"] == [14]; assert d["output_features"]["action"]["shape"] == [14]' "${CONFIG}" >/dev/null 2>&1; then
        fail "checkpoint" "LeRobot config does not match the bimanual absolute 50-action contract"
    fi
    if [[ "${checkpoint}" == "pi05_piper_bimanual_v1" ]]; then
        printf '%s  %s\n' "7b1cb16c1aeaa5f913807c0cc6e8b1c54ea8070d060c13b88012157b6abb92c3" \
            "${LEROBOT_SNAPSHOT}/policy_preprocessor.json" | sha256sum --check --strict >/dev/null 2>&1 \
            || fail "checkpoint" "LeRobot preprocessor hash does not match the pinned release"
        printf '%s  %s\n' "7e721aeab8736709ba60948f4c96ea9b14a88bb4c7baaf20afd7ba1eada8ed0d" \
            "${LEROBOT_SNAPSHOT}/policy_postprocessor.json" | sha256sum --check --strict >/dev/null 2>&1 \
            || fail "checkpoint" "LeRobot postprocessor hash does not match the pinned release"
        printf '%s  %s\n' "32789a34133894427cf748bc45d9be9b9bb13fa78c14bab289427e5b86622b8b" \
            "${LEROBOT_SNAPSHOT}/policy_postprocessor_step_0_unnormalizer_processor.safetensors" \
            | sha256sum --check --strict >/dev/null 2>&1 \
            || fail "checkpoint" "LeRobot unnormalizer hash does not match the pinned release"
        printf '%s  %s\n' "cfc880b3c0617b7acdb1c784c8bc6ca42299d1f1cefb088dd9c160a2d3467f9d" \
            "${LEROBOT_SNAPSHOT}/train_config.json" | sha256sum --check --strict >/dev/null 2>&1 \
            || fail "checkpoint" "LeRobot training config hash does not match the pinned release"
        if ! "${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert tuple(d["input_features"]) == ("observation.state", "observation.images.cam_front", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"); assert all(d["input_features"][k]["shape"] == [3,224,224] for k in tuple(d["input_features"])[1:]); assert d["normalization_mapping"] == {"VISUAL":"IDENTITY", "STATE":"QUANTILES", "ACTION":"QUANTILES"}; assert d["action_feature_names"] == ["motors"]' "${CONFIG}" >/dev/null 2>&1; then
            fail "checkpoint" "LeRobot config does not match the PiPER camera and quantile contract"
        fi
    fi
    if ! env PYTHONPATH="${ROBODOJO_ROOT_RESOLVED}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" -c 'import lerobot, torch, transformers; from transformers.models.siglip import check; assert transformers.__version__ == "4.53.3"; assert check.check_whether_transformers_replace_is_installed_correctly()' >/dev/null 2>&1; then
        fail "imports" "LeRobot PI0.5 requires the pinned OpenPI-compatible Transformers 4.53.3 runtime"
    fi
    pass "checkpoint" "pinned LeRobot PI0.5 model, config, normalization, and runtime checks passed"
elif [[ -n "${SNAPSHOT}" ]]; then
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
if [[ "${checkpoint}" == "pi05_yam_molmoact2" || "${checkpoint}" == "pi05_yam_abc_pickplace" \
    || "${checkpoint}" == "pi05_piper_bimanual_v1" ]]; then
    if ! timing_detail=$(env \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        PYTHONPATH="${ROBODOJO_ROOT_RESOLVED}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" -m XPolicyLab.policy.Pi_05.preflight_timing \
        --root "${ROBODOJO_ROOT_RESOLVED}" \
        --profile "${checkpoint}" 2>&1); then
        fail "timing" "${timing_detail}"
    fi
    pass "timing" "${timing_detail}"
fi
pass "contract" "dataset=${dataset} task=${task} env=${environment} action=${action}"

if [[ "${warning}" == true ]]; then exit 3; fi
