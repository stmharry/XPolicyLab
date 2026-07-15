#!/usr/bin/env bash
set -euo pipefail

ARX_PROFILE="pi05_arx5_multitask_v1"
YAM_PROFILE="pi05_yam_molmoact2"
PROFILE="${1:-${ARX_PROFILE}}"
DRY_RUN=false

if [[ "${PROFILE}" == "--dry-run" ]]; then
    PROFILE="${ARX_PROFILE}"
    DRY_RUN=true
else
    shift || true
    if [[ "${1:-}" == "--dry-run" ]]; then
        DRY_RUN=true
        shift
    fi
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [${ARX_PROFILE}|${YAM_PROFILE}] [--dry-run]" >&2
    exit 2
fi

case "${PROFILE}" in
    "${ARX_PROFILE}")
        REPO_ID="pravsels/pi05-arx5-multitask-v1"
        REVISION="880fa61406540d80b1c3b9824f12c19b903a233f"
        CHECKPOINT_STEP="55000"
        PARAMS_TAR_SHA256="7ee69681991cdc5e04b4759d3bf93bca5dac6bc98639ec7b00202d2f82fe5b2f"
        NORM_RELATIVE_PATH="assets/norm_stats.json"
        NORM_SHA256="c57c763df05f632b6912b9d1aefbd537d7f05f9c67360ac106451d5d6b9fa32c"
        DOWNLOAD_INCLUDES=(
            --include "checkpoints/${CHECKPOINT_STEP}/**"
            --include "assets/**"
        )
        ;;
    "${YAM_PROFILE}")
        REPO_ID="robocurve/pi05-yam-molmoact2"
        REVISION="df991e11e8f6540098338c56342b1143fac5b952"
        NORM_RELATIVE_PATH="assets/yam-bimanual-merged/norm_stats.json"
        NORM_SHA256="16daf28cec63d4829f01d7858bfed079ad18e183ce826a268f66c6669f323863"
        PARAMS_METADATA_SHA256="303a4e354814928e1d29b75e310f2c1ac7e7e29b62f48395b631045ca1cffc73"
        DOWNLOAD_INCLUDES=()
        ;;
    *)
        echo "Unknown PI0.5 checkpoint alias: ${PROFILE}" >&2
        echo "Expected ${ARX_PROFILE} or ${YAM_PROFILE}." >&2
        exit 2
        ;;
esac

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_robodojo_root() {
    if [[ -n "${ROBODOJO_ROOT:-}" ]]; then
        printf '%s\n' "$(cd "${ROBODOJO_ROOT}" && pwd)"
        return
    fi
    local candidate="${POLICY_DIR}"
    while [[ "${candidate}" != "/" ]]; do
        if [[ -d "${candidate}/configs/environment" && -d "${candidate}/XPolicyLab" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
        candidate="$(dirname "${candidate}")"
    done
    echo "Could not locate RoboDojo; set ROBODOJO_ROOT." >&2
    return 1
}

ensure_hf() {
    HF_BIN=""
    local candidate
    for candidate in "$(command -v hf 2>/dev/null || true)" "${HOME}/.local/bin/hf"; do
        if [[ -x "${candidate}" ]] && "${candidate}" cache verify --help >/dev/null 2>&1; then
            HF_BIN="${candidate}"
            break
        fi
    done
    if [[ -z "${HF_BIN}" ]]; then
        echo "Installing the Hugging Face CLI..."
        curl -LsSf https://hf.co/cli/install.sh | bash -s
        export PATH="${HOME}/.local/bin:${PATH}"
        HF_BIN="$(command -v hf)"
    fi
    if ! "${HF_BIN}" cache verify --help >/dev/null 2>&1; then
        echo "Hugging Face CLI is too old; expected support for 'hf cache verify'." >&2
        exit 1
    fi
    if [[ -n "${HF_TOKEN:-}" ]]; then
        "${HF_BIN}" auth whoami >/dev/null
        echo "Hugging Face authentication accepted from HF_TOKEN."
    elif "${HF_BIN}" auth whoami >/dev/null 2>&1; then
        echo "Using the active Hugging Face CLI login."
    else
        echo "No Hugging Face login found; continuing with public read access."
    fi
}

ROBODOJO_ROOT_RESOLVED="$(find_robodojo_root)"
STORAGE_ROOT="${ROBODOJO_STORAGE_ROOT:-${ROBODOJO_ROOT_RESOLVED}/.robodojo}"
DESTINATION="${STORAGE_ROOT}/model_weights/Pi_05/${PROFILE}/${REVISION}"

ensure_hf
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if [[ "${DRY_RUN}" == true ]]; then
    "${HF_BIN}" download \
        "${REPO_ID}" \
        --revision "${REVISION}" \
        "${DOWNLOAD_INCLUDES[@]}" \
        --dry-run
    echo "Destination: ${DESTINATION}"
    exit 0
fi

mkdir -p "${DESTINATION}"
"${HF_BIN}" download \
    "${REPO_ID}" \
    --revision "${REVISION}" \
    "${DOWNLOAD_INCLUDES[@]}" \
    --local-dir "${DESTINATION}"

if [[ "${PROFILE}" == "${ARX_PROFILE}" ]]; then
    # The model card prescribes `tar cf - ... | sha256sum`. A plain tar
    # archive includes local uid/gid, modes, mtimes, and traversal metadata,
    # so this digest is provenance rather than a reproducible integrity gate.
    PARAMS_SHA256="$({ tar cf - -C "${DESTINATION}/checkpoints/${CHECKPOINT_STEP}" params; } | sha256sum | awk '{print $1}')"
    echo "Model-card params tar digest: ${PARAMS_TAR_SHA256}"
    echo "Local plain-tar params digest: ${PARAMS_SHA256}"
    if [[ "${PARAMS_SHA256}" == "${PARAMS_TAR_SHA256}" ]]; then
        echo "Plain-tar params digest matches the model-card provenance."
    else
        echo "WARNING: metadata-sensitive params tar digest differs from model-card provenance." >&2
        echo "This is non-fatal because plain tar hashes local uid/gid/mode/mtime metadata." >&2
        echo "Pinned Hub object verification below is authoritative for downloaded files." >&2
    fi
fi

printf '%s  %s\n' "${NORM_SHA256}" "${DESTINATION}/${NORM_RELATIVE_PATH}" | sha256sum --check --strict

if [[ "${PROFILE}" == "${YAM_PROFILE}" ]]; then
    printf '%s  %s\n' "${PARAMS_METADATA_SHA256}" "${DESTINATION}/params/_METADATA" | sha256sum --check --strict
    "${HF_BIN}" cache verify \
        "${REPO_ID}" \
        --revision "${REVISION}" \
        --local-dir "${DESTINATION}" \
        --fail-on-missing-files
    echo "Prepared ${PROFILE}: ${DESTINATION}"
else
    "${HF_BIN}" cache verify \
        "${REPO_ID}" \
        --revision "${REVISION}" \
        --local-dir "${DESTINATION}"
    echo "Prepared ${PROFILE}: ${DESTINATION}/checkpoints/${CHECKPOINT_STEP}"
fi
