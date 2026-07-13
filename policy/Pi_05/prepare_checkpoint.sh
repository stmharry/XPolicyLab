#!/usr/bin/env bash
set -euo pipefail

REPO_ID="pravsels/pi05-arx5-multitask-v1"
REVISION="880fa61406540d80b1c3b9824f12c19b903a233f"
PROFILE="pi05_arx5_multitask_v1"
CHECKPOINT_STEP="55000"
PARAMS_TAR_SHA256="7ee69681991cdc5e04b4759d3bf93bca5dac6bc98639ec7b00202d2f82fe5b2f"
NORM_SHA256="c57c763df05f632b6912b9d1aefbd537d7f05f9c67360ac106451d5d6b9fa32c"
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
    if ! command -v hf >/dev/null 2>&1; then
        echo "Installing the Hugging Face CLI..."
        curl -LsSf https://hf.co/cli/install.sh | bash -s
        export PATH="${HOME}/.local/bin:${PATH}"
    fi
    if [[ -n "${HF_TOKEN:-}" ]]; then
        hf auth whoami >/dev/null
        echo "Hugging Face authentication accepted from HF_TOKEN."
    elif hf auth whoami >/dev/null 2>&1; then
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

if [[ "${1:-}" == "--dry-run" ]]; then
    hf download \
        "${REPO_ID}" \
        --revision "${REVISION}" \
        --include "checkpoints/${CHECKPOINT_STEP}/**" \
        --include "assets/**" \
        --dry-run
    echo "Destination: ${DESTINATION}"
    exit 0
fi

mkdir -p "${DESTINATION}"
hf download \
    "${REPO_ID}" \
    --revision "${REVISION}" \
    --include "checkpoints/${CHECKPOINT_STEP}/**" \
    --include "assets/**" \
    --local-dir "${DESTINATION}"

# The model card prescribes `tar cf - ... | sha256sum`. A plain tar archive
# includes local uid/gid, modes, mtimes, and traversal metadata, so this digest
# is reported as provenance but cannot be a reproducible integrity gate after
# `hf download`. Pinned per-file Hub verification below is the enforced gate.
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
printf '%s  %s\n' "${NORM_SHA256}" "${DESTINATION}/assets/norm_stats.json" | sha256sum --check --strict
hf cache verify \
    "${REPO_ID}" \
    --revision "${REVISION}" \
    --local-dir "${DESTINATION}"

echo "Prepared ${PROFILE}: ${DESTINATION}/checkpoints/${CHECKPOINT_STEP}"
