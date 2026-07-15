#!/usr/bin/env bash
set -euo pipefail

REPO_ID="allenai/MolmoAct2-BimanualYAM"
REVISION="8dcbed66f2380e4393189c303ea72488eb9e63c2"
PROFILE="molmoact2_bimanual_yam"
NORM_SHA256="6357a264ac820c5b775c884bd303ed13ffa18a36def28409312c842139879f4b"
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
DESTINATION="${STORAGE_ROOT}/model_weights/MolmoACT2/${PROFILE}/${REVISION}"

ensure_hf
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if [[ "${1:-}" == "--dry-run" ]]; then
    "${HF_BIN}" download "${REPO_ID}" --revision "${REVISION}" --dry-run
    echo "Destination: ${DESTINATION}"
    exit 0
fi

mkdir -p "${DESTINATION}"
"${HF_BIN}" download \
    "${REPO_ID}" \
    --revision "${REVISION}" \
    --local-dir "${DESTINATION}"

printf '%s  %s\n' "${NORM_SHA256}" "${DESTINATION}/norm_stats.json" | sha256sum --check --strict
"${HF_BIN}" cache verify \
    "${REPO_ID}" \
    --revision "${REVISION}" \
    --local-dir "${DESTINATION}" \
    --fail-on-missing-files

echo "Prepared ${PROFILE}: ${DESTINATION}"
