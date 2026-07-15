#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
mkdir -p "${TMP}/bin" "${TMP}/project/.venv/bin" "${TMP}/checkpoint"
touch "${TMP}/project/pyproject.toml" "${TMP}/project/uv.lock"

cat >"${TMP}/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
printf '0\n1\n'
EOF
cat >"${TMP}/bin/uv" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
cat >"${TMP}/project/.venv/bin/python" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "-c" && "${2:-}" == *"import XPolicyLab, molmo,"* ]]; then
    echo "unexpected import of nonexistent molmo module" >&2
    exit 1
fi
exit 0
EOF
chmod +x "${TMP}/bin/nvidia-smi" "${TMP}/bin/uv" "${TMP}/project/.venv/bin/python"

set +e
PATH="${TMP}/bin:${PATH}" "${ROOT}/policy/Pi_05/check_eval_policy.sh" \
    RoboDojo task "${TMP}/checkpoint" arx_x5 joint 0 0 "${TMP}/project" >"${TMP}/pi.log" 2>&1
pi_status=$?
set -e
[[ ${pi_status} -eq 3 ]]
grep -q '^WARN checkpoint:' "${TMP}/pi.log"

if PATH="${TMP}/bin:${PATH}" "${ROOT}/policy/Pi_05/check_eval_policy.sh" \
    RoboDojo task pi05_yam_molmoact2 arx_x5 joint 0 0 "${TMP}/project" >"${TMP}/pi-invalid.log" 2>&1; then
    echo "Pi_05 accepted the wrong environment" >&2
    exit 1
fi
grep -q 'requires env bimanual_yam' "${TMP}/pi-invalid.log"

set +e
PATH="${TMP}/bin:${PATH}" "${ROOT}/policy/MolmoACT2/check_eval_policy.sh" \
    RoboDojo task "${TMP}/checkpoint" bimanual_yam joint 0 1 "${TMP}/project" >"${TMP}/molmo.log" 2>&1
molmo_status=$?
set -e
[[ ${molmo_status} -eq 3 ]]
grep -q '^WARN checkpoint:' "${TMP}/molmo.log"

if PATH="${TMP}/bin:${PATH}" "${ROOT}/policy/MolmoACT2/check_eval_policy.sh" \
    RoboDojo task "${TMP}/missing" bimanual_yam joint 0 1 "${TMP}/project" >"${TMP}/molmo-missing.log" 2>&1; then
    echo "MolmoACT2 accepted a missing explicit checkpoint" >&2
    exit 1
fi
grep -q 'explicit checkpoint does not exist' "${TMP}/molmo-missing.log"

grep -q 'prepare_checkpoint.sh.*checkpoint' "${ROOT}/policy/Pi_05/prepare_eval_policy.sh"
grep -q 'install.sh.*infer' "${ROOT}/policy/MolmoACT2/prepare_eval_policy.sh"

echo "policy preflight hook tests passed"
