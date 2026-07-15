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

echo "[SETUP] MolmoACT2 dataset=${dataset} task=${task} checkpoint=${checkpoint} environment=${environment} action=${action} seed=${seed} gpu=${gpu} policy_env=${policy_env}"
if [[ "${checkpoint}" == "molmoact2_bimanual_yam" ]]; then
    bash "${POLICY_DIR}/prepare_checkpoint.sh"
    bash "${POLICY_DIR}/install.sh" infer
else
    bash "${POLICY_DIR}/install.sh"
fi
