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

echo "[SETUP] Pi_05 dataset=${dataset} task=${task} checkpoint=${checkpoint} environment=${environment} action=${action} seed=${seed} gpu=${gpu} policy_env=${policy_env}"
case "${checkpoint}" in
    pi05_arx5_multitask_v1|pi05_yam_molmoact2|pi05_yam_abc_pickplace|pi05_piper_bimanual_v1)
        bash "${POLICY_DIR}/prepare_checkpoint.sh" "${checkpoint}"
        ;;
esac
bash "${POLICY_DIR}/install.sh"
