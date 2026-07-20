#!/usr/bin/env bash
set -euo pipefail

POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XPOLICY_ROOT="$(cd "${POLICY_ROOT}/../.." && pwd)"
HEAD_SHA=$(git -C "${XPOLICY_ROOT}" rev-parse HEAD)
BRANCH_SHA=$(git -C "${XPOLICY_ROOT}" ls-remote origin refs/heads/train/pi05-robodojo-real-piper | awk '{print $1}')
if [[ -z "${BRANCH_SHA}" || "${HEAD_SHA}" != "${BRANCH_SHA}" ]]; then
  echo "XPolicyLab HEAD ${HEAD_SHA} is not the pushed train/pi05-robodojo-real-piper commit ${BRANCH_SHA:-<missing>}." >&2
  exit 1
fi

mkdir -p /home/harry/pi05-piper/logs
data_job=$(sbatch --parsable "${POLICY_ROOT}/slurm/pi05_piper_data.sbatch")
smoke_job=$(sbatch --parsable \
  --dependency="afterok:${data_job}" \
  --job-name=pi05-piper-smoke \
  --export=ALL,PI05_CKPT_NAME=real_piper_6task_smoke,OPENPI_NUM_TRAIN_STEPS=2,OPENPI_LOG_INTERVAL=1,OPENPI_SAVE_INTERVAL=1,OPENPI_KEEP_PERIOD=1 \
  "${POLICY_ROOT}/slurm/pi05_piper_train.sbatch")
train_job=$(sbatch --parsable \
  --dependency="afterok:${smoke_job}" \
  --export=ALL,PI05_CKPT_NAME=real_piper_6task \
  "${POLICY_ROOT}/slurm/pi05_piper_train.sbatch")

DATA_JOB="${data_job}" SMOKE_JOB="${smoke_job}" TRAIN_JOB="${train_job}" python - <<'PY'
import json
import os
from pathlib import Path

path = Path("/home/harry/pi05-piper/pipeline_jobs.json")
payload = {
    "data_job_id": os.environ["DATA_JOB"],
    "smoke_job_id": os.environ["SMOKE_JOB"],
    "training_job_id": os.environ["TRAIN_JOB"],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
