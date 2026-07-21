#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <training-job-id>" >&2
  exit 1
fi

POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_id=$(sbatch --parsable --dependency="afterok:$1" "${POLICY_ROOT}/slurm/pi05_piper_finalize.sbatch")
echo "${job_id}"
