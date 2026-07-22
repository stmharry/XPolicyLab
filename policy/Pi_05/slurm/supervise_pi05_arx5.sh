#!/usr/bin/env bash
set -euo pipefail

COORD_ROOT=${PI05_ARX5_COORD_ROOT:-/home/ubuntu/pi05-arx5}
LOGIN_HOST=${PI05_ARX5_LOGIN_HOST:-gmicloud-loki-g1-cpu-001}
POLL_SECONDS=${PI05_ARX5_POLL_SECONDS:-120}
LOG_PATH="${COORD_ROOT}/supervisor.log"
LOCK_PATH="${COORD_ROOT}/supervisor.lock"

mkdir -p "${COORD_ROOT}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "A pi05 ARX X5 supervisor already holds ${LOCK_PATH}." >&2
  exit 1
fi

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${LOG_PATH}"
}

snapshot() {
  ssh "${LOGIN_HOST}" python3 - <<'PY'
import json
from pathlib import Path
import subprocess

pipeline = Path("/home/harry/pi05-arx5/pipeline_jobs.json")
jobs = json.loads(pipeline.read_text())

def state(job_id):
    active = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"], text=True, capture_output=True, check=True
    ).stdout.strip()
    if active:
        return active.splitlines()[0]
    finished = subprocess.run(
        ["sacct", "-X", "-n", "-j", str(job_id), "-o", "State"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    return finished.splitlines()[0].split()[0].split("+")[0] if finished else "UNKNOWN"

run = Path("/home/harry/pi05-arx5/checkpoints/RoboDojo-real_arx_x5_6task-bimanual_arx_x5-joint-0")
steps = sorted(int(path.name) for path in run.iterdir() if path.is_dir() and path.name.isdigit()) if run.exists() else []
payload = {
    "jobs": jobs,
    "states": {key: state(value) for key, value in jobs.items() if key.endswith("_job_id")},
    "latest_checkpoint": steps[-1] if steps else None,
}
print(json.dumps(payload, sort_keys=True))
PY
}

resume_transient_failure() {
  ssh "${LOGIN_HOST}" bash -s <<'REMOTE'
set -euo pipefail
root=/home/harry/pi05-arx5
pipeline=${root}/pipeline_jobs.json
policy=${root}/RoboDojo/XPolicyLab/policy/Pi_05

readarray -t fields < <(python3 - "${pipeline}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
print(payload["training_job_id"])
print(payload["finalizer_job_id"])
PY
)
train_job=${fields[0]}
finalizer_job=${fields[1]}
state=$(sacct -X -n -j "${train_job}" -o State | awk 'NF {sub(/\+.*/, "", $1); print $1; exit}')
case "${state}" in
  PREEMPTED|NODE_FAIL|BOOT_FAIL|TIMEOUT) ;;
  *) echo "Training job ${train_job} is ${state}; no automatic resume."; exit 0 ;;
esac

run=${root}/checkpoints/RoboDojo-real_arx_x5_6task-bimanual_arx_x5-joint-0
latest=
while IFS= read -r step; do
  if [[ -d "${run}/${step}/params" && -d "${run}/${step}/assets" ]]; then
    latest=${step}
    break
  fi
done < <(find "${run}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -printf '%f\n' 2>/dev/null | sort -nr)
if [[ -z "${latest}" ]]; then
  echo "Refusing resume: nonempty run has no validated numeric checkpoint: ${run}" >&2
  exit 42
fi

scancel "${finalizer_job}" 2>/dev/null || true
new_train=$(sbatch --parsable \
  --export=ALL,PI05_CKPT_NAME=real_arx_x5_6task \
  "${policy}/slurm/pi05_arx5_train.sbatch")
new_finalizer=$(sbatch --parsable \
  --dependency="afterok:${new_train}" \
  "${policy}/slurm/pi05_arx5_finalize.sbatch")
TRAIN_JOB=${new_train} FINALIZER_JOB=${new_finalizer} LATEST=${latest} python3 - "${pipeline}" <<'PY'
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.setdefault("training_attempts", []).append(os.environ["TRAIN_JOB"])
payload.setdefault("resumes", []).append(
    {
        "from_checkpoint": int(os.environ["LATEST"]),
        "replaced_training_job_id": payload["training_job_id"],
        "replaced_finalizer_job_id": payload["finalizer_job_id"],
        "training_job_id": os.environ["TRAIN_JOB"],
        "finalizer_job_id": os.environ["FINALIZER_JOB"],
    }
)
payload["training_job_id"] = os.environ["TRAIN_JOB"]
payload["finalizer_job_id"] = os.environ["FINALIZER_JOB"]
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps(payload["resumes"][-1], sort_keys=True))
PY
REMOTE
}

log "Supervisor started for ${LOGIN_HOST}."
while true; do
  current=$(snapshot)
  log "${current}"
  train_state=$(python3 -c 'import json,sys; p=json.load(sys.stdin); print(p["states"].get("training_job_id", "UNKNOWN"))' <<<"${current}")
  final_state=$(python3 -c 'import json,sys; p=json.load(sys.stdin); print(p["states"].get("finalizer_job_id", "UNKNOWN"))' <<<"${current}")
  if [[ "${train_state}" == "COMPLETED" && "${final_state}" == "COMPLETED" ]]; then
    log "Training and finalizer completed. Supervisor exiting."
    exit 0
  fi
  case "${train_state}" in
    PREEMPTED|NODE_FAIL|BOOT_FAIL|TIMEOUT)
      log "Transient training state ${train_state}; attempting validated-checkpoint resume."
      recovery=$(resume_transient_failure)
      log "${recovery}"
      ;;
  esac
  sleep "${POLL_SECONDS}"
done
