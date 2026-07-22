#!/usr/bin/env bash
set -euo pipefail

COORD_ROOT=${PI05_ARX5_COORD_ROOT:-/home/ubuntu/pi05-arx5}
LOGIN_HOST=${PI05_ARX5_LOGIN_HOST:-gmicloud-loki-g1-cpu-001}
LOCAL_PORT=${PI05_ARX5_TENSORBOARD_PORT:-16007}
BIND_ADDRESS=${PI05_ARX5_TENSORBOARD_BIND_ADDRESS:-127.0.0.1}
POLL_SECONDS=${PI05_ARX5_TUNNEL_POLL_SECONDS:-20}
LOG_PATH="${COORD_ROOT}/tensorboard-tunnel.log"
LOCK_PATH="${COORD_ROOT}/tensorboard-tunnel.lock"

mkdir -p "${COORD_ROOT}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "A pi05 ARX X5 TensorBoard tunnel already holds ${LOCK_PATH}." >&2
  exit 1
fi

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${LOG_PATH}"
}

active_backend() {
  ssh -o BatchMode=yes "${LOGIN_HOST}" python3 - <<'PY'
import json
from pathlib import Path
import subprocess

pipeline = json.loads(Path("/home/harry/pi05-arx5/pipeline_jobs.json").read_text())
for key in ("training_job_id", "smoke_job_id", "viewer_job_id"):
    job_id = pipeline.get(key)
    if not job_id:
        continue
    result = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T|%N"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if result:
        state, node = result.splitlines()[0].split("|", 1)
        if state == "RUNNING" and node and node != "(null)":
            print(f"{job_id}|{node}")
            break
PY
}

tunnel_pid=
backend=
cleanup() {
  if [[ -n "${tunnel_pid}" ]]; then
    kill "${tunnel_pid}" 2>/dev/null || true
    wait "${tunnel_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

log "TensorBoard tunnel monitor started on ${BIND_ADDRESS}:${LOCAL_PORT}."
while true; do
  desired=$(active_backend || true)
  if [[ -z "${desired}" ]]; then
    if [[ -n "${tunnel_pid}" ]]; then
      log "No active TensorBoard allocation; closing tunnel for ${backend}."
      cleanup
      tunnel_pid=
      backend=
    fi
    sleep "${POLL_SECONDS}"
    continue
  fi

  job_id=${desired%%|*}
  node=${desired#*|}
  if [[ "${backend}" != "${desired}" ]] || [[ -z "${tunnel_pid}" ]] || ! kill -0 "${tunnel_pid}" 2>/dev/null; then
    cleanup
    ssh \
      -N \
      -o BatchMode=yes \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -L "${BIND_ADDRESS}:${LOCAL_PORT}:${node}:6006" \
      "${LOGIN_HOST}" >>"${LOG_PATH}" 2>&1 &
    tunnel_pid=$!
    sleep 2
    if ! kill -0 "${tunnel_pid}" 2>/dev/null; then
      wait "${tunnel_pid}" || true
      log "Tunnel startup failed for job ${job_id} on ${node}; retrying."
      tunnel_pid=
      backend=
      sleep "${POLL_SECONDS}"
      continue
    fi
    backend=${desired}
    log "Forwarding job ${job_id} ${node}:6006 to ${BIND_ADDRESS}:${LOCAL_PORT}."
  fi
  sleep "${POLL_SECONDS}"
done
