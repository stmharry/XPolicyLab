#!/usr/bin/env bash
#SBATCH --job-name=pi05-piper-tensorboard
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/harry/pi05-piper/logs/%x-%j.out
#SBATCH --error=/home/harry/pi05-piper/logs/%x-%j.err

set -euo pipefail

tensorboard_dir=${1:-/workspace/pi05-piper/tensorboard/RoboDojo-real_piper_6task-bimanual_piper-joint-0}
srun \
  --container-image=nvcr.io#nvidia/pytorch:24.10-py3 \
  --container-mounts=/home/harry/pi05-piper:/workspace/pi05-piper \
  --container-workdir=/workspace/pi05-piper/RoboDojo \
  bash /workspace/pi05-piper/RoboDojo/XPolicyLab/policy/Pi_05/slurm/run_viewer_in_container.sh "${tensorboard_dir}"
