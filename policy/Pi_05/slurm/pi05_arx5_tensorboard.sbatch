#!/usr/bin/env bash
#SBATCH --job-name=pi05-arx5-tensorboard
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/harry/pi05-arx5/logs/%x-%j.out
#SBATCH --error=/home/harry/pi05-arx5/logs/%x-%j.err

set -euo pipefail

tensorboard_dir=${1:-/workspace/pi05-arx5/tensorboard/RoboDojo-real_arx_x5_6task-bimanual_arx_x5-joint-0}
srun \
  --container-image=nvcr.io#nvidia/pytorch:24.10-py3 \
  --container-mounts=/home/harry/pi05-arx5:/workspace/pi05-arx5 \
  --container-workdir=/workspace/pi05-arx5/RoboDojo \
  bash /workspace/pi05-arx5/RoboDojo/XPolicyLab/policy/Pi_05/slurm/run_arx5_viewer_in_container.sh "${tensorboard_dir}"
