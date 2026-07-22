#!/usr/bin/env bash
#SBATCH --job-name=pi05-arx5-data
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/harry/pi05-arx5/logs/%x-%j.out
#SBATCH --error=/home/harry/pi05-arx5/logs/%x-%j.err

set -euo pipefail

srun \
  --container-image=nvcr.io#nvidia/pytorch:24.10-py3 \
  --container-mounts=/home/harry/pi05-arx5:/workspace/pi05-arx5 \
  --container-workdir=/workspace/pi05-arx5/RoboDojo \
  bash /workspace/pi05-arx5/RoboDojo/XPolicyLab/policy/Pi_05/slurm/run_arx5_data_in_container.sh
