#!/usr/bin/env bash
#SBATCH --job-name=pi05-piper-train
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/harry/pi05-piper/logs/%x-%j.out
#SBATCH --error=/home/harry/pi05-piper/logs/%x-%j.err

set -euo pipefail

srun \
  --container-image=nvcr.io#nvidia/pytorch:24.10-py3 \
  --container-mounts=/home/harry/pi05-piper:/workspace/pi05-piper \
  --container-workdir=/workspace/pi05-piper/RoboDojo \
  bash /workspace/pi05-piper/RoboDojo/XPolicyLab/policy/Pi_05/slurm/run_train_in_container.sh
