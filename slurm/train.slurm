#!/bin/bash
# Train one model on a GPU node. Submit from the project root, arguments go to src/train.py:
#   sbatch --job-name=apo_B  slurm/train.slurm --task apo  --variant B --epochs 60 --name apo_best
#   sbatch --job-name=fasc_B slurm/train.slurm --task fasc --variant B --epochs 60 --name fasc_best
# Use --partition=a100q (on the sbatch command line) if a30q queues for long.
#SBATCH --job-name=train
#SBATCH --partition=a30q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/%x_%j.out
set -e
module load anaconda3/3.11
source activate "${CONDA_ENV:?set CONDA_ENV to your conda environment path}"
export HF_HUB_OFFLINE=1   # compute nodes may have no internet: use the cached ImageNet weights
nvidia-smi --query-gpu=name,memory.total --format=csv
python src/train.py "$@"
