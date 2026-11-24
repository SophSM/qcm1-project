#!/bin/bash
# Run 2 (TRAINING_PLAN_2 §3.1): v2 aponeurosis ensemble on the fascicle validation images.
#   sbatch slurm/predict_fasc_apo.slurm
#SBATCH --job-name=fasc_apo
#SBATCH --partition=a30q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/%x_%j.out
set -e
module load anaconda3/3.11
source activate "${CONDA_ENV:?set CONDA_ENV to your conda environment path}"
export HF_HUB_OFFLINE=1
python src/predict.py --set fasc_val --apo models/apo_A.pt models/apo_B.pt --out val2_fasc_apo
echo PREDICT_DONE
