#!/bin/bash
# Full inference chain for one model pair. Submit from the project root:
#   sbatch --export=ALL,APO="models/apo_A.pt models/apo_B.pt",FASC=models/fasc_A.pt,TAG=final,OUT=submission.csv slurm/predict.slurm
# (several checkpoints per task, space-separated = ensemble: probabilities averaged)
# Steps: test masks -> validation masks -> validation metrics -> pixel scales -> measurements + submission.
#SBATCH --job-name=predict
#SBATCH --partition=a30q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/%x_%j.out
set -e
module load anaconda3/3.11
source activate "${CONDA_ENV:?set CONDA_ENV to your conda environment path}"
export HF_HUB_OFFLINE=1
python src/predict.py --set test     --apo $APO --fasc $FASC --out "test_$TAG"
python src/predict.py --set apo_val  --apo $APO              --out "val_$TAG"
python src/predict.py --set fasc_val              --fasc $FASC --out "val_$TAG"
python src/evaluate.py --pred "val_$TAG" --tag "$TAG"
python src/scale.py
python src/measure.py --pred "test_$TAG" --out "$OUT" --figs
echo PREDICT_DONE
