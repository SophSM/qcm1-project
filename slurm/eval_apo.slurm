#!/bin/bash
# Compare aponeurosis checkpoints on the validation split: sbatch slurm/eval_apo.slurm apo_A apo_B
#SBATCH --job-name=eval_apo
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
for m in "$@"; do
  python src/predict.py --set apo_val --apo models/$m.pt --out val_$m
  python src/evaluate.py --pred val_$m --tag $m | python -c "import sys,json; r=json.load(sys.stdin)['apo']; print('$m', 'dice', round(r['dice']['mean'],4), 'mt_px_edge', round(r['mt_err_px_edge']['mean'],2), round(r['mt_err_px_edge']['median'],2), 'mt_mm', round(r['mt_err_mm']['mean'],3), round(r['mt_err_mm']['median'],3), 'both', r['both_bands_found'])"
done
