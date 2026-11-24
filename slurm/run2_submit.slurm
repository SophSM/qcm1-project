#!/bin/bash
# Run 2 (TRAINING_PLAN_2 §3.4): chosen FL rule on the saved v3 and v2 test masks (CPU only, no GPU).
#   sbatch --export=ALL,RULE=<name> slurm/run2_submit.slurm
#SBATCH --job-name=run2_sub
#SBATCH --partition=debugq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/%x_%j.out
set -e
module load anaconda3/3.11
source activate "${CONDA_ENV:?set CONDA_ENV to your conda environment path}"
python src/measure.py --pred test_v3 --rule "$RULE" --out "submission_v4_${RULE}_v3models.csv" --same_mt submission_v3_alldata.csv --figs
python src/measure.py --pred test_v2 --rule "$RULE" --out "submission_v5_${RULE}_v2models.csv" --same_mt submission_v2_apoAB_fascA.csv
echo RUN2_SUBMIT_DONE
