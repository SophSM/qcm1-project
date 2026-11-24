#!/bin/bash
# CPU preprocessing: crop boxes, train/val split, enhanced-input cache. Submit from the project root:
#   sbatch slurm/prep.slurm
#SBATCH --job-name=prep
#SBATCH --partition=debugq
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/%x_%j.out
set -e
module load anaconda3/3.11
source activate "${CONDA_ENV:?set CONDA_ENV to your conda environment path}"
export OMP_NUM_THREADS=1
python src/crop.py
python src/split.py
python src/preprocess.py --task apo  --variants B C D --workers 16
python src/preprocess.py --task fasc --variants B C D --workers 16
python src/preprocess.py --figure
echo PREP_DONE
