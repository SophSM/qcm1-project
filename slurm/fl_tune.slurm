#!/bin/bash
# Run 2 (TRAINING_PLAN_2 §3.2): FL/PA rule tuning on the v2 validation proxy, plus a check that the default
# rule of measure.py (run1) still reproduces submission_v3 exactly.   sbatch slurm/fl_tune.slurm
#SBATCH --job-name=fl_tune
#SBATCH --partition=debugq
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
set -e
module load anaconda3/3.11
source activate "${CONDA_ENV:?set CONDA_ENV to your conda environment path}"
python src/fl_tune.py --workers 16
python src/measure.py --pred test_v3 --out check_run1_v3.csv
python - <<'PY'
import pandas as pd
a = pd.read_csv("outputs/check_run1_v3.csv"); b = pd.read_csv("outputs/submission_v3_alldata.csv")
same = a.equals(b)
print("REGEN_V3_IDENTICAL", same)
if not same:
    print((a.set_index("image_id") - b.set_index("image_id")).abs().max())
PY
echo FL_TUNE_DONE
