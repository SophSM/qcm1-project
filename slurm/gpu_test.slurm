#!/bin/bash
# Quick check that PyTorch sees a GPU. Submit from the project folder:  sbatch slurm/gpu_test.slurm
#SBATCH --job-name=gpu_test
#SBATCH --partition=a30q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/logs/%x_%j.out

module load anaconda3/3.11
source activate "${CONDA_ENV:?set CONDA_ENV to your conda environment path}"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
