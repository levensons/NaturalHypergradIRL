#!/bin/bash
#SBATCH --job-name="hopper/sketch_size_grid/64"
#SBATCH --partition=rocky
#SBATCH --constraint="type_d"
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --time=0-72:00
#SBATCH --mail-user=nvsevriukov@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -e

module purge
module load Python
conda activate nhd_env

export GIT_PYTHON_REFRESH=quiet
export SAC_RESET_MODE=policy_critics

python -m src.irl.train \
    --env hopper \
    --method fisher \
    --config config/hopper/sketch_size_grid/64.yaml \
    --checkpoint checkpoints/hopper/sketch_size_grid/64.pt \
    --run-name sketch_size_grid/64/irl_training \
    --log-dir logs/hopper/sketch_size_grid/64/irl_training \
    --n-jobs -1