#!/bin/bash
#SBATCH --job-name="hopper/reg_1e0"
#SBATCH --partition=rocky
#SBATCH --constraint="type_d"
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --time=0-24:00
#SBATCH --mail-user=nvsevriukov@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -e

module purge
module load Python
conda activate nhd_env

export GIT_PYTHON_REFRESH=quiet

python -m src.irl.train \
    --env hopper \
    --method fisher \
    --config config/hopper/reg_grid/1e0.yaml \
    --checkpoint checkpoints/hopper/reg_grid/1e0.pt \
    --run-name reg_grid_1e0_irl \
    --log-dir logs/hopper/reg_grid/1e0/irl_training \
    --n-jobs -1

python -m src.irl.policy_final_training \
    --env hopper \
    --config config/hopper/reg_grid/1e0.yaml \
    --checkpoint checkpoints/hopper/reg_grid/1e0.pt \
    --run-name reg_grid_1e0_final \
    --log-dir logs/hopper/reg_grid/1e0/final_policy_training

python -m src.evaluation.evaluate \
    --env hopper \
    --config config/hopper/reg_grid/1e0.yaml \
    --checkpoint checkpoints/hopper/reg_grid/1e0.pt \
    --report-dir reports/metrics/hopper/reg_grid/1e0
