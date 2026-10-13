#!/bin/bash
#SBATCH --job-name="hopper/fisher_diagnostics/512"
#SBATCH --partition=rocky
#SBATCH --constraint="type_d"
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --time=0-48:00
#SBATCH --mail-user=nvsevriukov@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -e

module purge
module load Python
conda activate nhd_env

export GIT_PYTHON_REFRESH=quiet
export SAC_RESET_MODE=policy_critics

NAME="512"

CONFIG="config/hopper/fisher_diagnostics/${NAME}.yaml"
CHECKPOINT="checkpoints/hopper/fisher_diagnostics/${NAME}.pt"
RUN_PREFIX="fisher_diagnostics/${NAME}"
LOG_PREFIX="logs/hopper/fisher_diagnostics/${NAME}"
REPORT_DIR="reports/metrics/hopper/fisher_diagnostics/${NAME}"

# 1. IRL training
python -m src.irl.train \
    --env hopper \
    --method fisher \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --run-name "$RUN_PREFIX/irl_training" \
    --log-dir "$LOG_PREFIX/irl_training"