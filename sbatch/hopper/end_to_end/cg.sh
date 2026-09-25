#!/bin/bash
#SBATCH --job-name="hopper/end_to_end/cg"
#SBATCH --partition=rocky
#SBATCH --constraint="type_d"
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --mail-user=nvsevriukov@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -e

module purge
module load Python
conda activate nhd_env

export GIT_PYTHON_REFRESH=quiet
export SAC_RESET_MODE=policy_critics

NAME="cg"

CONFIG="config/hopper/end_to_end/${NAME}.yaml"
CHECKPOINT="checkpoints/hopper/end_to_end/${NAME}.pt"
RUN_PREFIX="end_to_end/${NAME}"
LOG_PREFIX="logs/hopper/end_to_end/${NAME}"
REPORT_DIR="reports/metrics/hopper/end_to_end/${NAME}"

# 1. IRL training
timeout --signal=SIGINT 24h python -m src.irl.train \
    --env hopper \
    --method fisher \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --run-name "$RUN_PREFIX/irl_training" \
    --log-dir "$LOG_PREFIX/irl_training" \
    || [ $? -eq 124 ]

# 2. Final policy training on the learned reward
python -m src.irl.policy_final_training \
    --env hopper \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --run-name "$RUN_PREFIX/final_policy_training" \
    --log-dir "$LOG_PREFIX/final_policy_training"

# 3. Final evaluation
python -m src.evaluation.evaluate \
    --env hopper \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --report-dir "$REPORT_DIR"