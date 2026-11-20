#!/bin/bash
#SBATCH --job-name=hopper/fisher_diagnostics/all
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --time=0-48:00
#SBATCH --mail-user=nvsevriukov@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -e

module purge
module load Python
conda activate nhd_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export GIT_PYTHON_REFRESH=quiet
export SAC_RESET_MODE=policy_critics

for NAME in 1 2 4 8 16 32 64 128 256 512; do

    CONFIG="config/hopper/fisher_diagnostics/${NAME}.yaml"
    CHECKPOINT="checkpoints/hopper/fisher_diagnostics/${NAME}.pt"
    RUN_PREFIX="fisher_diagnostics/${NAME}"
    LOG_PREFIX="logs/hopper/fisher_diagnostics/${NAME}"

    echo "=================================="
    echo "Starting configuration: $NAME"
    echo "=================================="

    python -m src.irl.train \
        --env hopper \
        --method fisher \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --run-name "$RUN_PREFIX/irl_training" \
        --log-dir "$LOG_PREFIX/irl_training"

done