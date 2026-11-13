#!/bin/bash
#SBATCH --job-name=hopper/svd_vs_qr/all
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --time=0-72:00
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

for SIZE in 1 2 4 8 16 32 64 128 256 512; do
    for VARIANT in svd qr; do

        NAME="${SIZE}_${VARIANT}"

        CONFIG="config/hopper/svd_vs_qr/${NAME}.yaml"
        CHECKPOINT="checkpoints/hopper/svd_vs_qr/${NAME}.pt"
        RUN_PREFIX="svd_vs_qr/${NAME}"
        LOG_PREFIX="logs/hopper/svd_vs_qr/${NAME}"

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
done