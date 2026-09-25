#!/bin/bash
#SBATCH --job-name="hopper/end-to-end/fisher_explicit"
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

python -m src.irl.train \
    --env hopper \
    --method fisher \
    --config config/hopper/end_to_end/fisher_explicit.yaml \
    --checkpoint checkpoints/hopper/end_to_end/fisher_explicit.pt \
    --run-name end_to_end/fisher_explicit/irl_training \
    --log-dir logs/hopper/end_to_end/fisher_explicit/irl_training \
    --n-jobs -1

python -m src.irl.policy_final_training \
    --env hopper \
    --config config/hopper/end_to_end/fisher_explicit.yaml \
    --checkpoint checkpoints/hopper/end_to_end/fisher_explicit.pt \
    --run-name end_to_end/fisher_explicit/final_policy_training \
    --log-dir logs/hopper/end_to_end/fisher_explicit/final_policy_training

python -m src.evaluation.evaluate \
    --env hopper \
    --config config/hopper/end_to_end/fisher_explicit.yaml \
    --checkpoint checkpoints/hopper/end_to_end/fisher_explicit.pt \
    --report-dir reports/metrics/hopper/end_to_end/fisher_explicit
