#!/bin/bash
#SBATCH --job-name="hopper/reset/cg/policy_critics"
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
    --config config/hopper/reset/cg.yaml \
    --checkpoint checkpoints/hopper/reset/cg/policy_critics.pt \
    --run-name reset/cg/policy_critics/irl_training \
    --log-dir logs/hopper/reset/cg/policy_critics/irl_training \
    --n-jobs -1