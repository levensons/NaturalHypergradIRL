#!/bin/bash
#SBATCH --job-name="hopper/reset/sketch/policy_critics_replay_buffer"
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
export SAC_RESET_MODE=policy_critics_replay_buffer

python -m src.irl.train \
    --env hopper \
    --method fisher \
    --config config/hopper/reset/sketch.yaml \
    --checkpoint checkpoints/hopper/reset/sketch/policy_critics_replay_buffer.pt \
    --run-name reset/sketch/policy_critics_replay_buffer/irl_training \
    --log-dir logs/hopper/reset/sketch/policy_critics_replay_buffer/irl_training \
    --n-jobs -1