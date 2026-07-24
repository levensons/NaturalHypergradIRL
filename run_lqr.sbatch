#!/bin/bash
#SBATCH --job-name="fisher_lqr4d"
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-10:00:00
#SBATCH --output="logs/run_test_%j.out"
#SBATCH --partition=cpu-e-quick
#SBATCH --constraint="type_e"
#SBATCH --signal=TERM@60
#SBATCH --chdir="/home/ivlevin/HypergradIRL/NaturalHypergradIRL"

set -euo pipefail

module restore
export NO_PROXY=nexus.charisma
export no_proxy=nexus.charisma
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
source /opt/software/python/anaconda/2024_02/etc/profile.d/conda.sh
conda activate Py3.12
source ../python_env/bin/activate

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="/home/ivlevin/HypergradIRL/.runtime/osmesa/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Default MLflow logging is sqlite:///mlflow.db from config/neural_lqr.yaml.
# Start the UI on the server with:
#   mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
# Then tunnel from your laptop:
#   ssh -L 5000:127.0.0.1:5000 USER@SERVER

python -m src.irl.lqr.fisher --config config/neural_lqr.yaml
