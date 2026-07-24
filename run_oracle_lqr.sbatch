#!/bin/bash
#SBATCH --job-name="oracle_SAC_lqr4d"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-06:00:00
#SBATCH --output="logs/oracle_sac_neural_lqr_4d_%j.out"
#SBATCH --partition=normal
#SBATCH --constraint="type_c"
#SBATCH --signal=TERM@60

set -euo pipefail

module restore default
source /opt/software/python/anaconda/2024_02/etc/profile.d/conda.sh
conda activate Py3.12
source ../python_env/bin/activate

export NO_PROXY=nexus.charisma
export no_proxy=nexus.charisma

# Trains the custom SAC implementation directly on the 4D LQR environment reward.
# This is the oracle sanity check for the Fisher-NHD inner optimizer.
python -m src.irl.lqr.sac_oracle --config config/neural_lqr.yaml
