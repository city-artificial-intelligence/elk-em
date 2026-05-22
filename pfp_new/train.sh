#!/bin/bash
#SBATCH --job-name=pfp_new
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=pfp_new/logs/%x-%j.out
#SBATCH --error=pfp_new/logs/%x-%j.err

set -euo pipefail

cd /users/adgk852/ELK-Em
mkdir -p pfp_new/logs
mkdir -p pfp_new/checkpoints

module load miniforge3/25.3.0-3/none-none/a-j26s7cx
source /opt/gridware/subscribed/pkg/linux-x86_64_v3/miniforge3-25.3.0-3-j26s7cxrqjvmzwlr3ombbwlpkz57gahk/etc/profile.d/conda.sh
conda activate research

export PYTHONUNBUFFERED=1

srun python -m pfp_new.train