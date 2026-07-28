#!/bin/bash
# ESM-2 kNN transfer baseline (Table 4, last row).

#SBATCH --job-name=knn-baseline
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=paper/pfp/knn/logs/%x-%j.out
#SBATCH --error=paper/pfp/knn/logs/%x-%j.err

set -euo pipefail

: "${REPO_ROOT:=/users/adgk852/ELK-Em}"

cd "$REPO_ROOT"
mkdir -p paper/pfp/knn/logs

module load miniforge3/25.3.0-3/none-none/a-j26s7cx
source /opt/gridware/subscribed/pkg/linux-x86_64_v3/miniforge3-25.3.0-3-j26s7cxrqjvmzwlr3ombbwlpkz57gahk/etc/profile.d/conda.sh
conda activate research

export PYTHONUNBUFFERED=1

srun python -m paper.pfp.knn.knn_baseline
