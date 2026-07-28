#!/bin/bash
# §5.2 single-seed smoke run of ELK-Em on CAFA5 human protein-function data.
#
# For the paper's Table 4 numbers use `submit_multiseed.sh` instead — this
# script trains one seed only.

#SBATCH --job-name=elk-em-4pf
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=paper/pfp/logs/%x-%j.out
#SBATCH --error=paper/pfp/logs/%x-%j.err

set -euo pipefail

: "${REPO_ROOT:=/users/adgk852/ELK-Em}"

cd "$REPO_ROOT"
mkdir -p paper/pfp/logs
mkdir -p checkpoints/pfp

module load miniforge3/25.3.0-3/none-none/a-j26s7cx
source /opt/gridware/subscribed/pkg/linux-x86_64_v3/miniforge3-25.3.0-3-j26s7cxrqjvmzwlr3ombbwlpkz57gahk/etc/profile.d/conda.sh
conda activate research

export PYTHONUNBUFFERED=1

srun python -m paper.pfp.train
