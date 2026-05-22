#!/bin/bash
#SBATCH --job-name=box3el
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=model/ADC/%x-%j.out
#SBATCH --error=model/ADC/%x-%j.err

set -euo pipefail

cd /users/adgk852/ELK-Em
mkdir -p model/ADC

module load miniforge3/25.3.0-3/none-none/a-j26s7cx
source /opt/gridware/subscribed/pkg/linux-x86_64_v3/miniforge3-25.3.0-3-j26s7cxrqjvmzwlr3ombbwlpkz57gahk/etc/profile.d/conda.sh
conda activate research

export PYTHONUNBUFFERED=1

srun python -m model.train ANATOMY --device cuda