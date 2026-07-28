#!/bin/bash
# §5.1 ranking: one ontology, one seed per submission.
#
# Environment inputs:
#   ONT              GALEN | GO | ANATOMY   (required)
#   SEED             integer 0..4           (required)
#   CKPT_DIR         output root            (default: checkpoints/ranking)
#   N_NEG_PER_POS   corruptions per pos    (default: 5 for GO, 3 for GALEN/ANATOMY)
#   REPO_ROOT        path to ELK-Em/        (default: script's grand-parent dir)
#
# The paper's Table 3 numbers come from CKPT_DIR=checkpoints/ranking (the compat
# variant of the RI1 loss) across seeds 0..4.

#SBATCH --job-name=elk-em
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=paper/ranking/logs/%x-%j.out
#SBATCH --error=paper/ranking/logs/%x-%j.err

set -euo pipefail

: "${ONT:?ONT must be set (e.g. GALEN, GO, ANATOMY)}"
: "${SEED:?SEED must be set (integer)}"
: "${CKPT_DIR:=checkpoints/ranking}"
: "${REPO_ROOT:=/users/adgk852/ELK-Em}"

if [[ -z "${N_NEG_PER_POS:-}" ]]; then
  case "$ONT" in
    GO)                 N_NEG_PER_POS=5 ;;
    GALEN|ANATOMY)      N_NEG_PER_POS=3 ;;
    *)                  N_NEG_PER_POS=3 ;;
  esac
fi

cd "$REPO_ROOT"
mkdir -p paper/ranking/logs

module load miniforge3/25.3.0-3/none-none/a-j26s7cx
source /opt/gridware/subscribed/pkg/linux-x86_64_v3/miniforge3-25.3.0-3-j26s7cxrqjvmzwlr3ombbwlpkz57gahk/etc/profile.d/conda.sh
conda activate research

export PYTHONUNBUFFERED=1

srun python -m paper.ranking.train "$ONT" \
  --seed "$SEED" \
  --device cuda \
  --ckpt-dir "$CKPT_DIR" \
  --n-neg-per-pos "$N_NEG_PER_POS"
