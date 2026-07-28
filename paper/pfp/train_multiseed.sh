#!/bin/bash
# §5.2 multi-seed launcher.
#
# Reads the following from the environment (all optional; sensible defaults):
#   ELK_SEED             seed (default 42)
#   ELK_LEX_WEIGHT       lex regularisation weight (default 10.0)
#   ELK_EPOCHS           training epochs (default 15000)
#   ELK_CHECKPOINT_PATH  where to save the checkpoint + sidecar JSONs
#   ELK_WANDB_RUN_NAME   wandb run name (default init-model-smoke)
#   ELK_WANDB_PROJECT    wandb project      (default elk-em-4pf)
#   ELK_USE_WANDB        1 to enable wandb, 0 to disable (default 1)
#   REPO_ROOT            path to ELK-Em/    (default /users/adgk852/ELK-Em)
#
# Job name is set via `sbatch --job-name=...` from the submitter.

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=paper/pfp/logs/%x-%j.out
#SBATCH --error=paper/pfp/logs/%x-%j.err

set -euo pipefail

: "${REPO_ROOT:=/users/adgk852/ELK-Em}"

cd "$REPO_ROOT"
mkdir -p paper/pfp/logs
mkdir -p "$(dirname "${ELK_CHECKPOINT_PATH:-checkpoints/pfp/last_model.pt}")"

module load miniforge3/25.3.0-3/none-none/a-j26s7cx
source /opt/gridware/subscribed/pkg/linux-x86_64_v3/miniforge3-25.3.0-3-j26s7cxrqjvmzwlr3ombbwlpkz57gahk/etc/profile.d/conda.sh
conda activate research

export PYTHONUNBUFFERED=1

echo "== §5.2 multi-seed run =="
echo "  seed:          ${ELK_SEED:-42}"
echo "  lex_weight:    ${ELK_LEX_WEIGHT:-10.0}"
echo "  epochs:        ${ELK_EPOCHS:-15000}"
echo "  checkpoint:    ${ELK_CHECKPOINT_PATH:-checkpoints/pfp/last_model.pt}"
echo "  wandb_run:     ${ELK_WANDB_RUN_NAME:-init-model-smoke}"

srun python -m paper.pfp.train
