#!/bin/bash
# Submit the 6 SLURM jobs that produce the paper's Table 4 numbers:
# seeds {0, 1, 2} x configs {full, no_lex}.
#
#   full   — ELK-Em with lexical regularisation (lex_weight=10.0)
#   no_lex — ablation with lexical regularisation disabled (lex_weight=0.0)
#
# Checkpoints and sidecar JSONs land under:
#   checkpoints/pfp/{full,no_lex}/seed_{s}/last_model.pt
#   checkpoints/pfp/{full,no_lex}/seed_{s}/last_model.efficiency.json
#   checkpoints/pfp/{full,no_lex}/seed_{s}/last_model.eval.json
#
# After all 6 jobs have finished, aggregate with:
#   python -m paper.pfp.aggregate_results
#
# Usage:
#   bash paper/pfp/submit_multiseed.sh                     # submit all 6
#   ELK_EPOCHS=200 bash paper/pfp/submit_multiseed.sh      # smoke run
#   REPO_ROOT=/path/to/ELK-Em bash paper/pfp/submit_multiseed.sh

set -euo pipefail

: "${REPO_ROOT:=/users/adgk852/ELK-Em}"

cd "$REPO_ROOT"

SEEDS=(0 1 2)
CONFIGS=(full no_lex)

for cfg in "${CONFIGS[@]}"; do
  case "$cfg" in
    full)   lex_weight=10.0 ;;
    no_lex) lex_weight=0.0  ;;
    *) echo "unknown config $cfg" >&2; exit 1 ;;
  esac

  for seed in "${SEEDS[@]}"; do
    ckpt_dir="${CKPT_ROOT:-checkpoints}/pfp/${cfg}/seed_${seed}"
    ckpt_path="${ckpt_dir}/last_model.pt"
    run_name="elk-em-4pf-${cfg}-seed${seed}"
    mkdir -p "$ckpt_dir"

    echo "Submitting: cfg=${cfg} seed=${seed} lex_weight=${lex_weight}"

    sbatch \
      --job-name="${run_name}" \
      --export=ALL,ELK_SEED="${seed}",ELK_LEX_WEIGHT="${lex_weight}",ELK_CHECKPOINT_PATH="${ckpt_path}",ELK_WANDB_RUN_NAME="${run_name}",ELK_EPOCHS="${ELK_EPOCHS:-15000}",ELK_USE_WANDB="${ELK_USE_WANDB:-1}",REPO_ROOT="${REPO_ROOT}" \
      paper/pfp/train_multiseed.sh
  done
done
