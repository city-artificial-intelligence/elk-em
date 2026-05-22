#!/bin/bash
#SBATCH -D /users/adgk852/Box3EL
#SBATCH --job-name=go-gci0-closure
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=47G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
mkdir -p slurm_logs

module load miniforge3/25.3.0-3/none-none/a-j26s7cx
source /opt/gridware/subscribed/pkg/linux-x86_64_v3/miniforge3-25.3.0-3-j26s7cxrqjvmzwlr3ombbwlpkz57gahk/etc/profile.d/conda.sh
conda activate research

export PYTHONUNBUFFERED=1

# JVM heap: 80% of SLURM allocation
if [[ -z "${SEMELEL_JVM_HEAP:-}" ]]; then
    if [[ -n "${SLURM_MEM_PER_NODE:-}" && "${SLURM_MEM_PER_NODE}" != "0" ]]; then
        heap_mb=$(( SLURM_MEM_PER_NODE * 80 / 100 ))
        (( heap_mb < 1024 )) && heap_mb=1024
        export SEMELEL_JVM_HEAP="${heap_mb}m"
    else
        export SEMELEL_JVM_HEAP=200g
    fi
fi

printf '[%s] Job %s starting on %s\n' "$(date '+%F %T')" "${SLURM_JOB_ID:-local}" "$(hostname)"
printf '[%s] JVM heap: %s\n'          "$(date '+%F %T')" "$SEMELEL_JVM_HEAP"
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())"

srun python alt_data/tools/compute_go_gci0_closure.py \
    --data-dir      alt_data/GO \
    --ontology-file go.owl

printf '[%s] Job %s finished\n' "$(date '+%F %T')" "${SLURM_JOB_ID:-local}"
