#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 0-03:00:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH -J crop_bank
#SBATCH -o /scratch/%u/moco/logs/%x.%j.out
#SBATCH -e /scratch/%u/moco/logs/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# Phase 0 step 1: one pass over tensors/ producing the fixed evaluation crop bank.
# CPU only and comfortably under htc's 4h cap, so it does not need a GPU slot.
# Override: sbatch --export=CROPS=32 jobs/build_crop_bank.sh
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

CROPS="${CROPS:-16}"

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
mkdir -p "${EVAL_DIR}"

# --workers matches -c above: the pass is I/O bound on ~245 MB volume reads, so
# the win comes from having many reads outstanding against BeeGFS at once.
python scripts/eval/build_crop_bank.py \
    --tensor-dirs "${TENSOR_ACRIN}" "${TENSOR_PEDIATRIC}" \
    --output-dir "${EVAL_DIR}" \
    --crops "${CROPS}" \
    --workers 16

echo "Crop bank built. Next: sbatch jobs/eval_repr.sh"
