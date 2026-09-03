#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 0-00:30:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -J umap
#SBATCH -o /scratch/%u/moco/logs/%x.%j.out
#SBATCH -e /scratch/%u/moco/logs/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# UMAP projection of backbone features from a pretrained checkpoint.
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
mkdir -p "${UMAP_DIR}"

# Override: sbatch --export=CKPT=checkpoint_0099,CKPT_RUN=acrin jobs/run_umap.sh
CKPT="${CKPT:-checkpoint_0199}"
CKPT_RUN="${CKPT_RUN:-base}"

python scripts/visualize_umap.py \
    --checkpoint "${CKPT_ROOT}/${CKPT_RUN}/${CKPT}.pth.tar" \
    --data "${TENSOR_DIR}" \
    --output "${UMAP_DIR}/umap_${CKPT_RUN}_${CKPT}.png" \
    --crops 3
