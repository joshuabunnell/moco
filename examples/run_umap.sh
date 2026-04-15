#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 0-00:30:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# ===== EDIT THESE FOR YOUR ENVIRONMENT =====
PROJECT_DIR="/home/jpbunnel/moco"
DATA_DIR="/scratch/jpbunnel/cached-tensors"
CHECKPOINT_DIR="/scratch/jpbunnel/moco-checkpoints"
OUTPUT_DIR="/scratch/jpbunnel/umap-plots"
CONDA_ENV="moco_env"
# ============================================

set -e

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Default to latest checkpoint; override with: sbatch --export=CKPT=checkpoint_0099 examples/run_umap.sh
CKPT="${CKPT:-checkpoint_0199}"

python scripts/visualize_umap.py \
    --checkpoint "${CHECKPOINT_DIR}/${CKPT}.pth.tar" \
    --data "${DATA_DIR}" \
    --output "${OUTPUT_DIR}/umap_${CKPT}.png" \
    --crops 3
