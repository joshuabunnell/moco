#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH -t 3-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# ===== EDIT THESE FOR YOUR ENVIRONMENT =====
PROJECT_DIR="/home/jpbunnel/moco"
DATA_DIR="/scratch/jpbunnel/cached-tensors"
OUTPUT_DIR="/scratch/jpbunnel/moco-checkpoints"
LOG_DIR="/scratch/jpbunnel/logs"
CONDA_ENV="moco_env"
# ============================================

set -e

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
MASTER_PORT=$((10000 + RANDOM % 50000))

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# main_moco.py uses mp.spawn internally to launch one process per GPU.
# No torchrun needed — just pass --multiprocessing-distributed.
python main_moco.py "${DATA_DIR}" \
    --arch resnet50 \
    --mlp \
    --cos \
    --epochs 200 \
    --batch-size 256 \
    --lr 0.03 \
    --moco-dim 128 \
    --moco-k 16384 \
    --crops-per-volume 20 \
    --moco-m 0.999 \
    --moco-t 0.07 \
    --workers 32 \
    --save-freq 50 \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --dist-url "tcp://localhost:${MASTER_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    --print-freq 5
