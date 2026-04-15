#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 0-06:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# ===== EDIT THESE FOR YOUR ENVIRONMENT =====
PROJECT_DIR="/home/jpbunnel/moco"
DATA_DIR="/scratch/jpbunnel/cached-tensors/CT-Colonography"
CHECKPOINT_DIR="/scratch/jpbunnel/moco-checkpoints"
OUTPUT_DIR="/scratch/jpbunnel/lincls-checkpoints"
CSV_DIR="${PROJECT_DIR}/metadata/csv_metadata"
CONDA_ENV="moco_env"

# Which pretrained checkpoint to evaluate
CKPT="${CKPT:-checkpoint_0199}"
# ============================================

set -e

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
MASTER_PORT=$((10000 + RANDOM % 50000))

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

python main_lincls.py \
    --data "${DATA_DIR}" \
    --train-csv "${CSV_DIR}/labels_train.csv" \
    --val-csv "${CSV_DIR}/labels_val.csv" \
    --pretrained "${CHECKPOINT_DIR}/${CKPT}.pth.tar" \
    --num-classes 3 \
    --arch resnet50 \
    --epochs 100 \
    --lr 30.0 \
    --schedule 60 80 \
    --batch-size 256 \
    --crops-per-volume 5 \
    --workers 16 \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --dist-url "tcp://localhost:${MASTER_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    --print-freq 5
