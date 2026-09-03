#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH -t 3-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -J moco_train
#SBATCH -o /scratch/%u/moco/logs/%x.%j.out
#SBATCH -e /scratch/%u/moco/logs/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# MoCo v2 pretraining from scratch. DATASET picks the tensor pool and the
# checkpoint subdir under CKPT_ROOT: base = both collections (ACRIN + Pediatric),
# acrin/pediatric = one collection. Override: sbatch --export=DATASET=acrin jobs/train_moco.sh
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

DATASET="${DATASET:-base}"
case "${DATASET}" in
    base)      DATA_DIR="${TENSOR_DIR}" ;;
    acrin)     DATA_DIR="${TENSOR_ACRIN}" ;;
    pediatric) DATA_DIR="${TENSOR_PEDIATRIC}" ;;
    *) echo "ERROR: DATASET must be 'base', 'acrin', or 'pediatric', got '${DATASET}'"; exit 1 ;;
esac

# Pediatric-CT-SEG has ~354 volumes, so one epoch of crops is smaller than a
# 16384 queue. Shrink it for that run: sbatch --export=DATASET=pediatric,MOCO_K=4096 jobs/train_moco.sh
MOCO_K="${MOCO_K:-16384}"
EPOCHS="${EPOCHS:-200}"

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
MASTER_PORT=$((10000 + RANDOM % 50000))

cd "${PROJECT_DIR}"
mkdir -p "${CKPT_ROOT}/${DATASET}" "${LOG_DIR}"

# main_moco.py uses mp.spawn internally — no torchrun needed, just --multiprocessing-distributed.
python main_moco.py "${DATA_DIR}" \
    --arch resnet50 \
    --mlp \
    --cos \
    --epochs "${EPOCHS}" \
    --batch-size 256 \
    --lr 0.03 \
    --moco-dim 128 \
    --moco-k "${MOCO_K}" \
    --crops-per-volume 20 \
    --moco-m 0.999 \
    --moco-t 0.07 \
    --workers 32 \
    --save-freq 50 \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --dist-url "tcp://localhost:${MASTER_PORT}" \
    --output-dir "${CKPT_ROOT}/${DATASET}" \
    --print-freq 5
