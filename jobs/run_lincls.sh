#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 0-06:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -J lincls
#SBATCH -o /scratch/%u/moco/logs/%x.%j.out
#SBATCH -e /scratch/%u/moco/logs/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# Linear probe: freeze a pretrained backbone, train a 3-class polyp head on labeled ACRIN data.
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

# Override: sbatch --export=CKPT=checkpoint_0149,CKPT_RUN=acrin jobs/run_lincls.sh
CKPT="${CKPT:-checkpoint_0199}"
CKPT_RUN="${CKPT_RUN:-base}"

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
MASTER_PORT=$((10000 + RANDOM % 50000))

cd "${PROJECT_DIR}"
mkdir -p "${CKPT_ROOT}/lincls"

python main_lincls.py \
    --data "${TENSOR_ACRIN}" \
    --train-csv "${CSV_DIR}/labels_train.csv" \
    --val-csv "${CSV_DIR}/labels_val.csv" \
    --pretrained "${CKPT_ROOT}/${CKPT_RUN}/${CKPT}.pth.tar" \
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
    --output-dir "${CKPT_ROOT}/lincls" \
    --print-freq 5
