#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH -t 3-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -J moco_resume
#SBATCH -o /scratch/%u/moco/logs/%x.%j.out
#SBATCH -e /scratch/%u/moco/logs/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# Continue MoCo pretraining from the base checkpoint on ONE collection.
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

# DATASET name also becomes the checkpoint subdir under CKPT_ROOT. Override: sbatch --export=DATASET=pediatric jobs/resume_moco.sh
DATASET="${DATASET:-acrin}"
case "${DATASET}" in
    acrin)     DATA_DIR="${TENSOR_ACRIN}" ;;
    pediatric) DATA_DIR="${TENSOR_PEDIATRIC}" ;;
    *) echo "ERROR: DATASET must be 'acrin' or 'pediatric', got '${DATASET}'"; exit 1 ;;
esac

# Resumes from the base run's checkpoint. Override: sbatch --export=DATASET=pediatric,CKPT=checkpoint_0149 jobs/resume_moco.sh
CKPT="${CKPT:-checkpoint_0199}"
EPOCHS="${EPOCHS:-400}"  # total epochs to reach — checkpoint_0199 was saved at epoch 200

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
MASTER_PORT=$((10000 + RANDOM % 50000))

cd "${PROJECT_DIR}"
mkdir -p "${CKPT_ROOT}/${DATASET}" "${LOG_DIR}"

# --moco-k must match the checkpoint's value (16384) — the queue tensor is in the state_dict, so a mismatch fails to load.
python main_moco.py "${DATA_DIR}" \
    --resume "${CKPT_ROOT}/base/${CKPT}.pth.tar" \
    --arch resnet50 \
    --mlp \
    --cos \
    --epochs "${EPOCHS}" \
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
    --output-dir "${CKPT_ROOT}/${DATASET}" \
    --print-freq 5
