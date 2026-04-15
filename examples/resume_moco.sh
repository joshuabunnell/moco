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
DATA_DIR="/scratch/jpbunnel/cached-tensors/CT-Colonography"  # ACRIN only
OUTPUT_DIR="/scratch/jpbunnel/moco-checkpoints"
LOG_DIR="/scratch/jpbunnel/logs"
CONDA_ENV="moco_env"

# Resume from the last checkpoint of the mixed-dataset run.
# Set CKPT to override: sbatch --export=CKPT=checkpoint_0149 examples/resume_moco.sh
CKPT="${CKPT:-checkpoint_0199}"

# Total epochs (training runs from the checkpoint's saved epoch up to this).
# e.g. checkpoint_0199 saved at epoch 200, so EPOCHS=400 trains 200 more.
EPOCHS=400
# ============================================

set -e

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
MASTER_PORT=$((10000 + RANDOM % 50000))

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# NOTE: --moco-k must match the value used in the checkpoint being resumed.
# The first run used --moco-k 16384, so we must keep it here. Changing the
# queue size mid-training causes a shape mismatch on checkpoint load because
# the queue tensor is saved in the state_dict.
python main_moco.py "${DATA_DIR}" \
    --resume "${OUTPUT_DIR}/${CKPT}.pth.tar" \
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
    --output-dir "${OUTPUT_DIR}" \
    --print-freq 5
