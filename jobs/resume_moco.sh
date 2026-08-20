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

# Continue MoCo pretraining from the base checkpoint on ONE collection.
# Paths come from jobs/config.sh.
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

# Which collection to continue on: acrin (default) or pediatric. The name is
# also the checkpoint subdir written under CKPT_ROOT.
# Override: sbatch --export=DATASET=pediatric jobs/resume_moco.sh
DATASET="${DATASET:-acrin}"
case "${DATASET}" in
    acrin)     DATA_DIR="${TENSOR_ACRIN}" ;;
    pediatric) DATA_DIR="${TENSOR_PEDIATRIC}" ;;
    *) echo "ERROR: DATASET must be 'acrin' or 'pediatric', got '${DATASET}'"; exit 1 ;;
esac

# Resume from the last checkpoint of the mixed-dataset (base) run.
# Override: sbatch --export=DATASET=pediatric,CKPT=checkpoint_0149 jobs/resume_moco.sh
CKPT="${CKPT:-checkpoint_0199}"
# Total epochs to reach (checkpoint_0199 saved at epoch 200, so 400 = +200 more).
EPOCHS="${EPOCHS:-400}"

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
MASTER_PORT=$((10000 + RANDOM % 50000))

cd "${PROJECT_DIR}"
mkdir -p "${CKPT_ROOT}/${DATASET}" "${LOG_DIR}"

# NOTE: --moco-k must match the value used in the checkpoint being resumed.
# The base run used --moco-k 16384; changing the queue size mid-training causes
# a shape mismatch on load because the queue tensor is in the state_dict.
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
