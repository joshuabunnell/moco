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
# checkpoint subdir under CKPT_ROOT: acrin (default, the primary track) = CT
# COLONOGRAPHY only; base = both collections, kept only as the "does out-of-domain
# data help" ablation; pediatric = Pediatric-CT-SEG alone, which is out-of-domain
# for the polyp task. Override: sbatch --export=DATASET=base jobs/train_moco.sh
# RUN names the checkpoint subdir (default: DATASET). Give every ladder rung its
# own: sbatch --export=RUN=e0 jobs/train_moco.sh. SAVE_FREQ=10 gives the epoch-20
# checkpoint the kill gate is scored on.
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

DATASET="${DATASET:-acrin}"
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
RUN="${RUN:-${DATASET}}"
SAVE_FREQ="${SAVE_FREQ:-50}"
OUT_DIR="${CKPT_ROOT}/${RUN}"

# A fresh run never shares a directory: same-numbered checkpoints would overwrite,
# and different-numbered ones would pass for one run. Resuming is resume_moco.sh.
if compgen -G "${OUT_DIR}/checkpoint_*.pth.tar" > /dev/null; then
    echo "ERROR: ${OUT_DIR} already holds checkpoints; pick a new RUN"
    exit 1
fi

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
MASTER_PORT=$((10000 + RANDOM % 50000))

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

source jobs/stage_data.sh
trap 'rm -rf "${TMPDIR:-/tmp}/moco_stage"' EXIT
DATA_DIR=$(stage_data "${DATA_DIR}")

# The code that actually ran, captured at job start rather than submission, so
# edits made while the job sat in the queue are recorded too.
git rev-parse HEAD > "${OUT_DIR}/git_commit.txt"
git diff HEAD > "${OUT_DIR}/git_diff.patch"
git status --short > "${OUT_DIR}/git_status.txt"
echo "RUN=${RUN} DATASET=${DATASET} SLURM_JOB_ID=${SLURM_JOB_ID}" > "${OUT_DIR}/job.txt"

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
    --save-freq "${SAVE_FREQ}" \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --dist-url "tcp://localhost:${MASTER_PORT}" \
    --output-dir "${OUT_DIR}" \
    --print-freq 5
