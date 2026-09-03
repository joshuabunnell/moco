#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100.20gb:1
#SBATCH -t 0-02:00:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH -J eval_repr
#SBATCH -o /scratch/%u/moco/logs/%x.%j.out
#SBATCH -e /scratch/%u/moco/logs/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# Phase 0 step 2: score every encoder on the crop bank so the three-way control
# comparison lands in one log. Pure ResNet-50 inference over ~5 GB, so a 20 GB
# MIG slice is plenty and schedules faster than a whole A100.
# Override: sbatch --export=ENCODERS="imagenet moco",CKPT=checkpoint_0149,CKPT_RUN=acrin jobs/eval_repr.sh
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

ENCODERS="${ENCODERS:-random imagenet moco}"
CKPT="${CKPT:-checkpoint_0199}"
CKPT_RUN="${CKPT_RUN:-base}"

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

if [ ! -f "${EVAL_DIR}/bank.npy" ]; then
    echo "ERROR: no crop bank at ${EVAL_DIR}/bank.npy — run sbatch jobs/build_crop_bank.sh first"
    exit 1
fi

for enc in ${ENCODERS}; do
    echo "=============================================================="
    if [ "${enc}" = "moco" ]; then
        # Tag the JSON with the run and checkpoint so several MoCo runs can be
        # scored into the same EVAL_DIR without overwriting each other.
        python scripts/eval/eval_repr.py \
            --encoder moco \
            --checkpoint "${CKPT_ROOT}/${CKPT_RUN}/${CKPT}.pth.tar" \
            --bank-dir "${EVAL_DIR}" \
            --csv-dir "${CSV_DIR}" \
            --output "${EVAL_DIR}/eval_moco_${CKPT_RUN}_${CKPT}.json"
    else
        python scripts/eval/eval_repr.py \
            --encoder "${enc}" \
            --bank-dir "${EVAL_DIR}" \
            --csv-dir "${CSV_DIR}" \
            --output "${EVAL_DIR}/eval_${enc}.json"
    fi
done

echo "=============================================================="
echo "Reports written to ${EVAL_DIR}/eval_*.json"
