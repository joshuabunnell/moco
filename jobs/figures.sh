#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100.20gb:1
#SBATCH -t 0-01:30:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH -J figures
#SBATCH -o /scratch/%u/moco/logs/%x.%j.out
#SBATCH -e /scratch/%u/moco/logs/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# Embed encoders on the crop bank, then draw the explanatory figures into
# docs/figures (scripts/eval/figures.py). RUNS are MoCo run dirs at CKPT; the
# last one listed is the model compared against ImageNet.
# Override: sbatch --export=ALL,RUNS="e1 e2" jobs/figures.sh
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

RUNS="${RUNS:-e1}"
CKPT="${CKPT:-checkpoint_0199}"

module load mamba/latest
source activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_DIR}"

FIG=scripts/eval/figures.py
python "${FIG}" embed --bank-dir "${EVAL_DIR}" --encoder imagenet --name imagenet
for run in ${RUNS}; do
    python "${FIG}" embed --bank-dir "${EVAL_DIR}" --encoder moco \
        --checkpoint "${CKPT_ROOT}/${run}/${CKPT}.pth.tar" --name "${run}"
done
LAST=$(echo ${RUNS} | awk '{print $NF}')
python "${FIG}" plot --bank-dir "${EVAL_DIR}" --out-dir docs/figures --compare imagenet "${LAST}"
