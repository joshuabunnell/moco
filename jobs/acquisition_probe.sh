#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100.20gb:1
#SBATCH -t 0-02:00:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH -J acq_probe
#SBATCH -o /scratch/%u/moco/logs/%x.%j.out
#SBATCH -e /scratch/%u/moco/logs/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# Acquisition probe (experiments.md, "Acquisition probe"). Describes every ACRIN
# series once (skipped if acquisition.csv exists), then scores each encoder on
# the look-alike gallery. RUNS lists MoCo run dirs scored at CKPT.
# Override: sbatch --export=ALL,ENCODERS=moco,RUNS=e2 jobs/acquisition_probe.sh
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

ENCODERS="${ENCODERS:-random imagenet moco}"
RUNS="${RUNS:-base e0 e1 e1b}"
CKPT="${CKPT:-checkpoint_0199}"

module load mamba/latest
source activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_DIR}"

PROBE=scripts/eval/acquisition_probe.py
if [ ! -f "${EVAL_DIR}/acquisition.csv" ]; then
    python "${PROBE}" describe --bank-dir "${EVAL_DIR}" \
        --tensor-dir "${TENSOR_ACRIN}" --csv-dir "${CSV_DIR}"
fi

for enc in ${ENCODERS}; do
    if [ "${enc}" = "moco" ]; then
        for run in ${RUNS}; do
            python "${PROBE}" score --bank-dir "${EVAL_DIR}" --encoder moco \
                --checkpoint "${CKPT_ROOT}/${run}/${CKPT}.pth.tar" \
                --output "${EVAL_DIR}/acq_moco_${run}_${CKPT}.json"
        done
    else
        python "${PROBE}" score --bank-dir "${EVAL_DIR}" --encoder "${enc}" \
            --output "${EVAL_DIR}/acq_${enc}.json"
    fi
done
