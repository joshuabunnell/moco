#!/bin/bash
# Preprocess every raw DICOM series into the HU volume cache, one series per task.
# Submit once with no array: that run discovers the series, writes the manifest,
# resubmits this script as an array sized to the count, and queues a final run
# (FINALIZE=1) that writes each manifest.csv once the array ends. The resources
# below are sized for one series; the discovery and final runs fit inside them.
# Override the range to smoke-test a few series first: sbatch --export=ALL,ARRAY=0-4 jobs/prep_array.sh
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-00:30:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH -J prep
#SBATCH -o /scratch/%u/moco/logs/%x.%A_%a.out
#SBATCH -e /scratch/%u/moco/logs/%x.%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
cd "${PROJECT_DIR}"

MANIFEST="${TENSOR_DIR}/series_manifest.txt"

if [ -z "${SLURM_ARRAY_TASK_ID}" ] && [ -z "${FINALIZE}" ]; then
    # Old .pt caches used a different naming scheme; mixing the two in one
    # directory would give every series two files and two manifest rows.
    if compgen -G "${TENSOR_DIR}/*/*.pt" > /dev/null; then
        echo "ERROR: ${TENSOR_DIR} still holds a .pt cache. Move it aside before rebuilding."
        exit 1
    fi

    python scripts/data/prep_data.py --discover \
        --input-dirs "${RAW_ACRIN}" "${RAW_PEDIATRIC}" \
        --cache-dir "${TENSOR_DIR}" \
        --manifest "${MANIFEST}"

    TOTAL=$(wc -l < "${MANIFEST}")
    if [ "${TOTAL}" -eq 0 ]; then
        echo "No series found — check RAW_DIR in jobs/config.sh"
        exit 1
    fi

    ARRAY="${ARRAY:-0-$((TOTAL - 1))}"
    echo "Submitting array ${ARRAY} over ${TOTAL} series..."
    ARRAY_JOB=$(sbatch --parsable --array="${ARRAY}" jobs/prep_array.sh)
    # afterany, not afterok: a few series always fail, and the manifest should
    # still list every one that succeeded.
    sbatch --dependency=afterany:"${ARRAY_JOB}" --export=ALL,FINALIZE=1 jobs/prep_array.sh
elif [ -n "${FINALIZE}" ]; then
    python scripts/data/prep_data.py --finalize --manifest "${MANIFEST}"
else
    python scripts/data/prep_data.py \
        --process-index "${SLURM_ARRAY_TASK_ID}" \
        --manifest "${MANIFEST}"
fi
