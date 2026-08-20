#!/bin/bash
# Two-phase DICOM preprocessing: discover series (Phase 1), then process one
# series per SLURM array task (Phase 2). Paths come from jobs/config.sh.
# Discovery-phase resource limits below; array tasks get their own via the
# sbatch call in Phase 1.
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 0-01:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o slurm.discover.%j.out
#SBATCH -e slurm.discover.%j.err
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

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    # -------------------------------------------------------------------------
    # Phase 1 — discovery (runs once). Walks the raw DICOM dirs, writes the
    # manifest, then re-submits this script as an array job sized to the count.
    # The TCIA collection folder keeps its space ("CT COLONOGRAPHY"); prep_data
    # normalizes it to CT-COLONOGRAPHY in the tensor cache.
    # -------------------------------------------------------------------------
    python scripts/prep_data.py --discover \
        --input-dirs \
            "${RAW_ACRIN}" \
            "${RAW_PEDIATRIC}" \
        --cache-dir "${TENSOR_DIR}" \
        --manifest "${MANIFEST}"

    TOTAL=$(wc -l < "${MANIFEST}")
    if [ "${TOTAL}" -eq 0 ]; then
        echo "No series found — check RAW_DIR in jobs/config.sh"
        exit 1
    fi

    echo "Submitting array job for ${TOTAL} series..."
    sbatch \
        --array=0-$((TOTAL - 1)) \
        -N 1 \
        -c 4 \
        --mem=16G \
        -t 0-00:30:00 \
        -p htc \
        -q public \
        -o "slurm.prep.%A_%a.out" \
        -e "slurm.prep.%A_%a.err" \
        --mail-type=ALL \
        --mail-user=%u@asu.edu \
        jobs/prep_array.sh

else
    # -------------------------------------------------------------------------
    # Phase 2 — process one series (runs once per array task). SLURM sets
    # SLURM_ARRAY_TASK_ID automatically. No IPC, no shared memory.
    # -------------------------------------------------------------------------
    python scripts/prep_data.py \
        --process-index "${SLURM_ARRAY_TASK_ID}" \
        --manifest "${MANIFEST}"
fi
