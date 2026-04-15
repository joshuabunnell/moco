#!/bin/bash
set -e

# ===== EDIT THESE FOR YOUR ENVIRONMENT =====
PROJECT_DIR="/home/jpbunnel/moco"
DATA_DIR="/scratch/jpbunnel/cached-tensors"
CONDA_ENV="moco_env"
# ============================================

# Discovery phase resource limits (array tasks get their own via the sbatch call below)
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

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

MANIFEST="${DATA_DIR}/series_manifest.txt"

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    # -----------------------------------------------------------------------
    # Phase 1 — discovery (runs once, submitted manually)
    # Walks input dirs, writes manifest, then re-submits this same script
    # as a SLURM array job sized to the number of series found.
    #
    # Update --input-dirs below to point to your DICOM dataset directories.
    # -----------------------------------------------------------------------
    python scripts/prep_data.py --discover \
        --input-dirs \
            "/scratch/jpbunnel/downloads/manifest/CT COLONOGRAPHY" \
            "/scratch/jpbunnel/downloads/manifest/Pediatric-CT-SEG" \
        --cache-dir "${DATA_DIR}" \
        --manifest "${MANIFEST}"

    TOTAL=$(wc -l < "${MANIFEST}")
    if [ "${TOTAL}" -eq 0 ]; then
        echo "No series found — check --input-dirs paths"
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
        -q htc \
        -o "slurm.prep.%A_%a.out" \
        -e "slurm.prep.%A_%a.err" \
        --mail-type=ALL \
        --mail-user=%u@asu.edu \
        examples/prep_array.sh

else
    # -----------------------------------------------------------------------
    # Phase 2 — process one series (runs once per array task)
    # SLURM sets SLURM_ARRAY_TASK_ID automatically. No IPC, no shared memory.
    # -----------------------------------------------------------------------
    python scripts/prep_data.py \
        --process-index "${SLURM_ARRAY_TASK_ID}" \
        --manifest "${MANIFEST}"
fi
