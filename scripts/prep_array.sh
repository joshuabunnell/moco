#!/bin/bash
set -e

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
source activate moco_env

PYTHON=/home/jpbunnel/.conda/envs/moco_env/bin/python

export PYTHONUNBUFFERED=1

cd /home/jpbunnel/moco/

MANIFEST=/scratch/jpbunnel/cached-tensors/series_manifest.txt

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    # -----------------------------------------------------------------------
    # Phase 1 — discovery (runs once, submitted manually)
    # Walks input dirs, writes manifest, then re-submits this same script
    # as a SLURM array job sized to the number of series found.
    # -----------------------------------------------------------------------
    git stash
    git pull

    "${PYTHON}" scripts/prep_data.py --discover --manifest "${MANIFEST}"

    TOTAL=$(wc -l < "${MANIFEST}")
    if [ "${TOTAL}" -eq 0 ]; then
        echo "No series found — check INPUT_DIRS in prep_data.py"
        exit 1
    fi

    echo "Submitting array job for ${TOTAL} series..."
    sbatch \
        --array=0-$((TOTAL - 1)) \
        -N 1 \
        -c 4 \
        --mem=16G \
        -t 0-00:30:00 \
        -p public \
        -q public \
        -o "slurm.prep.%A_%a.out" \
        -e "slurm.prep.%A_%a.err" \
        --mail-type=ALL \
        --mail-user=%u@asu.edu \
        scripts/prep_array.sh

else
    # -----------------------------------------------------------------------
    # Phase 2 — process one series (runs once per array task)
    # SLURM sets SLURM_ARRAY_TASK_ID automatically. No IPC, no shared memory.
    # -----------------------------------------------------------------------
    "${PYTHON}" scripts/prep_data.py \
        --process-index "${SLURM_ARRAY_TASK_ID}" \
        --manifest "${MANIFEST}"
fi
