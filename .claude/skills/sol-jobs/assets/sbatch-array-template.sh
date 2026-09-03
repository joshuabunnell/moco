#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-00:30:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH -o logs/%x.%A_%a.out
#SBATCH -e logs/%x.%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu
# Submit with the array range, sized to the manifest:
#   N=$(wc -l < manifest.txt)
#   sbatch --array=0-$((N-1))%50 jobs/<name>.sh
#
# Resources above are PER TASK, not for the whole array. Size one task, then use
# %50 (or similar) to cap concurrency so one array does not take the whole
# partition and to be gentle on shared scratch I/O. Put the job wherever a single
# task fits: a sub-4h task belongs on -p htc, not -p public. MaxArraySize is
# 50000, so the top index is 49999.

set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

MANIFEST="${MANIFEST:-${PROJECT_DIR}/manifest.txt}"

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
cd "${PROJECT_DIR}"
mkdir -p logs

# One line of the manifest per task. SLURM_ARRAY_TASK_ID is 0-based; sed is 1-based.
INPUT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
if [ -z "${INPUT}" ]; then
    echo "No manifest line for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

python scripts/process_one.py --input "${INPUT}"
