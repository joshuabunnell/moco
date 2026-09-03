#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 2-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -J dicom_organize
#SBATCH -o /scratch/%u/moco/logs/%x.%j.out
#SBATCH -e /scratch/%u/moco/logs/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# OPTIONAL — builds a tidy symlinked DICOM view; not needed by prep_data.py. Requires dicom-organizer (github.com/joshuabunnell/dicom-organizer) in its own conda env.
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

ORGANIZED_DIR="${DATA_ROOT}/organized_ref"
ORGANIZER_ENV="${ORGANIZER_ENV:-dicom_organizer_env}"

module load mamba/latest
source activate "${ORGANIZER_ENV}"

# src = raw collection dir, dst basename = its sanitized tensor-cache name.
for pair in "${RAW_ACRIN}:${TENSOR_ACRIN}" "${RAW_PEDIATRIC}:${TENSOR_PEDIATRIC}"; do
    src="${pair%%:*}"
    dst="${ORGANIZED_DIR}/$(basename "${pair##*:}")"
    dicom-organizer organize "${src}" "${dst}" \
        --mode symlink \
        --follow-symlinks \
        --workers 8 \
        --collision error \
        --progress \
        -v
done
