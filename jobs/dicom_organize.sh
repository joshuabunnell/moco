#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 2-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o slurm.dicom_organize.%j.out
#SBATCH -e slurm.dicom_organize.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# OPTIONAL — not part of the default pipeline. prep_data.py already walks the
# raw TCIA layout directly, so this step is only for building a tidy, human-
# browsable symlinked view of the DICOM series. Requires the separate
# `dicom-organizer` tool (github.com/joshuabunnell/dicom-organizer), installed
# into its own conda env. Paths from jobs/config.sh.
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

ORGANIZED_DIR="${DATA_ROOT}/organized_ref"
ORGANIZER_ENV="${ORGANIZER_ENV:-dicom_organizer_env}"

module load mamba/latest
source activate "${ORGANIZER_ENV}"

# src = raw collection dir, dst basename = its sanitized tensor-cache name — both
# from config.sh so the collection identity is never hardcoded here.
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
