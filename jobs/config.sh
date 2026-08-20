#!/bin/bash
# Canonical paths for the moco project on ASU Sol — the single source of truth.
# Every job script in jobs/ sources this, so paths are defined in ONE place.
#
# Everything is derived from $USER and $HOME, so another student in the group can
# run these scripts UNCHANGED — no per-file editing. ($USER resolves to whoever
# submits the job; the author's alias "jpbunnel" appears nowhere below.)
#
# Layout mirror:  code -> $HOME/moco   |   data -> /scratch/$USER/moco
#
# Override any value by exporting it before sourcing, e.g.:
#   CONDA_ENV=other_env sbatch jobs/train_moco.sh

# --- code (durable, in $HOME, git-tracked) ---
: "${PROJECT_DIR:=$HOME/moco}"
: "${CSV_DIR:=$PROJECT_DIR/metadata/csv_metadata}"
: "${MANIFEST_TCIA:=$PROJECT_DIR/metadata/manifest.tcia}"  # TCIA download spec (durable copy)
: "${CONDA_ENV:=moco_env}"

# --- data (disposable, in /scratch, regenerable from TCIA) ---
: "${DATA_ROOT:=/scratch/$USER/moco}"
: "${RAW_DIR:=$DATA_ROOT/raw}"          # raw DICOM from TCIA (NBIA output)
: "${TENSOR_DIR:=$DATA_ROOT/tensors}"   # preprocessed .pt cache
: "${CKPT_ROOT:=$DATA_ROOT/checkpoints}" # base/ acrin/ pediatric/
: "${UMAP_DIR:=$DATA_ROOT/umap}"
: "${LOG_DIR:=$DATA_ROOT/logs}"
: "${TOOLS_DIR:=$DATA_ROOT/tools}"      # NBIA retriever RPM + extracted JDK

# --- per-dataset subdirs (raw keeps TCIA's spaces; tensors normalize to hyphens) ---
: "${RAW_ACRIN:=$RAW_DIR/CT COLONOGRAPHY}"          # TCIA keeps the space in this name
: "${RAW_PEDIATRIC:=$RAW_DIR/Pediatric-CT-SEG}"
: "${TENSOR_ACRIN:=$TENSOR_DIR/CT-COLONOGRAPHY}"
: "${TENSOR_PEDIATRIC:=$TENSOR_DIR/Pediatric-CT-SEG}"
