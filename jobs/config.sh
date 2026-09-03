#!/bin/bash
#
# Single source of truth for every path the job scripts use. Nothing here is
# hardcoded to a username: everything derives from $USER (or $HOME), so any
# member of grp_vkodibag can run the committed scripts unchanged. Each value is
# assigned with := so a caller can override it in the environment.

: "${PROJECT_DIR:=$HOME/moco}"
: "${CSV_DIR:=$PROJECT_DIR/metadata/csv_metadata}"
: "${MANIFEST_TCIA:=$PROJECT_DIR/metadata/manifest.tcia}"
: "${SERIES_CATALOG:=$PROJECT_DIR/metadata/series_catalog.csv}"
: "${CONDA_ENV:=moco_env}"
# Deliberately under $HOME, not scratch: the retriever is a build dependency, not
# disposable data, and nothing reads it often enough to survive the 90-day purge
# on its own. A hand-extracted JDK kept in scratch rotted exactly this way.
: "${TOOLS_DIR:=$PROJECT_DIR/tools}"

: "${DATA_ROOT:=/scratch/$USER/moco}"
: "${RAW_DIR:=$DATA_ROOT/raw}"
: "${TENSOR_DIR:=$DATA_ROOT/tensors}"
: "${CKPT_ROOT:=$DATA_ROOT/checkpoints}"
: "${UMAP_DIR:=$DATA_ROOT/umap}"
: "${EVAL_DIR:=$DATA_ROOT/eval}"
: "${LOG_DIR:=$DATA_ROOT/logs}"

# Every job's #SBATCH -o/-e points here (/scratch/%u/moco/logs). SLURM won't create
# the dir and evaluates the redirect before this script runs, so make it eagerly:
# this covers every job after the first on a fresh scratch (see CLAUDE.md Rebuild
# for the one-time bootstrap mkdir).
mkdir -p "$LOG_DIR"

: "${RAW_ACRIN:=$RAW_DIR/CT COLONOGRAPHY}"
: "${RAW_PEDIATRIC:=$RAW_DIR/Pediatric-CT-SEG}"
: "${TENSOR_ACRIN:=$TENSOR_DIR/CT-COLONOGRAPHY}"
: "${TENSOR_PEDIATRIC:=$TENSOR_DIR/Pediatric-CT-SEG}"
