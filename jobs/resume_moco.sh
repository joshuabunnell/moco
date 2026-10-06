#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH -t 3-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -J moco_resume
#SBATCH -o /scratch/%u/moco/logs/%x.%j.out
#SBATCH -e /scratch/%u/moco/logs/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# Continue an interrupted train_moco.sh run in its own directory.
# sbatch --export=ALL,RUN=e3 jobs/resume_moco.sh resumes from the newest
# checkpoint in CKPT_ROOT/RUN; CKPT=checkpoint_0149 picks one, EPOCHS raises
# the target. The recipe (crop overlap/scale, window jitter, z shift, queue
# size, save frequency) is read from the args the checkpoint saved, so a
# resumed run cannot silently fall back to another recipe. Checkpoints from
# before 2026-09-16 carry no args and are refused.
set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

RUN="${RUN:?set RUN to the run dir to resume, e.g. RUN=e3}"
OUT_DIR="${CKPT_ROOT}/${RUN}"
if [ -z "${CKPT:-}" ]; then
    CKPT=$(basename "$(ls "${OUT_DIR}"/checkpoint_*.pth.tar | sort | tail -1)" .pth.tar)
fi
CKPT_PATH="${OUT_DIR}/${CKPT}.pth.tar"

module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
MASTER_PORT=$((10000 + RANDOM % 50000))

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

# Recipe from the checkpoint, as shell assignments. Assigned first so a
# refusal stops the job under set -e (eval "$(...)" would swallow it).
RECIPE=$(python - "${CKPT_PATH}" <<'EOF'
import os, shlex, sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
args = ckpt.get("args")
if args is None:
    sys.exit("ERROR: %s saved no args; resume it by hand" % sys.argv[1])
def sh(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join("%g" % x for x in v)
    return str(v)
recipe = {
    "DATA_NAME": os.path.basename(os.path.normpath(args["data"])),
    "CROP_OVERLAP": sh(args.get("crop_overlap")),
    "CROP_SCALE": sh(args.get("crop_scale")),
    "WINDOW_JITTER": sh(args.get("window_jitter")),
    "PAIR_Z_SHIFT": sh(args.get("pair_z_shift")),
    "MOCO_K": sh(args["moco_k"]),
    "CROPS_PER_VOLUME": sh(args["crops_per_volume"]),
    "SAVE_FREQ": sh(args["save_freq"]),
    "CKPT_EPOCHS": sh(args["epochs"]),
}
for k, v in recipe.items():
    print("%s=%s" % (k, shlex.quote(v)))
EOF
)
eval "${RECIPE}"
EPOCHS="${EPOCHS:-${CKPT_EPOCHS}}"

# The staged path in args ends in the collection dir name (or "tensors" for base).
case "${DATA_NAME}" in
    "$(basename "${TENSOR_ACRIN}")")     DATA_DIR="${TENSOR_ACRIN}" ;;
    "$(basename "${TENSOR_PEDIATRIC}")") DATA_DIR="${TENSOR_PEDIATRIC}" ;;
    "$(basename "${TENSOR_DIR}")")       DATA_DIR="${TENSOR_DIR}" ;;
    *) echo "ERROR: cannot map checkpoint data dir '${DATA_NAME}' to a tensor dir"; exit 1 ;;
esac

source jobs/stage_data.sh
trap 'rm -rf "${TMPDIR:-/tmp}/moco_stage"' EXIT
DATA_DIR=$(stage_data "${DATA_DIR}")

# Provenance of this leg, beside the original run's files rather than over them.
TAG="resume_${SLURM_JOB_ID}"
git rev-parse HEAD > "${OUT_DIR}/git_commit.${TAG}.txt"
git diff HEAD > "${OUT_DIR}/git_diff.${TAG}.patch"
git status --short > "${OUT_DIR}/git_status.${TAG}.txt"
cat > "${OUT_DIR}/job.${TAG}.txt" <<EOF
RUN=${RUN} CKPT=${CKPT} SLURM_JOB_ID=${SLURM_JOB_ID} DATA=${DATA_NAME}
CROP_OVERLAP=${CROP_OVERLAP} CROP_SCALE=${CROP_SCALE} WINDOW_JITTER=${WINDOW_JITTER} PAIR_Z_SHIFT=${PAIR_Z_SHIFT} MOCO_K=${MOCO_K} EPOCHS=${EPOCHS} SAVE_FREQ=${SAVE_FREQ}
EOF
cat "${OUT_DIR}/job.${TAG}.txt"

# Fixed flags match train_moco.sh. --moco-k must match the checkpoint: the
# queue tensor is in the state_dict, so a mismatch fails to load.
python main_moco.py "${DATA_DIR}" \
    --resume "${CKPT_PATH}" \
    --arch resnet50 \
    --mlp \
    --cos \
    --epochs "${EPOCHS}" \
    --batch-size 256 \
    --lr 0.03 \
    --moco-dim 128 \
    --moco-k "${MOCO_K}" \
    --crops-per-volume "${CROPS_PER_VOLUME}" \
    ${CROP_OVERLAP:+--crop-overlap ${CROP_OVERLAP}} \
    ${PAIR_Z_SHIFT:+--pair-z-shift ${PAIR_Z_SHIFT}} \
    ${CROP_SCALE:+--crop-scale ${CROP_SCALE}} \
    ${WINDOW_JITTER:+--window-jitter ${WINDOW_JITTER}} \
    --moco-m 0.999 \
    --moco-t 0.07 \
    --workers 32 \
    --save-freq "${SAVE_FREQ}" \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --dist-url "tcp://localhost:${MASTER_PORT}" \
    --output-dir "${OUT_DIR}" \
    --print-freq 5
