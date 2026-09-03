#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 0-04:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# One-line description of what this job does.
#
# Partition/QoS: -p htc -q public if it fits in 4h (faster to schedule, no
# preemption, more GPU types); -p public -q public for 4h-7d; -p general -q
# private for >7d (preemptable). GRES must match `sinfo -o "%P %G"` exactly:
# gpu:a100:N | gpu:a100.20gb:N (MIG, ~20GB) | gpu:h100:N (htc/general only).

set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

# Override at submit time: sbatch --export=FOO=bar jobs/<name>.sh
FOO="${FOO:-default}"

# Load the environment in the body: a batch job does not inherit your login
# shell (Sol templates set --export=NONE), and `conda activate` needs shell
# hooks a non-interactive job lacks.
module load mamba/latest
source activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

python scripts/your_script.py \
    --foo "${FOO}" \
    --out "${DATA_ROOT}/out"
