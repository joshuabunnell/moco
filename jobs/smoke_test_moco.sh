#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a30:2
#SBATCH -t 0-01:00:00
#SBATCH -p general
#SBATCH -q general
#SBATCH -o /scratch/jpbunnel/logs/moco-smoke.%j.out
#SBATCH -e /scratch/jpbunnel/logs/moco-smoke.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

set -e

module load mamba/latest
source activate moco_env

PYTHON=/home/jpbunnel/.conda/envs/moco_env/bin/python
export PYTHONUNBUFFERED=1

cd /home/jpbunnel/moco/

DATA_DIR="/scratch/jpbunnel/cached-tensors"
OUTPUT_DIR="/scratch/jpbunnel/moco-smoke-checkpoints"
mkdir -p "${OUTPUT_DIR}" /scratch/jpbunnel/logs

MASTER_PORT=$((RANDOM % 10000 + 20000))

echo "=== Smoke Test: MoCo v2 ==="
echo "Data dir:   ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Port:       ${MASTER_PORT}"
echo "GPUs:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python:     $(${PYTHON} --version)"
echo "PyTorch:    $(${PYTHON} -c 'import torch; print(torch.__version__)')"
echo "CUDA avail: $(${PYTHON} -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count:  $(${PYTHON} -c 'import torch; print(torch.cuda.device_count())')"
echo "Data files: $(find ${DATA_DIR} -name '*.pt' | wc -l) .pt files"
echo "==========================="

"${PYTHON}" main_moco.py "${DATA_DIR}" \
    --arch resnet50 \
    --mlp \
    --cos \
    --epochs 3 \
    --batch-size 64 \
    --lr 0.03 \
    --moco-dim 128 \
    --moco-k 65536 \
    --moco-m 0.999 \
    --moco-t 0.07 \
    --workers 4 \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --dist-url "tcp://localhost:${MASTER_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    --print-freq 1

echo "=== Smoke test complete ==="
echo "Checkpoints written:"
ls -lh "${OUTPUT_DIR}"/*.pth.tar 2>/dev/null || echo "No checkpoints found — training failed"
