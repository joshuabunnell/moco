#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH -t 3-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o /scratch/jpbunnel/logs/moco-train.%j.out
#SBATCH -e /scratch/jpbunnel/logs/moco-train.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

set -e

module load mamba/latest
source activate moco_env

PYTHON=/home/jpbunnel/.conda/envs/moco_env/bin/python
export PYTHONUNBUFFERED=1
MASTER_PORT=$((10000 + RANDOM % 50000))

cd /home/jpbunnel/moco/

DATA_DIR="/scratch/jpbunnel/cached-tensors"
OUTPUT_DIR="/scratch/jpbunnel/moco-checkpoints"
mkdir -p "${OUTPUT_DIR}" /scratch/jpbunnel/logs

# main_moco.py uses mp.spawn internally to launch one process per GPU.
# No torchrun needed — just pass --multiprocessing-distributed.
"${PYTHON}" main_moco.py "${DATA_DIR}" \
    --arch resnet50 \
    --mlp \
    --cos \
    --epochs 200 \
    --batch-size 256 \
    --lr 0.03 \
    --moco-dim 128 \
    --moco-k 16384 \
    --crops-per-volume 20 \
    --moco-m 0.999 \
    --moco-t 0.07 \
    --workers 32 \
    --save-freq 50 \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --dist-url "tcp://localhost:${MASTER_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    --print-freq 5
