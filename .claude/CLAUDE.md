# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoCo v2 self-supervised pretraining on unlabeled CT colonoscopy data. The goal is to learn robust feature representations from two CT datasets (ACRIN 6664 and Pediatric CT-SEG) without labels, then fine-tune on labeled ACRIN 6664 data for polyp detection/classification.

## Commands

### Environment
```bash
conda activate moco
```

### Phase 1: Preprocess DICOM → .pt tensors
```bash
python scripts/prep_data.py \
    --input-dirs /path/to/CT-Colonography /path/to/Pediatric-CT-SEG \
    --cache-dir /path/to/cached-tensors
```

### Phase 2: MoCo Pretraining
```bash
# Multi-GPU DDP (required — single-GPU / DataParallel intentionally disabled)
python main_moco.py /path/to/cached-tensors \
    --arch resnet50 --mlp --cos \
    --epochs 200 --batch-size 256 --lr 0.03 \
    --moco-dim 128 --moco-k 16384 --moco-m 0.999 --moco-t 0.07 \
    --crops-per-volume 20 --workers 32 \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --dist-url "tcp://localhost:10001" \
    --output-dir /path/to/checkpoints
```

### Phase 3: Linear Probing Evaluation
```bash
python main_lincls.py /path/to/labeled-data --arch resnet50 --epochs 100 --resume checkpoint_0199.pth.tar
```

## Architecture

### Data Flow
```
DICOM files → [prep_data.py] → .pt tensor cache → [ct_dataset.py] → MoCo training → pretrained encoder → [main_lincls.py] → downstream task
```

### Key Files
- `moco/builder.py` — MoCo v2 dual-encoder architecture (query + momentum key encoder, 65k-sample queue, InfoNCE loss)
- `moco/ct_dataset.py` — Loads cached 3D volumes, extracts random 2.5D crops (224×224×3), applies dual augmentations for contrastive pairs
- `main_moco.py` — Training loop with DDP support, SGD optimizer, LR scheduling (drop at epochs 120/160 or cosine)
- `main_lincls.py` — Linear probing: frozen backbone + trainable classification head, SGD lr=30.0
- `scripts/prep_data.py` — DICOM → RAS reorientation → 1mm isotropic resampling → HU windowing → cached .pt tensors

### MoCo v2 Training Details
- Backbone: ResNet-50 with 2048→128 projection head (MLP)
- Queue: 65,536 negatives maintained across batches
- Momentum: 0.999 for key encoder updates
- Temperature: 0.07 for InfoNCE loss
- Expected: loss drops rapidly then plateaus; oscillation indicates augmentation or LR issue

## Medical Imaging Constraints

**Critical — never violate these:**
- **No color jitter** — destroys Hounsfield Unit (HU) semantics; HU values carry physical meaning
- **Soft-tissue windowing: -150 to +250 HU** — isolates colon wall, fat, muscle; not bone/air
- **1mm isotropic resampling** — normalizes scanner resolution differences across acquisitions
- **Patient-level data splitting** — no patient can appear in both train and val sets
- **Reorient to RAS** — consistent anatomical orientation across scanners

**Allowed augmentations:** spatial transforms (rotation ≤15°, flips on x/y/z), Gaussian noise (σ=0.05), Gaussian blur (σ=0.5–1.5). Flips are valid because colon anatomy is orientation-agnostic for polyp detection.

## Code Conventions

- Use **MONAI** for all medical image transforms (`RandRotated`, `RandFlipd`, `RandGaussianNoised`, `RandGaussianSmoothd`, `Orientationd`, `Spacingd`)
- Use **PyTorch** for model logic
- Use `'spawn'` multiprocessing context for HPC/SLURM compatibility (Linux ITK requirement)
- Comments should explain *why* in terms of medical physics, not just *what*
- SLURM integration: script respects `SLURM_CPUS_PER_TASK` for worker count
