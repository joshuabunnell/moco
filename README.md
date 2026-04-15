# MoCo v2 for CT Colonoscopy

Self-supervised pretraining on unlabeled CT colonoscopy data using [Momentum Contrast (MoCo) v2](https://arxiv.org/abs/2003.04297), adapted for 3D medical imaging.

The goal is to learn robust visual representations from two unlabeled CT datasets — [ACRIN 6664](https://www.cancerimagingarchive.net/collection/ct-colonography/) and [Pediatric CT-SEG](https://www.cancerimagingarchive.net/collection/pediatric-ct-seg/) — that transfer to downstream tasks such as polyp detection and classification.

## Overview

```
DICOM files
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  prep_data   │────▶│  .pt tensor  │────▶│   MoCo v2    │────▶│   Pretrained │
│  (preprocess)│     │    cache     │     │  pretraining  │     │   encoder    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │   Linear     │
                                                               │   probing    │
                                                               └──────────────┘
```

**Pipeline stages:**

1. **Preprocessing** — Load DICOM series, reorient to RAS, resample to 1 mm isotropic, apply soft-tissue HU windowing, and cache as `.pt` tensors.
2. **MoCo v2 Pretraining** — Train a ResNet-50 backbone with momentum contrast on 2.5D crops (224x224x3) extracted from the cached volumes.
3. **Linear Probing** — Freeze the pretrained backbone and train a linear classifier on labeled data to evaluate representation quality.
4. **Visualization** *(optional)* — Extract backbone features and project to 2D with UMAP.

## Medical Imaging Adaptations

Standard MoCo v2 was designed for natural images (ImageNet). This project adapts it for CT data with several domain-specific constraints:

| Adaptation | Rationale |
|---|---|
| **No color jitter** | Hounsfield Unit (HU) values encode physical tissue density — color jitter would destroy this signal |
| **Soft-tissue HU window [-150, +250]** | Isolates colon wall, mesenteric fat, and polyp tissue; excludes bone (>400 HU) and air (<-500 HU) |
| **1 mm isotropic resampling** | Normalizes variable slice thickness and in-plane resolution across scanners |
| **RAS reorientation** | Consistent anatomical coordinate system regardless of scanner manufacturer |
| **2.5D crops (224x224x3)** | Three adjacent axial slices mapped to RGB channels for compatibility with 2D ResNet architectures |
| **Medical-safe augmentations only** | Spatial flips, small rotations (≤15°), Gaussian noise/blur — no transforms that alter HU relationships |

## Installation

**Option 1: Minimal (pip)**
```bash
pip install -r requirements.txt
```

**Option 2: Full reproducible environment (conda)**
```bash
conda env create -f environment.yml
conda activate moco
```

## Usage

### Step 1: Preprocess DICOM to Tensors

```bash
# Sequential (single machine)
python scripts/prep_data.py \
    --input-dirs /path/to/CT-Colonography /path/to/Pediatric-CT-SEG \
    --cache-dir /path/to/cached-tensors

# Or use SLURM array jobs for large datasets (see examples/prep_array.sh)
```

### Step 2: MoCo v2 Pretraining

```bash
# Multi-GPU (recommended — DDP is required)
python main_moco.py /path/to/cached-tensors \
    --arch resnet50 \
    --mlp --cos \
    --epochs 200 \
    --batch-size 256 \
    --lr 0.03 \
    --moco-dim 128 \
    --moco-k 16384 \
    --moco-m 0.999 \
    --moco-t 0.07 \
    --crops-per-volume 20 \
    --workers 32 \
    --multiprocessing-distributed \
    --world-size 1 --rank 0 \
    --dist-url "tcp://localhost:10001" \
    --output-dir /path/to/checkpoints
```

Training uses `mp.spawn` internally — no `torchrun` required.

**Expected behavior:** Loss drops rapidly in the first ~50 epochs, then plateaus. Persistent oscillation may indicate an augmentation or learning rate issue.

### Step 3: Linear Probing Evaluation

```bash
python main_lincls.py /path/to/labeled-data \
    --arch resnet50 \
    --pretrained /path/to/checkpoints/checkpoint_0199.pth.tar \
    --epochs 100 \
    --lr 30.0
```

> **Note:** `main_lincls.py` uses ImageNet-style data loading (`torchvision.datasets.ImageFolder` with `train/` and `val/` subdirectories). It serves as a reference template — adaptation for CT downstream tasks will require replacing the data loader and normalization.

### Step 4: UMAP Visualization (Optional)

```bash
python scripts/visualize_umap.py \
    --checkpoint /path/to/checkpoints/checkpoint_0199.pth.tar \
    --data /path/to/cached-tensors \
    --output umap.png
```

## HPC / SLURM

Example SLURM job scripts are provided in [`examples/`](examples/):

| Script | Purpose | Resources |
|---|---|---|
| `prep_array.sh` | Two-phase DICOM preprocessing (discover + array jobs) | 2-4 CPUs, 4-16 GB per task |
| `train_moco.sh` | MoCo pretraining | 32 CPUs, 128 GB, 2x A100 GPUs |
| `run_umap.sh` | UMAP visualization | 4 CPUs, 32 GB, 1x A100 GPU |

Edit the configuration block at the top of each script to set paths for your environment.

## Repository Structure

```
moco/
├── main_moco.py              # MoCo v2 pretraining entry point (DDP)
├── main_lincls.py             # Linear probing evaluation (template)
├── moco/
│   ├── __init__.py            # Shared utilities (to_resnet_format)
│   ├── builder.py             # MoCo v2 model (dual encoders, queue, momentum)
│   └── ct_dataset.py          # CT volume dataset with contrastive augmentations
├── scripts/
│   ├── prep_data.py           # DICOM → .pt tensor preprocessing
│   └── visualize_umap.py      # UMAP feature visualization
├── examples/
│   ├── train_moco.sh          # SLURM job: pretraining
│   ├── prep_array.sh          # SLURM job: preprocessing array
│   └── run_umap.sh            # SLURM job: UMAP visualization
├── notebooks/
│   ├── data_information.ipynb # Dataset characterization
│   ├── data_transforms.ipynb  # Transform pipeline validation
│   └── tensor_prepare.ipynb   # Preprocessing architecture demo
├── requirements.txt           # Minimal pip dependencies
├── environment.yml            # Full conda environment
└── LICENSE                    # MIT
```

## Citation

If you use this code, please cite the original MoCo papers:

```bibtex
@inproceedings{he2020momentum,
  title={Momentum Contrast for Unsupervised Visual Representation Learning},
  author={He, Kaiming and Fan, Haoqi and Wu, Yuxin and Xie, Saining and Girshick, Ross},
  booktitle={CVPR},
  year={2020}
}

@article{chen2020improved,
  title={Improved Baselines with Momentum Contrastive Learning},
  author={Chen, Xinlei and Fan, Haoqi and Girshick, Ross and He, Kaiming},
  journal={arXiv preprint arXiv:2003.04297},
  year={2020}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

Original MoCo implementation by Meta Platforms, Inc. Adapted for CT colonoscopy self-supervised pretraining.

## Acknowledgments

- [Meta AI Research](https://github.com/facebookresearch/moco) — Original MoCo v2 implementation
- [MONAI](https://monai.io/) — Medical image transforms and data loading
- [ACRIN 6664](https://www.cancerimagingarchive.net/collection/ct-colonography/) — CT Colonography dataset
- [Pediatric CT-SEG](https://www.cancerimagingarchive.net/collection/pediatric-ct-seg/) — Pediatric CT dataset
