# MoCo v2 for CT Colonoscopy

Self-supervised pretraining on unlabeled CT colonoscopy data using [Momentum Contrast (MoCo) v2](https://arxiv.org/abs/2003.04297), adapted for 3D medical imaging.

The goal is to learn robust visual representations from two unlabeled CT datasets — [ACRIN 6664](https://www.cancerimagingarchive.net/collection/ct-colonography/) and [Pediatric CT-SEG](https://www.cancerimagingarchive.net/collection/pediatric-ct-seg/) — that transfer to downstream polyp detection and size classification on labeled ACRIN 6664 data.

## Pipeline

```
DICOM files → [prep_data.py] → .pt tensors + manifest.csv
                                     ↓
XLSX metadata → [convert_metadata.py] → acrin_combined.csv
                                     ↓
manifest.csv + acrin_combined.csv → [split_data.py] → labels_{train,val,test}.csv
                                     ↓
.pt tensors (unlabeled)   → [main_moco.py]   → pretrained encoder
.pt tensors + label CSVs  → [main_lincls.py] → polyp classification
```

**Stages:**

1. **Preprocessing** (`scripts/prep_data.py`) — DICOM series → RAS reorientation → 1 mm isotropic resampling → soft-tissue HU windowing [-150, +250] → cached `.pt` tensors with patient-identifiable filenames and `manifest.csv`.
2. **Metadata** (`scripts/convert_metadata.py`) — ACRIN 6664 XLSX → clean CSVs mapping patient IDs to polyp size categories.
3. **Splitting** (`scripts/split_data.py`) — Patient-level stratified train/val/test split joining the manifest with metadata. Outputs `labels_{train,val,test}.csv` for the linear probe.
4. **MoCo v2 Pretraining** (`main_moco.py`) — ResNet-50 backbone with momentum contrast on 2.5D crops (224x224x3). Multi-GPU DDP required.
5. **Linear Probing** (`main_lincls.py`) — Freeze pretrained backbone, train a linear head on labeled ACRIN data for 3-class polyp classification (no polyp / 6-9 mm / >=10 mm).

## Medical Imaging Adaptations

| Adaptation | Rationale |
|---|---|
| **No color jitter** | HU values encode physical tissue density — jitter destroys this signal |
| **Soft-tissue HU window [-150, +250]** | Isolates colon wall, mesenteric fat, polyp tissue; excludes bone and air |
| **1 mm isotropic resampling** | Normalizes variable slice thickness across scanners |
| **RAS reorientation** | Consistent anatomical coordinates regardless of scanner manufacturer |
| **2.5D crops (224x224x3)** | Three adjacent axial slices mapped to RGB channels for 2D ResNet compatibility |
| **Spatial-only augmentations** | Flips, rotations (<=15 deg), Gaussian noise/blur — nothing that alters HU relationships |
| **Patient-level splitting** | No patient appears in both train and val — prevents data leakage |

## Installation

```bash
# Option 1: pip
pip install -r requirements.txt

# Option 2: conda (full reproducible environment)
conda env create -f environment.yml
conda activate moco
```

## Usage

### Preprocess DICOM to Tensors

```bash
python scripts/prep_data.py \
    --input-dirs /data/CT-Colonography /data/Pediatric-CT-SEG \
    --cache-dir /scratch/cached-tensors

# SLURM array mode for large datasets — see examples/prep_array.sh
```

### Prepare Labels

```bash
# Convert XLSX metadata to CSV (one-time)
python scripts/convert_metadata.py \
    --input-dir raw_metadata/ACRIN_6664 \
    --output-dir csv_metadata

# Generate patient-level train/val/test splits
python scripts/split_data.py \
    --manifest /scratch/cached-tensors/CT-Colonography/manifest.csv \
    --metadata csv_metadata/acrin_combined.csv \
    --output-dir csv_metadata \
    --label-scheme three \
    --val-frac 0.15 --test-frac 0.15 --seed 42
```

### MoCo v2 Pretraining

```bash
python main_moco.py /scratch/cached-tensors \
    --arch resnet50 --mlp --cos \
    --epochs 200 --batch-size 256 --lr 0.03 \
    --moco-dim 128 --moco-k 65536 --moco-m 0.999 --moco-t 0.07 \
    --crops-per-volume 20 --workers 32 \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --dist-url "tcp://localhost:10001" \
    --output-dir /scratch/moco-checkpoints

# Resume from a checkpoint (e.g. to continue on ACRIN-only data):
python main_moco.py /scratch/cached-tensors/CT-Colonography \
    --resume /scratch/moco-checkpoints/checkpoint_0199.pth.tar \
    --epochs 400 \
    ...  # keep --moco-k the same as the original run
```

Training uses `mp.spawn` internally — no `torchrun` required. Loss drops rapidly in the first ~50 epochs then plateaus.

### Linear Probing

```bash
python main_lincls.py \
    --data /scratch/cached-tensors/CT-Colonography \
    --train-csv csv_metadata/labels_train.csv \
    --val-csv csv_metadata/labels_val.csv \
    --pretrained /scratch/moco-checkpoints/checkpoint_0199.pth.tar \
    --num-classes 3 \
    --epochs 100 --lr 30.0 --batch-size 256 \
    --output-dir /scratch/lincls-checkpoints
```

## HPC / SLURM

Example job scripts in [`examples/`](examples/):

| Script | Purpose | Resources |
|---|---|---|
| `prep_array.sh` | Two-phase DICOM preprocessing (discover + array) | 2-4 CPUs, 4-16 GB per task |
| `train_moco.sh` | MoCo pretraining from scratch | 32 CPUs, 128 GB, 2x A100 |
| `resume_moco.sh` | Continue pretraining from a checkpoint | 32 CPUs, 128 GB, 2x A100 |
| `run_lincls.sh` | Linear probing evaluation | 16 CPUs, 64 GB, 1x A100 |
| `run_umap.sh` | UMAP feature extraction | 4 CPUs, 32 GB, 1x A100 |

Edit the configuration block at the top of each script for your environment.

## Repository Structure

```
├── main_moco.py                          # MoCo v2 pretraining (DDP, multi-GPU)
├── main_lincls.py                        # Linear probing on labeled ACRIN data
├── moco/
│   ├── __init__.py                       # Shared utils (to_resnet_format)
│   ├── builder.py                        # MoCo model (dual encoders, queue, InfoNCE)
│   └── ct_dataset.py                     # CTMoCoDataset (contrastive) + CTLinClsDataset (labeled)
├── scripts/
│   ├── prep_data.py                      # DICOM → .pt preprocessing + manifest
│   ├── convert_metadata.py               # ACRIN XLSX → CSV metadata
│   ├── split_data.py                     # Patient-level stratified train/val/test splits
│   ├── reorganize_cache.py               # One-time migration: MD5 filenames → patient IDs
│   └── visualize_umap.py                 # UMAP projection of backbone features
├── examples/
│   ├── train_moco.sh                     # SLURM: pretraining from scratch
│   ├── resume_moco.sh                    # SLURM: continue pretraining from checkpoint
│   ├── run_lincls.sh                     # SLURM: linear probing evaluation
│   ├── prep_array.sh                     # SLURM: preprocessing array jobs
│   └── run_umap.sh                       # SLURM: UMAP visualization
├── metadata/
│   ├── raw_metadata/                     # ACRIN 6664 XLSX files (no-polyp, 6-9mm, >=10mm)
│   └── csv_metadata/                     # Processed CSVs + split label files
├── notebooks/
│   ├── data_information.ipynb            # Dataset characterization
│   ├── data_transforms.ipynb             # Transform pipeline validation
│   └── tensor_prepare.ipynb              # Preprocessing architecture demo
├── requirements.txt
├── environment.yml
└── LICENSE
```

## Citation

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

[MIT License](LICENSE). Original MoCo implementation by Meta Platforms, Inc. Adapted for CT colonoscopy self-supervised pretraining.
