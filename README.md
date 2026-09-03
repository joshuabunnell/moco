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

1. **Preprocessing** (`scripts/data/prep_data.py`) — DICOM series → RAS reorientation → 1 mm isotropic resampling → soft-tissue HU windowing [-150, +250] → cached `.pt` tensors with patient-identifiable filenames and `manifest.csv`.
2. **Metadata** (`scripts/data/convert_metadata.py`) — ACRIN 6664 XLSX → clean CSVs mapping patient IDs to polyp size categories.
3. **Splitting** (`scripts/data/split_data.py`) — Patient-level stratified train/val/test split joining the manifest with metadata. Outputs `labels_{train,val,test}.csv` for the linear probe.
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

## Data

Two public collections from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/):

| Collection | Role | Subjects | Series (downloaded) | Cached tensors |
|---|---|---|---|---|
| [CT COLONOGRAPHY (ACRIN 6664)](https://www.cancerimagingarchive.net/collection/ct-colonography/) | pretraining + labeled downstream | 825 | 3,451 | 1,720 |
| [Pediatric-CT-SEG](https://www.cancerimagingarchive.net/collection/pediatric-ct-seg/) | pretraining only (unlabeled) | 359 | 718 | 354 |

Cached-tensor counts are lower than series counts because `prep_data.py` drops
series with fewer than 10 slices. For Pediatric-CT-SEG about half the catalogued
series are single-file RTSTRUCT organ segmentations rather than CT, so they fall
out at the same threshold; those annotations are currently unused. Downstream **labels** come from the ACRIN 6664
polyp-size spreadsheets in [`metadata/raw_metadata/`](metadata/raw_metadata/)
(no-polyp / 6–9 mm / ≥10 mm), converted to the CSVs in `metadata/csv_metadata/`.

The exact download set is pinned by [`metadata/manifest.tcia`](metadata/manifest.tcia)
(committed here so it survives a `/scratch` purge). TCIA data is de-identified and
publicly available under each collection's license — cite the collections if you use them.

> **On Sol, data lives in `/scratch/$USER/moco/` and is treated as a disposable
> cache** (the filesystem purges files unread for 90 days). It is fully
> reproducible from TCIA — see [Reproducing the data](#reproducing-the-data).
> The code (`/home/$USER/moco`) is the durable, version-controlled half.

## Installation

```bash
# Option 1: pip
pip install -r requirements.txt

# Option 2: conda (full reproducible environment)
conda env create -f environment.yml   # creates env "moco_env"
conda activate moco_env
```

> On ASU Sol: `module load mamba/latest && source activate moco_env`. Job scripts
> that run Python do this for you. See [`CLAUDE.md`](CLAUDE.md) for canonical
> paths and conventions.

## Reproducing the data

If `/scratch` is wiped (or you are a new student starting fresh), rebuild the
entire dataset from TCIA. Everything is parameterized by `$USER` via
[`jobs/config.sh`](jobs/config.sh), so these run unchanged for anyone in the group:

```bash
# 1. Download raw DICOM from TCIA (uses metadata/manifest.tcia + NBIA retriever).
#    Installs the retriever from its RPM on first run. ~549 GB, many hours.
#    Resumable: it diffs the manifest against what is on disk, so rerun to continue.
sbatch jobs/tcia_download.sh          # → /scratch/$USER/moco/raw/

# 2. Preprocess DICOM → .pt tensors (discover, then a SLURM array over series).
sbatch jobs/prep_array.sh             # → /scratch/$USER/moco/tensors/

# 3. Build labels: XLSX → CSV, then patient-level train/val/test split.
python scripts/data/convert_metadata.py \
    --input-dir metadata/raw_metadata --output-dir metadata/csv_metadata
python scripts/data/split_data.py \
    --manifest /scratch/$USER/moco/tensors/CT-COLONOGRAPHY/manifest.csv \
    --metadata metadata/csv_metadata/acrin_combined.csv \
    --output-dir metadata/csv_metadata \
    --label-scheme three --val-frac 0.15 --test-frac 0.15 --seed 42
```

The one-time NBIA retriever RPM extraction is handled inside `jobs/tcia_download.sh`
(Java comes from Sol's `module load`, not a bundled copy); see its comments if the
retriever's jar path differs for a newer version. The `metadata/csv_metadata/` label CSVs are also committed, so
step 3 only needs re-running if the cache is rebuilt.

## Usage

All jobs are submitted from the repo and pull their paths from `jobs/config.sh` —
no editing per user or per run:

```bash
sbatch jobs/train_moco.sh                                  # pretrain from scratch (DATASET=acrin by default)
sbatch jobs/resume_moco.sh                                 # continue on ACRIN (DATASET=pediatric for the other)
sbatch jobs/build_crop_bank.sh                             # fixed evaluation crop bank (run once)
sbatch jobs/eval_repr.sh                                   # score random / imagenet / moco on that bank
sbatch --export=CKPT=checkpoint_0199 jobs/run_lincls.sh    # linear probe a checkpoint
sbatch --export=CKPT_RUN=acrin,CKPT=checkpoint_0249 jobs/run_umap.sh   # UMAP a checkpoint
```

Training uses `mp.spawn` internally — no `torchrun` required. Loss drops rapidly
in the first ~50 epochs then plateaus. To run a stage by hand instead of via
SLURM, read the corresponding job script — it shows the exact `python` invocation.

## HPC / SLURM

Job scripts live in [`jobs/`](jobs/). Paths are centralized in `jobs/config.sh`
(derived from `$USER`), so a new student on Sol runs them unchanged.

| Script | Purpose | Resources |
|---|---|---|
| `config.sh` | Shared path/env definitions sourced by every job | — |
| `tcia_download.sh` | Download raw DICOM from TCIA (NBIA retriever) | 1 CPU, 8 GB |
| `dicom_organize.sh` | *Optional* tidy symlinked DICOM view (needs `dicom-organizer`) | 8 CPUs |
| `prep_array.sh` | Two-phase DICOM preprocessing (discover + array) | 2–4 CPUs, 4–16 GB/task |
| `train_moco.sh` | MoCo pretraining from scratch (`--export=DATASET=acrin\|base\|pediatric`) | 32 CPUs, 128 GB, 2× A100 |
| `resume_moco.sh` | Continue pretraining on one collection (`--export=DATASET=acrin\|pediatric`) | 32 CPUs, 128 GB, 2× A100 |
| `build_crop_bank.sh` | Fixed uint8 evaluation crop bank from the tensor cache | 16 CPUs, 64 GB, `htc` |
| `eval_repr.sh` | Frozen-encoder metric battery over the crop bank | 8 CPUs, 32 GB, 1× A100 MIG |
| `run_lincls.sh` | Linear probing evaluation | 16 CPUs, 64 GB, 1× A100 |
| `run_umap.sh` | UMAP feature extraction | 4 CPUs, 32 GB, 1× A100 |
| `refresh_scratch.sh` | Touch the whole scratch tree (plus any RC-flagged path) to dodge the 90-day purge | 2 CPUs, 2 GB, `lightwork` |

## Repository Structure

```
├── main_moco.py                          # MoCo v2 pretraining (DDP, multi-GPU)
├── main_lincls.py                        # Linear probing on labeled ACRIN data
├── moco/
│   ├── __init__.py                       # Shared utils (to_resnet_format)
│   ├── builder.py                        # MoCo model (dual encoders, queue, InfoNCE)
│   └── ct_dataset.py                     # CTMoCoDataset (contrastive) + CTLinClsDataset (labeled)
├── scripts/
│   ├── pending_series.py                 # Diff manifest.tcia against disk → resumable download list
│   ├── data/
│   │   ├── prep_data.py                  # DICOM → .pt preprocessing + manifest
│   │   ├── convert_metadata.py           # ACRIN XLSX → CSV metadata
│   │   └── split_data.py                 # Patient-level stratified train/val/test splits
│   └── eval/
│       ├── build_crop_bank.py            # Deterministic uint8 crop bank for encoder comparison
│       ├── eval_repr.py                  # Frozen-encoder metric battery → JSON report
│       └── visualize_umap.py             # UMAP projection of backbone features
├── jobs/                                 # SLURM job scripts — the job source of truth
│   ├── config.sh                         # Canonical $USER-derived paths (sourced by all)
│   ├── tcia_download.sh                  # Download raw DICOM from TCIA
│   ├── dicom_organize.sh                 # (optional) symlinked DICOM view
│   ├── prep_array.sh                     # Preprocessing discover + array
│   ├── train_moco.sh                     # Pretraining from scratch
│   ├── resume_moco.sh                    # Continue pretraining (DATASET=acrin|pediatric)
│   ├── build_crop_bank.sh                # Build the evaluation crop bank
│   ├── eval_repr.sh                      # Score encoders on the crop bank
│   ├── run_lincls.sh                     # Linear probing
│   ├── run_umap.sh                       # UMAP visualization
│   └── refresh_scratch.sh                # Dodge the 90-day /scratch purge
├── metadata/
│   ├── manifest.tcia                     # TCIA download spec (pins the exact data)
│   ├── series_catalog.csv                # Per-series paths, sizes, scanner, supine/prone
│   ├── raw_metadata/                     # ACRIN 6664 XLSX files (no-polyp, 6-9mm, >=10mm)
│   └── csv_metadata/                     # Processed CSVs + split label files
├── tools/                                # NBIA retriever RPM (git-ignored; extracted here on first download)
├── notebooks/                            # Dataset characterization + transform validation
├── requirements.txt
├── environment.yml                       # conda env "moco_env"
├── CLAUDE.md                             # operational notes (paths, env, conventions)
└── LICENSE
```

> ASU Sol documentation is kept as a local snapshot at `~/sol-docs/` (outside this
> repo — it is environment reference, not project code). Start from `~/sol-docs/INDEX.md`.

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
