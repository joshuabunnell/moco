# MoCo v2 for CT Colonography

Pretrain an image encoder on **unlabelled** CT colonography scans with
[MoCo v2](https://arxiv.org/abs/2003.04297), then test whether it helps find
polyps, especially when few labelled patients are available.

## How it works

```
TCIA DICOM ──prep──▶ 1 mm raw-HU volumes ──MoCo──▶ encoder ──eval──▶ metrics
                                                     │
                     polyp labels (345 patients) ────┴──▶ polyp probes
```

1. **Data.** Public [TCIA](https://www.cancerimagingarchive.net/) collections:
   [CT COLONOGRAPHY (ACRIN 6664)](https://www.cancerimagingarchive.net/collection/ct-colonography/)
   (1,744 usable scans, supine and prone per patient, polyp-size labels for 345
   patients) and [Pediatric-CT-SEG](https://www.cancerimagingarchive.net/collection/pediatric-ct-seg/)
   (out-of-domain, optional). Each scan is reoriented and resampled to 1 mm, and
   stored as raw Hounsfield units.
2. **Pretraining.** A ResNet-50 sees 2.5D crops: three adjacent axial slices as
   the three channels, windowed to soft tissue ([-150, 250] HU) when read. MoCo
   learns by pulling two views of the same scan region together and pushing
   other regions away. Only augmentations that keep HU meaningful are used, so
   no colour jitter.
3. **Evaluation.** The encoder is frozen and scores one fixed bank of crops
   (16 per scan), so runs are compared like for like against two controls: an
   untrained network and ImageNet weights.

## What the metrics mean

| Metric | Question it answers |
|---|---|
| **Cross-position retrieval** (main) | Given a patient's supine scan, is their prone scan the most similar of all prone scans? The body has shifted between scans, so this rewards recognising the patient's anatomy, not the same pixels. |
| Any-series retrieval | Same, against every scan. Easier, because alternate reconstructions of one acquisition count. |
| Depth (`knn_zpos`) | From one crop, how well can its position along the scan be recovered? |
| Polyp probe (`knn_polyp`) | Nearest-neighbour vote for none / 6-9 mm / >=10 mm. Weak by design: a whole-scan vector hardly sees a 6 mm polyp. |
| Confounders | Can the features tell collection, bowel prep or contrast apart? Higher means more risk of sorting scans by acquisition instead of anatomy. |
| RankMe | Effective dimension of the features. Only flags collapse. |

Label-free metrics choose the pretraining recipe. The polyp question is then
tested separately with labels, via slice-level and patient-level probes and
label-efficiency curves (see `docs/experiments.md`).

## Running it (ASU Sol)

```bash
module load mamba/latest && source activate moco_env   # or: conda env create -f environment.yml
mkdir -p /scratch/$USER/moco/logs                       # once

# Data (only if /scratch was wiped; everything rebuilds from TCIA)
sbatch jobs/tcia_download.sh      # raw DICOM, pinned by metadata/manifest.tcia
sbatch jobs/prep_array.sh         # DICOM → volumes
sbatch jobs/build_crop_bank.sh    # fixed evaluation bank

# One experiment: train, then score automatically when training succeeds
JOB=$(sbatch --parsable --export=ALL,RUN=myrun,SAVE_FREQ=10,CROP_OVERLAP="0.3 0.7" jobs/train_moco.sh)
sbatch --dependency=afterok:$JOB --export=ALL,ENCODERS=moco,CKPT_RUN=myrun,CKPT=checkpoint_0199 jobs/eval_repr.sh

# Record it
python scripts/eval/log_experiment.py --run-id MYRUN --note "..." /scratch/$USER/moco/eval/eval_moco_myrun_*.json
python scripts/eval/check_ledger.py   # docs == ledger == JSONs
```

All paths come from `jobs/config.sh` (derived from `$USER`), so jobs run
unchanged for anyone. Recipe options (crop overlap, scale, window jitter) are
environment variables documented at the top of `jobs/train_moco.sh`. Each run
directory records the commit, diff and recipe it ran with.

Other jobs: `acquisition_probe.sh` (is retrieval matching scanner/body?),
`figures.sh` (plain-language figures into `docs/figures/`), `run_lincls.sh`
(linear probe), `resume_moco.sh`, `refresh_scratch.sh` (avoid the 90-day
`/scratch` purge).

## Where things are

| Path | Contents |
|---|---|
| `main_moco.py`, `moco/` | Training, the MoCo model (unchanged from Meta), CT crop reading |
| `scripts/data/` | DICOM prep, label conversion, patient-level splits |
| `scripts/eval/` | Crop bank, metric battery, ledger tools, probes, figures |
| `jobs/` | Slurm scripts; the source of truth for how anything is run |
| `metadata/` | TCIA manifest, series catalog, polyp labels, clinical table |
| `docs/experiments.md` | Protocol, pre-registered experiments, results and verdicts |
| `docs/experiment_results.csv` | Machine-written metrics, one row per scored model |
| `docs/research_notes.md` | Current status, and standing findings about the data and code |

Data lives on `/scratch/$USER/moco/` as a disposable, reproducible cache; code
and results live in this repo.

## Citation and license

He et al., *Momentum Contrast for Unsupervised Visual Representation Learning*,
CVPR 2020. Chen et al., *Improved Baselines with Momentum Contrastive Learning*,
arXiv:2003.04297. Cite the TCIA collections if you use the data.

[MIT License](LICENSE). Original MoCo implementation by Meta Platforms, Inc.
