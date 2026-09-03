# CLAUDE.md

Operating notes for this repo on **ASU Sol** (Research Computing HPC). Single
source of truth for paths, environment, and project conventions. Keep it current.

Nothing is hardcoded to a username: paths use `$USER` (original author `jpbunnel`;
any member of `grp_vkodibag` runs everything unchanged).

## What this is

MoCo v2 self-supervised pretraining on unlabeled CT colonography, adapted for 3D
medical imaging, with a linear-probe downstream for polyp size classification.
See `README.md` for the science; this file is the operational layer.

## Environment

- **Cluster:** ASU Sol. Account/group: `grp_vkodibag`.
- **Conda env:** `moco_env` (from `environment.yml`):
  ```bash
  module load mamba/latest
  source activate moco_env
  ```
- Never run training/prep on the login node. Submit `sbatch jobs/<name>.sh` or use
  `salloc`. For anything SLURM (partition/QoS/GPU/walltime, why a job is pending,
  `seff` right-sizing), use the **`sol-jobs`** skill — it carries the current Sol
  facts and defers to the committed scripts in `jobs/`.

## Layout: code vs data

Decoupled, mirrored by name:

- **Code** (durable, git-tracked): `/home/$USER/moco`
- **Data** (disposable, regenerable): `/scratch/$USER/moco`

```
/home/$USER/moco/tools/     NBIA retriever RPM (git-ignored; extracted on first download)

/scratch/$USER/moco/
├── raw/            raw DICOM from TCIA (NBIA output) + metadata.csv
│   ├── CT COLONOGRAPHY/     (TCIA's name — keeps the space)
│   └── Pediatric-CT-SEG/
├── tensors/        preprocessed .pt cache (spaces normalized to hyphens)
├── checkpoints/    base/ (200ep) · acrin/ · pediatric/ · lincls/
├── umap/           UMAP output plots
└── logs/           SLURM .out/.err — every job in jobs/ is routed here
```

- **All scratch paths are defined once in `jobs/config.sh`**, derived from `$USER`.
  Job scripts source it; nothing else hardcodes a scratch path.
- **Naming rule:** TCIA collection folders may contain spaces (`CT COLONOGRAPHY`);
  `prep_data.py` normalizes spaces to hyphens for cache subdirs so tensors are
  shell-safe (`CT-COLONOGRAPHY/`).
- `tools/` is on the durable side deliberately (build dependency, not disposable
  data): a copy kept under `/scratch` gets silently purged. Java comes from
  `module load openjdk-17.0.3_7-gcc-12.1.0`, not a kept local copy.

## Pipeline

| Stage | Script | Job |
|---|---|---|
| Download raw DICOM from TCIA | (NBIA retriever) + `scripts/pending_series.py` | `jobs/tcia_download.sh` |
| (optional) tidy symlinked view | external `dicom-organizer` | `jobs/dicom_organize.sh` |
| DICOM → `.pt` tensors + manifest | `scripts/data/prep_data.py` | `jobs/prep_array.sh` |
| XLSX → CSV metadata | `scripts/data/convert_metadata.py` | — (one-time) |
| Patient-level train/val/test split | `scripts/data/split_data.py` | — |
| MoCo pretraining | `main_moco.py` | `jobs/train_moco.sh` (`DATASET=base\|acrin\|pediatric`, `MOCO_K`, `EPOCHS`) |
| Resume (one collection) | `main_moco.py --resume` | `jobs/resume_moco.sh` (`DATASET=acrin\|pediatric`) |
| Evaluation crop bank | `scripts/eval/build_crop_bank.py` | `jobs/build_crop_bank.sh` |
| Frozen-encoder metrics | `scripts/eval/eval_repr.py` | `jobs/eval_repr.sh` |
| Linear probe | `main_lincls.py` | `jobs/run_lincls.sh` |
| UMAP of features | `scripts/eval/visualize_umap.py` | `jobs/run_umap.sh` |
| Dodge the 90-day scratch purge | — | `jobs/refresh_scratch.sh` |

- Scripts `cd` to `$HOME/moco` themselves; submit from anywhere.
- **Never paste a job body into the OnDemand web UI** — the committed script in
  `jobs/` is the source of truth so runs stay reproducible.
- **Job logs:** every script sets `#SBATCH -o/-e /scratch/%u/moco/logs/%x.%j` (and
  `-J <name>` so `%x` is meaningful), so `.out`/`.err` land in `$LOG_DIR`, never the
  repo. `config.sh` `mkdir -p`s it; SLURM evaluates the redirect before that runs,
  so on a fresh scratch create it once (see Rebuild).
- Pretraining runs 2×A100 (`--gres=gpu:a100:2`); lincls/UMAP 1×. `main_moco.py`
  uses `mp.spawn` — pass `--multiprocessing-distributed`, not torchrun.
  MoCo needs >=2 ranks for shuffle-BN: a single-GPU run collapses the pretext
  task (Acc@1 pins near 100%, loss near 0). Confirm the log shows `Use GPU: 0`
  AND `Use GPU: 1`.

## Data durability — scratch is DISPOSABLE

`/scratch` is not backed up; **files unread for 90 days are purged** (RC emails
first, drops `scratch-dirs-{inactive,pending-removal}.csv` in `$HOME`). Treat it
as a regenerable cache:

1. **Refresh:** `sbatch jobs/refresh_scratch.sh` reads those CSVs and `touch`es
   the tree to reset the clock. Run on a purge warning.
2. **Rebuild:** `mkdir -p /scratch/$USER/moco/logs` (one-time, so the first job's
   `#SBATCH -o` has somewhere to write), then `jobs/tcia_download.sh` →
   `jobs/prep_array.sh` → `scripts/data/split_data.py` (README "Reproducing the data").

**Checkpoints are the only non-regenerable artifact** — copy any run worth keeping
off scratch before a purge.

## Trusting a checkpoint or the cache

- **A pretraining checkpoint is a valid baseline only if its run used >=2 GPU
  ranks and its curve did not collapse.** Acc@1 pinning near 100% (loss near 0)
  means the pretext task collapsed; usual causes are a single-GPU run (shuffle-BN
  becomes a no-op, so BN stats leak q/k identity) or near-identical q/k views.
  Check `logs/moco-train.*.out` before building on any checkpoint or running
  UMAP/lincls against it. UMAP plots inherit the checkpoint's validity.
- **Counting ranks from the log: `Use GPU: N` lines do NOT work.** `main_worker`
  swaps `builtins.print` for a no-op on every rank except 0, so a healthy 2-GPU
  run still logs only `Use GPU: 0`. Infer the per-rank batch from `Acc@1`
  granularity instead: values are multiples of `100 / per_rank_batch`, so all
  multiples of 0.78125 means batch 128 (2 ranks at `-b 256`), whereas multiples
  of 0.390625 mean batch 256 (1 rank). Cross-check `sacct -j <id> --format=AllocTRES`.
- **The `tensors/` cache is regenerable only while `raw/` still holds the full
  TCIA download** and `series_manifest.txt`'s source paths resolve. After a
  `/scratch` purge of `raw/`, a rebuild starts from `jobs/tcia_download.sh` +
  `metadata/manifest.tcia` (**549 GB**: CT COLONOGRAPHY 485 GB / 3451 series /
  825 patients, Pediatric-CT-SEG 64 GB / 718 series / 359 patients; many hours).
  Good-cache invariants per file: MetaTensor `(1,H,W,D)`, RAS, 1 mm isotropic,
  float32 in [0,1], depth >=3.
- **Do not trust a `.pt` filename as a patient identifier.** `series_filename`
  builds `<patient_id>_<4 hex>.pt`, but for the 40 CT COLONOGRAPHY patients using
  TCIA's `CTC-…` subject convention an older `prep_data.py` wrote an md5 as the
  `patient_id`, hashed **per series** — so 102 of 1720 cached files fragment one
  patient into several. Anything grouping by patient (splits, retrieval galleries,
  leakage checks) must resolve identity through `tensors/<collection>/manifest.csv`
  instead: its `series_path` still names the real patient directory at
  `parts[-3]`. `build_crop_bank.py`'s `load_series_index` does this. Those paths
  predate the current scratch layout and no longer resolve on disk, which does not
  matter as they are parsed as strings.

## Known issues / where work left off

- **`openpyxl` is pinned but not installed** in `moco_env`, so
  `scripts/data/convert_metadata.py` cannot currently run. Install it before any
  work that re-parses `metadata/raw_metadata/*.xlsx` (Phase 3's slice indices).
  The committed CSVs in `metadata/csv_metadata/` mean nothing else is blocked.
- **Augmentation axis, RESOLVED (8c4be1a, 2026-04-15):** the bug was in
  `moco/ct_dataset.py` `moco_augs`, not `prep_data.py` (which has no rotation).
  `RandRotated(range_x)` rotated the A-S plane against the 3-voxel depth axis;
  fixed to `range_z` (in-plane axial). Any pre-April run is suspect on this axis.
- **Primary track is `DATASET=acrin`** (CT-COLONOGRAPHY only); `base` (both
  collections) is an ablation. Pediatric-CT-SEG is out-of-domain (pediatric
  organ-at-risk body CT, no bowel prep or insufflation) and unlabeled; if
  pretraining on it alone, set `MOCO_K=4096` (only 354 volumes).
- **Status 2026-09-02 (corrected):** all current `checkpoints/` collapsed, but
  *not* for the reason previously recorded here. Jobs 49491542 / 49495423 /
  51208743 each ran on **2 A100s** (`sacct` AllocTRES `gres/gpu:a100=2`; every
  `Acc@1` in 51208743 is a multiple of 100/128, never an odd multiple of
  100/256). Shuffle-BN was working. The confirmed cause is identical q/k views
  (see the notes section below). Re-running on 2 GPUs alone will not fix it.
- **`raw/` is partially restored; resubmit `jobs/tcia_download.sh` until it
  reports 0 pending.** Three retriever (v4.4.3) bugs hit in sequence, all now
  worked around in the job script:
  1. Job 62500909 died in `DataRetrieverCLI.scanDataDir`
     (`NoSuchElementException: No line found`): a single `printf 'Y\nM\n'`
     starved the retriever's second `Scanner`. Fixed by padding stdin with 50k
     `A` lines.
  2. Jobs 62515055 / 62515887 died in `DataRetrieverCLI.performDownload`
     (`StringIndexOutOfBoundsException: begin 37, end 23, length 50`), before any
     series downloaded, deterministically (even a 39-series manifest). Root
     cause: the retriever builds its nest dir as
     `manifestPath.substring(lastSlash+1, firstDot)`, so a manifest under a
     dot-prefixed dir (`.tcia_pending/`) puts the first `.` at index 23, before
     the last `/` at index 37, and the substring underflows. Fixed by moving the
     pending manifest to a plain dir: `${DATA_ROOT}/tcia_pending/manifest.tcia`.
     (A brief chunking mitigation was tried and reverted; it was the path, not
     request volume.)
  Disk now: ~33 of 879 CT COLONOGRAPHY patient dirs (16 GB), complete
  Pediatric-CT-SEG. **`tensors/` is intact** (1720 + 354 volumes, 468 GB), so
  nothing downstream is blocked; `raw/` is needed only to re-run `prep_data.py`
  (e.g. to change the HU window).

## Reference

- `~/sol-docs/` — local mirror of `docs.rc.asu.edu` (environment reference, not in
  this repo). Update it with the **`sol-docs-refresh`** skill.

## Research notes (state-dependent; re-check before trusting)

Written 2026-09-02 from a full read of the code, the `tensors/` cache, the
training logs, `checkpoints/base/checkpoint_0199`, and `metadata/raw_metadata/`.
Everything below is a finding plus the evidence for it, so a later session can
re-verify rather than re-derive.

### Why the base run collapsed

`moco/ct_dataset.py:109-115` extracts **one** crop, then deep-copies it twice:

```python
base_crop = self.extract_crop(volume)
view_q = self.moco_augs(copy.deepcopy(base_crop))["image"]
view_k = self.moco_augs(copy.deepcopy(base_crop))["image"]
```

Both views are therefore the same pixels, differing only by flip, +/-15 deg
rotation, Gaussian noise and blur, none of which change the crop's texture
fingerprint. There is no RandomResizedCrop analogue, so the model never has to
learn position or scale invariance, and colour jitter was deliberately dropped
(HU semantics) with nothing substituted in its place. MoCo v2's own ablations
find jitter necessary precisely because without it the task falls to low-level
statistics.

Evidence from `logs/moco-train.51208743.out` (200 epochs, 2d5h, COMPLETED):

| Epoch | Loss | Acc@1 vs 16384 negatives |
|---|---|---|
| 0 | 7.76 | 1.6% |
| 20 | 4.54 | 39.8% |
| 50 | 0.62 | **99.1%** |
| 200 | 0.081 | **99.7%** |

Saturated by epoch 50, so epochs 50-400 (the `acrin/` and `pediatric/` resumes)
added nothing. The saved 128-d queue is near-perfectly uniform (mean pairwise
cosine 0.000, std 0.100, effective rank 121.5/128), which with a saturated
pretext task is the signature of instance fingerprinting rather than collapse.
This also explains why `umap/*.png` is one featureless blob: uniformly spread
fingerprint features have no cluster structure to find.

**Do not resume from these checkpoints.** Keep them as the "before" baseline.

### The cache is sound; the *format* is the bottleneck

Verified against real volumes: RAS orientation confirmed, 1 mm isotropic
confirmed, `roi_size=(224,224,3)` really is an axial 224x224 mm patch by 3
adjacent slices, `RandRotated(range_z)` really is in-plane, `to_resnet_format`
gives `(3,224,224)` correctly. Over 1000 sampled crops: mean 30% zero voxels,
**zero** crops >90% blank or near-constant. No reason to re-run `prep_data.py`.

The cost is I/O, not preprocessing. `__getitem__` does `torch.load` on a whole
volume (~245 MB float32, ~356x356x424) to take one 602 KB crop, roughly 400x
read amplification. In the log, `Data` averages **5.4 s/iter** against `Time`
0.22 s of GPU compute, so ~96% of wall clock is I/O: 53 hours of job for about
2 hours of GPU work.

Fix without touching preprocessing: convert the `.pt` cache to `(z,y,x)` **uint8
`.npy` memmaps**. A 3-slice slab becomes one ~380 KB contiguous read, and disk
drops 468 GB to ~117 GB. The 400 HU window across 256 levels is 1.56 HU/level,
far under CT's ~10-20 HU noise floor, so uint8 is lossless in practice. Pure
tensor-to-tensor, embarrassingly parallel as a SLURM array, one read per volume.

**Deferred, do not act yet:** the HU window `[-150, 250]` clips all gas to
exactly 0 (37-61% of each volume). Standard for soft tissue, but CTC is
specifically about soft-tissue protrusions into a gas-insufflated lumen. The
air/wall edge survives clipping so it is defensible. If the polyp task
underperforms, re-prep only the ~350 labelled ACRIN patients to test a wider
window, not all 2074 volumes. Requires `raw/` to be restored first.

### Unused signal already on disk

- **Polyp slice numbers.** `metadata/raw_metadata/*.xlsx` columns B and C are
  `Slice# polyp Supine` / `Slice# polyp Prone`. `scripts/data/convert_metadata.py`
  reads past them (`row[3:]`) and never saves them. **74 of 104 polyp patients**
  have a usable slice index for at least one view, some as bare integers, some
  as `147/275` (slice-of-total, resolution-independent). This is weak 3D
  localisation and it is the highest-value unused asset in the repo.
- **Colon segment code.** `LESION x.1` is a 1-6 segment code, present for all
  104 polyp patients and well spread across segments.
- **Supine/prone position, directly labelled.** CTC scans each patient face-up
  then face-down minutes apart. The position is *recorded*, not inferred: the
  series directory name (never the study name, which is often
  `SupineandProneColon` and matches both) carries SUPI/PRON. Parsed from
  `tensors/<collection>/manifest.csv`, the cache holds 618 supine / 659 prone /
  13 decubitus / 430 unknown series, and **587 patients have both**. Free
  positive pairs (Phase 2) and a label-free retrieval benchmark (Phase 0).
  Crop-level pairing needs rough z-alignment or you match liver to pelvis.
  Note most patients have **4 series, not 2** (642 of 825 in the catalogue),
  because reconstructions are catalogued separately; two of a patient's series
  are often the same acquisition reconstructed twice.
- **Scanner manufacturer**, per series, from `metadata/series_catalog.csv`:
  SIEMENS 2025, GE 970, Philips 144, TOSHIBA 110, blank 202. The confounder
  control — an encoder that clusters by scanner rather than anatomy is a
  failure mode UMAP can show directly.
- **Pediatric-CT-SEG organ segmentations.** 253 RTSTRUCT DICOM files are on
  disk under `raw/Pediatric-CT-SEG/`, one per patient series. `prep_data.py`
  drops them silently via `min_slices >= 10` (a single-file series never
  qualifies), not by any modality check. Real annotations, entirely unparsed.
  Deferred, not scheduled: out-of-domain for the polyp task.

### Why the lincls design as written cannot work

A 6-10 mm polyp inside a 224 mm crop is 2.7-4.5% of the width and ~0.1% of the
area. ResNet-50's stride-32 output is 7x7, so the polyp is under a third of one
cell, then global-average-pooled into 2048 dims, contributing on the order of
0.2% of the vector. A **linear** head then has to predict a **patient-level**
max-polyp-size label from a **random** crop that probably does not contain the
polyp. Separately, `jobs/run_lincls.sh` carries ImageNet defaults that do not
transfer (`--lr 30.0`, `wd=0`, tuned for a 1000-way head on 1.28M images against
437 training files and 3 classes), and class balance is 243/69/35, so plain
accuracy has a **70% majority-class floor**. Expect ~70% and a wrong conclusion.
The right framing for patient-level labels is attention-MIL: encode many crops
per volume, attention-pool, classify.

### Planned phases

- **Phase 0 (~1 GPU-hour, no retraining).** Build the evaluation harness and get
  baseline numbers on existing checkpoints. k-NN probe on frozen features;
  RankMe / alignment / uniformity; supine-prone patient retrieval. Three
  controls evaluated identically: random init, ImageNet-init ResNet-50, MoCo.
  **The ImageNet control is the bar that matters**; if MoCo does not beat it
  there is no result yet. No single metric suffices: the current checkpoint
  scores well on RankMe and uniformity and is still useless.
  Any Phase 0 script must load each volume **once** and take all crops from it
  while resident, the way `scripts/eval/visualize_umap.py`'s `extract_features`
  already does. At one read per volume the I/O is tolerable, so Phase 0 does not
  need Phase 1. Implemented as two steps so that pass happens only once ever:
  `build_crop_bank.py` writes a deterministic uint8 bank (16 depth-stratified
  slabs per volume, ~5 GB) and `eval_repr.py` scores each encoder over it, so
  every comparison runs on byte-identical input in minutes.
  Retrieval is reported twice: `any_series` over the whole ACRIN pool, and
  `cross_position` (supine query, prone-only gallery). Prefer the latter: with
  ~4 series per patient the former can be won by matching an alternate
  reconstruction of the same acquisition, which is the shortcut again.
- **Phase 1.** uint8 memmap re-cache (SLURM array), bounded by reading 468 GB once.
- **Phase 2.** Fix the positive pair, in priority order: (a) two **independent**
  crops with controlled overlap instead of one deep-copied crop, the single most
  important change; (b) scale jitter (crop 160-320 mm, resize to 224), since
  every crop is currently exactly 224 mm FOV; (c) HU window jitter (+/-30 HU
  shift and width) as the HU-respecting stand-in for colour jitter; (d) optional
  supine/prone positives; (e) exclude same-volume keys from the negatives. On
  (e): 41,480 samples/epoch against a 16,384 queue means the queue holds ~40% of
  an epoch, so with 20 crops/volume roughly **8 crops of the same scan sit in
  the negatives** whenever the query comes from that scan. Adjacent slabs of one
  colon are being pushed apart while identical pixels are being pulled together;
  both pressures favour instance fingerprinting.
  Watch `Acc@1`: still >95% by epoch 20 means the augmentations are still too
  weak. Healthy is roughly 60-85%.
- **Phase 3.** Evaluate with the localisation data: polyp-slab probe from the 74
  slice indices (annotated slab positive, random slabs from no-polyp patients
  negative), reporting **AUROC and balanced accuracy** on a patient-level split.
  Then attention-MIL for the patient-level size labels.

Use UMAP throughout as a *picture of metrics already computed*, never as the
metric. Colour by polyp status, size, colon segment, patient ID, z-position,
scanner. Separation by z-position and scanner but not pathology tells you
exactly what was learned. Its global structure is not trustworthy.

### Deferred

- Use the size/localisation metadata for **evaluation only** for now. 104
  positives is too few to train on and doing so burns the only test set.
- Keep MoCo. The failure was augmentation design, not the method. If it plateaus
  after Phase 2, the principled next step is a dense or masked objective
  (DenseCL, or MAE/SparK), not more MoCo epochs: global contrastive learning is
  known to be weak for small-object tasks because pooling discards spatial detail.
