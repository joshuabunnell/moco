# Research notes

State-dependent findings and the phase plan, started 2026-09-15. Re-check before trusting: this is a point-in-time
record, not automatically kept current.

Run-by-run results and the pre-registered Phase 2 ladder live in
[`experiments.md`](experiments.md), with the numbers themselves in
[`experiment_results.csv`](experiment_results.csv). This file holds standing
findings about the data and the code; that one holds what each run did.

## Status and next steps

Kept current message to message: what is in flight, why, and what comes next.
Settled findings move into the sections below; superseded ones are marked, not deleted.

**As of 2026-09-25.** Two workstreams: **A** makes the data trustworthy and fast,
**B** runs the experiments on it. B's training runs wait on A; B's code does not.

### A. Data: rebuild the cache from `raw/` (Phase 1)

**Done (2026-09-16, code only, nothing submitted yet):**

- `prep_data.py` writes int16 raw-HU `.npy` volumes, `(z, y, x)`, with no window.
  Orientation (RAS) and 1 mm resampling are unchanged and were verified sound.
  The `[-150, 250]` window moved to `moco.HU_WINDOW` and is applied when a crop
  is read, so E0-E2 see what the old runs saw and E3 can vary it.
- One file per series, memory-mapped; not one file per slab. Slabs overlap and
  crops take any (z, y, x) offset, so pre-cut slabs would fix crop positions and
  mean ~700k files. A crop read touches only the crop's bytes.
- The cache stays at `tensors/` (`TENSOR_DIR` unchanged), so no path anywhere
  changes. `prep_array.sh` refuses to run while `.pt` files are still there.
- Every reader of the old `.pt` files now goes through `moco/__init__.py`
  (`list_volumes`, `load_volume`, `random_crop`, `apply_window`): `ct_dataset.py`
  (both datasets), `build_crop_bank.py`, `visualize_umap.py`. Verified on a
  synthetic volume that crops match the old MONAI crop exactly, in the same
  orientation, for both training crops and bank slabs.
- `prep_array.sh` simplified: the header holds the per-series resources and the
  discovery run resubmits the same script as the array, so there is one set of
  resources, not two. `ARRAY=0-4` override for a smoke test.
- `split_data.py` default paths pointed at a nonexistent `csv_metadata/`; fixed to
  `metadata/csv_metadata/`.
- The benchmark script drafted on 2026-09-15 was deleted unsubmitted. It priced
  rebuild vs convert, and that decision is made.

**Verified findings (2026-09-16):**

- **40 labelled patients were silently missing from every split.** TCIA names 40
  ACRIN subjects `CTC-<10 digits>` instead of `1.3.6.1.4.1.9328.50.4.<n>`. The old
  `extract_patient_id` matched only the second form by regex and hashed the path
  otherwise, per *series*, so each of those patients became 2-6 unrelated fake
  patients. All 40 are labelled (28 medium, 12 large, 0 no-polyp), and
  `split_data.py` found none of them in the metadata, so `labels_*.csv` hold 305
  of 345 labelled patients: **41% of medium and 34% of large polyp patients were
  lost**. The crop bank was unaffected (`build_crop_bank.py` already recovered the
  real ID from the manifest path), so the Phase 0 retrieval numbers stand. Fixed:
  the patient ID is now the subject directory, which covers every TCIA naming
  scheme (825/825 ACRIN and 359/359 Pediatric subjects resolve, 0 filename
  collisions). The same bug hashed every Pediatric-CT-SEG ID.
- **26 ACRIN series were never cached** (1746 real volumes on disk, 1720 in the
  old cache; every cached one maps to a raw series). 24 are full scans of 388-626
  slices from 17 patients, 2 are 51-54 slice series of `CTC-3174825007`. The old
  manifest has no record of them, so they were most likely absent from the July
  download. Pediatric: 359 on disk vs 354 cached. The rebuild covers both.
- **The other 1705 ACRIN series are scouts/localizers** (under 10 images) and are
  correctly skipped.
- **MONAI random transforms were identical across DataLoader workers.** Each
  MONAI random transform keeps a private `RandomState` that PyTorch does not
  reseed, so all 32 workers drew the same sequence of crop positions, flips,
  rotations and noise, and the same sequence again every epoch (workers restart).
  Confirmed with a 4-worker test: every worker returned the identical crop.
  Fixed with `seed_worker_transforms` in both training scripts. Crop positions no
  longer use MONAI at all. All existing checkpoints trained with the bug.
- **Filenames were tied to the scratch layout.** The series hash was of the
  absolute path, so moving `raw/` renamed every file and orphaned the label CSVs.
  Now hashed on the path relative to the collection directory.
- `tensors/series_manifest.txt` pointed at `/scratch/$USER/cached-tensors` and
  `/scratch/$USER/downloads/manifest`, the layout from before everything moved
  under `/scratch/$USER/moco/`. Rewritten by the next discovery run.

`CLAUDE.md` was removed from the repo on 2026-09-16 along with every mention of
it; the one-time `mkdir` of the logs dir moved into README "Reproducing the data".

**Next, in order:**

1. Done 2026-09-16: old cache renamed to `/scratch/$USER/moco/tensors_pt_legacy`.
2. Done 2026-09-16: smoke test (array job 63439672, 5 ACRIN series). Discovery found
   2105 series (1746 + 359, as predicted). Each series took 1.0-1.7 min with a
   peak of 5.0-6.3 GB RSS against 16 GB requested. Against the legacy `.pt` of
   the same series: identical shape and orientation, max difference 0.00125 in
   windowed units, exactly the +/-0.5 HU int16 rounding bound. HU range
   [-1024, 3071]. Files are half the legacy size (100-148 MB), so the full cache
   should be about 250 GB. The first attempt (63439440) failed on `import numpy`:
   the array inherited the discovery job's active conda env, and `source activate`
   skipped an env it thought was active. Fixed for every job in `config.sh`.
3. First full run (array 63439967, 2026-09-16): 2103 of 2105 series cached in
   about 30 min, 238 GB. Two tasks fail deterministically and are correctly
   excluded: series 7 and 10 of `CTC-3174825007` (51 and 54 slices), the same two
   the legacy cache lacked. Each mixes slices of different in-plane spacing
   (0.9375 mm with 0.879 / 0.850 mm), so they are not one coherent volume and
   `Orientationd` rejects them. That accounts for 2 of the 26 series the legacy
   cache was missing.
   Verification then caught two defects, and that output was deleted:
   - **Volumes were written in Fortran order.** `transpose` only reverses
     strides, `astype` keeps the layout and `np.save` records it, so on disk a
     3-slice z-slab was spread through the whole file. A crop took 0.9-1.3 s
     instead of ~2 ms, no better than the `.pt` cache. The synthetic round-trip
     test checked values and orientation but not byte layout. Fixed with
     `np.ascontiguousarray` in `save_volume`, and `load_volume` now refuses a
     non-C-ordered file.
   - **`manifest.csv` lost a row** (1743 rows, 1744 volumes): concurrent appends
     from the array on the shared filesystem. Tasks no longer append; a final
     `FINALIZE=1` run, queued by discovery with `afterany` on the array, writes
     each manifest once from the volumes present.
   Also noted, not defects: patient `0766` has two short partial series (13 and
   35 slices after resampling), present in the legacy cache too; 94 Pediatric
   volumes are under 224 px in-plane and get padded, as before.
   Second full run (array 63445814, finalize 63445815, 2026-09-16): 2103 of
   2105 cached, the same two failures, finalize ran by itself.
4. Done 2026-09-16, verified: 1744 ACRIN + 359 Pediatric volumes, every one
   int16, 3D and C-ordered; each `manifest.csv` matches its files exactly with no
   duplicates; 825/825 and 359/359 subjects, every filename prefixed by its
   subject ID. 238 GB (legacy 469 GB).
   **Read speed, measured from the login node on cold files:** opening a file
   costs ~29 ms and a 1 MB read ~25 ms on scratch, so latency, not bytes,
   is the floor. A crop through a memory map took ~105 ms median (one network
   round trip per page fault), so `random_crop` now reads the crop's byte range
   in one seek and read: ~56 ms median, 95 ms p90, identical output (checked on
   8 real volumes). Augmenting both views adds ~25 ms. Estimate at 32 workers:
   256 x ~90 ms / 32 = **~0.7 s per batch against 5.4 s before, about 7x, not
   the ~300x predicted** from byte counts alone. That puts a 200-epoch run near
   5 h rather than 2 h. E0's log reports `Data` time per iteration, which is the
   real measurement; if it still dominates, raise `--workers` (the work is I/O
   wait, not CPU).
5. Done 2026-09-16: labels re-split (`split_data.py`, seed 42, three-class,
   15/15). All 345 labelled patients match the cache (302 before). Train 243 /
   val 51 / test 51 patients (516 / 108 / 108 series), each 170-36-36 no-polyp,
   48-10-10 medium, 25-5-5 large; patient-disjoint, verified. Adding 43 patients
   reshuffled the stratified order, so this is a new split, not an extension:
   only 177 of the old 302 kept their side. **From here the test split is frozen**:
   nothing reads `labels_test.csv` until Phase 3's final numbers.
6. Done 2026-09-16: crop bank rebuilt (job 63459066, 7.3 min, 2103 volumes, 0
   failed, 33,648 crops) and P0 re-scored (job 63459067) as **P0r** in
   `experiments.md` and the ledger. Nothing moved beyond its standard error
   (ImageNet cross-position 0.494, MoCo 0.315), so the cache rebuild is neutral.
   The Phase 0 bank and JSONs are in `eval/p0_pt_legacy/`.
7. **Keep `tensors_pt_legacy/`** (469 GB, the old `.pt` cache). Decided
   2026-09-17: scratch has room and it is the only copy of what the pre-2026-09
   runs actually read.

### B. Experiments (`experiments.md`)

- **Decided (2026-09-16):** E0-E5 as pre-registered. E0 also carries the
  worker-seeding fix (an implementation bug, not a recipe choice; recorded in its
  diff). E1 is trained twice with different seeds, to measure retrain noise.
- **Before E0, done 2026-09-16 (tested end to end on a stub encoder):**
  - `main_moco.py` saves `args` and `git_commit` (with `-dirty` when the tree
    differs) in every checkpoint.
  - `eval_repr.py` prints balanced accuracy beside `chance_balanced` (1 /
    n_classes); `majority_floor` stays in the JSON for plain accuracy.
  - `eval_repr.py` adds `knn_prep` (3 prep protocols; the single Magnesium
    citrate subject is dropped, classes under 10 patients are) and
    `knn_contrast` (iodinated contrast as directed, yes/no). Same held-out
    patient half as the collection probe. The training side is class-balanced
    by subsampling: unbalanced, the 692/60 contrast split made k=15 k-NN vote
    "Yes" for every query and score exactly 0.500 regardless of features.
    Read both as confounders: higher means the encoder separates prep/tagging.
    `log_experiment.py` records both.
- **E0, first attempt (job 63459386), cancelled** after 10 epochs / 72 min for
  speed only; its dir is kept as `checkpoints/e0_cancelled_unstaged/`. A 1-epoch
  smoke test on a MIG slice (63459385) had passed first, including a checkpoint
  carrying `args` and `git_commit`. `train_moco.sh` refuses a `RUN` dir that
  already holds checkpoints and records `git_commit.txt`, `git_diff.patch` and
  `git_status.txt` there at job start.
- **Why it was slow: I/O, not compute.** Reading from BeeGFS ran ~4.2 s/iter at
  first (~9.5 min/epoch), easing to ~7 min/epoch as caches warmed. On the node
  workers sat at 20-50% CPU, several in `D`, one GPU at 0% waiting on the other
  rank. The ~0.7 s/iter estimate from single-process reads did not survive 32
  concurrent readers.
- **Fixes (2026-09-17):**
  - `jobs/stage_data.sh` copies the tensor dir to node-local SSD at job start
    (229 GB in 456 s on sg038), falling back to scratch if there is no room.
    Test job 63462339: 2.9 min/epoch staged.
  - `persistent_workers=True` in `main_moco.py`: respawning workers cost ~80 s
    at every epoch start, about half of a staged epoch.
  **Confirmed on E0 (job 63465601, sg042):** staging 390 s, then ~0.69 s/iter
  from epoch 1 (~1.6 min/epoch; epoch 0 is slower while page cache fills), so
  200 epochs take ~5.3 h, against ~53 h for the original run. Data is still
  ~0.3-0.5 s of each iter, so more is available, but not worth chasing now.
- **E0 done 2026-09-17** (job 63465601, 4 h 50 m, 200 epochs, ~1.6 min/epoch).
  Scored at epochs 20 and 200 (jobs 63529379 / 63529380); result and verdict in
  `experiments.md`. Cross-position 0.356 against the parent's 0.315 and
  ImageNet's 0.494. The epoch-20 `Acc@1` kill gate did not fire (0.4%, chance is
  0.39%) although the run still saturated at 99.9%, so that gate moved to epoch
  50; the alignment gate fired correctly at epoch 20.
- **E1 submitted 2026-09-17** (job 63530506, `RUN=e1 SAVE_FREQ=10
  CROP_OVERLAP="0.3 0.7"`). Parameters and prediction pre-registered in
  `experiments.md` before implementation. `random_crop_pair` reads both crops
  from one span; checked against brute-force search on a synthetic volume.
  Watch `Acc@1` at epoch 50 (gate) and alignment cosine on `checkpoint_0019`.
- **E1 completed and scored (2026-09-21).** 200 epochs in 5 h 11 m; scored at
  epochs 20 and 200 (jobs 63759913 / 63759914). Cross-position **0.873**, the
  first encoder here to beat ImageNet's 0.494, against E0's 0.356. Result and
  verdict in `experiments.md`. Two gate consequences: the epoch-50 `Acc@1` gate
  worked as amended (37.8%), and the **alignment cosine gate is withdrawn**,
  because E1 scores higher on it than E0 (0.9936 vs 0.9917 at epoch 20) while
  being better on everything downstream.
- **E1b submitted 2026-09-22** (job 63806648, same command and commit as E1,
  `seed=None` so the draw differs). It exists to price retrain noise: without it
  no gap between any two ladder rungs is interpretable. Still pending on
  Priority as of 2026-09-22 (Slurm estimates a 2026-09-23 02:00 start).
  Verified against E1's submission: same script, same `CROP_OVERLAP`,
  `SAVE_FREQ`, `DATASET`. Scoring is queued behind it (jobs 63811030 /
  63811031, `afterok`), so both checkpoints are scored when training ends.
  **Until E1b starts, do not edit `main_moco.py`, `moco/` or `jobs/`**: the job
  reads them from the working tree at start, so an edit would make E1b a
  different experiment from E1.
- **Docs brought in line with E1 (2026-09-22):** alignment gate struck from
  Kill gates, ImageNet bar marked cleared, later rungs judged against their
  parent, RankMe read as a floor detector only, E4's prediction marked
  superseded. Query count corrected to 613 (671 gallery) where the measurement
  section still said 601 from the old bank.
- **Next after E1b:** E2 (scale jitter), unchanged. Three questions are open for
  the scope discussion and listed under E1's result in `experiments.md`:
  headroom on the main metric (E1 at 0.873), E4's already-met prediction, and
  how many rungs to run before Phase 3. Added the same day: E5 would train on
  the retrieval metric's own supine/prone pairs, so it cannot be scored on it
  as things stand.
- **Recorded 2026-09-22 (questions from the report):** how the evaluation scans
  were chosen and how that differs from a train/val/test split
  (`experiments.md`, "Where the evaluation scans come from"), and the HU window
  history (Research notes, "The HU window"). Neither changes the plan. The first
  surfaced the E5 issue above, and a caveat for reporting E1: MoCo was
  pretrained on the scans it is scored on, while ImageNet never saw them.
- **Verification added (2026-09-22).** `scripts/eval/check_ledger.py` checks that
  every results table in `experiments.md` matches the ledger and that every
  ledger row matches its eval JSON. Run it before committing results; it exits
  nonzero on a mismatch (tested by planting a typo). Separately, a cold-start
  agent audited the E0/E1 write-ups against raw artifacts: numbers and provenance
  all held, and five interpretation claims were corrected in place (see "Audit,
  2026-09-22" under E1's result). The one that matters most for reporting: "the
  match has to come from anatomy" is untested, because supine and prone scans
  share scanner and body size. That is open question 4.
- **E1b done and scored (2026-09-25 write-up; trained 2026-09-22 20:09 to
  09-23 01:46, sg027).** Cross-position 0.874 against E1's 0.873: E1
  replicates, and retrain noise on the main metric is far below sampling noise.
  Retrain moves `any_series` ~0.04 and `knn_polyp` ~0.02. Epoch 20 does *not*
  replicate (0.501 vs 0.160: E1b's loss sat near uniform far longer), so early
  checkpoints cannot rank recipes. Logged to the ledger; `check_ledger.py` OK.
  Details in `experiments.md`, "E1b — result".
- **Decided 2026-09-25 (by Claude, on Josh's instruction to keep the ladder
  moving; reversible):** run E2 next as pre-registered, without waiting on the
  scope questions. None of them blocks E2: E4 and E5 are the rungs they affect,
  and the headroom question is answered well enough by E1b to judge E2.
  Josh's advisor sets direction at the level of "use unlabelled data for better
  pretraining for early detection", so the ladder's technical scope calls are
  made here and recorded, not escalated.
- **Queued for after E1b starts (touches `jobs/`):** have `train_moco.sh` write
  `CROP_OVERLAP` and the other recipe variables into `job.txt`. Today only sacct
  records them.
- **Phase 3 design** (the polyp claim, label-efficiency curves, where the linear
  probe comes in) is written up in `experiments.md`.

## Known issues / where work left off

- **Spreadsheet readers, RESOLVED (2026-09-15):** `openpyxl` and `xlrd` are now
  installed in `moco_env` and both are pinned in `environment.yml` /
  `requirements.txt`. `xlrd` is required because TCIA ships the polyp
  spreadsheets as legacy `.xls`; `openpyxl` covers the re-saved `v1/` `.xlsx`
  copies and the human-readable data dictionary.
- **Polyp slice indices are still unparsed.** `convert_metadata.py` reads
  `row[3:]` and so drops columns B/C (`Slice# polyp Supine` / `Prone`).
  Re-verified against the canonical v2 files: 35/35 large and 39/69 medium
  patients carry an index, 74 of 104 total. This is Phase 3's localisation
  signal and nothing reads it yet.
- **TCIA Version 2 (2026-08-24) changed no polyp data.** Diffed 2026-09-15:
  `raw_metadata/v1/*.xlsx` and `raw_metadata/v2_2026-08-24/*.xls` have identical
  columns and **zero differing cells** (35 large / 69 medium / 243 no-polyp).
  `convert_metadata.py` regenerates all four committed CSVs byte-identically from
  the v2 files. Do not re-diff these; the only new content in Version 2 is the
  clinical table (see below). `series_catalog.csv` is likewise byte-identical to
  `/scratch/$USER/moco/raw/metadata.csv` and current.
- **Two contradictory polyp labels in the TCIA source.** Patient `...0011` is
  listed in both `no-polyp` and `6-9mm` (9.0 mm); patient `...0216` is in both
  `6-9mm` (as 10.0 mm, the wrong bucket) and `large-10-mm` (as 0 mm). Present in
  v1 and v2 alike, so it is TCIA's error, not a parsing bug. `convert_metadata.py`
  does not validate, so each patient appears twice in `acrin_combined.csv`
  (347 rows, 345 unique). `split_data.py` resolved both to a single label and put
  both in **train**, so the Phase 0 evaluation is uncontaminated. Fixing the
  labels would change the splits and so invalidate the Phase 0 baseline numbers;
  leave them until a deliberate re-split. **That re-split is now scheduled**
  (Status A step 5), because the missing `CTC-` patients force one anyway.
  **Resolved 2026-09-16:** `convert_metadata.py` keeps the larger finding, so
  `0011` is 6-9 mm (9.0 mm) and `0216` is large (10.0 mm). `acrin_combined.csv`
  is now one row per patient: 242 no-polyp / 68 medium / 35 large = 345.
  The per-category CSVs still list both, as TCIA published them.
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
- **`raw/` is fully restored (job 62516121, finished 2026-09-05).** Verified
  2026-09-15 against `metadata/series_catalog.csv`: 3451 CT COLONOGRAPHY series
  across 825 subjects plus 718 Pediatric-CT-SEG series, matching TCIA's Version 2
  listing exactly. Three retriever (v4.4.3) bugs were hit in sequence getting
  there, all worked around in the job script and worth keeping for the next
  rebuild:
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
  `raw/` is needed only to re-run `prep_data.py` (e.g. to change the HU window);
  `tensors/` (1720 + 354 volumes, 468 GB) is intact independently of it.

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
**zero** crops >90% blank or near-constant. No reason to change the geometry steps of `prep_data.py`. (Superseded 2026-09-16 on the *window*: the cache is being rebuilt as raw HU, see Status.)

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

**Superseded 2026-09-16 by the rebuild decision (see Status); kept for the reasoning.** The HU window `[-150, 250]` clips all gas to
exactly 0 (37-61% of each volume). Standard for soft tissue, but CTC is
specifically about soft-tissue protrusions into a gas-insufflated lumen. The
air/wall edge survives clipping so it is defensible. If the polyp task
underperforms, re-prep only the ~350 labelled ACRIN patients to test a wider
window, not all 2074 volumes. Requires `raw/` to be restored first.

### The HU window: value never changed, where it is applied did

Recorded 2026-09-22, traced through git history.

- **The value has been `[-150, 250]` HU since the first prep script**
  (`9e735a6`, 2026-02-25). It is a standard soft-tissue window: colon wall, fat
  and muscle keep contrast, while gas (below -150) clips to 0 and bone (above
  250) clips to 1. It rescales linearly to [0, 1]. No run has used any other
  value, and ImageNet normalisation is never applied to CT input for MoCo runs.
- **Until 2026-09-16 it was baked into the cache.** `prep_data.py` applied it
  with MONAI `ScaleIntensityRanged` before saving, so the `.pt` files held only
  windowed values. The original HU outside the window was gone, and changing
  the window meant re-running prep from DICOM.
- **Since 2026-09-16 (`9643f7a`) it is applied when a crop is read.** The
  cache stores raw int16 HU. The window lives in `moco.HU_WINDOW` and is
  applied by `moco.apply_window` inside `random_crop` / `random_crop_pair`
  (training) and in `build_crop_bank.py` (evaluation). Checked against the
  legacy cache: identical up to +/-0.5 HU of int16 rounding. P0r confirmed
  the move is neutral.
- **Why it moved:** two planned experiments need HU outside the current
  window. E3 jitters the window centre and width per view, which on a
  pre-clipped cache would only shuffle levels inside an already-clipped band.
  And the open question of whether a wider window helps polyp detection
  (clipping sends 37-61% of each volume, all the gas, to exactly 0) can now be
  tested without re-prepping.
- **Consequence to keep in mind:** the crop bank is stored as uint8 *after*
  windowing, so it is tied to `[-150, 250]`. A run trained on a different
  window has to be scored on a bank rebuilt with that window, and then every
  prior run re-scored on it (protocol rule 4). E3 jitters around the same
  centre, so it can still use the current bank. A change to the base window
  could not.

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
- **Clinical table, new in TCIA Version 2 (added 2026-08-24, pulled 2026-09-15).**
  `metadata/csv_metadata/clinical_data.csv` (from `convert_clinical.py`), 752 of
  825 subjects, joined on `patient_id`. All 345 polyp-labelled subjects have a
  clinical record; 407 clinical subjects have no polyp label; 74 imaged subjects
  have no clinical record.
  **It is not a second label set** — there is no age, no BMI, and no diagnosis or
  outcome field. What it is good for:
  - *Bowel prep and contrast compliance* (`type_of_colon_prep_utilizied`,
    `cathartic_laxative_taken_as_directed`, `barium_sulfate_taken_as_directed`,
    `iodinated_oral_contrast_taken_as_directed`). Oral contrast tags residual
    stool and fluid bright, so these change image appearance directly. Three prep
    protocols split the cohort 401 / 195 / 155, and 60 subjects did not take the
    iodinated contrast as directed. That is a **visible, label-free confounder**
    and a sharper probe than scanner make: an encoder that sorts by tagging
    rather than anatomy will show it here.
  - *Gender* (391 F / 361 M), race, ethnicity, insurance status, ZIP3, and family
    history of colon cancer (73 yes) — stratification and fairness reporting.
  Codes are meaningless without
  `raw_metadata/v2_2026-08-24/CT-Colonography_machine_readable_data_dictionary_*.tsv`;
  `convert_clinical.py` is the join.
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
- **Phase 1.** ~~uint8 memmap re-cache~~ Superseded 2026-09-16: int16 raw-HU rebuild from DICOM, see Status A.
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
  Watch `Acc@1`: still >95% by epoch 50 (moved from epoch 20 after E0; see the
  kill gates in `experiments.md`) means the augmentations are still too weak.
  Healthy is roughly 60-85%.
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
