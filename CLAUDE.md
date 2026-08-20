# CLAUDE.md

Operating notes for this repo on **ASU Sol** (Research Computing HPC). Read this
first; it is the single source of truth for paths, environment, and conventions
so a session doesn't have to re-derive them. Keep it current when those change.

> **Author alias:** paths below use `$USER`, which resolves to whoever is logged in.
> The original author's alias is `jpbunnel`; a different student in `grp_vkodibag`
> runs everything unchanged because nothing is hardcoded to a username.

## What this is

MoCo v2 self-supervised pretraining on unlabeled CT colonography, adapted for 3D
medical imaging, with a linear-probe downstream for polyp size classification.
See `README.md` for the science; this file is the operational layer.

## Environment

- **Cluster:** ASU Sol. Group: `grp_vkodibag`.
- **Conda env:** `moco_env` (from `environment.yml`). Activate on Sol:
  ```bash
  module load mamba/latest
  source activate moco_env
  ```
- **Do not** run training/prep on the login node. Submit with `sbatch jobs/<name>.sh`
  or use an interactive `salloc`.

## Layout: code vs data

Code and data are deliberately decoupled and mirror each other by name:

- **Code** (durable, git-tracked): `/home/$USER/moco`
- **Data** (disposable, regenerable): `/scratch/$USER/moco`

```
/home/$USER/moco/
└── tools/          NBIA retriever RPM (git-ignored; extracted here on first download)

/scratch/$USER/moco/
├── raw/            raw DICOM from TCIA (NBIA output) + metadata.csv
│   ├── CT COLONOGRAPHY/     (TCIA's name — keeps the space)
│   └── Pediatric-CT-SEG/
├── tensors/        preprocessed .pt cache (spaces normalized to hyphens)
│   ├── CT-COLONOGRAPHY/
│   └── Pediatric-CT-SEG/
├── checkpoints/    base/ (200ep) · acrin/ · pediatric/ · lincls/
├── umap/           UMAP output plots
└── logs/           job logs
```

`tools/` moved to the durable side on 2026-08-20 — it's a build dependency (retriever
jar), not disposable data, and nothing reads it often enough to dodge the scratch
purge on its own. That's exactly how a hand-extracted JDK previously kept here
silently rotted (its `lib/` got purged, breaking `tcia_download.sh` with no
warning until the next real download attempt); Java now comes from `module load
openjdk-17.0.3_7-gcc-12.1.0` instead and isn't kept as a local copy at all.

**All these paths are defined once in `jobs/config.sh`** and derived from `$USER`.
Job scripts source it; nothing else should hardcode a scratch path. Naming rule:
TCIA collection folders can contain spaces (`CT COLONOGRAPHY`); `prep_data.py`
normalizes spaces to hyphens when creating cache subdirs, so tensors are shell-safe.

## Pipeline

| Stage | Script | Job |
|---|---|---|
| Download raw DICOM from TCIA | (NBIA retriever) | `jobs/tcia_download.sh` |
| (optional) tidy symlinked view | external `dicom-organizer` | `jobs/dicom_organize.sh` |
| DICOM → `.pt` tensors + manifest | `scripts/prep_data.py` | `jobs/prep_array.sh` |
| XLSX → CSV metadata | `scripts/convert_metadata.py` | — (one-time) |
| Patient-level train/val/test split | `scripts/split_data.py` | — |
| MoCo pretraining | `main_moco.py` | `jobs/train_moco.sh` |
| Resume (one collection) | `main_moco.py --resume` | `jobs/resume_moco.sh` (`DATASET=acrin\|pediatric`) |
| Linear probe | `main_lincls.py` | `jobs/run_lincls.sh` |
| UMAP of features | `scripts/visualize_umap.py` | `jobs/run_umap.sh` |
| Dodge the 90-day scratch purge | — | `jobs/refresh_scratch.sh` |

Submit with `sbatch jobs/<name>.sh` from anywhere; scripts `cd` to `$HOME/moco`
themselves. **Never paste job bodies into the OnDemand web UI** — the committed
script in `jobs/` is the source of truth so runs stay reproducible.

## SLURM conventions (from `~/sol-docs/`)

- Partitions we use: `public` / `-q public` (default, ≤7-day GPU+CPU);
  `htc` / `-q public` (jobs ≤4h, no preemption — used by prep array tasks);
  `lightwork` / `-q public` (idle/light jobs ≤24h — env builds, file ops, tunnels).
- If the lab owns nodes, `general -q grp_vkodibag` gives up-to-30-day walltime with
  no fairshare cost — check `myaccounts` before long pretraining (`public` caps at 7d).
- GPUs: `--gres=gpu:a100:N` (pretraining 2×, lincls/UMAP 1×).
- `main_moco.py` uses `mp.spawn` — pass `--multiprocessing-distributed`, no torchrun.

## Data durability — scratch is DISPOSABLE

`/scratch` is not backed up and **files unread for 90 days are purged** (RC emails
first and drops `scratch-dirs-{inactive,pending-removal}.csv` in `$HOME`). Treat
scratch as a regenerable cache. Two defenses:

1. **Refresh (convenience):** `sbatch jobs/refresh_scratch.sh` reads those CSVs and
   `touch`es our scratch tree to reset the 90-day clock. Run it on a purge warning.
2. **Rebuild (authoritative):** everything is reproducible from TCIA —
   `jobs/tcia_download.sh` → `jobs/prep_array.sh` → `scripts/split_data.py`. See the
   README "Reproducing the data" section. Checkpoints are the ONLY thing not
   auto-regenerable — copy any run worth keeping somewhere durable before a purge.

## Docs

ASU Sol docs are a local mirror of the whole docs.rc.asu.edu site at
**`~/sol-docs/`** (133 pages as of the 2026-08-19 full refresh — see
`~/sol-docs/INDEX.md`). They live at user root — not in this repo — because
they are environment reference, not project code. The key facts are already
distilled above; consult `~/sol-docs/` before guessing Sol behavior. If it's
been a while, ask for a "sol docs refresh" — the `sol-docs-refresh` skill
(`~/.claude/skills/sol-docs-refresh/`) re-crawls the site and rewrites only
what changed.

**Account requests now go through Voyager** (`voyager.rc.asu.edu`, VPN required),
not the old `links.asu.edu/gethpc` form — see `~/sol-docs/voyager-accounts.md`.
Not relevant to this repo's own SLURM jobs (those need an existing, already-
provisioned Sol account), but relevant if a new lab member needs HPC access.

## Current state (scratch artifacts — snapshot 2026-07-17)

What's actually in `/scratch/$USER/moco/` right now — the "where I left off" refs.
Checkpoints are the ONLY non-regenerable artifacts; copy any you care about off
scratch before a purge.

| Run | Location | Checkpoints saved | Notes |
|---|---|---|---|
| base (ACRIN + Pediatric) | `checkpoints/base/` | `checkpoint_0049/0099/0149/0199` | mixed pretraining, `--moco-k 16384` |
| acrin (resumed from base@199) | `checkpoints/acrin/` | `checkpoint_0249` | ACRIN-only; one save then stopped |
| pediatric (resumed from base@199) | `checkpoints/pediatric/` | `checkpoint_0249/0299/0349/0399` | Pediatric-only |
| lincls | `checkpoints/lincls/` | — | not run yet |

UMAP projections of the **base** run already exist in `umap/` (checkpoints 50–200) —
that is the existing reference point for re-analysis. `raw/` (~75 GB DICOM) and
`tensors/` (~468 GB, 2074 volumes) are fully regenerable from TCIA.

## Known issues / where work left off

- **Augmentation axis (unverified):** prior runs may have augmented tensors on the
  wrong axis. `moco/ct_dataset.py` crops `(224,224,3)` in MONAI `(1,H,W,D)` layout,
  then `to_resnet_format` permutes `(2,0,1)` → `(3,224,224)`. Validate flips/rotations
  act on the intended anatomical axes before trusting new experiments.
- Immediate research direction undecided; UMAP re-analysis on the existing base-run
  checkpoints (see table above) is the low-risk way to re-establish a reference point.
