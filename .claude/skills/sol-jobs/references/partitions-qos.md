# Partitions, QoS, GPUs, fairshare

Everything here was verified on Sol on **2026-09-02** as user `jpbunnel`
(account `grp_vkodibag`), cross-checked against `~/sol-docs/partitions-and-qos.md`,
`~/sol-docs/resource-limits.md`, and `~/sol-docs/fairshare.md`. Re-verify with
`myaccounts -p`, `sinfo`, and `sacctmgr show qos` if a decision depends on an
exact number, and offer `sol-docs-refresh` if `~/sol-docs/INDEX.md` shows an old
date.

## Contents

- [Partitions (live `sinfo`)](#partitions-live-sinfo)
- [QoS available to this account](#qos-available-to-this-account)
- [There is no 30-day owned-node path](#there-is-no-30-day-owned-node-path)
- [GPU GRES strings (live `sinfo -o %G`)](#gpu-gres-strings-live-sinfo--o-g)
- [Preemption model](#preemption-model)
- [Fairshare and CHE cost](#fairshare-and-che-cost)
- [Queue and array limits](#queue-and-array-limits)
- [Storage](#storage)

## Partitions (live `sinfo`)

| Partition | Walltime cap | Nodes | GPUs present | Notes |
|---|---|---|---|---|
| `htc` | **4 h** | 212 | a100, a100.20gb, a100.40gb, a30, h100, l40, h200 | Cluster **default** partition. No preemption with `-q public`. Widest GPU selection. Best home for anything that fits in 4 h. |
| `public` | **7 d** | 166 | a100 (`:4`), a100.20gb (`:16`), a30 (`:3`) | General-purpose RC-owned CPU+GPU. Only A100 and A30 here, no H100/L40/H200. |
| `general` | **14 d** | 100 | a100, a100.40gb, a30, h100, l40, h200 | Privately-owned nodes. Needs `-q private` (preemptable) for this account, since no `grp_` QoS is granted. |
| `highmem` | 7 d | 11 | none | Up to ~2 TB RAM/node. Use `-q public`. RC can extend past 7 d case-by-case. |
| `lightwork` | 1 d | 3 | a100.20gb (`:16`) | Env builds, compiles, file ops, tunnels. **Max 8 cores/node.** Sustained ~100% CPU here is subject to cancellation. |
| `arm` | 7 d | 4 | gh200 (`:1`) | Grace Hopper, `aarch64`. Software must be built for ARM. |
| `fpga` | 7 d | 4 | none | FPGA / vector-engine accelerators. |
| `gaudi` | 7 d | 10 | hl225 (`:8`) | Habana Gaudi accelerators. |

Source: `~/sol-docs/partitions-and-qos.md` (prose descriptions; that page still
says `general` is "30 days for the owning group", which is the QoS cap, not the
partition cap, and not what this account gets, see below).

## QoS available to this account

`myaccounts -p` on 2026-09-02:

```
User     Def Acct      Account       QOS
jpbunnel grp_vkodibag  grp_vkodibag  debug,htc,private,public
```

| QoS | Cap | When to use |
|---|---|---|
| `public` | Follows the partition (7 d on `public`, 4 h on `htc`, 1 d on `lightwork`). **No per-job CPU/GPU cap**; only a per-*user* aggregate `cpu=7500` (surfaces as `QOSMaxCpuPerUserLimit`, never `...PerJobLimit`). | The default choice for almost everything. |
| `htc` | With `-p htc`, 4 h, no preemption. No per-job cap. | Effectively the same as `-q public` on `htc`; either works. |
| `private` | Follows partition (14 d on `general`); user cap **161280 GPU-minutes** = 16 GPU x 7 d aggregate | The only way this account reaches `general` / runs longer than 7 d. Jobs are **preemptable**, so the job must checkpoint and resume. |
| `debug` | **15 min**, can exceed partition limits (`OverPartQOS`), small TRES caps, **max 2 jobs queued** | Smoke-test a script's syntax, modules, and paths on a tiny input before a real submit. A `-q debug` left in by accident is the usual cause of a surprise `QOSMax*PerJobLimit`. |

**Not granted to `grp_vkodibag`:** `long` (14-day RC-owned, case-by-case grant,
request from RC with a job ID showing efficient use), `class` (course accounts),
and any `grp_<lab>` QoS. There is **no `grp_vkodibag` QoS anywhere in
`sacctmgr show qos`**: this lab has not bought nodes, so there is no
fairshare-free owned-node path. Verify before assuming otherwise.

## There is no 30-day owned-node path

Older project notes claimed `general -q grp_vkodibag` gives up-to-30-day walltime
with no fairshare cost if the lab owns nodes. As of 2026-09-02 that path does not
exist for this account: no `grp_vkodibag` QoS, and `general`'s partition cap is
14 days, not 30. The real ">7 days" option today is `-p general -q private
-t <=14-0` (preemptable). Check `myaccounts` before assuming it has changed.

## GPU GRES strings (live `sinfo -o %G`)

Request as `--gres=gpu:<type>:<count>` (repo style) or `-G <type>:<count>`.

| GRES string | Card | Where |
|---|---|---|
| `gpu:a100:N` | A100 80 GB | `public`, `htc`, `general` |
| `gpu:a100.40gb:N` | A100 40 GB | `htc`, `general` |
| `gpu:a100.20gb:N` | A100 MIG ~20 GB slice | `public`, `htc`, `lightwork` |
| `gpu:a30:N` | A30 24 GB | `public` (`:3`), `htc`, `general` |
| `gpu:h100:N` | H100 | `htc`, `general` (up to `:8`) |
| `gpu:l40:N` | L40 | `htc`, `general` |
| `gpu:h200:N` / `gpu:h200.35gb:N` / `gpu:h200.71gb:N` | H200 (+ MIG slices) | `htc`, `general` |
| `gpu:gh200:1` | Grace Hopper | `arm` only |
| `gpu:hl225:8` | Habana Gaudi | `gaudi` only |

Practical consequences:
- Whole-card GPU + `-q public` + fits in 4 h -> `-p htc` and you can pick from
  a100 / h100 / l40 / a30.
- Whole-card GPU + longer than 4 h -> `-p public`, and you are limited to A100 or
  A30.
- Sub-20-GB workload (linear probe, UMAP, inference) -> `gpu:a100.20gb:1`;
  faster to schedule, cheaper on fairshare.
- The `moco` pretraining scripts use `--gres=gpu:a100:2` on `-p public`; lincls
  and UMAP use `--gres=gpu:a100:1`. Keep those unless the job's shape changes.

## Preemption model

- `-q public` jobs on `htc` that are <= 4 h are **protected**: never preempted.
- `-q private` on `general` is **preemptable**, but only by a job from the
  node-owning `grp_<lab>` that cannot otherwise be placed. Checkpoint any
  `private` GPU job.
- `-q public` on `public` runs on RC-owned nodes and is not preempted (it can
  still wait a long time behind higher-priority jobs).

## Fairshare and CHE cost

Source: `~/sol-docs/fairshare.md`.

Fairshare score is `0.0`-`1.0`; it **halves for every 10,000 CHE** of recent
usage and decays with a **one-week half-life**. Higher score = shorter queue
waits. Requesting fewer resources does not move you up the line, but a smaller
job is far more likely to **backfill** (start early in a gap ahead of a bigger
higher-priority job).

Core-Hour Equivalents for a job:

```
CHE = ( cores
        + RAM_GiB / 4
        + 3  * MIG_slices
        + 20 * A30_GPUs
        + 25 * A100_GPUs
        + 40 * H100_GPUs
      ) * runtime_hours
```

Why this matters for job design:
- "300 cores for 7 days" is ~50,000 CHE in one shot and waits a very long time
  even at top fairshare. The fix is many small `-p htc` jobs.
- Submitting those small jobs in a `sbatch` loop takes a fairshare deduction
  **per submission**. A single `--array` job takes **one** deduction for all
  sub-jobs. Always prefer the array.
- This account's `RawUsage_CHE` was ~0 on 2026-09-02 (`RealFairShare` ~0.994),
  so queue waits now are dominated by cluster load, not by our history.

## Queue and array limits

- `MaxArraySize = 50000`, `MaxJobCount = 300000` (live `scontrol show config`).
- `--array=0-999%20` throttles to 20 concurrent sub-jobs.
- `debug` QoS: 2 jobs queued max, 15-minute walltime.

## Storage

Source: `~/sol-docs/resource-limits.md`, `~/sol-docs/scratch.md`.

- `$HOME` (`/home/$USER`): 100 GiB. Code, envs, software. Not for high-I/O job
  output.
- `$SCRATCH` (`/scratch/$USER`): no fixed quota (this account: ~545 GiB / ~284 k
  files on 2026-09-02). **Files not accessed for 90 days are deleted. No
  backups.** RC emails first and drops
  `scratch-dirs-{inactive,pending-removal}.csv` in `$HOME`. The `moco` repo's
  `jobs/refresh_scratch.sh` reads those and `touch`es the tree to reset the
  clock.
- `/data/grp_vkodibag`: 100 GiB shared lab storage, for anything that must
  survive the scratch purge (checkpoints worth keeping).
