---
name: sol-jobs
description: Configure, submit, monitor, and debug SLURM jobs on ASU Sol (Research Computing HPC). Use this whenever the user wants to run something on Sol from the command line: writing or adjusting an sbatch script, choosing a partition / QoS / GPU / walltime / memory, submitting a job, starting an interactive salloc/interactive session, checking why a job is PENDING, reading seff/sacct output, cancelling or resizing a queued job, or turning a python/shell command into a batch job. Trigger even when the user just says "run this on Sol", "submit the training job", "kick off pretraining", "why is my job stuck", or "how much memory should I ask for", or names a script in jobs/. They do not have to say "SLURM" or "sbatch". Also use it before hand-writing any #SBATCH header so partition/QoS/GRES strings match what Sol actually accepts.
---

# sol-jobs

## Why this exists

Everything the user runs on ASU Sol goes through SLURM, and the knowledge needed
to configure a job correctly is scattered: the 130-page docs mirror at
`~/sol-docs/`, a repo's `CLAUDE.md`, and whatever `sbatch` scripts already live in
that repo. A session without that context in front of it re-derives partition and
QoS choices from memory and gets them wrong (invents a `gpu` partition, asks for a
`long` QoS this account does not have, pads walltime into a maintenance
reservation so the job never starts).

This skill carries the **durable decision procedure**: which partition and QoS fit
a given job shape, how to size `-c` / `--mem` / `-t`, how to read a `PENDING`
reason, how to right-size the next run from `seff`. The **volatile numbers**
(exact partition walltime caps, the live GPU inventory, per-QoS limits) live in
`references/partitions-qos.md` with citations back to `~/sol-docs/<file>`, because
those change and freezing them here would rot silently. When a specific number
decides the outcome and the mirror looks stale (check the date in
`~/sol-docs/INDEX.md`), offer to run the `sol-docs-refresh` skill first.

**A repo's own files win when they exist.** If the working directory has a
`jobs/` (or `examples/`) dir of committed `sbatch` scripts, a `config.sh`, and a
SLURM section in `CLAUDE.md`, those are the source of truth for paths and
conventions. This skill defers to them and only supplies what no repo file does.

## On-cluster facts (verified 2026-09-02, account `grp_vkodibag`)

These were checked directly on Sol and are safe to rely on until the next
`sol-docs-refresh`. Re-verify with `myaccounts -p` and `sinfo` if a decision
hinges on them.

- **Account:** `grp_vkodibag` (the only account, and the default).
- **QoS granted to this user:** `public`, `htc`, `private`, `debug`. There is
  **no `long` QoS and no `grp_vkodibag` QoS** on this account today. Any doc or
  note that says "use `-q grp_vkodibag` for 30-day walltime" is not currently
  actionable here: confirm with `myaccounts -p` before believing it.
- **Partition walltime caps (live `sinfo`):** `public` 7 days · `htc` 4 hours
  (this is the cluster default partition) · `general` 14 days · `highmem` 7 days ·
  `lightwork` 24 hours · `arm` / `fpga` / `gaudi` 7 days.
- **GPU GRES strings (live `sinfo -o %G`):** `gpu:a100:N` (80 GB) ·
  `gpu:a100.20gb:N` (MIG slice) · `gpu:a100.40gb:N` · `gpu:a30:N` · `gpu:h100:N` ·
  `gpu:h200:N` · `gpu:l40:N` · `gpu:gh200:1` (arm) · `gpu:hl225:8` (gaudi).
  **`public` carries only `a100`, `a30`, and `a100.20gb`.** H100 / L40 / H200 are
  on `general` and `htc`.
- `MaxArraySize = 50000`, `MaxJobCount = 300000`. `debug` QoS: 15 min walltime,
  1 job running / 2 queued.
- **Sol has no `gpu` partition** and never uses a bare `--gres=gpu:1`. GPUs are
  always `--gres=gpu:<type>:<count>` or `-G <type>:<count>`. If you see a doc
  example with a `gpu` partition (e.g. in `~/sol-docs/tutorials-slurm-script-generator.md`),
  it is a generic teaching example, not Sol.

## Verify live when a specific number decides the answer

The facts above are a snapshot. Sol changes what QoS an account holds, what a
partition caps walltime at, and what GPUs sit where. When the answer turns on one
of those, run the check, do not trust this file or `~/sol-docs/`:

| Question | Command |
|---|---|
| What QoS / accounts does the user actually have? | `myaccounts -p` |
| Partition walltime caps and up/down state | `sinfo -o "%P %l %a %D"` |
| Which GPU types live in which partition | `sinfo -o "%P %G"` |
| The real per-job / per-user ceiling on a QoS | `sacctmgr show qos format=Name,MaxWall,MaxTRESPerJob%40,MaxTRESPU%40` |
| What a specific job actually requested | `scontrol show job <id>` |
| Would this header even be accepted | `sbatch --test-only <script>` |

If a check contradicts this skill or a repo's `CLAUDE.md`, the live check wins.
Say so, and (for `~/sol-docs/` drift) offer to run the `sol-docs-refresh` skill.

## Step 0: read the room

Figure out which of three situations you are in before writing anything.

1. **Inside a repo with committed job scripts** (a `jobs/` dir, a `config.sh`, a
   SLURM section in `CLAUDE.md`). The committed script is the source of truth.
   Adjust or add a script in `jobs/`, `source` the repo's `config.sh` for paths,
   and follow that repo's naming and `--export=` override patterns. Submit with
   `sbatch jobs/<name>.sh`. **Never paste a job body into the OnDemand web UI**:
   the committed script is what keeps runs reproducible. Match the style already
   in the neighboring scripts (flag form, log-name pattern, `set -e`); do not
   "modernize" a working script.

2. **A standalone script or a bare command to batch.** Scaffold from
   `assets/sbatch-template.sh` (single job) or `assets/sbatch-array-template.sh`
   (one task per input over many inputs) and fill in resources with Step 1.

3. **No new job at all** (monitor, cancel, resize, or explain a running/queued
   job). Go straight to `references/monitoring-debugging.md`.

Then in Step 1, watch for three shapes that change the partition/QoS answer: a
**fan-out over many inputs** (use one array job, not a loop), a run **longer than
7 days** (only `-p general -q private`, preemptable, so it must checkpoint), and a
**GPU type that `public` does not have** (H100 / L40 / H200 mean `-p htc` for
&le; 4 h or `-p general -q private`).

## Step 1: size the job

This is a decision procedure, not a lookup table. The exact caps are in
`references/partitions-qos.md`.

### Partition and QoS

The rows below assume the QoS grant verified 2026-09-02 (`debug, htc, private,
public`). If a longer walltime or a fairshare-free path matters, run
`myaccounts -p` first: a `long` or `grp_<lab>` QoS would change the answer.

| Job shape | Partition / QoS | Why |
|---|---|---|
| Fits in 4 hours | `-p htc -q public` | `htc` has no preemption and the shortest queue wait; 4 h is the hard cap. Widest GPU selection. |
| 4 hours to 7 days | `-p public -q public` | The general-purpose GPU+CPU partition. Only A100 / A30 GPUs here. |
| Env build, file copy, data move, SSH tunnel (light, ≤ 24 h) | `-p lightwork -q public` | Keeps light work off the compute partitions. Max 8 cores. |
| Smoke-test syntax and paths first | add `-q debug -t 15` | 15-minute cap, near-instant scheduling; catches a bad module or path before you burn a real slot. |
| Longer than 7 days, **or** needs H100 / L40 / H200 | `-p general -q private` | Privately-owned nodes, **preemptable** (cancelled if the owning lab needs the node), up to 14 days. The job **must** checkpoint and be resumable. `-q long` (RC-owned, 14 d) and a `grp_<lab>` QoS are **not** on this account; `myaccounts -p` to confirm before assuming either. |

If a repo script already picks a partition/QoS for a given job type, keep it
unless the user is explicitly changing the job's shape.

### GPU

- Use the **same GRES form the surrounding scripts use.** The `moco` repo writes
  `#SBATCH --gres=gpu:a100:2`; `-G a100:2` is equivalent. Do not switch forms in
  an existing script.
- **A100 (80 GB) is the only full GPU in `public`.** For H100 / L40 you must move
  to `-p general` or `-p htc` (see the GRES table in `references/partitions-qos.md`).
- `gpu:a100.20gb` is a ~20 GB MIG slice: right for lincls / UMAP / inference that
  does not need a whole card, and it schedules faster and costs less fairshare.
- A GPU job showing low CPU utilization in `seff` is normal; `seff` reports no GPU
  stats at all.

### CPU, memory, walltime

- Prefer `--mem=<total>` over `--mem-per-cpu` unless the code is OpenMP and scales
  memory per thread. `--mem=0` requests the whole node's memory.
- Unspecified memory defaults to roughly 2 GB per core; be explicit for anything
  data-heavy.
- `-N > 1` only helps genuine MPI. A single Python process (even multi-GPU via
  `mp.spawn`) is `-N 1`.
- Give walltime real headroom over the expected runtime, but do **not** pad it so
  far that it collides with a maintenance reservation: the job then sits with
  `ReqNodeNotAvail, Reserved_for_maintenance` until you lower `-t`.
- After the first run, right-size the next one from `seff <jobid>` (peak memory,
  CPU-time efficiency, elapsed vs. requested).

### Arrays (fan-out over many inputs)

When the task is "do the same thing to N files / series / seeds", it is one array
job, not a loop:

- **One `--array=0-<N-1>` job, never a shell loop of `sbatch` calls.** An array
  takes a single fairshare deduction for the whole set; a loop of N `sbatch`
  calls takes N, and later submissions wait behind their own earlier ones.
- **Per-task index** is `$SLURM_ARRAY_TASK_ID`. Map it to an input via a manifest
  file (`sed -n "$((SLURM_ARRAY_TASK_ID+1))p" manifest.txt`) or pass it straight
  to the script (`--index "$SLURM_ARRAY_TASK_ID"`).
- **`#SBATCH` resources are per task**, not for the whole array. Size one task.
- **Put each task where a single task fits**: an 8-minute task belongs on
  `-p htc -q public`, not `-p public`.
- **Throttle** with `%K`: `--array=0-3999%50` runs at most 50 at once, which keeps
  one array from monopolizing the partition and is kinder to shared scratch I/O.
- **Logs** use `%A_%a` (`%A` array job id, `%a` task index):
  `-o logs/%x.%A_%a.out`.
- `MaxArraySize` is 50000, so the highest index is 49999. For more tasks than
  that, batch them (each array task processes a slice) or submit multiple arrays.
- Local pattern: the `moco` repo's `jobs/prep_array.sh` runs a discovery job that
  writes a manifest, then self-submits the array sized to the line count.
  `assets/sbatch-array-template.sh` is a minimal standalone version. More
  variants: `~/sol-docs/slurm-job-array-examples.md`.

## Step 2: write the script

Structural contract for a Sol batch script (see `assets/sbatch-template.sh`, or
`assets/sbatch-array-template.sh` for a fan-out array):

- `#!/bin/bash` on the **first** line, then the `#SBATCH` block with no code above
  it.
- **Load your environment inside the script body, every time.** Sol's own
  templates set `#SBATCH --export=NONE`, which means the job starts with none of
  your login shell's environment: nothing is loaded by default. Even scripts that
  do not set `--export=NONE` (the `moco` repo's do not) still `module load` in the
  body so the job does not depend on what happened to be loaded at submit time.
- Conda: `module load mamba/latest` then `source activate <env>`. **Never**
  `conda activate` / `mamba activate` in a batch script: they need shell hooks
  that a non-interactive job does not have.
- `cd` to the project directory explicitly (`cd "$PROJECT_DIR"`); do not assume
  the job starts there.
- Log naming: the `moco` repo uses `-o %x.%j.out` / `-e %x.%j.err` (`%x` job
  name, `%j` job id; `%A_%a` for array master/index). Match the local pattern.
- `--mail-type=ALL --mail-user=%u@asu.edu` if the neighboring scripts have it.
- If submission fails with "DOS line breaks" / "cannot execute", run
  `dos2unix <script>` (the file has CRLF line endings).

## Step 3: validate before spending a real slot

- `sbatch --test-only jobs/<name>.sh` reports whether it would be accepted and an
  estimated start time, without queuing anything. Use it whenever you have
  changed a `#SBATCH` header.
- For a brand-new script, offer a `-q debug -t 15` run against a tiny input first;
  it catches a missing module, a bad path, or an import error in a minute instead
  of after a multi-hour queue wait.

## Step 4: submit and report back

- **Do not submit without the user's explicit go-ahead.** In an eval or test
  context, stop at the finished script plus the exact `sbatch` command, or at
  `sbatch --test-only`.
- Submit form: `sbatch jobs/<name>.sh`, or with overrides
  `sbatch --export=CKPT=checkpoint_0149,CKPT_RUN=acrin jobs/run_lincls.sh` (the
  `moco` repo's env-var pattern; each script documents its own overrides in a
  comment near the top).
- After submitting, report: the job id, the log file paths it will write, and the
  one-liner to watch it (`myjobs`, or `sq -u $USER`).

## Step 5: monitor, debug, manage

All of this is in `references/monitoring-debugging.md`: the command quick
reference (`myjobs`, `sq`, `thisjob`, `seff`, `mysacct`, `scontrol`, `scancel`),
how to inspect a *running* job on its node, the `PENDING`-reason decoder, the
submission-error decoder, and the `seff`-driven right-sizing loop.

## Interactive sessions

- `interactive` is a Sol wrapper for `salloc -c 1 -p htc -q public -t 0-4`: a
  quick 4-hour CPU shell.
- `aux-interactive` (or `interactive -p lightwork`) for light work.
- For a custom interactive allocation:
  `salloc -N 1 -p public -q public -c 8 -t 0-6:00:00 --gres=gpu:a100:1`.
- **Do not run compute on the login node.** It is an explicit policy violation
  (`~/sol-docs/policies.md`); the scheduler is the only correct path.

## Doc traps (recorded so a future session does not rediscover them)

- `~/sol-docs/tutorials-slurm-script-generator.md` embeds a `cluster-constraints.md`
  block describing a `gpu` partition and `--gres=gpu:1`. It is a generic teaching
  example. Sol has no `gpu` partition.
- `~/sol-docs/cxfel-nodes.md` has a `--gres=gpu2` typo. `~/sol-docs/python-example.md`
  has a duplicated `-p` in an `interactive` line.
- The docs mirror was last refreshed per the date in `~/sol-docs/INDEX.md`.
  Partitions, QoS grants, GPU inventory, and queue limits are the parts that
  drift. Offer `sol-docs-refresh` when one of those numbers is load-bearing.
