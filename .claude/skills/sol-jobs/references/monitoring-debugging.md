# Monitoring, debugging, and managing jobs

Commands verified present on Sol 2026-09-02. Sources: `~/sol-docs/helpful-slurm-commands.md`,
`~/sol-docs/jobstates.md`, `~/sol-docs/slurm-sbatch.md`,
`~/sol-docs/evaluating-job-performance.md`.

## Contents

- [Command quick reference](#command-quick-reference)
- [Inspecting a running job](#inspecting-a-running-job)
- [PENDING reason decoder](#pending-reason-decoder)
- [Submission and runtime error decoder](#submission-and-runtime-error-decoder)
- [Resizing or fixing a queued job](#resizing-or-fixing-a-queued-job)
- [Right-sizing the next run from seff](#right-sizing-the-next-run-from-seff)

## Command quick reference

| Command | Standard / Sol wrapper | Use |
|---|---|---|
| `myjobs` | wrapper | Your queued + running jobs, with `NODELIST(REASON)`. First thing to run. |
| `sq -u $USER` | wrapper (`squeue`) | Same idea, squeue formatting; drop `-u` to see the whole queue. |
| `thisjob <id>` | wrapper | Job detail including **estimated start time**. |
| `seff <id>` | wrapper | Efficiency of a **completed** job: CPU%, peak memory vs requested, elapsed vs requested. No GPU stats. |
| `mysacct [--starttime=YYYY-MM-DD] [--long]` | wrapper (`sacct`) | History: past jobs, states, exit codes, per-step breakdown. (`myacct` does not exist.) |
| `sacct -j <id> -o JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,ReqTRES%40` | standard | Raw fields when you need exact numbers. |
| `scontrol show job <id>` | standard | Full record of a pending/running job: `Reason=`, `StartTime=`, requested TRES, `NodeList`. |
| `scancel <id>` | standard | Cancel. `scancel -u $USER` cancels all yours; `scancel <id>_<idx>` one array task. |
| `myquota` | wrapper (`beegfs-ctl --getquota`) | `$SCRATCH` usage and file count. |
| `myfairshare` / `mybalance` | wrapper | Fairshare score and `RawUsage_CHE`. |
| `ns` | wrapper | CLI cluster status (node/partition load) to gauge queue pressure. |

## Inspecting a running job

`seff` only works after a job finishes. For a job running **now**:

1. `myjobs` -> read the node name from `NODELIST`.
2. `ssh <node>` (allowed only while you have a job on that node).
3. On the node: `htop` (per-core CPU; one fully-used core shows as 100%),
   `nvidia-smi` or `nvtop` for GPU utilization and memory, `free -g` for memory.
4. Job logs stream to the `-o` / `-e` files as it runs: `tail -f <name>.<jobid>.out`.

## PENDING reason decoder

From `myjobs` the reason is the parenthetical in `NODELIST(REASON)`; from
`scontrol show job <id>` it is the `Reason=` field.

| Reason | Meaning | Action |
|---|---|---|
| `Priority` | Waiting behind higher-priority jobs (fairshare line). | Normal. A smaller job may backfill past it. If urgent, cut resources or move to `-p htc` (<= 4 h). |
| `Resources` | You are #1 for the partition; waiting for hardware to free up. No `StartTime` yet. | Wait. Nothing to fix. |
| `ReqNodeNotAvail, May be reserved for other job` | You are the next job on a specific named node; start is imminent. `scontrol show job` shows a `StartTime`. | Wait. |
| `ReqNodeNotAvail,_Reserved_for_maintenance` | Requested walltime does not fit before a scheduled maintenance window. | **Lower `-t`** so the job fits before maintenance, or wait until maintenance ends. |
| `QOSMax*PerJobLimit` (e.g. `QOSMaxCpuPerJobLimit`) | This one job's request exceeds a **per-job** ceiling on the QoS it was submitted under. It will sit `PENDING` forever; waiting does not clear it. | See "`QOSMax*` limits" below: identify the binding QoS and ceiling, then shrink the request or move to a QoS without that cap. |
| `QOSMax*PerUserLimit` (e.g. `QOSMaxCpuPerUserLimit`) | Your jobs **in aggregate** exceed a per-user cap (on `-q public` the cap is `cpu=7500` across all your running + pending jobs). | Nothing is broken. Let running jobs drain, or cancel/shrink the least important; the held job starts automatically as headroom frees. |
| `Nodes required for job are DOWN, DRAINED or reserved...` | The scarce resource you asked for (rare GPU/FPGA, or a specific `-w` node) is unavailable. | Drop the `-w`, or pick a GPU type that actually exists in your partition (see `partitions-qos.md`). |
| `launch_failure_limit_exceeded_requeued_held` (state `REQUEUEHOLD`) | Job repeatedly failed to launch, usually a node-level environment problem. | Open an RC ticket; admins can release it. |
| `None` | Transient, the job is starting right now. | Wait a moment. |

### `QOSMax*` limits: find the binding ceiling before acting

Do not guess which resource is over. Read the actual numbers:

```bash
scontrol show job <id> | grep -E 'QOS=|Partition=|ReqTRES=|NumCPUs=|TimeLimit='
sacctmgr show qos format=Name,MaxWall,MaxTRESPerJob%40,MaxTRESPU%40 | grep -E 'Name|<the QOS from above>'
```

Then compare the job's `ReqTRES` against that QoS's `MaxTRESPerJob` (per-job) or
`MaxTRESPU` (per-user). Where they collide is the cause.

On the 2026-09-02 grant, only some QoS carry a **per-job** ceiling:

- `debug`: 15-minute walltime and small TRES caps. A leftover `-q debug` from a
  smoke test is the single most common cause of `QOSMaxCpuPerJobLimit`.
- `class` (course accounts): `cpu=32, gpu=4, mem=320G`, 24 h. Only reachable if
  the job used a `class_*` account (`-A`), which is not this account's default.
- `public`, `htc`, `private`: **no per-job CPU cap.** `-q public` has only a
  per-*user* `cpu=7500` aggregate, which surfaces as `QOSMaxCpuPerUserLimit`, not
  `...PerJobLimit`. So `QOSMaxCpuPerJobLimit` on a job you think is `-q public`
  means it is really on `debug` / `class` / a typo'd QoS. Confirm with
  `scontrol show job <id>`'s `QOS=` field.

Fix: put the job back on the right QoS (usually `-q public`) and resubmit, or
shrink the request to fit the QoS it is on.

## Submission and runtime error decoder

| Symptom | Cause | Fix |
|---|---|---|
| `sbatch: error: Batch script contains DOS line breaks` / `cannot execute` | File has CRLF endings. | `dos2unix <script>`. |
| `sbatch: error: This does not look like a batch script` | Missing or misplaced `#!/bin/bash` on line 1, or `#SBATCH` lines after real code. | Shebang first line, `#SBATCH` block immediately after, nothing executable above it. |
| `Invalid qos specification` / `Invalid partition name specified` | QoS or partition not granted / not real. | `myaccounts -p` for QoS (`debug,htc,private,public` here); `sinfo` for partitions. |
| `Invalid feature specification` | Bad `--constraint` / GRES string. | Match GRES exactly to `sinfo -o "%P %G"` (e.g. `gpu:a100:2`, never bare `gpu:2`). |
| Job dies instantly, log says `ModuleNotFoundError` / `command not found` | Environment not loaded because `--export=NONE` (or just not inherited). | `module load mamba/latest` + `source activate <env>` **in the script body**, not `conda activate`. |
| Exit code 125 | Out of memory (OOM killer). | Raise `--mem`; confirm with `seff` (`MaxRSS` near `ReqMem`). |
| Exit code 127 | Executable not found. | Wrong `cd`, wrong path, or env not active. |
| Exit code `128 + n` | Killed by signal `n` (137 = SIGKILL/OOM or walltime, 139 = segfault). | 137 with `State=TIMEOUT` means raise `-t`; 137 otherwise usually OOM. |
| `State=TIMEOUT` in `seff`/`mysacct` | Hit the walltime wall. | Raise `-t`, or checkpoint and resume. |

## Resizing or fixing a queued job

While a job is still `PENDING`, **shrinking** it in place avoids losing queue
position and taking another fairshare hit:

```bash
scontrol update job <id> TimeLimit=2-00:00:00   # lower -t to clear a maintenance-reservation hold
scontrol update job <id> NumCPUs=8              # shrink core request
scontrol update job <id> MinMemoryNode=32G
```

What in-place editing can and cannot do:

- **Reliable:** lowering `TimeLimit`, `NumCPUs`, `MinMemoryNode`. You can never
  raise a value above what the QoS/partition allows.
- **Often refused for unprivileged users, and site-dependent:** changing
  `Partition=` or `QOS=`. Try it; if `scontrol` returns
  `Access/permission denied` or the reason does not clear, cancel and resubmit
  with a corrected `#SBATCH` header instead.
- **Not editable:** GRES / GPU count on most Slurm builds. Changing the GPU
  request means cancel + resubmit.

For a `QOSMax*PerJobLimit` caused by the wrong QoS, cancel + resubmit is usually
the only path, and it is fine: a job that can never start is not a queue position
worth keeping.

## Right-sizing the next run from seff

After a job completes, `seff <id>` reports CPU efficiency, memory used vs
requested, and elapsed vs requested walltime. Pull the next submit toward actual
usage:

- Memory used was 35 % of requested -> cut `--mem` to ~1.3x the peak.
- CPU efficiency low on a **CPU** job -> the code is not parallel; fewer cores
  will not slow it and costs less fairshare. (Low CPU % on a **GPU** job is
  expected; `seff` cannot see the GPU.)
- Elapsed was 40 % of `-t` -> tighten walltime (keep headroom, but not so much
  that it collides with maintenance).

Every trimmed resource lowers the job's CHE cost and improves its odds of
backfilling early. See `partitions-qos.md` for the CHE formula.
