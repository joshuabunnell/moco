<!-- source: https://docs.rc.asu.edu/jobstates -->
# Understanding Job States | ASU RC Docs

View job state with `myjobs`.

## States
- **RUNNING**: resources allocated, job executing.
- **PENDING**: accepted by scheduler, valid request, waiting for resources. Reason codes (via `scontrol show job <id>`):
  - `ReqNodeNotAvail, May be reserved for other job` — assigned to a specific resource, next in line for it.
  - `ReqNodeNotAvail,_Reserved_for_maintenance` — job's requested walltime wouldn't finish before scheduled maintenance; reduce walltime if possible.
  - `Resources` — highest-priority job in its partition, waiting on next available matching resource; can occasionally be displaced by an even-higher-priority private-QOS submission.
  - `Priority` — waiting in the fairshare-ordered queue.
  - `QOSMaxCpuPerJobLimit` — valid request, but running now would exceed aggregate QoS limits (mainly class accounts).
  - `None` — no blocking reason; about to transition to RUNNING.
  - `Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions` — required nodes aren't accepting jobs (often a failed scarce resource like an uncommon GPU/FPGA, or a specific-node request `-w`).
- **REQUEUEHOLD**: e.g. `launch_failure_limit_exceeded_requeued_held` — valid resources but something prevented execution (often node-level issue); needs RC admin re-entry into queue.

## Backfilling
Jobs requesting comparatively small resources can run ahead of schedule ("backfill") if doing so doesn't delay any already-scheduled higher-priority job — e.g. 3 of 4 GPUs idle for 2 hours before a 4-GPU job starts can be backfilled by a job needing only 1-3 GPUs for ≤2 hours.
