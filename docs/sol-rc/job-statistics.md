<!-- source: https://docs.rc.asu.edu/job-statistics -->
# Job Statistics | ASU RC Docs

## Running jobs (seff/sacct don't work yet — job must be finished)
Find the node via `myjobs` (NODELIST column), then SSH directly to it (only possible while your job is running there):
```
ssh <node>
htop     # CPU/memory of your job's processes
nvtop    # GPU usage on GPU nodes
```

## Completed jobs
- `seff <jobID>` — CPU/memory efficiency vs. requested resources (see evaluating-job-performance.md).
- `sacct --jobs=<id>[,<id>...]` or `sacct --user=<username>` — accounting DB query. Format variables: account, allocTRES, avecpu, cputime, elapsed, state, jobid, jobname, maxdiskread, maxdiskwrite, maxrss, ncpus, nnodes, ntasks, priority, qos, user.
- `mysacct` — shortcut alias equivalent to `sacct --user=$USER --format=jobid,avecpu,maxrss,cputime,allocTRES%42,state`; accepts `--starttime=YYYY-MM-DD`, `--endtime=YYYY-MM-DD`, `--long`.
- A `+` at the end of a truncated field means widen it: e.g. `allocTRES%42`.
