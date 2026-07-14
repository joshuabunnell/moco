<!-- source: https://docs.rc.asu.edu/partitions-and-qos -->
# Partitions and QoS | ASU RC Docs

Partitions are separate queues for jobs to be allocated to different subsets of the supercomputer's hardware. Quality-of-Service ("QoS") refers to the length of time jobs may run, their preemption status, and any limitations on resources.

As a general rule the `public` partition will satisfy most users' needs (CPU and GPU nodes, up to 7 day long jobs).

## Partitions

### `public`
All Research Computing-owned nodes. Wall time limit 7 days. QoS options: `public`, `long`, `debug`, `class`.
```
#SBATCH -p public
salloc -p public
```

### `general`
All privately-owned nodes. 30-day wall time limit for users within the owning group (`grp_` prefixed QoS). Non-privileged users can use it with the `private` QoS but risk preemption.
```
#SBATCH -p general -q grp_<mygrpname> -t 30-0
#SBATCH -p general -q private -t 7-0
```

### `htc` (High-Throughput Computing)
Jobs completing within 4-hour walltime. Includes RC-owned and privately-owned nodes; runs without preemption risk when submitted with `public` QoS.
```
#SBATCH -p htc -q public -t 0-4
```

### `highmem`
For jobs needing more than 512GB. Up to 2TB. Capped at 48 hours by default; longer available via special QoS by request.
```
#SBATCH -p highmem -q public -t 0-48
```

### `lightwork`
**For jobs requiring relatively less computing power and/or that may stay idle for large amounts of time — explicitly listed examples: creating mamba environments, compiling software, VSCode tunnels, or bulk file operations.**
```
#SBATCH -p lightwork -q public -t 0-24
```
Max job time: **24 hours**. Max CPU cores per node: **8**.
> Jobs using full cores >99% for a continued duration, or requesting excessive resources, are **subject to cancellation**. Repeated misuse results in ineligibility from `lightwork`.

### `fpga` (Sol only)
Mix of Intel/AMD FPGA and Vector Engine accelerators.

### `arm` (Sol only)
ARM `aarch64` workloads; all Sol arm hosts are also Grace Hopper GPU nodes. Software not automatically ARM-compatible — check first.

## QOS

### `public`
Default/preferred QoS for public RC resources.

### `debug`
For testing sbatch syntax/setup with quick turnaround and short expected start times. Works with `general` and `htc` partitions for walltimes up to **15 minutes**.
```
#SBATCH -p public -q debug -t 15
```

### `private`
Willingness to be preempted in exchange for using privately-owned resources beyond the protected 4-hour `htc` window. User-level limit: **161280 GPU running minutes** (≈16 GPUs for 7 days).

### `grp_labname`
Given to labs that purchased hardware. No fairshare impact for group members. Preempts `private` jobs but not protected `htc`/`public` jobs ≤4h.

### `long`
Extends RC-owned hardware runtime from 7 to **14 days**. SBATCH only, not interactive. Granted case-by-case — be ready to justify with a job ID showing effective use of existing allocations.

### `class`
For academic-course accounts.
- Max **32 CPU cores, 320GB memory, 4 GPUs** per job
- Max wall time **24 hours**
- Max **2 concurrent jobs**, **10 queued jobs** per user
- Max **960 GPU running minutes** per user
Use `myfairshare` / `myaccounts` to see which Slurm accounts you have; specify with `-A` if you have both class and research accounts.

## Community Use of Privately-Owned Nodes
Three contexts: `grp_labname` (owner priority, no fairshare impact), `htc` partition with `public` QoS (protected 4h runtime even on private nodes, can backfill onto idle lab nodes without delaying higher-priority work), and `private` partition/QoS (preemptable). `grp_` jobs are always considered higher-priority regardless of fairshare.
