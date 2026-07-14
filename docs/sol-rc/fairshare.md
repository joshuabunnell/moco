<!-- source: https://docs.rc.asu.edu/fairshare -->
# Slurm Fairshare Score | ASU RC Docs

Jobs are prioritized by a Fairshare score based on recent usage (Core-Hour Equivalents, CHE).

```
FairShare = 2^(-current_CHE_usage / 10,000)
```
Halves every 10,000 CHE used; decays with a one-week half-life (20,000 CHE today ≈ 10,000 CHE in a week, ≈1.07 CHE after 26 weeks).

```
CHE = (cores + RAM_GiB/4 + 3*MIG_slices + 20*A30_GPUs + 25*A100_GPUs + 40*H100_GPUs) * runtime_hours
```
E.g. 1 core + 4GiB RAM + 1 A100 for 4 hours ≈ 108 CHE.

All jobs eventually run; higher recent usage (lower score) means longer waits, not exclusion. Requesting more/fewer resources doesn't change queue position, but backfilling can let a job run ahead of schedule if it doesn't block higher-priority jobs.

## Checking score
```
myfairshare   # or: mybalance
myaccounts    # -p flag if output is truncated
```

## Practical workarounds for a low score
- Break one giant job (e.g. 300 CPUs × 7 days) into many small `htc`-partition jobs (1 CPU × 1h) — much less fairshare penalty per unit of work.
- Don't submit thousands of small jobs one-by-one via a script (each submission dings fairshare) — use a **job array** instead, which takes a single one-time deduction for the whole array.

## Submitting to a specific Slurm account
```
#SBATCH -A grp_mylab
#SBATCH -A class_asu101spring2025
salloc -A grp_mylab
```
Default account (used if `-A` omitted) is the first account you joined.
