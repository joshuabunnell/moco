<!-- source: https://docs.rc.asu.edu/evaluating-job-performance -->
# Evaluating Job Performance | ASU RC Docs

## HPC vs HTC
HPC = large monolithic workloads, substantial resources. HTC = thousands of independent concurrent jobs, minimal resources each. Choice depends on whether your computation can be split.

## Code runs faster on a workstation than the supercomputer
Common cause: job isn't parallelized (many Python/R scripts don't use multiple cores automatically — needs explicit OpenMP/multi-threading/multi-processing). Also: individual Sol/Phoenix cores run ~2GHz vs 3-4+GHz on a workstation — supercomputers win via core *count*, not per-core speed.

## Serial vs parallel
Some software parallelizes automatically (MATLAB), some offers options (SAS), some needs manual recoding (Python, R). Dependent/timestep-based computations (agent-based models, genome evolution sims) have inherent serial bottlenecks with diminishing returns from extra cores.

## `seff` for efficiency
```
seff <jobID>
```
Shows CPU/memory efficiency relative to what was requested. Example inefficient job: 12.55% CPU efficiency, 16.85% memory efficiency — over-requested resources hurt fairshare. Note: `seff` excludes GPU utilization, so GPU jobs may show misleadingly low CPU efficiency.

## Computational Research Accelerator
RC team offering code optimization/parallelization/hardware consultation — short sessions or long-term embedded projects, requested via the RTO Request Help page.
