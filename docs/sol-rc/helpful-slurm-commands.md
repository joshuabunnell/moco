<!-- source: https://docs.rc.asu.edu/helpful-slurm-commands -->
# Helpful Slurm Commands

| Command | Purpose |
|---------|---------|
| `myjobs` | Display your current jobs in the queue |
| `sq` | Display queue info with filtering (wrapper for squeue) |
| `thisjob <jobID>` | View info about a job, including estimated start time |
| `seff <jobID>` | View Slurm efficiency for a completed job (CPU/memory used) |
| `mkjupy <envName>` | Turn a mamba environment into a Jupyter kernel |
| `myfairshare` | View your real FairShare score |
| `myquota` | View your $SCRATCH quota |
| `showsimg` | List available apptainer sif files |
| `sbatch` | Submit batch script |
| `salloc` | Execute command on compute node (interactive) |
| `sinfo` | View info about the cluster |
| `ns` | Command-line cluster status tool (by Prof. Jay Oswald) |

Note: `myquota` reports the $SCRATCH quota specifically, not home. (Confirms: the 605GB/100TB figure seen in our earlier probe was scratch usage, not home — home is the separate 100GiB quota reported by `df -h $HOME`.)
