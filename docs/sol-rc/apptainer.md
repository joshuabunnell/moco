<!-- source: https://docs.rc.asu.edu/apptainer -->
# Apptainer Usage | ASU RC Docs

Apptainer (formerly Singularity) container binaries are available on all **compute** nodes (not directly usable on login nodes — you start it from a login node but it launches on a compute node via `interactive`).

## Usage
```
showsimg   # list available .sif images
SIMG=ubuntu-22.04 classic-interactive
SIMG=fmriprep.sif classic-interactive -c 4
SIMG=tensorflow_latest-gpu.sif classic-interactive --gres=gpu:a100:4
```
Once allocated, you're inside the container (`Apptainer>` prompt) but still your own ASURITE user — the container can read/write `$HOME` and scratch; only files saved there persist after leaving the container.
