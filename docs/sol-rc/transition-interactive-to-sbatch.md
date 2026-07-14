<!-- source: https://docs.rc.asu.edu/transition-interactive-to-sbatch -->
# Transitioning Interactive Jobs to Batch Jobs | ASU RC Docs

Batch jobs let the scheduler queue/prioritize/allocate resources efficiently without continuous user participation.

## Jupyter workflows
1. Convert notebook to script: via Jupyter UI (`File -> Save and Export Notebook As... -> Executable script`) or terminal (`aux-interactive` to get a lightwork node, `module load jupyter/latest`, `jupyter nbconvert --to script example_notebook.ipynb`).
2. Adjust script for non-interactive execution (remove widgets/plots, explicit paths, save plots to files).
3. Create sbatch script (loads `mamba/latest`, `source activate myEnv`, `python3 myscript.py`).
4. Submit: `sbatch submit_job.sbatch`; monitor `squeue -u $USER`; check `slurm.<jobid>.out/.err`. Test first with `interactive -q debug -t 15`.

## R workflows
Same pattern — convert `.Rmd` via `knitr::purl()` (using `aux-interactive` + `module load r-4.4.0-gcc-12.1.0`), adjust for non-interactive execution, then sbatch with `module load r-4.4.0-gcc-12.1.0` + `Rscript example.R`.

Note: `aux-interactive` is used here as the command to request a lightwork compute node (appears to be an alias related to `interactive -p lightwork`).
