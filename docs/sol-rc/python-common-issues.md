<!-- source: https://docs.rc.asu.edu/python-common-issues -->
# Python Common Issues | ASU RC Docs

**Issue 1 - Installation Location**: "Do not install packages on the login nodes or inside a Jupyter Notebook. It has to be done in a terminal/shell" — use `interactive` (with appropriate parameters) instead.

**Issue 2 - Base Environment Usage**: Don't install packages in the base environment. If `(base)` appears in your prompt, run `source deactivate` first.

**Issue 3 - Conda Configuration Contamination**: Improper `conda` command use injects code into `~/.bashrc`, causing problems. Fix: run `remove_conda_from_bashrc`, then `source ~/.bashrc`.

**Issue 4 - Pip Cache Folder Problems**: Incorrect pip usage creates hidden cache folders that degrade Python functionality. Fix: remove python3 directories under `~/.local/lib`, then `mamba clean --all`.

**Issue 5 - Safe Environment Removal**: In an interactive session, `module load mamba/latest`, `mamba remove -n ENV_NAME --all`, clean cache, then manually remove the directory from `~/.conda/envs`.
