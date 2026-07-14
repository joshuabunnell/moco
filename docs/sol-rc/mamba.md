<!-- source: https://docs.rc.asu.edu/mamba -->
# Python Envs and Mamba | ASU RC Docs

Supercomputer uses `Mamba` (drop-in conda/pip replacement) for Python environments.

> Use `mamba` via the module system (`ml mamba`) rather than installing your own — multiple python managers conflict, and admins can only support the supercomputer-installed mamba.

## Why environments / why Mamba
Fresh shell `python`/`python3` points to system Python (`/usr/bin`), fixed version, minimal libs. Mamba has an advanced dependency solver (Python + non-Python libs), runs natively without a Rust toolchain (unlike pixi), and solves faster than conda. RC discourages `pip` except when necessary — it can break an environment's packages.

## Using Mamba
```
module load mamba/latest   # or: ml mamba
```

### Finding environments
Public/admin envs live under `/packages/envs` (read-only, version-fixed). User envs default to `~/.conda/envs`.
```
mamba info --envs
```

### Loading environments
```
source activate <name-or-path>
```
Only affects the current shell — multiple envs can run in different shells/jobs simultaneously without interference.

> Only use `source activate` — avoid `conda activate`/`mamba activate` even if a tutorial suggests it; `source activate` is proven most compatible here.

## Creating Environments
```
interactive
module load mamba/latest
mamba create -n <environment_name> -c conda-forge -c <channel> <packages>
# or, for a shared/group location:
mamba create -p /data/example_group/ENV_NAME -c conda-forge [-c <channel>] [packages]
```
- `-n ENVNAME`: creates in `$HOME`
- `-p PATH`: creates at an arbitrary path
- `-c CHANNEL`: e.g. `conda-forge`, `bioconda`. **Avoid the `defaults` channel.**

Tip: install all needed packages in one command, not incrementally — better dependency resolution.

## Adding packages
Public/global envs are **read-only** — clone them first:
```
source activate <public_environment_name>
mamba env export --from-history --no-builds -n <public_environment_name> > /your/preferred/path/env_recreate_file
source deactivate
mamba env create -n <your_environment_name> --file /your/preferred/path/env_recreate_file
```
Cloning is approximate (bugfix/security-patch versions may differ) unless a package was pinned to an exact version/hash, which is preserved.

For your own envs:
```
source activate <your_environment_name>
mamba install -c <channel> <packages>
```

## Jupyter
```
mkjupy <env_name>
```

## GitHub-sourced packages
```
git clone <url>
```
Then install inside an appropriate mamba env. Dependency files are often overspecified/fragile — try removing all but first-order dependencies if a build fails.

## pip
"pip is generally discouraged because it is a naive package adder, rather than a managed package adder/remover" — can silently break other packages' dependency versions. Prefer mamba everywhere; pip is acceptable only where dependencies are minimal/self-contained (e.g. pytorch's own installer). See [Python Package Installation Method Comparison](/python-package-installation).
