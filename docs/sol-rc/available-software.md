<!-- source: https://docs.rc.asu.edu/available-software -->
# Available Software | ASU RC Docs

By default no software modules are loaded (clean session). Tip: avoid putting `module load` in shell init scripts, so job submissions and interactive shells behave identically/reproducibly.

## Listing modules
```
module avail
module avail rust     # keyword search
```
Web portal listings: [Sol modules](https://links.asu.edu/sol-modules), [Phoenix modules](https://links.asu.edu/phx-modules)

Manual builds: `software/version.number`. Auto-built: `software-version.number-compiler-version.number`. Usage identical once loaded.

## Loading / unloading
```
module load aspect/2.3.0   # or: ml aspect/2.3.0
module list
module unload aspect/2.3.0
module purge                # unload everything — useful before an sbatch job
```

If software isn't yet installed as a module, see [Building Software](/building-software).
