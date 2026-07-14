<!-- source: https://docs.rc.asu.edu/interactive-sessions -->
# Interactive Sessions | ASU RC Docs

Interactive sessions let users directly interact with compute nodes — step-by-step experimentation, iterative testing, creating Python environments, bulk file operations.

> "we have an entire partition of resources set aside for immediate allocation to users wanting to do these tasks, `lightwork`"

## Starting an Interactive Session

Basic session:
```
interactive
```
This is a shortcut for `salloc -c 1 -p htc -q public -t 0-4` (1 CPU core, 4 hours, `htc` partition).

**For light work, get a core faster with:**
```
interactive -p lightwork
```

After the session begins, work directly from the terminal.

## Customizing Interactive Session Resources

Use `salloc` directly for specific allocations, e.g.:
```
salloc -p public -c 8 -N 1 -t 0-6:00
salloc -p public -t 1-12:00
```
Full options: [official salloc documentation](https://slurm.schedmd.com/salloc.html); see also [Requesting Resources](/requesting-resources).
