<!-- source: https://docs.rc.asu.edu/building-software -->
# Building Software | ASU RC Docs

RC staff make a best-effort attempt to install requested software as modules, but it's ultimately the user's responsibility to find a viable solution (compile, use prebuilt binaries/containers, or find alternatives).

## Installation locations
- **Single-user**: install to your `HOME` directory, invoke by filepath.
- **Group/lab**: install to `/data/grp_XXXX` (see your groups via `groups` command) — shared read/write for the group.
- **Public** (wide appeal): built in `/packages/apps` with a public module, `module load <package/version>`.

## Requesting software
Submit a [support ticket](https://rto.asu.edu/request-help/) with: URL to source, version requirements, plugin/addon needs, source location.

> "The supercomputer runs 100% Linux using Rocky Linux 8.x, a Redhat variant (RHEL)." No Windows software support. Debian/Ubuntu-only software may be attempted if it compiles; otherwise use a container.

## Compiling from source
No `sudo` access for users — can't install to `/usr/local` or use `dnf`/`apt`. Even with sudo on one node, subsequent connections may land on a different node where the install is inaccessible, breaking jobs.

Always redirect install prefix to `HOME`, scratch, or group project storage (available on every node):
```
./configure --prefix=/home/[asurite]/.local/opt/python-3.9.7
make
make install
```

## Containers
Apptainer (Singularity) is available; many `.sif` files can be uploaded to `HOME` and run directly. Building a container from a recipe requires building on your **local workstation** (root/sudo required), not on the supercomputer.

> **"Docker is not supported in the supercomputer environment, as it was architected to use `root`"** — not accessible to end-users. Apptainer typically works with Docker images without issue.
