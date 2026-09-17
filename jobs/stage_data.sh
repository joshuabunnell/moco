#!/bin/bash
# Source from a job script: stage_data <dir> copies a tensor dir to node-local disk
# and prints the path to train from. Scratch is BeeGFS, where every file open and
# first read pays a network round trip; with 20 crops per volume per epoch that
# latency, not bytes, bounded training at ~4 s/iter (E0, job 63459386) with both
# GPUs waiting on data. Falls back to the source dir when local disk is too small,
# so a job on a node without room still runs, only slower.

stage_data() {
    local src="$1"
    local dest="${TMPDIR:-/tmp}/moco_stage/$(basename "$src")"
    local need have
    need=$(du -sb "$src" | cut -f1)
    have=$(df -B1 --output=avail "${TMPDIR:-/tmp}" | tail -1)
    # 10% headroom: other jobs on a shared node write to the same disk.
    if [ "$have" -lt $((need + need / 10)) ]; then
        echo "stage_data: $((have / 1000000000)) GB free locally, need $((need / 1000000000)) GB; reading from $src" >&2
        echo "$src"
        return
    fi
    mkdir -p "$dest"
    local t0=$SECONDS
    # Parallel per-file copies: BeeGFS serves many streams far faster than one.
    # --parents keeps subdirs, which DATASET=base (both collections) has.
    (cd "$src" && find . -type f -print0 | xargs -0 -n 8 -P 16 cp --parents -t "$dest")
    echo "stage_data: copied $((need / 1000000000)) GB to $dest in $((SECONDS - t0)) s" >&2
    echo "$dest"
}
