#!/bin/bash
#SBATCH -J refresh_scratch
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH -t 0-02:00:00
#SBATCH -p lightwork
#SBATCH -q public
#SBATCH -o /scratch/%u/moco/logs/%x.%j.out
#SBATCH -e /scratch/%u/moco/logs/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

# RC drops at-risk /scratch paths into $HOME as scratch-dirs-{inactive,pending-removal}.csv; this touches every path in our own $SCRATCH tree so they register as active again.
set -uo pipefail

CSV_FILES=(
    "$HOME/scratch-dirs-pending-removal.csv"
    "$HOME/scratch-dirs-inactive.csv"
)

for csv in "${CSV_FILES[@]}"; do
    if [ ! -f "$csv" ]; then
        echo "Not found, skipping: $csv"
        continue
    fi
    echo "=== $csv ==="
    tail -n +2 "$csv" | while IFS=',' read -r dir last_used days_last oldest days_oldest file_count size_gib; do
        if [[ "$dir" != "$SCRATCH"/* ]]; then
            echo "SKIP (not our scratch tree): $dir"
            continue
        fi
        if [ ! -d "$dir" ]; then
            echo "SKIP (no longer exists): $dir"
            continue
        fi
        echo "TOUCH: $dir (${file_count} files, ${size_gib} GiB, last used ${days_last}d ago)"
        find "$dir" -exec touch {} +
    done
done

echo "Refresh complete: $(date)"
