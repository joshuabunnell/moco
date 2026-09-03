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

# Resets the 90-day purge clock on our /scratch data. Two passes, because RC's
# at-risk CSVs are reactive: by the time a path is listed in one, it is already
# close to deletion. The unconditional pass keeps DATA_ROOT alive regardless.
set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/moco}"
source "${PROJECT_DIR}/jobs/config.sh"

if [ -d "${DATA_ROOT}" ]; then
    echo "=== unconditional refresh: ${DATA_ROOT} ==="
    find "${DATA_ROOT}" -exec touch {} +
    echo "TOUCHED: ${DATA_ROOT} ($(du -sh "${DATA_ROOT}" 2>/dev/null | cut -f1))"
else
    echo "SKIP (no such dir): ${DATA_ROOT}"
fi

# Second pass: anything else of ours RC has flagged, which may sit outside DATA_ROOT.
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
        # Never touch another user's directory, even if one shows up in the report.
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
