"""Rename existing MD5-hashed .pt files to patient-identifiable filenames.

Run this once on the cluster where both the original DICOM directories and
the cached .pt tensors exist.  It re-walks the DICOM source directories,
computes the old MD5 hash for each series, finds the corresponding .pt file
in the cache, and renames it to the new ``<patient_id>_<short_hash>.pt``
convention.  Also writes a ``manifest.csv`` in each cache subdirectory.

Usage:
    python scripts/reorganize_cache.py \
        --input-dirs /data/CT-Colonography /data/Pediatric-CT-SEG \
        --cache-dir /scratch/jpbunnel/cached-tensors

This is idempotent — already-renamed files are detected and skipped.
"""

import argparse
import csv
import hashlib
import os
import re
import sys
from pathlib import Path

# Reuse helpers from prep_data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.prep_data import (
    extract_patient_id,
    find_series_dirs,
    series_filename,
    DEFAULT_MIN_SLICES,
)

# Match old-style MD5 filenames (32 hex chars + .pt)
_MD5_FILENAME_RE = re.compile(r"^[0-9a-f]{32}\.pt$")


def _dataset_cache_dir(input_dir, cache_dir):
    return os.path.join(cache_dir, os.path.basename(input_dir.rstrip("/")))


def reorganize(input_dir, cache_dir, min_slices, dry_run=False):
    """Rename MD5-hashed .pt files and write a manifest.

    Args:
        input_dir: Original DICOM root directory.
        cache_dir: Cache subdirectory containing .pt files for this dataset.
        min_slices: Minimum DICOM slices per series.
        dry_run: If True, only print what would be done.
    """
    series = find_series_dirs(input_dir, min_slices)
    print("Found %d series in %s" % (len(series), input_dir))

    renamed = 0
    skipped = 0
    missing = 0
    manifest_rows = []

    for series_dir in series:
        old_hash = hashlib.md5(str(series_dir).encode()).hexdigest()
        old_path = os.path.join(cache_dir, old_hash + ".pt")

        patient_id = extract_patient_id(series_dir, input_dir)
        new_fname = series_filename(patient_id, series_dir)
        new_path = os.path.join(cache_dir, new_fname)

        if os.path.exists(new_path):
            # Already renamed
            manifest_rows.append([new_fname, patient_id, str(series_dir)])
            skipped += 1
            continue

        if not os.path.exists(old_path):
            # Never cached or already renamed differently
            missing += 1
            continue

        if dry_run:
            print("  RENAME %s → %s" % (old_hash + ".pt", new_fname))
        else:
            os.rename(old_path, new_path)
            print("  RENAMED %s → %s" % (old_hash + ".pt", new_fname))

        manifest_rows.append([new_fname, patient_id, str(series_dir)])
        renamed += 1

    # Write manifest
    manifest_path = os.path.join(cache_dir, "manifest.csv")
    if not dry_run and manifest_rows:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "patient_id", "series_path"])
            writer.writerows(manifest_rows)
        print("Wrote manifest with %d entries to %s" % (len(manifest_rows), manifest_path))

    print("Done %s — renamed: %d, already done: %d, not cached: %d"
          % (cache_dir, renamed, skipped, missing))


def main():
    parser = argparse.ArgumentParser(
        description="Rename MD5-hashed cached tensors to patient-identifiable names")
    parser.add_argument("--input-dirs", nargs="+", required=True,
                        help="Original DICOM source directories")
    parser.add_argument("--cache-dir", required=True,
                        help="Root cache directory (contains per-dataset subdirs)")
    parser.add_argument("--min-slices", type=int, default=DEFAULT_MIN_SLICES,
                        help="Min DICOM slices per series (default: %d)" % DEFAULT_MIN_SLICES)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without changing anything")
    args = parser.parse_args()

    for d in args.input_dirs:
        out_dir = _dataset_cache_dir(d, args.cache_dir)
        if not os.path.isdir(out_dir):
            print("WARNING: cache subdir %s does not exist, skipping %s" % (out_dir, d))
            continue
        reorganize(d, out_dir, args.min_slices, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
