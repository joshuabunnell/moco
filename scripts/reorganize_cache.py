"""Rename existing MD5-hashed .pt files to patient-identifiable filenames.

Run this once on the cluster where the cached .pt tensors and the original
series_manifest.txt exist.  It reads the manifest written by prep_data.py's
discover phase (which records the exact paths used to compute MD5 hashes),
re-derives each hash, finds the corresponding .pt file in the cache, and
renames it to the new ``<patient_id>_<short_hash>.pt`` convention.  Also
writes a ``manifest.csv`` in each cache subdirectory.

Usage:
    python scripts/reorganize_cache.py \
        --manifest /scratch/cached-tensors/series_manifest.txt \
        --cache-dir /scratch/cached-tensors

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
    series_filename,
)

# Match old-style MD5 filenames (32 hex chars + .pt)
_MD5_FILENAME_RE = re.compile(r"^[0-9a-f]{32}\.pt$")


def reorganize(manifest_path, cache_dir, dry_run=False):
    """Rename MD5-hashed .pt files using paths from the series manifest.

    The series_manifest.txt is tab-separated with columns:
        <cache_subdir>  <original_series_path>

    The MD5 hash was computed from the original series path string, so we
    can reconstruct it even if that path no longer exists on disk.

    Args:
        manifest_path: Path to series_manifest.txt from prep_data --discover.
        cache_dir: Root cache directory containing per-dataset subdirs.
        dry_run: If True, only print what would be done.
    """
    # Read manifest entries grouped by cache subdirectory.
    # Old discover format: <output_dir>\t<series_path>  (2 columns)
    # New discover format: <output_dir>\t<input_root>\t<series_path>  (3 columns)
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                cache_subdir, input_root, series_path = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                cache_subdir, series_path = parts[0], parts[1]
                input_root = None
            else:
                continue
            entries.append((cache_subdir, input_root, series_path))

    print(f"Read {len(entries)} entries from {manifest_path}")

    # Group by cache subdirectory
    by_subdir = {}
    for cache_subdir, input_root, series_path in entries:
        by_subdir.setdefault(cache_subdir, []).append((input_root, series_path))

    for cache_subdir, entries_list in by_subdir.items():
        if not os.path.isdir(cache_subdir):
            print(f"WARNING: cache subdir {cache_subdir} does not exist, skipping")
            continue

        dataset_name = os.path.basename(cache_subdir)

        renamed = 0
        skipped = 0
        missing = 0
        manifest_rows = []

        for input_root, series_path in entries_list:
            # Reconstruct the MD5 hash that prep_data originally used.
            # The old code hashed str(Path(series_path)) — Path normalizes
            # trailing slashes, so we do the same.
            normalized = str(Path(series_path))
            old_hash = hashlib.md5(normalized.encode()).hexdigest()
            old_path = os.path.join(cache_subdir, old_hash + ".pt")

            # Derive input_root if not provided (2-column manifest format).
            if input_root is None:
                idx = series_path.find(dataset_name)
                if idx >= 0:
                    input_root = series_path[:idx + len(dataset_name)]
                else:
                    input_root = series_path

            patient_id = extract_patient_id(series_path, input_root)
            new_fname = series_filename(patient_id, series_path)
            new_path = os.path.join(cache_subdir, new_fname)

            if os.path.exists(new_path):
                # Already renamed
                manifest_rows.append([new_fname, patient_id, series_path])
                skipped += 1
                continue

            if not os.path.exists(old_path):
                missing += 1
                continue

            if dry_run:
                print(f"  RENAME {old_hash}.pt → {new_fname}")
            else:
                os.rename(old_path, new_path)
                print(f"  RENAMED {old_hash}.pt → {new_fname}")

            manifest_rows.append([new_fname, patient_id, series_path])
            renamed += 1

        # Write manifest.csv
        manifest_csv_path = os.path.join(cache_subdir, "manifest.csv")
        if not dry_run and manifest_rows:
            with open(manifest_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "patient_id", "series_path"])
                writer.writerows(manifest_rows)
            print(f"Wrote manifest with {len(manifest_rows)} entries to {manifest_csv_path}")

        print(f"Done {cache_subdir} — renamed: {renamed}, already done: {skipped}, not cached: {missing}")


def main():
    parser = argparse.ArgumentParser(
        description="Rename MD5-hashed cached tensors to patient-identifiable names")
    parser.add_argument("--manifest", required=True,
                        help="Path to series_manifest.txt from prep_data --discover")
    parser.add_argument("--cache-dir", required=True,
                        help="Root cache directory (contains per-dataset subdirs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without changing anything")
    args = parser.parse_args()

    reorganize(args.manifest, args.cache_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
