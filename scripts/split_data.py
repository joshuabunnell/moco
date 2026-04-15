"""Generate patient-level train/val/test splits for linear probing.

Joins the cache manifest (mapping .pt filenames to patient IDs) with the
ACRIN metadata (mapping patient IDs to polyp categories) and produces
per-split CSVs that ``CTLinClsDataset`` reads directly.

Patient-level splitting ensures no patient appears in more than one split —
critical for medical imaging to prevent data leakage across scanner-specific
characteristics.

Class labels (configurable via ``--label-scheme``):

    binary:    0 = no polyp, 1 = polyp (6-9mm or >=10mm)
    three:     0 = no polyp, 1 = medium (6-9mm), 2 = large (>=10mm)

Usage:
    python scripts/split_data.py \
        --manifest /scratch/cached-tensors/CT-Colonography/manifest.csv \
        --metadata csv_metadata/acrin_combined.csv \
        --output-dir csv_metadata \
        --label-scheme three \
        --val-frac 0.15 --test-frac 0.15 \
        --seed 42
"""

import argparse
import csv
import os
import random


LABEL_SCHEMES = {
    "binary": {
        "no_polyp": 0,
        "medium_6_9mm": 1,
        "large_10mm_plus": 1,
    },
    "three": {
        "no_polyp": 0,
        "medium_6_9mm": 1,
        "large_10mm_plus": 2,
    },
}


def load_manifest(path):
    """Load a cache manifest CSV.  Returns dict: patient_id → [filenames]."""
    patient_files = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            patient_files.setdefault(pid, []).append(row["filename"])
    return patient_files


def load_metadata(path):
    """Load the combined ACRIN metadata CSV.  Returns dict: patient_id → category."""
    meta = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta[row["patient_id"]] = row["category"]
    return meta


def stratified_patient_split(patient_labels, val_frac, test_frac, seed):
    """Split patients into train/val/test preserving class proportions.

    Groups patients by label, then within each group allocates the
    requested fractions to val and test.  This ensures even rare classes
    appear in all splits when possible.

    Args:
        patient_labels: Dict of patient_id → integer label.
        val_frac: Fraction of patients for validation.
        test_frac: Fraction of patients for test.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_pids, val_pids, test_pids) as sets.
    """
    rng = random.Random(seed)

    # Group patients by label
    by_label = {}
    for pid, label in patient_labels.items():
        by_label.setdefault(label, []).append(pid)

    train, val, test = set(), set(), set()

    for label in sorted(by_label):
        pids = sorted(by_label[label])  # sorted for reproducibility
        rng.shuffle(pids)
        n = len(pids)
        n_test = max(1, round(n * test_frac)) if test_frac > 0 else 0
        n_val = max(1, round(n * val_frac)) if val_frac > 0 else 0
        # Ensure we don't over-allocate
        n_test = min(n_test, n - 1)
        n_val = min(n_val, n - n_test - 1)

        test.update(pids[:n_test])
        val.update(pids[n_test:n_test + n_val])
        train.update(pids[n_test + n_val:])

    return train, val, test


def write_split_csv(path, rows):
    """Write a split CSV with columns: filename, label."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for fname, label in sorted(rows):
            writer.writerow([fname, label])


def main():
    parser = argparse.ArgumentParser(
        description="Generate patient-level train/val/test splits")
    parser.add_argument("--manifest", required=True,
                        help="Path to cache manifest.csv from prep_data or reorganize_cache")
    parser.add_argument("--metadata", default="csv_metadata/acrin_combined.csv",
                        help="Path to combined ACRIN metadata CSV")
    parser.add_argument("--output-dir", default="csv_metadata",
                        help="Directory for output split CSVs")
    parser.add_argument("--label-scheme", default="three",
                        choices=list(LABEL_SCHEMES.keys()),
                        help="Label mapping scheme (default: three)")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Fraction of patients for validation (default: 0.15)")
    parser.add_argument("--test-frac", type=float, default=0.15,
                        help="Fraction of patients for test (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split reproducibility")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scheme = LABEL_SCHEMES[args.label_scheme]

    # Load data
    patient_files = load_manifest(args.manifest)
    metadata = load_metadata(args.metadata)

    # Match cached files to metadata labels
    matched = {}
    unmatched_cache = []
    for pid, fnames in patient_files.items():
        if pid in metadata:
            matched[pid] = (fnames, scheme[metadata[pid]])
        else:
            unmatched_cache.append(pid)

    if unmatched_cache:
        print("WARNING: %d cached patients not found in metadata (non-ACRIN data?): %s..."
              % (len(unmatched_cache), unmatched_cache[:3]))

    unmatched_meta = set(metadata) - set(patient_files)
    if unmatched_meta:
        print("NOTE: %d metadata patients have no cached tensors yet"
              % len(unmatched_meta))

    # Build patient → label mapping for splitting
    patient_labels = {pid: label for pid, (_, label) in matched.items()}

    # Split
    train_pids, val_pids, test_pids = stratified_patient_split(
        patient_labels, args.val_frac, args.test_frac, args.seed)

    # Generate per-split file lists
    splits = {"train": train_pids, "val": val_pids, "test": test_pids}
    for split_name, pids in splits.items():
        rows = []
        for pid in pids:
            fnames, label = matched[pid]
            for fname in fnames:
                rows.append((fname, label))

        path = os.path.join(args.output_dir, "labels_%s.csv" % split_name)
        write_split_csv(path, rows)
        # Count per-class distribution
        from collections import Counter
        dist = Counter(label for _, label in rows)
        print("%-5s: %3d patients, %4d files  %s"
              % (split_name, len(pids), len(rows), dict(sorted(dist.items()))))

    print("\nLabel scheme '%s': %s" % (args.label_scheme, scheme))
    print("Splits written to %s/" % args.output_dir)


if __name__ == "__main__":
    main()
