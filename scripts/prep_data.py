"""Preprocess DICOM CT series into cached .pt tensors for MoCo training.

Walks one or more input directories containing DICOM series, applies medical
image transforms (reorientation, resampling, windowing), and saves each
processed volume as a PyTorch tensor.  Supports two execution modes:

Sequential (single machine):
    python scripts/prep_data.py \\
        --input-dirs /data/CT-Colonography /data/Pediatric-CT-SEG \\
        --cache-dir /scratch/cached-tensors

SLURM array jobs (HPC):
    # Phase 1 — discover series and write manifest (login node, no DICOM I/O)
    python scripts/prep_data.py --discover \\
        --input-dirs /data/CT-Colonography /data/Pediatric-CT-SEG \\
        --cache-dir /scratch/cached-tensors \\
        --manifest /scratch/cached-tensors/series_manifest.txt

    # Phase 2 — process one series per array task
    python scripts/prep_data.py \\
        --process-index $SLURM_ARRAY_TASK_ID \\
        --manifest /scratch/cached-tensors/series_manifest.txt
"""

import argparse
import csv
import hashlib
import os
import re
import sys
import time
from pathlib import Path

import torch
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd

# ---------------------------------------------------------------------------
# Transforms — the static preprocessing pipeline applied to every volume
# ---------------------------------------------------------------------------
TRANSFORMS = Compose(
    [
        LoadImaged(keys=["image"], reader="PydicomReader", image_only=False),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),

        # Reorient to Right-Anterior-Superior (RAS) axes so that all volumes
        # share a consistent anatomical coordinate system regardless of how
        # the original scanner encoded orientation.
        Orientationd(keys=["image"], axcodes="RAS", labels=None),

        # Resample to 1 mm isotropic voxels.  CT scanners vary in slice
        # thickness (commonly 1-3 mm) and in-plane resolution; resampling
        # normalizes these differences so the model sees uniform geometry.
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),

        # Soft-tissue HU window: [-150, +250] isolates colon wall, mesenteric
        # fat, and polyp tissue while excluding bone (>400 HU) and air
        # (<-500 HU).  Values are rescaled to [0, 1] for network input.
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-150,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
)

# Default minimum slices to consider a directory a valid DICOM series
DEFAULT_MIN_SLICES = 10

# TCIA ACRIN 6664 patient IDs follow this OID pattern
_TCIA_UID_RE = re.compile(r"1\.3\.6\.1\.4\.1\.9328\.50\.4\.\d+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_patient_id(series_dir, input_root):
    """Extract the TCIA patient ID from a DICOM series path.

    Walks the path components between *input_root* and *series_dir* looking
    for the TCIA OID pattern (``1.3.6.1.4.1.9328.50.4.XXXX``).  Falls back
    to an MD5 hash if no recognisable ID is found (e.g., non-TCIA data).

    Args:
        series_dir: Full path to the DICOM series directory.
        input_root: Top-level input directory that was passed to the script.

    Returns:
        A patient-identifiable string suitable for use in filenames.
    """
    try:
        rel = Path(series_dir).relative_to(input_root)
    except ValueError:
        rel = Path(series_dir)

    for part in rel.parts:
        if _TCIA_UID_RE.fullmatch(part):
            return part

    # Fallback: use MD5 (handles non-TCIA datasets like Pediatric-CT-SEG)
    return hashlib.md5(str(series_dir).encode()).hexdigest()


def series_filename(patient_id, series_dir):
    """Build a .pt filename that is unique per series within a patient.

    Patients may have multiple series (e.g., supine + prone).  We append a
    short hash of the series path to disambiguate.

    Returns:
        String like ``1.3.6.1.4.1.9328.50.4.0007_a3b2.pt``.
    """
    short_hash = hashlib.md5(str(series_dir).encode()).hexdigest()[:4]
    return f"{patient_id}_{short_hash}.pt"


def write_manifest(manifest_path, rows):
    """Write or append to a manifest CSV mapping cached files to source paths.

    Columns: ``filename``, ``patient_id``, ``series_path``.
    """
    file_exists = os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "patient_id", "series_path"])
        writer.writerows(rows)


def find_series_dirs(root, min_slices):
    """Walk a directory tree and return paths containing enough DICOM files.

    Args:
        root: Top-level directory to search.
        min_slices: Minimum number of .dcm files for a directory to qualify.

    Returns:
        List of ``Path`` objects pointing to valid series directories.
    """
    series = []
    for dirpath, _dirs, files in os.walk(
        root
    ):  # no followlinks — circular symlinks loop forever
        dcm_count = 0
        for f in files:
            if f.lower().endswith(".dcm"):
                dcm_count += 1
                if dcm_count >= min_slices:
                    series.append(Path(dirpath))
                    break
    return series


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def preprocess(input_dir, cache_dir, min_slices):
    """Process all series in *input_dir* sequentially and cache as .pt files.

    Saves each volume with a patient-identifiable filename and writes a
    manifest CSV (``manifest.csv``) in the cache directory mapping each
    file back to its source DICOM series path.

    Args:
        input_dir: Directory containing DICOM series subdirectories.
        cache_dir: Output directory for cached .pt tensors.
        min_slices: Minimum DICOM slices to consider a valid series.
    """
    os.makedirs(cache_dir, exist_ok=True)

    series = find_series_dirs(input_dir, min_slices)
    total = len(series)
    print("Found %d series in %s" % (total, input_dir))

    if not series:
        return

    skipped = 0
    failed = 0
    saved = 0
    manifest_rows = []
    manifest_path = os.path.join(cache_dir, "manifest.csv")

    for i, series_dir in enumerate(series):
        patient_id = extract_patient_id(series_dir, input_dir)
        fname = series_filename(patient_id, series_dir)
        out_path = os.path.join(cache_dir, fname)

        # Already cached — still record in manifest for completeness
        if os.path.exists(out_path):
            manifest_rows.append([fname, patient_id, str(series_dir)])
            skipped += 1
            continue

        t0 = time.time()
        try:
            data = TRANSFORMS({"image": str(series_dir)})
            torch.save(data["image"], out_path)
            elapsed = time.time() - t0
            saved += 1
            manifest_rows.append([fname, patient_id, str(series_dir)])
            print("  [%d/%d] saved %s (%.1fs)" % (i + 1, total, fname, elapsed))
        except Exception as exc:
            failed += 1
            print(
                "  [%d/%d] FAILED %s -- %s" % (i + 1, total, series_dir, exc),
                file=sys.stderr,
            )

    # Write manifest — overwrites per input_dir so the final manifest
    # reflects exactly what is in the cache directory
    write_manifest(manifest_path, manifest_rows)

    print(
        "Done %s — saved: %d, skipped: %d, failed: %d"
        % (input_dir, saved, skipped, failed)
    )


# ---------------------------------------------------------------------------
# SLURM job array support
# ---------------------------------------------------------------------------
def _dataset_cache_dir(input_dir, cache_dir):
    """Derive a per-dataset output subdir from the input directory's basename."""
    return os.path.join(cache_dir, os.path.basename(input_dir.rstrip("/")))


def discover(input_dirs, cache_dir, min_slices, manifest_path):
    """Phase 1: walk input dirs and write every valid series path to a manifest.

    Each line is tab-separated: ``<output_dir>\\t<input_root>\\t<series_path>``.
    The input_root is included so that Phase 2 can extract patient IDs by
    computing the relative path from the dataset root.

    No DICOM I/O occurs here — safe to run on a login node.

    Args:
        input_dirs: List of top-level directories to scan for DICOM series.
        cache_dir: Root output directory for cached tensors.
        min_slices: Minimum DICOM slices per series.
        manifest_path: File path to write the series manifest.
    """
    os.makedirs(cache_dir, exist_ok=True)
    total = 0
    with open(manifest_path, "w") as f:
        for d in input_dirs:
            out_dir = _dataset_cache_dir(d, cache_dir)
            found = find_series_dirs(d, min_slices)
            print("Found %d series in %s → %s" % (len(found), d, out_dir))
            for p in found:
                f.write("%s\t%s\t%s\n" % (out_dir, d, p))
            total += len(found)
    print("Wrote %d total series to %s" % (total, manifest_path))
    if total:
        print("Submit with: sbatch --array=0-%d examples/prep_array.sh" % (total - 1))


def process_one(index, manifest_path):
    """Phase 2: process the single series at position *index* in the manifest.

    Called by each SLURM array task via ``SLURM_ARRAY_TASK_ID``.  Each task is
    fully independent — no shared memory or inter-task communication.

    The discover manifest uses tab-separated format:
    ``<output_dir>\\t<input_root>\\t<series_path>``.

    Args:
        index: Zero-based position in the manifest file.
        manifest_path: Path to the manifest written by :func:`discover`.
    """
    with open(manifest_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    if index >= len(lines):
        print(
            "ERROR: index %d out of range (manifest has %d entries)"
            % (index, len(lines)),
            file=sys.stderr,
        )
        sys.exit(1)

    parts = lines[index].split("\t")
    if len(parts) == 3:
        out_dir, input_root, series_path = parts
    else:
        # Backward compat with old 2-column discover manifests
        out_dir, series_path = parts[0], parts[-1]
        input_root = out_dir

    series_dir = Path(series_path)
    patient_id = extract_patient_id(series_dir, input_root)
    fname = series_filename(patient_id, series_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)

    if os.path.exists(out_path):
        print("Already cached, skipping: %s" % fname)
        return

    t0 = time.time()
    try:
        data = TRANSFORMS({"image": str(series_dir)})
        torch.save(data["image"], out_path)
        # Append to the cache manifest so downstream scripts can map files
        cache_manifest = os.path.join(out_dir, "manifest.csv")
        write_manifest(cache_manifest, [[fname, patient_id, str(series_dir)]])
        print("Saved %s (%.1fs)" % (fname, time.time() - t0))
    except Exception as exc:
        print("FAILED %s -- %s" % (series_dir, exc), file=sys.stderr)
        sys.exit(2)  # non-zero so SLURM marks this task as FAILED


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT colonoscopy DICOM preprocessing")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help="One or more directories containing DICOM series to preprocess",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help="Output directory for cached .pt tensors",
    )
    parser.add_argument(
        "--min-slices",
        type=int,
        default=DEFAULT_MIN_SLICES,
        help="Minimum DICOM slices per series (default: %d)" % DEFAULT_MIN_SLICES,
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Phase 1: walk input dirs and write series manifest (no DICOM I/O)",
    )
    parser.add_argument(
        "--process-index",
        type=int,
        default=None,
        metavar="N",
        help="Phase 2: process series at index N from the manifest (set by SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to the series manifest file (default: <cache-dir>/series_manifest.txt)",
    )
    args = parser.parse_args()

    # Derive manifest default from cache-dir if both are provided
    if args.manifest is None and args.cache_dir is not None:
        args.manifest = os.path.join(args.cache_dir, "series_manifest.txt")

    if args.discover:
        if not args.input_dirs or not args.cache_dir:
            parser.error("--discover requires --input-dirs and --cache-dir")
        if args.manifest is None:
            parser.error("--discover requires --cache-dir or --manifest")
        discover(args.input_dirs, args.cache_dir, args.min_slices, args.manifest)
    elif args.process_index is not None:
        if args.manifest is None:
            parser.error("--process-index requires --manifest (or --cache-dir)")
        process_one(args.process_index, args.manifest)
    else:
        # Sequential mode
        if not args.input_dirs or not args.cache_dir:
            parser.error("Sequential mode requires --input-dirs and --cache-dir")
        print("Starting preprocessing (sequential)")
        for d in args.input_dirs:
            preprocess(d, _dataset_cache_dir(d, args.cache_dir), args.min_slices)
        print("All done.")
