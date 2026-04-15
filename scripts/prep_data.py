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
import hashlib
import os
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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

    for i, series_dir in enumerate(series):
        uid = hashlib.md5(str(series_dir).encode()).hexdigest()
        out_path = os.path.join(cache_dir, uid + ".pt")

        # Already cached — skip
        if os.path.exists(out_path):
            skipped += 1
            continue

        t0 = time.time()
        try:
            data = TRANSFORMS({"image": str(series_dir)})
            torch.save(data["image"], out_path)
            elapsed = time.time() - t0
            saved += 1
            print("  [%d/%d] saved %s (%.1fs)" % (i + 1, total, uid[:12], elapsed))
        except Exception as exc:
            failed += 1
            print(
                "  [%d/%d] FAILED %s -- %s" % (i + 1, total, series_dir, exc),
                file=sys.stderr,
            )

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

    Each line is tab-separated: ``<output_dir>\\t<series_path>``.
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
                f.write("%s\t%s\n" % (out_dir, p))
            total += len(found)
    print("Wrote %d total series to %s" % (total, manifest_path))
    if total:
        print("Submit with: sbatch --array=0-%d examples/prep_array.sh" % (total - 1))


def process_one(index, manifest_path):
    """Phase 2: process the single series at position *index* in the manifest.

    Called by each SLURM array task via ``SLURM_ARRAY_TASK_ID``.  Each task is
    fully independent — no shared memory or inter-task communication.

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

    out_dir, series_path = lines[index].split("\t", 1)
    series_dir = Path(series_path)
    uid = hashlib.md5(str(series_dir).encode()).hexdigest()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, uid + ".pt")

    if os.path.exists(out_path):
        print("Already cached, skipping: %s" % uid[:12])
        return

    t0 = time.time()
    try:
        data = TRANSFORMS({"image": str(series_dir)})
        torch.save(data["image"], out_path)
        print("Saved %s (%.1fs)" % (uid[:12], time.time() - t0))
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
