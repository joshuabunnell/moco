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
# Config
# ---------------------------------------------------------------------------
INPUT_DIRS = [
    "/scratch/jpbunnel/downloads/manifest/CT COLONOGRAPHY",
    "/scratch/jpbunnel/downloads/manifest/Pediatric-CT-SEG",
]
CACHE_DIR = "/scratch/jpbunnel/cached-tensors/"
MIN_SLICES = 10

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
TRANSFORMS = Compose(
    [
        LoadImaged(keys=["image"], reader="PydicomReader", image_only=False),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_series_dirs(root, min_slices):
    """Return directories containing at least min_slices .dcm files."""
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
    """
    Phase 1: walk all input dirs, write every valid series path to a manifest file.
    Each line is tab-separated: <output_dir>\\t<series_path>
    No DICOM I/O occurs here — safe to run on a login node.
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
        print("Submit with: sbatch --array=0-%d scripts/prep_array.sh" % (total - 1))


def process_one(index, manifest_path):
    """
    Phase 2: process the single series at position `index` in the manifest.
    Called by each SLURM array task via SLURM_ARRAY_TASK_ID — no shared memory,
    no semaphores, each task is fully independent.
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
        default=os.path.join(CACHE_DIR, "series_manifest.txt"),
        help="Path to the series manifest file",
    )
    args = parser.parse_args()

    if args.discover:
        discover(INPUT_DIRS, CACHE_DIR, MIN_SLICES, args.manifest)
    elif args.process_index is not None:
        process_one(args.process_index, args.manifest)
    else:
        # Backward-compatible sequential mode
        print("Starting preprocessing (sequential)")
        for d in INPUT_DIRS:
            preprocess(d, _dataset_cache_dir(d, CACHE_DIR), MIN_SLICES)
        print("All done.")
