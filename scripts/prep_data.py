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
    "/scratch/jpbunnel/organized_ref/CT_COLONOGRAPHY",
    "/scratch/jpbunnel/organized_ref/Pediatric-CT-SEG",
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
        Orientationd(keys=["image"], axcodes="RAS"),
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
    for dirpath, _dirs, files in os.walk(root, followlinks=True):
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


if __name__ == "__main__":
    print("Starting preprocessing (sequential)")

    for d in INPUT_DIRS:
        preprocess(d, CACHE_DIR, MIN_SLICES)

    print("All done.")
