import hashlib
import os
from multiprocessing import Pool
from pathlib import Path

import torch
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from tqdm import tqdm


TRANSFORMS = Compose([
    # image_only=False preserves the affine/spacing metadata needed by the spatial transforms below.
    LoadImaged(keys=["image"], reader="PydicomReader", image_only=False),
    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    # Reorient to RAS so all volumes share a consistent anatomical coordinate frame.
    Orientationd(keys=["image"], axcodes="RAS", labels=None),
    # 1mm isotropic voxels normalize slice-thickness differences across scanners.
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    # Soft-tissue HU window: captures colon wall, fat, and muscle; suppresses bone and air.
    ScaleIntensityRanged(keys=["image"], a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True),
])


def find_series_dirs(root: Path, min_slices: int = 10) -> list[Path]:
    """Return directories that contain at least min_slices DICOM files."""
    series = []
    for dirpath, _, files in os.walk(root, followlinks=True):
        if sum(1 for f in files if f.lower().endswith(".dcm")) >= min_slices:
            series.append(Path(dirpath))
    return series


def process_one(args: tuple[Path, Path]) -> str | None:
    """
    Load, transform, and cache a single DICOM series as a .pt tensor.
    Returns the output path on success, or None if the series was already cached.
    """
    series_dir, cache_dir = args

    # Derive a stable filename from the series path so reruns skip completed volumes.
    uid = hashlib.md5(str(series_dir).encode()).hexdigest()
    out_path = cache_dir / f"{uid}.pt"
    if out_path.exists():
        return None

    data = TRANSFORMS({"image": str(series_dir)})
    torch.save(data["image"], out_path)
    return str(out_path)


def preprocess(input_dir: str, cache_dir: str, num_workers: int, min_slices: int = 10):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    series = find_series_dirs(Path(input_dir), min_slices)
    print(f"Found {len(series)} series in {input_dir}")

    args = [(s, cache_path) for s in series]

    # ProcessPool maps one series per worker process — clean, no shared memory,
    # no DataLoader semaphores that can hang under SLURM resource limits.
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_one, args), total=len(args), desc="Caching"):
            pass


if __name__ == "__main__":
    # Use the CPU count allocated by SLURM, or fall back to 8 for local runs.
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    print(f"Using {num_workers} workers")

    # INPUT_DIR1 = "/scratch/jpbunnel/organized_ref/CT_COLONOGRAPHY"
    # INPUT_DIR2 = "/scratch/jpbunnel/organized_ref/Pediatric-CT-SEG"
    # CACHE_DIR  = "/scratch/jpbunnel/cached-tensors/"

    INPUT_DIR1 = "/Users/joshuabunnell/Projects/data/dicom/ct-colonography_organized"
    INPUT_DIR2 = "/Users/joshuabunnell/Projects/data/dicom/pediatric-ct-seg_organized"
    CACHE_DIR  = "/Users/joshuabunnell/Projects/data/dicom/cached-tensors/"

    preprocess(INPUT_DIR1, CACHE_DIR, num_workers)
    preprocess(INPUT_DIR2, CACHE_DIR, num_workers)
