import os
from pathlib import Path

from monai.data.dataset import PersistentDataset
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_transforms():
    return Compose(
        [
            # Load the full DICOM series from a directory; image_only=False preserves
            # the spatial metadata (affine, spacing) needed by Orientationd and Spacingd.
            LoadImaged(keys=["image"], reader="PydicomReader", image_only=False),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            # Reorient to RAS so all volumes share a consistent anatomical coordinate frame.
            Orientationd(keys=["image"], axcodes="RAS", labels=None),
            # Resample to 1mm isotropic voxels so slice thickness differences across
            # scanners don't leak into the learned representations.
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            # Soft-tissue HU window (-150 to 250 HU) captures the colon wall and
            # surrounding fat/muscle while suppressing bone and air signal.
            ScaleIntensityRanged(
                keys=["image"], a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
        ]
    )


def build_dataset_standard(input_base_dir, cache_dir, min_slices=10, num_workers=10):
    input_path = Path(input_base_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Walk the input tree and collect any directory that contains enough DICOM slices
    # to constitute a valid 3D volume.
    print(f"Scanning {input_path} for DICOM series...")
    data_dicts = []
    for root, dirs, files in os.walk(input_path, followlinks=True):
        dcm_files = [f for f in files if f.endswith((".dcm", ".DCM"))]
        if len(dcm_files) >= min_slices:
            data_dicts.append({"image": root})

    print(f"Found {len(data_dicts)} valid 3D volumes.")

    # PersistentDataset runs get_transforms() on first access and writes the result
    # to cache_dir as a .pt file. Subsequent runs skip processing and load from disk.
    dataset = PersistentDataset(
        data=data_dicts, transform=get_transforms(), cache_dir=cache_dir
    )

    # spawn is the safest multiprocessing start method for MONAI/ITK on Linux/HPC.
    # batch_size=1 keeps per-worker RAM manageable for large 3D volumes.
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        multiprocessing_context="spawn",
    )

    print(f"Processing and caching volumes using {num_workers} workers...")
    for _ in tqdm(dataloader, desc="Caching Volumes"):
        pass


if __name__ == "__main__":
    # Respect the CPU allocation assigned by SLURM; fall back to 8 for local runs.
    cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    print(f"Using {cores} CPU cores for processing.")

    INPUT_DIR1 = "/scratch/jpbunnel/organized_ref/CT_COLONOGRAPHY"
    INPUT_DIR2 = "/scratch/jpbunnel/organized_ref/Pediatric-CT-SEG"
    CACHE_DIR = "/scratch/jpbunnel/cached-tensors/"

    build_dataset_standard(INPUT_DIR1, CACHE_DIR, num_workers=cores)
    build_dataset_standard(INPUT_DIR2, CACHE_DIR, num_workers=cores)
