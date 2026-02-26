import os
from pathlib import Path

import torch
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from tqdm import tqdm


def build_hpc_dataset(input_base_dir, output_dir, min_slices=10):
    """
    Crawls DICOM directories, applies static preprocessing, and saves full 3D volumes as fast-loading .pt files.
    """
    input_path = Path(input_base_dir)
    output_path = Path(output_dir)

    # Create 'train' subdirectory for MoCo dataset structure
    train_output_path = output_path / "train"
    train_output_path.mkdir(parents=True, exist_ok=True)

    # Static preprocessing pipeline (heavy transforms applied once, reused for all augmentations)
    prep_transforms = Compose(
        [
            # Load DICOM series as 3D volumes
            LoadImaged(keys=["image"], reader="ITKReader", image_only=False),
            # Ensure (C, H, W, D) format for consistency
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            # Standardize to right-anterior-superior orientation
            Orientationd(keys=["image"], axcodes="RAS"),
            # Resample to 1mm isotropic spacing for consistent voxel dimensions
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            # Soft tissue window: HU range [-150, 250] normalized to [0, 1]
            # Clips outliers (air < -150 HU, dense bone > 250 HU) for better contrast
            ScaleIntensityRanged(
                keys=["image"], a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
        ]
    )

    print(f"Scanning {input_path} for DICOM series...")

    # Discover all series directories with sufficient slices
    # (filters out scout images and empty folders)
    series_dirs = []
    for root, dirs, files in os.walk(input_path):
        dcm_files = [f for f in files if f.endswith(".dcm") or f.endswith(".DCM")]

        if len(dcm_files) >= min_slices:
            series_dirs.append(root)

    print(f"Found {len(series_dirs)} valid 3D volumes to process.")

    success_count = 0
    error_count = 0

    # Process each DICOM series
    for series_dir in tqdm(series_dirs, desc="Processing Volumes"):
        try:
            # Extract unique identifiers from directory path
            # (assumes organized structure: .../StudyInstanceUID/SeriesInstanceUID)
            parts = Path(series_dir).parts
            series_uid = parts[-1]
            study_uid = parts[-2]

            save_name = f"{study_uid}_{series_uid}.pt"
            save_file = train_output_path / save_name

            # Skip if already processed (allows resuming interrupted runs)
            if save_file.exists():
                success_count += 1
                continue

            # Apply preprocessing transforms
            data = {"image": series_dir}
            processed_data = prep_transforms(data)

            image_tensor = processed_data["image"]

            # Convert to float16 to reduce storage by 50% while preserving image quality
            image_tensor = image_tensor.to(torch.float16)

            torch.save(image_tensor, save_file)
            success_count += 1

        except Exception as e:
            print(f"\n[ERROR] Failed to process {series_dir}")
            print(f"Reason: {e}")
            error_count += 1

    print("\n--- Pipeline Complete ---")
    print(f"Successfully processed and saved: {success_count} volumes")
    print(f"Failed volumes: {error_count}")


if __name__ == "__main__":
    INPUT_DICOM_DIR_1 = "/scratch/jpbunnel/organized_ref/CT_COLONOGRAPHY"
    INPUT_DICOM_DIR_2 = "/scratch/jpbunnel/organized_ref/Pediatric-CT-SEG"

    OUTPUT_PT_DIR = "/scratch/jpbunnel/pretraining-dataset"

    build_hpc_dataset(INPUT_DICOM_DIR_1, OUTPUT_PT_DIR)
    build_hpc_dataset(INPUT_DICOM_DIR_2, OUTPUT_PT_DIR)
