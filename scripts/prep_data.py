import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from tqdm import tqdm

prep_transforms = Compose(
    [
        LoadImaged(keys=["image"], reader="ITKReader", image_only=False),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
    ]
)


def process_single_volume(series_dir, train_output_path):
    try:
        parts = Path(series_dir).parts
        series_uid = parts[-1]
        study_uid = parts[-2]

        save_name = f"{study_uid}_{series_uid}.pt"
        save_file = train_output_path / save_name

        # Skip if already processed
        if save_file.exists():
            return "skipped"

        # Apply preprocessing transforms
        data = {"image": series_dir}
        processed_data = prep_transforms(data)

        image_tensor = processed_data["image"]
        image_tensor = image_tensor.to(torch.float16)

        torch.save(image_tensor, save_file)
        return "success"

    except Exception as e:
        return f"error: {series_dir} -> {e}"


def build_hpc_dataset(input_base_dir, output_dir, min_slices=10, max_workers=8):
    input_path = Path(input_base_dir)
    output_path = Path(output_dir)
    train_output_path = output_path / "train"
    train_output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {input_path} for DICOM series...")

    series_dirs = []
    for root, dirs, files in os.walk(input_path, followlinks=True):
        dcm_files = [f for f in files if f.endswith(".dcm") or f.endswith(".DCM")]
        if len(dcm_files) >= min_slices:
            series_dirs.append(root)

    print(f"Found {len(series_dirs)} valid 3D volumes to process.")

    success_count = 0
    skipped_count = 0
    error_count = 0

    print(f"Starting parallel processing with {max_workers} CPU cores...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_volume, s_dir, train_output_path): s_dir
            for s_dir in series_dirs
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing Volumes"
        ):
            result = future.result()

            if result == "success":
                success_count += 1
            elif result == "skipped":
                skipped_count += 1
            else:
                print(f"\n[ERROR] {result}")
                error_count += 1

    print("\n--- Pipeline Complete ---")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (Already existed): {skipped_count}")
    print(f"Failed volumes: {error_count}")


if __name__ == "__main__":
    # Adjust max_workers to match the number of CPUs requested from the HPC
    CPU_CORES = 16

    INPUT_DICOM_DIR_1 = "/scratch/jpbunnel/organized_ref/CT_COLONOGRAPHY"
    INPUT_DICOM_DIR_2 = "/scratch/jpbunnel/organized_ref/Pediatric-CT-SEG"
    OUTPUT_PT_DIR = "/scratch/jpbunnel/pretraining-dataset"

    build_hpc_dataset(INPUT_DICOM_DIR_1, OUTPUT_PT_DIR, max_workers=CPU_CORES)
    build_hpc_dataset(INPUT_DICOM_DIR_2, OUTPUT_PT_DIR, max_workers=CPU_CORES)
