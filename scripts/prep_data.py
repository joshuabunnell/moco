import multiprocessing
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


def _build_transforms():
    """Instantiate transforms fresh per-worker process.

    Defined as a function (not module-level) so that each spawned worker
    builds its own clean instance. This avoids deadlocks caused by forking
    a process that already holds C-extension locks (ITK, OpenMP, etc.).
    """
    return Compose(
        [
            # PydicomReader is MONAI's built-in DICOM reader — no ITK required.
            # It accepts a directory path and assembles slices into a 3-D volume.
            LoadImaged(keys=["image"], reader="PydicomReader", image_only=False),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            # labels=None: use the meta-tensor's own 'space' field (suppresses FutureWarning).
            Orientationd(keys=["image"], axcodes="RAS", labels=None),
            # Resample to 1 mm isotropic so features are scale-invariant across scanners.
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            # Colon HU window: -150 HU (fat) to 250 HU (soft tissue/polyp).
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

        # Build transforms inside the worker so each spawned process has its
        # own instance — avoids shared C-extension state across processes.
        transforms = _build_transforms()

        # Apply preprocessing transforms
        data = {"image": series_dir}
        processed_data = transforms(data)

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
    multiprocessing.set_start_method("spawn", force=True)

    CPU_CORES = 1

    INPUT_DICOM_DIR_1 = (
        "/Users/joshuabunnell/Projects/data/dicom/ct-colonography_organized"
    )
    OUTPUT_PT_DIR = (
        "/Users/joshuabunnell/Projects/data/dicom/pt-ct-colonography_organized"
    )

    build_hpc_dataset(INPUT_DICOM_DIR_1, OUTPUT_PT_DIR, max_workers=CPU_CORES)
