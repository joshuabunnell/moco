import glob
import os

import torch
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import RandSpatialCropd
from monai.transforms.intensity.dictionary import (
    RandGaussianNoised,
    RandGaussianSmoothd,
)
from monai.transforms.spatial.dictionary import RandFlipd, RandRotated
from monai.transforms.utility.dictionary import Lambdad
from torch.utils.data import Dataset


class CTMoCoDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "**/*.pt"), recursive=True)
        print(f"Found {len(self.files)} 3D volumes for Pretraining.")

        # Base Extraction: Get the raw 2.5D chunk
        self.extract_crop = RandSpatialCropd(
            keys=["image"], roi_size=(224, 224, 3), random_size=False
        )

        # MoCo Augmentations: Strict medical transforms
        self.moco_augs = Compose(
            [
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),  # Up/Down
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),  # Left/Right
                RandFlipd(
                    keys=["image"], prob=0.5, spatial_axis=2
                ),  # Channel/Z-Axis Reversal
                RandRotated(keys=["image"], prob=0.5, range_x=0.26),  # ~15 degrees
                RandGaussianNoised(keys=["image"], prob=0.5, std=0.05),
                RandGaussianSmoothd(keys=["image"], prob=0.5, sigma_x=(0.5, 1.5)),
                # Format for ResNet: (1, 224, 224, 3) -> (3, 224, 224)
                Lambdad(keys=["image"], func=lambda x: x[0].permute(2, 0, 1)),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the saved 3D volume
        volume = {"image": torch.load(self.files[idx])}

        # Extract the base crop
        base_crop = self.extract_crop(volume)

        # Branch into Query and Key with independent random augmentations
        view_q = self.moco_augs(base_crop)["image"]
        view_k = self.moco_augs(base_crop)["image"]

        # Return as a list of two views, plus a dummy target '0'
        return [view_q, view_k], 0
