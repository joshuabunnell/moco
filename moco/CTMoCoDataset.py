import copy
import glob
import os

import numpy as np
import torch
import torch.serialization
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CenterSpatialCropd, RandSpatialCropd, ResizeWithPadOrCropd
from monai.transforms.intensity.dictionary import (
    RandGaussianNoised,
    RandGaussianSmoothd,
)
from monai.transforms.spatial.dictionary import RandFlipd, RandRotated
from monai.transforms.utility.dictionary import Lambdad
from torch.utils.data import Dataset

# MONAI MetaTensor serializes numpy affine/metadata alongside the tensor.
# PyTorch 2.6+ validates globals even with weights_only=False, so we must
# allowlist the types used by MONAI before any torch.load call.
from monai.data.meta_tensor import MetaTensor

torch.serialization.add_safe_globals([
    np.ndarray,
    np.dtype,
    MetaTensor,
])


def _to_resnet_format(x):
    # (1, 224, 224, 3) -> (3, 224, 224) for ResNet input
    return x[0].permute(2, 0, 1)


class CTMoCoDataset(Dataset):
    def __init__(self, data_dir, crops_per_volume=20):
        self.files = glob.glob(os.path.join(data_dir, "**/*.pt"), recursive=True)
        self.crops_per_volume = crops_per_volume
        print(f"Found {len(self.files)} 3D volumes for Pretraining "
              f"({len(self.files) * crops_per_volume} effective samples "
              f"with {crops_per_volume} crops/volume).")

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
                # Rotation can slightly change output size; pad/crop to guarantee fixed shape
                ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, 3)),
                RandGaussianNoised(keys=["image"], prob=0.5, std=0.05),
                RandGaussianSmoothd(keys=["image"], prob=0.5, sigma_x=(0.5, 1.5)),
                # Format for ResNet: (1, 224, 224, 3) -> (3, 224, 224)
                Lambdad(keys=["image"], func=_to_resnet_format),
            ]
        )

    def __len__(self):
        return len(self.files) * self.crops_per_volume

    def __getitem__(self, idx):
        # Map virtual index back to a real volume (different crops sampled randomly)
        file_idx = idx % len(self.files)
        volume = {"image": torch.load(self.files[file_idx], weights_only=False)}

        # Extract the base crop
        base_crop = self.extract_crop(volume)

        # Deep copy so each view gets independent random augmentations
        # (MONAI dict transforms mutate in place — without copies, view_k
        # would be a double-augmented version of view_q, not a separate view)
        view_q = self.moco_augs(copy.deepcopy(base_crop))["image"]
        view_k = self.moco_augs(copy.deepcopy(base_crop))["image"]

        # Return as a list of two views, plus a dummy target '0'
        return [view_q, view_k], 0
