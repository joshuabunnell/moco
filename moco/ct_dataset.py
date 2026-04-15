"""Dataset for MoCo v2 pretraining on cached CT volume tensors.

Loads preprocessed 3D CT volumes (.pt files) produced by ``scripts/prep_data.py``,
extracts random 2.5D crops (224x224x3), and generates two independently augmented
views for contrastive learning.  Each volume is sampled multiple times per epoch
via the ``crops_per_volume`` multiplier so that the effective dataset size exceeds
the number of physical volumes.

Augmentations are restricted to those that preserve Hounsfield Unit (HU) semantics:
spatial transforms (flips, small rotations) and mild intensity perturbations
(Gaussian noise, Gaussian blur).  Color jitter is intentionally excluded because
HU values encode physical tissue density.
"""

import copy
import glob
import os

import numpy as np
import torch
import torch.serialization
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import RandSpatialCropd, ResizeWithPadOrCropd
from monai.transforms.intensity.dictionary import (
    RandGaussianNoised,
    RandGaussianSmoothd,
)
from monai.transforms.spatial.dictionary import RandFlipd, RandRotated
from monai.transforms.utility.dictionary import Lambdad
from torch.utils.data import Dataset

from moco import to_resnet_format

# MONAI MetaTensor serializes numpy affine/metadata alongside the tensor.
# PyTorch 2.6+ validates globals even with weights_only=False, so we must
# allowlist the types used by MONAI before any torch.load call.
from monai.data.meta_tensor import MetaTensor

torch.serialization.add_safe_globals([
    np.ndarray,
    np.dtype,
    MetaTensor,
])


class CTMoCoDataset(Dataset):
    """PyTorch Dataset that yields contrastive pairs from cached CT volumes.

    Args:
        data_dir: Root directory containing preprocessed ``.pt`` tensor files.
            Files are discovered recursively via ``**/*.pt``.
        crops_per_volume: Number of random crops to draw from each volume per
            epoch.  Multiplies the effective dataset length so the model sees
            diverse spatial regions without reloading new volumes.
    """

    def __init__(self, data_dir, crops_per_volume=20):
        self.files = glob.glob(os.path.join(data_dir, "**/*.pt"), recursive=True)
        self.crops_per_volume = crops_per_volume
        print(f"Found {len(self.files)} 3D volumes for Pretraining "
              f"({len(self.files) * crops_per_volume} effective samples "
              f"with {crops_per_volume} crops/volume).")

        # Base extraction: random 2.5D crop (224x224 axial plane x 3 adjacent slices)
        self.extract_crop = RandSpatialCropd(
            keys=["image"], roi_size=(224, 224, 3), random_size=False
        )

        # MoCo augmentations — only transforms that preserve HU semantics
        self.moco_augs = Compose(
            [
                # Spatial flips: valid because colon anatomy is orientation-agnostic
                # for polyp detection (polyps can appear in any orientation)
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),  # up/down
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),  # left/right
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),  # depth reversal

                # Small in-plane rotation (~15 deg) in the axial (H-W) plane
                RandRotated(keys=["image"], prob=0.5, range_z=0.26),
                # Rotation can slightly change output size; pad/crop back to fixed shape
                ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, 3)),

                # Mild intensity perturbations that don't destroy HU relationships
                RandGaussianNoised(keys=["image"], prob=0.5, std=0.05),
                RandGaussianSmoothd(keys=["image"], prob=0.5, sigma_x=(0.5, 1.5)),

                # Convert from MONAI format (1, H, W, D) to ResNet format (D, H, W)
                Lambdad(keys=["image"], func=to_resnet_format),
            ]
        )

    def __len__(self):
        return len(self.files) * self.crops_per_volume

    def __getitem__(self, idx):
        """Load a volume and return two independently augmented views.

        Args:
            idx: Virtual index. Mapped back to a physical volume via modulo so
                that the same volume is cropped at different random locations.

        Returns:
            Tuple of ([view_q, view_k], 0) where each view is a (3, 224, 224)
            tensor and 0 is a dummy label (MoCo is self-supervised).
        """
        file_idx = idx % len(self.files)
        volume = {"image": torch.load(self.files[file_idx], weights_only=False)}

        base_crop = self.extract_crop(volume)

        # Deep copy so each view gets independent random augmentations.
        # MONAI dict transforms mutate in place — without copies, view_k
        # would be a double-augmented version of view_q, not a separate view.
        view_q = self.moco_augs(copy.deepcopy(base_crop))["image"]
        view_k = self.moco_augs(copy.deepcopy(base_crop))["image"]

        return [view_q, view_k], 0
