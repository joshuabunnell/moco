# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MoCo v2 adapted for CT colonoscopy self-supervised pretraining."""

import glob
import os

import numpy as np
import torch

# The soft-tissue window every run so far trained on.  Applied when a crop is
# read, not baked into the cache, so a run can change it without re-running prep.
HU_WINDOW = (-150.0, 250.0)


def list_volumes(root):
    """Return every cached volume under *root*, sorted for a stable order."""
    return sorted(glob.glob(os.path.join(root, "**", "*.npy"), recursive=True))


def load_volume(path):
    """Open a cached ``(z, y, x)`` int16 HU volume without reading it into memory."""
    return np.load(path, mmap_mode="r")


def apply_window(hu, window=HU_WINDOW):
    """Clip HU to *window* and rescale to [0, 1] float32."""
    lo, hi = window
    return np.clip((np.asarray(hu, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)


def random_crop(volume, size=224, depth=3, window=HU_WINDOW):
    """Read one random 2.5D crop and return it in MONAI ``(1, x, y, z)`` layout.

    Only the crop's bytes are read from the memory-mapped volume.  Axes shorter
    than the crop are taken whole and left for ``ResizeWithPadOrCropd`` to pad,
    which is what ``RandSpatialCropd`` did on the old in-memory volumes.  Uses
    the global numpy RNG, which PyTorch reseeds in every DataLoader worker.
    """
    starts, lengths = [], []
    for extent, want in zip(volume.shape, (depth, size, size)):
        length = min(extent, want)
        starts.append(np.random.randint(0, extent - length + 1))
        lengths.append(length)
    (z0, y0, x0), (dz, dy, dx) = starts, lengths
    slab = apply_window(volume[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx], window)
    return torch.from_numpy(np.ascontiguousarray(slab.transpose(2, 1, 0)[None]))


def to_resnet_format(x):
    """Convert a MONAI 4D volume crop to a 3-channel 2D tensor for ResNet.

    MONAI dictionary transforms produce tensors with shape (1, H, W, D) where
    D is the depth (number of slices). This function squeezes the channel
    dimension and moves the depth axis to the channel position, yielding
    (D, H, W) — compatible with ResNet's expected (C, H, W) input when D=3.

    Args:
        x: Tensor of shape (1, H, W, D).

    Returns:
        Tensor of shape (D, H, W), typically (3, 224, 224).
    """
    return x[0].permute(2, 0, 1)
