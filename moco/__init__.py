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
    volume = np.load(path, mmap_mode="r")
    if not volume.flags.c_contiguous:
        # Still readable, but a crop then touches most of the file: ~1.3 s instead
        # of milliseconds. Caches written before the 2026-09-16 fix are laid out so.
        raise ValueError("%s is not C-ordered on disk; rebuild it with prep_data.py" % path)
    return volume


def apply_window(hu, window=HU_WINDOW):
    """Clip HU to *window* and rescale to [0, 1] float32."""
    lo, hi = window
    return np.clip((np.asarray(hu, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)


def random_crop(path, size=224, depth=3, window=HU_WINDOW):
    """Read one random 2.5D crop of a cached volume, in MONAI ``(1, x, y, z)`` layout.

    Reads only the byte range from the crop's first row to its last with a single
    seek and read. On Sol's scratch filesystem that is ~65 ms from a cold file,
    against ~105 ms for the same bytes through a memory map, which pays a network
    round trip per page fault. Axes shorter than the crop are taken whole and left
    for ``ResizeWithPadOrCropd`` to pad. Uses the global numpy RNG, which PyTorch
    reseeds in every DataLoader worker.
    """
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        shape, fortran_order, dtype = read_header(fh)
        if fortran_order:
            raise ValueError("%s is not C-ordered on disk; rebuild it with prep_data.py" % path)
        nz, ny, nx = shape
        dz, dy, dx = min(nz, depth), min(ny, size), min(nx, size)
        z0 = np.random.randint(0, nz - dz + 1)
        y0 = np.random.randint(0, ny - dy + 1)
        x0 = np.random.randint(0, nx - dx + 1)
        first = z0 * ny * nx + y0 * nx
        count = (dz - 1) * ny * nx + dy * nx
        fh.seek(fh.tell() + first * dtype.itemsize)
        span = np.fromfile(fh, dtype=dtype, count=count)

    # Lay the span back into its planes; bytes outside it are never indexed.
    planes = np.empty(dz * ny * nx, dtype=dtype)
    planes[y0 * nx:y0 * nx + count] = span
    slab = planes.reshape(dz, ny, nx)[:, y0:y0 + dy, x0:x0 + dx]
    slab = apply_window(slab, window)
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
