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


def _open_volume(path):
    """Open a cached volume and return (handle, shape, dtype) with the data next."""
    fh = open(path, "rb")
    version = np.lib.format.read_magic(fh)
    read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                   else np.lib.format.read_array_header_2_0)
    shape, fortran_order, dtype = read_header(fh)
    if fortran_order:
        fh.close()
        raise ValueError("%s is not C-ordered on disk; rebuild it with prep_data.py" % path)
    return fh, shape, dtype


def _read_planes(fh, shape, dtype, z0, nz_read, y0, ny_read):
    """Read rows ``y0:y0+ny_read`` of planes ``z0:z0+nz_read`` in one seek and read.

    The span runs from the first row wanted to the last, so it includes the rows
    between them on intermediate planes; those bytes are never indexed. Returns a
    ``(nz_read, ny, nx)`` array in which only the requested rows are populated.
    """
    _, ny, nx = shape
    count = (nz_read - 1) * ny * nx + ny_read * nx
    fh.seek(fh.tell() + (z0 * ny * nx + y0 * nx) * dtype.itemsize)
    span = np.fromfile(fh, dtype=dtype, count=count)

    planes = np.empty(nz_read * ny * nx, dtype=dtype)
    planes[y0 * nx:y0 * nx + count] = span
    return planes.reshape(nz_read, ny, nx)


def _jittered_window(window, jitter):
    """*window* with centre and width each moved by U(-jitter, jitter) HU (E3)."""
    if not jitter:
        return window
    lo, hi = window
    centre = (lo + hi) / 2.0 + np.random.uniform(-jitter, jitter)
    width = (hi - lo) + np.random.uniform(-jitter, jitter)
    return centre - width / 2.0, centre + width / 2.0


def _to_monai(slab, window):
    """Window a ``(z, y, x)`` slab and return it as MONAI ``(1, x, y, z)``."""
    return torch.from_numpy(
        np.ascontiguousarray(apply_window(slab, window).transpose(2, 1, 0)[None]))


def random_crop(path, size=224, depth=3, window=HU_WINDOW):
    """Read one random 2.5D crop of a cached volume, in MONAI ``(1, x, y, z)`` layout.

    Reads only the byte range from the crop's first row to its last with a single
    seek and read. On Sol's scratch filesystem that is ~65 ms from a cold file,
    against ~105 ms for the same bytes through a memory map, which pays a network
    round trip per page fault. Axes shorter than the crop are taken whole and left
    for ``ResizeWithPadOrCropd`` to pad. Uses the global numpy RNG, which PyTorch
    reseeds in every DataLoader worker.
    """
    fh, shape, dtype = _open_volume(path)
    with fh:
        nz, ny, nx = shape
        dz, dy, dx = min(nz, depth), min(ny, size), min(nx, size)
        z0 = np.random.randint(0, nz - dz + 1)
        y0 = np.random.randint(0, ny - dy + 1)
        x0 = np.random.randint(0, nx - dx + 1)
        planes = _read_planes(fh, shape, dtype, z0, dz, y0, dy)
    return _to_monai(planes[:, y0:y0 + dy, x0:x0 + dx], window)


def _shift_within(pos, span, limit, shift):
    """Offset *pos* by *shift* along an axis of length *limit*, staying in bounds.

    Tries the requested direction, then the opposite one, so a crop near an edge
    keeps the intended overlap instead of being pulled back toward the centre;
    only when neither fits does it clip, which raises the realised overlap.
    """
    for candidate in (pos + shift, pos - shift):
        if 0 <= candidate <= limit - span:
            return candidate
    return int(np.clip(pos + shift, 0, limit - span))


def _scaled_extent(start, span, limit, side):
    """Place a crop of *side* about the centre of ``start:start+span``, in bounds.

    Returns ``(lo, hi, pad)``: the slice actually read, and how many rows to pad
    on each side so the crop is *side* long when the axis is shorter than it.
    """
    if side >= limit:
        return 0, limit, ((side - limit) // 2, side - limit - (side - limit) // 2)
    lo = int(np.clip(start + span // 2 - side // 2, 0, limit - side))
    return lo, lo + side, (0, 0)


def _resize_inplane(slab, size):
    """Resize a ``(z, s, s)`` float slab to ``(z, size, size)``, bilinear with antialias."""
    t = torch.from_numpy(np.ascontiguousarray(slab))[:, None]
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                        align_corners=False, antialias=True)
    return t[:, 0].numpy()


def random_crop_pair(path, size=224, depth=3, window=HU_WINDOW,
                     overlap=(0.3, 0.7), z_shift=2, scale=None, window_jitter=0):
    """Read two overlapping 2.5D crops of one volume in a single read.

    The second crop is offset in-plane so the two share a fraction of their area
    drawn uniformly from *overlap*, split at random between the two axes, plus an
    independent z shift of up to *z_shift* slices. Both crops come from one span
    of the file, since they overlap, so a pair costs about what one crop costs.

    With *scale* ``(low, high)`` (E2), each view's in-plane side is drawn
    independently from it, in voxels (1 mm), about that view's centre from the
    draw above, and the windowed crop is resized to *size*. Positions are drawn
    first and identically, so the only difference from ``scale=None`` is scale.

    With *window_jitter* (E3), each view gets its own window, centre and width
    each moved by up to that many HU, drawn after every position and scale draw.
    """
    fh, shape, dtype = _open_volume(path)
    with fh:
        nz, ny, nx = shape
        dz, dy, dx = min(nz, depth), min(ny, size), min(nx, size)
        z0 = np.random.randint(0, nz - dz + 1)
        y0 = np.random.randint(0, ny - dy + 1)
        x0 = np.random.randint(0, nx - dx + 1)

        # Overlap is the product of the two axes' retained fractions, so split it
        # between them with a random exponent: f_y * f_x = f, either axis free to
        # carry most of the shift.
        fraction = np.random.uniform(*overlap)
        split = np.random.uniform()
        shift_y = int(round(dy * (1.0 - fraction ** split)))
        shift_x = int(round(dx * (1.0 - fraction ** (1.0 - split))))
        y1 = _shift_within(y0, dy, ny, np.random.choice([-1, 1]) * shift_y)
        x1 = _shift_within(x0, dx, nx, np.random.choice([-1, 1]) * shift_x)
        z1 = _shift_within(z0, dz, nz, np.random.randint(-z_shift, z_shift + 1))

        if scale is not None:
            return _read_scaled_pair(fh, shape, dtype, size, window, scale,
                                     (z0, y0, x0), (z1, y1, x1), (dz, dy, dx),
                                     window_jitter)

        z_lo, z_hi = min(z0, z1), max(z0, z1) + dz
        y_lo, y_hi = min(y0, y1), max(y0, y1) + dy
        planes = _read_planes(fh, shape, dtype, z_lo, z_hi - z_lo, y_lo, y_hi - y_lo)

    def crop(z, y, x):
        return planes[z - z_lo:z - z_lo + dz, y:y + dy, x:x + dx]

    w0, w1 = _jittered_window(window, window_jitter), _jittered_window(window, window_jitter)
    return _to_monai(crop(z0, y0, x0), w0), _to_monai(crop(z1, y1, x1), w1)


def _read_scaled_pair(fh, shape, dtype, size, window, scale, origin0, origin1, dims,
                      window_jitter=0):
    """The *scale* branch of ``random_crop_pair``: resize each view's own FOV to *size*."""
    nz, ny, nx = shape
    dz, dy, dx = dims
    views = []
    for z, y, x in (origin0, origin1):
        side = int(round(np.random.uniform(*scale)))
        views.append((z,) + _scaled_extent(y, dy, ny, side) + _scaled_extent(x, dx, nx, side))

    z_lo = min(v[0] for v in views)
    z_hi = max(v[0] for v in views) + dz
    y_lo = min(v[1] for v in views)
    y_hi = max(v[2] for v in views)
    planes = _read_planes(fh, shape, dtype, z_lo, z_hi - z_lo, y_lo, y_hi - y_lo)

    windows = [_jittered_window(window, window_jitter) for _ in views]
    out = []
    for (z, ya, yb, ypad, xa, xb, xpad), w in zip(views, windows):
        slab = apply_window(planes[z - z_lo:z - z_lo + dz, ya:yb, xa:xb], w)
        slab = np.pad(slab, ((0, 0), ypad, xpad))
        slab = _resize_inplane(slab, size)
        out.append(torch.from_numpy(np.ascontiguousarray(slab.transpose(2, 1, 0)[None])))
    return out[0], out[1]


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
