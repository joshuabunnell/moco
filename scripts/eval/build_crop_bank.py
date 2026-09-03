"""Extract a fixed evaluation crop bank from the cached .pt volumes.

Phase 0 evaluation compares several encoders (random init, ImageNet init, one or
more MoCo checkpoints) against each other.  Re-deriving crops per encoder would
mean a full ~468 GB pass over ``tensors/`` every time and would feed each model
slightly different pixels.  This script does that pass once and writes a compact
uint8 bank so every later evaluation is a forward pass over ~5 GB of byte
identical input.

Crops are fully deterministic, so the bank can be rebuilt and compared across
runs:

* ``--crops K`` slabs per volume, taken at depth fractions ``(i + 0.5) / K``.
  Stratifying over depth (rather than sampling uniformly at random) gives every
  patient comparable anatomical coverage and lets supine/prone series be matched
  slab-for-slab by fraction.
* In-plane the crop is centred, not random.  Volumes are ~356 px wide against a
  224 px crop, so centring keeps the body core in frame and removes the run to
  run variance that a random offset would add to every reported metric.

Values are stored as uint8.  ``prep_data.py`` maps a 400 HU window onto [0, 1],
so a uint8 level is 1.56 HU, well under CT's ~10-20 HU noise floor.

Usage:
    python scripts/eval/build_crop_bank.py \\
        --tensor-dirs /scratch/$USER/moco/tensors/CT-COLONOGRAPHY \\
            /scratch/$USER/moco/tensors/Pediatric-CT-SEG \\
        --output-dir /scratch/$USER/moco/eval \\
        --crops 16 --workers 16
"""

import argparse
import csv
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.serialization
from monai.data.meta_tensor import MetaTensor

torch.serialization.add_safe_globals([np.ndarray, np.dtype, MetaTensor])

CROP_HW = 224
SLAB_D = 3


def center_crop_or_pad(plane, size):
    """Centre a ``(D, H, W)`` slab onto a ``(D, size, size)`` canvas.

    Crops the dimensions that are larger than *size* and zero-pads those that
    are smaller.  Zero is the HU window's floor (air), so padding introduces
    background rather than an artificial tissue value.
    """
    d, h, w = plane.shape
    out = np.zeros((d, size, size), dtype=plane.dtype)

    sy = max((h - size) // 2, 0)
    sx = max((w - size) // 2, 0)
    ch = min(h, size)
    cw = min(w, size)
    dy = max((size - h) // 2, 0)
    dx = max((size - w) // 2, 0)

    out[:, dy:dy + ch, dx:dx + cw] = plane[:, sy:sy + ch, sx:sx + cw]
    return out


def extract_volume(task):
    """Extract every slab for one volume.  Runs in a worker process.

    Args:
        task: Tuple of (row_index, filepath, crops_per_volume).

    Returns:
        Tuple of (row_index, crops, meta) where *crops* is a
        ``(K, 3, 224, 224)`` uint8 array and *meta* is a list of per-crop
        ``(z_frac, z_index, depth)`` tuples.  Returns ``(row_index, None, err)``
        if the volume could not be read.
    """
    idx, fpath, k = task
    try:
        vol = torch.load(fpath, weights_only=False)
        arr = vol[0] if vol.ndim == 4 else vol
        arr = np.asarray(arr, dtype=np.float32)
    except Exception as exc:
        return idx, None, "%s: %s" % (type(exc).__name__, exc)

    depth = arr.shape[2]
    crops = np.zeros((k, SLAB_D, CROP_HW, CROP_HW), dtype=np.uint8)
    meta = []

    for i in range(k):
        z_frac = (i + 0.5) / k
        z0 = int(round(z_frac * depth)) - SLAB_D // 2
        z0 = max(0, min(z0, max(depth - SLAB_D, 0)))
        slab = arr[:, :, z0:z0 + SLAB_D]

        # Depth-last (H, W, D) to channel-first (D, H, W), matching to_resnet_format.
        slab = np.transpose(slab, (2, 0, 1))
        if slab.shape[0] < SLAB_D:
            pad = np.zeros((SLAB_D - slab.shape[0],) + slab.shape[1:], dtype=slab.dtype)
            slab = np.concatenate([slab, pad], axis=0)

        slab = center_crop_or_pad(slab, CROP_HW)
        crops[i] = np.clip(np.rint(slab * 255.0), 0, 255).astype(np.uint8)
        meta.append((z_frac, z0, depth))

    return idx, crops, meta


def patient_id_of(filename):
    """Recover the patient ID from a cache filename.

    ``prep_data.series_filename`` builds ``<patient_id>_<4 hex>.pt``, so the ID
    is everything before the final underscore.  Only a fallback: prefer
    ``load_series_index`` where a manifest exists (see its docstring).
    """
    stem = os.path.basename(filename)[:-3]
    return stem.rsplit("_", 1)[0]


def load_series_index(tensor_dirs):
    """Map cached filename -> (patient_id, position) using each dir's manifest.csv.

    Two things come out of the manifest that the filename alone cannot give:

    * **The true patient ID.**  An older ``prep_data.py`` wrote an md5 as the
      ``patient_id`` for the 40 patients using TCIA's ``CTC-…`` subject naming,
      and it hashed per *series*, so those patients fragment into one bogus
      patient each.  The manifest's ``series_path`` still holds the real patient
      directory, which regroups them.
    * **Scan position.**  CT colonography scans each patient supine and prone;
      the series directory name records which.  Only the series dir is inspected,
      never the study dir, because study names like ``SupineandProneColon``
      describe the whole exam and would match both.

    Source paths in the manifest predate the current scratch layout and no longer
    resolve, but they are parsed as strings only.  Layout is uniformly
    ``.../<collection>/<patient>/<study>/<series>``.
    """
    index = {}
    for d in tensor_dirs:
        manifest = os.path.join(d, "manifest.csv")
        if not os.path.isfile(manifest):
            print("no manifest.csv in %s, falling back to filename parsing" % d)
            continue
        with open(manifest, newline="") as fh:
            for row in csv.DictReader(fh):
                parts = row["series_path"].rstrip("/").split("/")
                if len(parts) < 3:
                    continue
                series = parts[-1].lower()
                if "supi" in series:
                    position = "supine"
                elif "pron" in series:
                    position = "prone"
                elif "decub" in series:
                    position = "decubitus"
                else:
                    position = "unknown"
                index[row["filename"]] = (parts[-3], position)
    return index


def main():
    parser = argparse.ArgumentParser(description="Build the Phase 0 evaluation crop bank")
    parser.add_argument("--tensor-dirs", nargs="+", required=True, metavar="DIR",
                        help="One or more directories of cached .pt volumes")
    parser.add_argument("--output-dir", required=True, metavar="DIR",
                        help="Destination for bank.npy and bank_index.csv")
    parser.add_argument("--crops", type=int, default=16,
                        help="Slabs per volume, stratified over depth (default: 16)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel reader processes (default: 16)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N volumes (smoke test)")
    args = parser.parse_args()

    files = []
    for d in args.tensor_dirs:
        found = sorted(glob.glob(os.path.join(d, "**/*.pt"), recursive=True))
        print("%5d volumes in %s" % (len(found), d))
        files.extend(found)
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error("no .pt files found under the given --tensor-dirs")

    series_index = load_series_index(args.tensor_dirs)
    covered = sum(1 for f in files if os.path.basename(f) in series_index)
    print("%5d of %d volumes resolved against a manifest" % (covered, len(files)))

    os.makedirs(args.output_dir, exist_ok=True)
    bank_path = os.path.join(args.output_dir, "bank.npy")
    index_path = os.path.join(args.output_dir, "bank_index.csv")

    n_rows = len(files) * args.crops
    nbytes = n_rows * SLAB_D * CROP_HW * CROP_HW
    print("Building %d crops (%d volumes x %d) -> %s (%.1f GB)"
          % (n_rows, len(files), args.crops, bank_path, nbytes / 1e9))

    bank = np.lib.format.open_memmap(
        bank_path, mode="w+", dtype=np.uint8,
        shape=(n_rows, SLAB_D, CROP_HW, CROP_HW),
    )

    rows = [None] * len(files)
    failed = []
    tasks = [(i, f, args.crops) for i, f in enumerate(files)]
    t0 = time.time()

    with Pool(args.workers) as pool:
        for done, (idx, crops, meta) in enumerate(
            pool.imap_unordered(extract_volume, tasks, chunksize=1), start=1
        ):
            fpath = files[idx]
            if crops is None:
                failed.append((fpath, meta))
            else:
                bank[idx * args.crops:(idx + 1) * args.crops] = crops
                name = os.path.basename(fpath)
                pid, position = series_index.get(
                    name, (patient_id_of(fpath), "unknown"))
                rows[idx] = [
                    (idx * args.crops + i, name,
                     pid, os.path.basename(os.path.dirname(fpath)), position,
                     "%.5f" % zf, zi, dp)
                    for i, (zf, zi, dp) in enumerate(meta)
                ]

            if done % 100 == 0 or done == len(files):
                rate = done / (time.time() - t0)
                eta = (len(files) - done) / rate if rate else 0
                print("  %d/%d volumes  %.1f vol/s  ETA %.1f min"
                      % (done, len(files), rate, eta / 60), flush=True)

    bank.flush()

    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "filename", "patient_id", "collection",
                         "position", "z_frac", "z_index", "depth"])
        for group in rows:
            if group:
                writer.writerows(group)

    ok = sum(1 for r in rows if r)
    print("\nWrote %s and %s" % (bank_path, index_path))
    print("Volumes: %d ok, %d failed  (%.1f min)"
          % (ok, len(failed), (time.time() - t0) / 60))
    for fpath, err in failed[:10]:
        print("  FAILED %s -- %s" % (fpath, err))
    if len(failed) > 10:
        print("  ... and %d more" % (len(failed) - 10))

    # Rows for a failed volume stay zero-filled; the index omits them so the
    # evaluator never reads a blank crop as if it were data.
    if failed:
        print("NOTE: %d volumes are zero-filled in bank.npy and absent from the index."
              % len(failed))


if __name__ == "__main__":
    main()
