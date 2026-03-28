#!/usr/bin/env python
"""
Extract 2048-dim features from a pretrained MoCo checkpoint and visualize with UMAP.

Usage:
    python scripts/visualize_umap.py \
        --checkpoint /scratch/jpbunnel/moco-checkpoints/checkpoint_0199.pth.tar \
        --data /scratch/jpbunnel/cached-tensors \
        --output umap.png
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import umap
from monai.transforms.croppad.dictionary import RandSpatialCropd, ResizeWithPadOrCropd

# MONAI MetaTensor safe globals (same as CTMoCoDataset)
from monai.data.meta_tensor import MetaTensor
import torch.serialization

torch.serialization.add_safe_globals([np.ndarray, np.dtype, MetaTensor])


def _to_resnet_format(x):
    return x[0].permute(2, 0, 1)


def build_encoder(checkpoint_path):
    """Load the query encoder from a MoCo checkpoint, removing the MLP head."""
    # Build the same architecture used during training
    encoder = models.resnet50(num_classes=128)
    dim_mlp = encoder.fc.weight.shape[1]  # 2048
    encoder.fc = nn.Sequential(
        nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), encoder.fc
    )

    # Load weights from checkpoint (keys are prefixed with "module.encoder_q.")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    encoder_state = {}
    for k, v in state.items():
        if k.startswith("module.encoder_q."):
            encoder_state[k.replace("module.encoder_q.", "")] = v

    encoder.load_state_dict(encoder_state)

    # Replace MLP projection head with identity to get 2048-dim features
    encoder.fc = nn.Identity()
    encoder.eval()
    return encoder


def extract_features(encoder, data_dir, crops_per_volume=1, device="cuda"):
    """Extract one feature vector per volume (or multiple crops per volume)."""
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.pt"), recursive=True))
    print(f"Extracting features from {len(files)} volumes "
          f"({crops_per_volume} crop(s) each)...")

    crop_transform = RandSpatialCropd(
        keys=["image"], roi_size=(224, 224, 3), random_size=False
    )
    pad_crop = ResizeWithPadOrCropd(
        keys=["image"], spatial_size=(224, 224, 3)
    )

    encoder = encoder.to(device)
    all_features = []
    all_labels = []  # index of which file

    with torch.no_grad():
        for i, fpath in enumerate(files):
            volume = {"image": torch.load(fpath, weights_only=False)}
            for _ in range(crops_per_volume):
                crop = crop_transform(volume)
                crop = pad_crop(crop)
                # Convert to ResNet format: (1, 224, 224, 3) -> (3, 224, 224)
                img = _to_resnet_format(crop["image"]).unsqueeze(0).to(device)
                feat = encoder(img).squeeze().cpu().numpy()
                all_features.append(feat)
                all_labels.append(i)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(files)}")

    return np.stack(all_features), np.array(all_labels), files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to MoCo .pth.tar")
    parser.add_argument("--data", required=True, help="Path to cached .pt tensors")
    parser.add_argument("--output", default="umap.png", help="Output image path")
    parser.add_argument("--crops", default=1, type=int,
                        help="Crops per volume (1 is usually enough for UMAP)")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    encoder = build_encoder(args.checkpoint)
    features, labels, files = extract_features(
        encoder, args.data, crops_per_volume=args.crops, device=args.device
    )
    print(f"Feature matrix: {features.shape}")

    # Fit UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    embedding = reducer.fit_transform(features)

    # Color by parent directory (separates datasets if stored in subdirs)
    dir_names = [os.path.basename(os.path.dirname(f)) for f in files]
    unique_dirs = sorted(set(dir_names))
    dir_to_idx = {d: i for i, d in enumerate(unique_dirs)}
    colors = [dir_to_idx[dir_names[label]] for label in labels]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1],
        c=colors, cmap="tab10", s=5, alpha=0.7
    )
    if len(unique_dirs) <= 10:
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=plt.cm.tab10(i / max(len(unique_dirs) - 1, 1)),
                              markersize=8, label=d)
                   for i, d in enumerate(unique_dirs)]
        plt.legend(handles=handles, title="Source")
    plt.title("UMAP of MoCo Pretrained Features (2048-dim)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
