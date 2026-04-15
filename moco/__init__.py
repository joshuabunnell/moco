# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MoCo v2 adapted for CT colonoscopy self-supervised pretraining."""


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
