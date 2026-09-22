# BSD 3-Clause License

# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Various model utilities."""

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def panoptic_image(masks: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Create panoptic label image from panoptic masks and labels.

    :param masks: panoptic binary masks (NxHxW)
    :param labels: tensor of N labels
    :return: panoptic image (HxW)
    """
    ids = torch.arange(labels.shape[0]) + 1
    return (
        (masks * ids.to(masks.device).view(masks.shape[0], 1, 1))
        .sum(dim=0)
        .to(torch.int)
    )


def scale_masks(
    masks: np.ndarray, im0_shape: tuple[int, int], ratio_pad=None
) -> np.ndarray:
    """
    Rescale masks to original image size.

    Takes resized and padded masks and rescales them back
    to the original image dimensions, removing any padding
    that was applied during preprocessing.
    :param masks: Masks to be rescaled (HxW or HxWxN).
    :param im0_shape: Original image shape (height, width).
    :param ratio_pad: Optional tuple containing the scaling ratio
                      and padding applied during preprocessing.
    :return: Rescaled masks matching the original image size.
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(
            im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1]
        )  # gain  = old / new
        pad = (
            (im1_shape[1] - im0_shape[1] * gain) / 2,
            (im1_shape[0] - im0_shape[0] * gain) / 2,
        )  # wh padding
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(
            f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}'
        )
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def pad_image(
    img: torch.Tensor, mean: list[float], std: list[float], crop_size: int
) -> torch.Tensor:
    """Pad image to the desired crop size.
    :param img: Input image tensor (B,C,H,W)
    :param mean: Mean for each channel.
    :param std: Standard deviation for each channel.
    :param crop_size: Desired crop size.
    :return: Padded image tensor (B,C,H',W')
    """
    b, c, h, w = img.shape  # .size()
    assert c == 3
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b, c, h + padh, w + padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:, i, :, :] = F.pad(
            img[:, i, :, :], (0, padw, 0, padh), value=pad_values[i]
        )
    assert img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size
    return img_pad


def crop_image(img: torch.Tensor, h0: int, h1: int, w0: int, w1: int) -> torch.Tensor:
    """Crop image to the desired size.
    :param img: Input image tensor (B,C,H,W)
    :param h0: Starting height index.
    :param h1: Ending height index.
    :param w0: Starting width index.
    :param w1: Ending width index.
    :return: Cropped image tensor (B,C,h1-h0,w1-w0)
    """
    return img[:, :, h0:h1, w0:w1]
