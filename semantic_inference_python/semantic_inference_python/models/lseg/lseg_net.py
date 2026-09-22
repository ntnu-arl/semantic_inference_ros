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
#
# Code adapted from VLMAPS: https://github.com/vlmaps/vlmaps
# Original LICENSE:
# MIT License

# Copyright (c) 2023 Tom-Huang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""LSeg network implementation."""

from pathlib import Path
from typing import Optional

import clip
import numpy as np
import torch
import torch.nn as nn

from semantic_inference_python.models.lseg.lseg_blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
)
from semantic_inference_python.models.lseg.lseg_vit import forward_vit


class depthwise_clipseg_conv(nn.Module):
    """Depthwise convolution module for ClipSeg."""

    def __init__(self) -> None:
        """Initialize depthwise convolution."""
        super().__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def depthwise_clipseg(self, x: torch.Tensor, channels: int) -> torch.Tensor:
        """Depthwise convolution for ClipSeg.
        :param x: Input tensor
        :param channels: Number of channels
        :return: Output tensor
        """
        x = torch.cat(
            [self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1
        )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        :param x: Input tensor
        :return: Output tensor
        """
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    """Depthwise convolution module."""

    def __init__(self, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        """
        Initialize depthwise convolution.
        :param kernel_size: Kernel size
        :param stride: Stride
        :param padding: Padding
        """
        super().__init__()
        self.depthwise = nn.Conv2d(
            1, 1, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: Input tensor
        :return: Output tensor
        """
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    """Depthwise block with depthwise convolution."""

    def __init__(self, activation: str = "relu") -> None:
        """
        Initialize depthwise block.
        :param activation: Activation function
        """
        super().__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor, act: bool = True) -> torch.Tensor:
        """
        Forward pass.
        :param x: Input tensor
        :param act: Whether to apply activation
        :return: Output tensor
        """
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    """Bottleneck block with depthwise convolution."""

    def __init__(self, activation: str = "relu") -> None:
        """
        Initialize bottleneck block.
        :param activation: Activation function
        """
        super().__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor, act: bool = True) -> torch.Tensor:
        """
        Forward pass.
        :param x: Input tensor
        :param act: Whether to apply activation
        :return: Output tensor
        """
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x


class BaseModel(torch.nn.Module):
    """Base model class."""

    def load(self, path: Path) -> None:
        """
        Load model parameters from a file.
        :param path: Path to the model file
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


def _make_fusion_block(features: int, use_bn: bool) -> nn.Module:
    """
    Create a feature fusion block.
    :param features: Number of features
    :param use_bn: Whether to use batch normalization
    :return: FeatureFusionBlock instance
    """
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class LSeg(BaseModel):
    """LSeg network for semantic segmentation."""

    def __init__(
        self,
        head: nn.Module,
        labels: list[str],
        features: int = 256,
        backbone: str = "clip_vitl16_384",
        readout: str = "project",
        channels_last: bool = False,
        use_bn: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize LSeg network.
        :param head: Head module
        :param labels: List of class labels
        :param features: Number of features
        :param backbone: Backbone model
        :param readout: Readout type
        :param channels_last: Use channels last format
        :param use_bn: Use batch normalization
        """
        super().__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        assert backbone in hooks, f"Backbone {backbone} not supported."

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs["block_depth"]
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs["block_depth"]

        self.scratch.output_conv = head

        self.text = clip.tokenize(labels)

    def forward(self, x: torch.Tensor, labelset: str = "") -> torch.Tensor:
        """
        Forward pass.
        :param x: Input tensor
        :param labelset: Label set
        :return: Output tensor
        """
        text = self.text if labelset == "" else clip.tokenize(labelset)

        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        text = text.to(x.device)
        self.logit_scale = self.logit_scale.to(x.device)
        text_features = self.clip_pretrained.encode_text(text)

        image_features = self.scratch.head1(path_1)

        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        pixel_encoding = self.logit_scale * image_features.half()

        logits_per_image = pixel_encoding @ text_features.t()

        out = (
            logits_per_image.float()
            .view(imshape[0], imshape[2], imshape[3], -1)
            .permute(0, 3, 1, 2)
        )

        if self.arch_option in [1, 2]:
            for _ in range(self.block_depth - 1):
                out = self.scratch.head_block(out)
            out = self.scratch.head_block(out, False)

        out = self.scratch.output_conv(out)

        return out


class LSegNet(LSeg):
    """Network for semantic segmentation."""

    def __init__(
        self,
        labels: list[str],
        path: Optional[str] = None,
        scale_factor: int = 2,
        **kwargs,
    ) -> None:
        """
        Initialize LSeg network.
        :param labels: List of class labels
        :param path: Path to model weights
        :param scale_factor: Scale factor for input images
        """
        assert len(labels) > 0, "Labels list must not be empty."
        kwargs["use_bn"] = True
        head = nn.Sequential(
            Interpolate(scale_factor=scale_factor, mode="bilinear", align_corners=True),
        )

        super().__init__(head, labels, **kwargs)

        if path is not None:
            self.load(path)


class LSegEnc(BaseModel):
    """Encoder network for semantic segmentation."""

    def __init__(
        self,
        head: nn.Module,
        labels: Optional[list[str]] = None,
        features: int = 256,
        backbone: str = "clip_vitl16_384",
        readout: str = "project",
        channels_last: bool = False,
        use_bn: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize LSeg encoder.
        :param head: Head module
        :param features: Number of features
        :param backbone: Backbone model
        :param readout: Readout type
        :param channels_last: Use channels last format
        :param use_bn: Use batch normalization"""
        super().__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        assert backbone in hooks, f"Backbone {backbone} not supported."

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs["block_depth"]
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs["block_depth"]

        self.scratch.output_conv = head

        self.text = clip.tokenize(labels) if labels is not None else None

    def forward(
        self, x: torch.Tensor, refine: bool = False, labelset: str = ""
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        :param x: Input tensor
        :param refine: Whether to refine output
        :param labelset: Label set
        :return: Output tensor
        """
        text = self.text if labelset == "" else clip.tokenize(labelset)

        if self.channels_last:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        self.logit_scale = self.logit_scale.to(x.device)

        image_features = self.scratch.head1(path_1)

        if not refine:
            return image_features, None

        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        pixel_encoding = self.logit_scale * image_features.half()
        pixel_encoding = (
            pixel_encoding.float()
            .view(imshape[0], imshape[2], imshape[3], -1)
            .permute(0, 3, 1, 2)
        )
        pixel_encoding = self.scratch.output_conv(pixel_encoding)

        out = None
        if text is not None:
            text = text.to(x.device)
            text_features = self.clip_pretrained.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_per_image = pixel_encoding @ text_features.t()
            out = (
                logits_per_image.float()
                .view(imshape[0], imshape[2], imshape[3], -1)
                .permute(0, 3, 1, 2)
            )
            out = self.scratch.output_conv(out)

        return pixel_encoding, out


class LSegEncNet(LSegEnc):
    """Network for semantic segmentation."""

    def __init__(
        self,
        labels: Optional[list[str]] = None,
        path: Optional[Path] = None,
        scale_factor: float = 2,
        **kwargs,
    ) -> None:
        """
        Initialize LSeg network.
        :param labels: List of class labels
        :param path: Path to model weights
        :param scale_factor: Scale factor for input images
        """

        kwargs["use_bn"] = True
        head = nn.Sequential(
            Interpolate(scale_factor=scale_factor, mode="bilinear", align_corners=True),
        )

        super().__init__(head, labels, **kwargs)

        if path is not None:
            self.load(path)
