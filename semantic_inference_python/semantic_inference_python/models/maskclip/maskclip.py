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
# -------------------------------------------------------------------------------
# Modified version of CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology

# Copyright (c) OpenMMLab. All rights reserved.
# Modified version of the original MaskCLIP code:
# https://github.com/chongzhou96/MaskCLIP/tree/master
# -------------------------------------------------------------------------------
"""MaskCLIP model implementation."""

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_inference_python.misc import Logger


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning and size is not None and align_corners:
        input_h, input_w = tuple(int(x) for x in input.shape[2:])
        output_h, output_w = tuple(int(x) for x in size)
        if (output_h > input_h or output_w > input_w) and (
            (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
            and (output_h - 1) % (input_h - 1)
            and (output_w - 1) % (input_w - 1)
        ):
            Logger.info(
                f"When align_corners={align_corners}, "
                "the output would more aligned if "
                f"input size {(input_h, input_w)} is `x+1` and "
                f"out size {(output_h, output_w)} is `nx+1`"
            )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MaskClip(nn.Module):
    """MaskCLIP model."""

    def __init__(
        self, clip_model: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"
    ) -> None:
        """
        Initialize MaskCLIP model.
        :param clip_model: CLIP model name
        :param pretrained: Pretrained weights name
        """
        super().__init__()

        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=pretrained
        )
        model.eval()

        self.patch_size = model.visual.patch_size[0]
        self.img_size = tuple([preprocess.transforms[0].size] * 2)
        self.clip_T = preprocess.transforms[-1]

        self.hook_features = {}
        self.backbone = model

        def hook_fn_forward(module, input, output):
            self.hook_features["v"] = output

        self.backbone.visual.transformer.resblocks[-2].register_forward_hook(
            hook_fn_forward
        )
        self._positional_embd = nn.Parameter(
            self.backbone.visual.positional_embedding.data.clone()
        )

        self.proj = nn.Conv2d(
            model.visual.proj.shape[0], model.visual.output_dim, 1, bias=False
        )
        self.proj.weight = nn.Parameter(model.visual.proj.t()[:, :, None, None])

    @torch.no_grad()
    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        :param inputs: Input image tensor
        :return: Extracted feature tensor
        """
        pos_embed = self.backbone.visual.positional_embedding

        B, _, H, W = inputs.shape
        hw_shape = (H // self.patch_size, W // self.patch_size)
        x_len, pos_len = hw_shape[0] * hw_shape[1], pos_embed.shape[0]

        if x_len != pos_len:
            if (
                pos_len
                == (self.img_size[0] // self.patch_size)
                * (self.img_size[1] // self.patch_size)
                + 1
            ):
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(f"{x_len}, {pos_len}")

            self.backbone.visual.positional_embedding.data = self.resize_pos_embed(
                self._positional_embd[None], hw_shape, (pos_h, pos_w), "bicubic"
            )[0]

        _ = self.backbone(inputs)
        v = self.hook_features["v"]
        v = v.permute(1, 0, 2)
        v = self.extract_v(v, self.backbone.visual.transformer.resblocks[-1]).permute(
            1, 0, 2
        )
        assert v.shape[-1] == 768, v.shape
        v = self.backbone.visual.ln_post(v)
        v = v[:, 1:]
        v = v.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()

        self.backbone.visual.positional_embedding.data = self._positional_embd
        return v

    def extract_v(self, x: torch.Tensor, block: torch.nn.Module) -> torch.Tensor:
        """
        Extract features from the vision transformer block.
        :param x: Input tensor
        :param block: Vision transformer block
        :return: Output tensor
        """
        y = block.ln_1(x)
        y = torch.nn.functional.linear(
            y, block.attn.in_proj_weight, block.attn.in_proj_bias
        )
        B, N, C = y.shape
        y = y.view(B, N, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * B, N, C // 3)
        y = F.linear(y, block.attn.out_proj.weight, block.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v += block.mlp(block.ln_2(v))
        return v

    @staticmethod
    def resize_pos_embed(
        pos_embed: torch.Tensor,
        input_shape: tuple,
        pos_shape: tuple,
        mode: str = "bicubic",
    ) -> torch.Tensor:
        """
        Resize pos_embed weights.
        :param pos_embed: Position embedding weights.
        :param input_shape: Tuple for (downsampled input image height,
                            downsampled input image width).
        :param pos_shape: The resolution of downsampled origin training image.
        :param mode: Algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'``. Default: ``'nearest'``
        :return: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, "shape of pos_embed must be [B, L, C]"
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w) :]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]
        ).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shape, align_corners=False, mode=mode
        )
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, inputs: torch.Tensor, interpolate: bool = False) -> torch.Tensor:
        """
        Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
        :param inputs: Input image tensor
        :param interpolate: Whether to interpolate the output to the input size
        :return: Output feature tensor
        """
        h, w = inputs.shape[-2:]
        inputs = self.clip_T(inputs)
        x = self.extract_feat(inputs)
        feats = self.proj(x)
        if interpolate:
            feats = F.interpolate(
                feats, size=(h, w), mode="bilinear", align_corners=False
            )
        return feats
