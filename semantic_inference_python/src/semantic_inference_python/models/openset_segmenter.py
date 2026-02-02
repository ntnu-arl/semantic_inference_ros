# Portions of the following code and their modifications are originally from
# https://github.com/MIT-SPARK/semantic_inference and are licensed under the following
# license:
# -----------------------------------------------------------------------------
# BSD 3-Clause License

# Copyright (c) 2021-2024, Massachusetts Institute of Technology.

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
# --------------------------------------------------------------------------

# Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
# Technology All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree

"""Model to segment an image and encode segments with CLIP embeddings."""

from semantic_inference_python.config import Config, config_field
from semantic_inference_python.models.segment_refinement import SegmentRefinement
from semantic_inference_python.models.mask_functions import ConstantMask
from semantic_inference_python.models.patch_extractor import PatchExtractor
from semantic_inference_python.models.patch_extractor import get_image_preprocessor
from semantic_inference_python.models.patch_extractor import (
    default_normalization_parameters,
)
from semantic_inference_python.models.patch_extractor import center_crop
from semantic_inference_python.models.wrappers import vis_result_fast

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import supervision as sv

from typing import Any, List
from supervision.draw.color import ColorPalette
from dataclasses import dataclass, field
import time
import rospy


def _default_extractor():
    return PatchExtractor.Config(crop_padding=4)


def _map_opt(values, f):
    return {k: v if v is None else f(v) for k, v in values.items()}


def pool_masked_features(embeddings, masks, use_area=False):
    """
    Compute averaged features where masked elements are valid.

    Args:
        embeddings (torch.Tensor): Tensor of shape [1, H, W, L] where L is feature size
        masks (torch.Tensor): Tensor of shape [N, H, W] where N is number of masks

    Returns:
        torch.Tensor: Pooled features of shape [N, L]
    """
    target_size = (embeddings.size(1), embeddings.size(2))
    masks = masks.type(torch.uint8).unsqueeze(1)
    if use_area:
        downscaled = F.interpolate(
            masks.to(torch.float16), size=target_size, mode="area"
        ).squeeze()
        downscaled = (downscaled >= 0.5).to(torch.uint8)
    else:
        downscaled = F.interpolate(masks, size=target_size, mode="nearest").squeeze()

    downscaled = downscaled.unsqueeze(3)
    num_valid = torch.sum(downscaled, dim=(1, 2))
    valid = num_valid > 0

    features = downscaled * embeddings
    features = torch.sum(features, dim=(1, 2))
    num_valid[num_valid == 0] = 1
    features /= num_valid
    return features, valid


@dataclass
class Results:
    """Openset Segmentation Results."""

    masks: torch.Tensor
    panoptic_image: torch.Tensor
    boxes: torch.Tensor
    features: torch.Tensor
    boxed_patches: torch.Tensor
    masked_patches: torch.Tensor
    image_embedding: torch.Tensor
    labels: torch.Tensor = None
    panoptic_ids: torch.Tensor = None
    feature_image: torch.Tensor = None

    @property
    def instances(self):
        """Get instance image (if it exists)."""
        if self.masks is None:
            return None
        if self.masks.shape[0] == 0:
            return None

        np_masks = self.masks.numpy()
        img = np.zeros(np_masks[0].shape, dtype=np.uint16)
        for i in range(self.masks.shape[0]):
            # instance ids are 1-indexed
            img[np_masks[i, ...] > 0] = i + 1

        return img

    def get_labels(self):
        """Get labels."""
        if self.labels is None:
            return list(range(1, len(self.features) + 1))
        return self.labels.cpu().numpy().tolist()

    def get_ids(self):
        """Get ids."""
        if self.panoptic_ids is None:
            return list(range(1, len(self.features) + 1))
        return self.panoptic_ids.cpu().numpy().tolist()

    def cpu(self):
        """Move results to CPU."""
        return Results(
            masks=self.masks.cpu() if self.masks is not None else None,
            panoptic_image=self.panoptic_image.cpu()
            if self.panoptic_image is not None
            else None,
            boxes=self.boxes.cpu() if self.boxes is not None else None,
            features=self.features.cpu() if self.features is not None else None,
            boxed_patches=self.boxed_patches.cpu()
            if self.boxed_patches is not None
            else None,
            masked_patches=self.masked_patches.cpu()
            if self.masked_patches is not None
            else None,
            image_embedding=self.image_embedding.cpu()
            if self.image_embedding is not None
            else None,
            labels=self.labels.cpu() if self.labels is not None else None,
            panoptic_ids=self.panoptic_ids.cpu()
            if self.panoptic_ids is not None
            else None,
            feature_image=self.feature_image.cpu()
            if self.feature_image is not None
            else None,
        )

    def to(self, *args, **kwargs):
        """Forward to to all tensors."""
        return Results(
            masks=self.masks.to(*args, **kwargs) if self.masks is not None else None,
            panoptic_image=self.panoptic_image.to(*args, **kwargs)
            if self.panoptic_image is not None
            else None,
            boxes=self.boxes.to(*args, **kwargs) if self.boxes is not None else None,
            features=self.features.to(*args, **kwargs)
            if self.features is not None
            else None,
            boxed_patches=self.boxed_patches.to(*args, **kwargs)
            if self.boxed_patches is not None
            else None,
            masked_patches=self.masked_patches.to(*args, **kwargs)
            if self.masked_patches is not None
            else None,
            image_embedding=self.image_embedding.to(*args, **kwargs)
            if self.image_embedding is not None
            else None,
            labels=self.labels.to(*args, **kwargs) if self.labels is not None else None,
            panoptic_ids=self.panoptic_ids.to(*args, **kwargs)
            if self.panoptic_ids is not None
            else None,
            feature_image=self.feature_image.to(*args, **kwargs)
            if self.feature_image is not None
            else None,
        )


@dataclass
class OpensetSegmenterConfig(Config):
    """Main config for openset segmenter."""

    clip_model: Any = config_field("clip", default="clip")
    segmentation: Any = config_field("segmentation", default="fastsam")
    use_dense: bool = False
    dense_ratio: float = 0.9
    max_batch: int = 10
    text_embeddings: bool = False
    mask_embeddings: bool = True
    box_embeddings: bool = True
    text_embeddings_weight: float = 0.33
    box_embeddings_weight: float = 0.33
    mask_embeddings_weight: float = 0.33
    use_dense_area_interpolation: bool = False
    refinement: SegmentRefinement.Config = field(
        default_factory=SegmentRefinement.Config
    )
    patches: PatchExtractor.Config = field(default_factory=_default_extractor)
    object_labels: List[int] = field(default_factory=list)
    cuda: bool = True
    pub_detections: bool = False
    dense_representation_radius_m: float = 8
    min_depth: float = 0.05


class OpensetSegmenter(nn.Module):
    """Module to segment and encode an image."""

    def __init__(self, config, id_to_name=None):
        """Construct an openset segmenter."""
        super(OpensetSegmenter, self).__init__()
        # for detecting model device
        self._canary_param = nn.Parameter(torch.empty(0))

        self.config = config
        self.id_to_name = id_to_name
        self.segmenter = self.config.segmentation.create()
        self.segment_refinement = SegmentRefinement(config.refinement)
        self.encoder = self.config.clip_model.create()

        # previous code normalized after masking, so make sure we "normalize" 0
        # to be consistent
        mean, std = default_normalization_parameters()
        mask_value = -mean / std
        self.patch_extractor = PatchExtractor(
            self.encoder.input_size,
            config.patches,
            mask_function=ConstantMask(mask_value),
        )
        self.preprocess = get_image_preprocessor(self.encoder.input_size)

        self.dense_encoder = None
        if config.use_dense:
            self.dense_encoder = self.encoder.get_dense_encoder()

    def to_device(self, device):
        """Move model to device."""
        self._canary_param = nn.Parameter(torch.empty(0).to(device))
        self.segmenter.to_device(device)
        self.encoder.to_device(device)
        self.segment_refinement.to(device)
        self.patch_extractor.to(device)
        self.preprocess.to(device)

        if self.dense_encoder is not None:
            self.dense_encoder.to(device)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = OpensetSegmenterConfig()
        config.update(kwargs)
        return cls(config)

    @torch.no_grad()
    def segment(
        self,
        rgb_img,
        depth,
        colors: ColorPalette = ColorPalette.DEFAULT,
        is_rgb_order=True,
    ):
        """
        Segment image and compute language embeddings for each mask.

        Args:
            img (np.ndarry): uint8 image of shape (R, C, 3) in rgb order
            depth (np.ndarray): depth image of shape (R, C)
            colors (ColorPalette): Color palette for recoloring
            is_rgb_order (bool): whether the image is rgb order or not

        Returns:
            Encoded image
        """
        img = rgb_img if is_rgb_order else rgb_img[:, :, ::-1].copy()
        # TODO(nathan) tensor if we can get around FastSAM
        return self(img, depth, colors)

    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device

    def encode(self, img, depth, masks, boxes, labels, feature_image, panoptic_image):
        """Compute language embeddings for each segment."""
        start_time = time.time()
        img = img.permute((2, 0, 1))

        # Get only masks with labels associates to objects
        object_masks = []
        object_boxes = []
        object_labels = []

        use_depth = np.any(depth > 0)
        if self.config.segmentation.verbose:
            rospy.loginfo(
                f"[Open vocabulary node] CLIP initial preprocessing: {(time.time() - start_time) * 1000:.3f} ms"
            )
        start_time = time.time()
        for i, label in enumerate(labels):
            if use_depth:
                mean_depth = depth[masks[i].cpu().numpy()].mean()
                if not (
                    mean_depth > self.config.min_depth
                    and mean_depth < self.config.dense_representation_radius_m
                ):
                    continue
            if label.item() in self.config.object_labels:
                object_masks.append(masks[i])
                object_boxes.append(boxes[i])
                object_labels.append(label)
        if len(object_masks) == 0:
            return Results(
                None,
                panoptic_image.to(self.device),
                None,
                torch.empty(0),
                None,
                torch.empty(0),
                None,
                None,
                None,
            )

        object_boxes = torch.stack(object_boxes)
        object_masks = torch.stack(object_masks)
        if self.config.segmentation.verbose:
            rospy.loginfo(
                f"[Open vocabulary node] Mask filtering time: {(time.time() - start_time) * 1000:.3f} ms"
            )
        masks_to_use = object_masks if self.dense_encoder is None else None
        start_time = time.time()
        patch_boxes, patch_masks = self.patch_extractor(
            img, bboxes=object_boxes, masks=masks_to_use
        )
        if self.config.segmentation.verbose:
            rospy.loginfo(
                f"[Open vocabulary node] Patch extraction time: {(time.time() - start_time) * 1000:.3f} ms"
            )
        start_time = time.time()
        choice = torch.arange(self.config.max_batch).to(self.device)
        num_patches = self.config.max_batch
        if patch_boxes.shape[0] > self.config.max_batch:
            choice = torch.from_numpy(
                np.random.choice(
                    patch_boxes.shape[0], self.config.max_batch, replace=False
                )
            ).to(self.device)
            object_labels = (
                torch.stack(object_labels)[choice]
                if object_labels is not None
                else None
            )
            object_masks = object_masks[choice]
            object_boxes = object_boxes[choice]

        else:
            # Add extra placeholder patches if we have less than max_batch
            object_labels = (
                torch.stack(object_labels) if object_labels is not None else None
            )
            num_patches = patch_boxes.shape[0]
            patch_boxes = torch.cat(
                [
                    patch_boxes,
                    torch.zeros(
                        (
                            self.config.max_batch - patch_boxes.shape[0],
                            *patch_boxes.shape[1:],
                        )
                    ).to(self.device),
                ]
            )
            if patch_masks is not None:
                patch_masks = torch.cat(
                    [
                        patch_masks,
                        torch.zeros(
                            (
                                self.config.max_batch - patch_masks.shape[0],
                                *patch_masks.shape[1:],
                            )
                        ).to(self.device),
                    ]
                )
        patch_boxes = patch_boxes[choice]
        patch_masks = patch_masks[choice]

        features = torch.zeros(num_patches, self.encoder.output_dim_visual).to(
            self.device
        )
        if self.config.segmentation.verbose:
            rospy.loginfo(
                f"[Open vocabulary node] Patch selection time: {(time.time() - start_time) * 1000:.3f} ms"
            )
        start_time = time.time()
        img = self.preprocess(img)
        if self.config.segmentation.verbose:
            rospy.loginfo(
                f"[Open vocabulary node] Image preprocessing time: {(time.time() - start_time) * 1000:.3f} ms"
            )

        # dense clip doesn't use center-crop, so we have to apply it ourselves
        start_time = time.time()
        clip_img = center_crop(img, self.encoder.input_size)
        if self.config.segmentation.verbose:
            rospy.loginfo(
                f"[Open vocabulary node] CLIP center crop time: {(time.time() - start_time) * 1000:.3f} ms"
            )
        start_time = time.time()
        if self.dense_encoder is None:
            assert patch_masks is not None
            result = self.encoder(torch.cat([patch_boxes, clip_img.unsqueeze(0)]))
            features += result[:num_patches] * self.config.box_embeddings_weight
            img_embedding = torch.squeeze(result[-1])
            if self.config.text_embeddings:
                class_names = [self.id_to_name[label.item()] for label in object_labels]
                features += (
                    self.encoder.embed_text(class_names)[:num_patches]
                    * self.config.text_embeddings_weight
                )
            if self.config.mask_embeddings:
                features += (
                    self.encoder(patch_masks)[:num_patches]
                    * self.config.mask_embeddings_weight
                )
        else:
            dense_embeddings = self.dense_encoder(img)
            dense_features, valid = pool_masked_features(
                dense_embeddings,
                object_masks,
                use_area=self.config.use_dense_area_interpolation,
            )
            ratios = self.config.dense_ratio * valid
            features = (1.0 - ratios) * self.encoder(
                patch_boxes
            ) + ratios * dense_features
        if self.config.segmentation.verbose:
            rospy.loginfo(
                f"[Open vocabulary node] CLIP model inference time: {(time.time() - start_time) * 1000:.3f} ms"
            )
        panoptic_ids = (
            torch.Tensor(
                [panoptic_image[mask.to(torch.bool)][0] for mask in object_masks]
            )
            .to(torch.int)
            .to(self.device)
        )
        res = Results(
            object_masks.detach().cpu(),
            panoptic_image.detach().cpu(),
            object_boxes.detach().cpu(),
            features.detach().cpu(),
            patch_boxes.detach().cpu(),
            patch_masks.detach().cpu(),
            img_embedding.detach().cpu(),
            object_labels.detach().cpu() if labels is not None else None,
            panoptic_ids.detach().cpu(),
            feature_image.detach().cpu() if feature_image is not None else None,
        )
        return res

    def forward(self, rgb_img, depth, colors: ColorPalette = ColorPalette.DEFAULT):
        """
        Segment image and compute language embeddings for each mask.

        Args:
            img (np.ndarray): uint8 image of shape (R, C, 3) in rgb order
            depth (np.ndarray): depth image of shape (R, C)
            colors (ColorPalette): Color palette for recoloring

        Returns:
            Encoded image
        """
        start_time = time.time()
        masks, boxes, labels, feature_image, panoptic_image, confs = self.segmenter(
            rgb_img, device=self.device
        )
        if self.config.segmentation.verbose:
            rospy.loginfo(
                f"[Open vocabulary node] Segmentation inference time: {(time.time() - start_time) * 1000:.3f} ms"
            )

        if masks is None:
            return (
                Results(
                    None,
                    torch.zeros((rgb_img.shape[0], rgb_img.shape[1])).to(int),
                    None,
                    torch.empty(0),
                    None,
                    torch.empty(0),
                    None,
                    None,
                    None,
                ),
                None,
            )
        start_time = time.time()
        (
            masks,
            boxes,
            labels,
            feature_image,
            panoptic_image,
            confs,
        ) = self.segment_refinement(masks, boxes, labels, feature_image, confs)
        if self.config.segmentation.verbose:
            rospy.loginfo(
                f"[Open vocabulary node] Segmentation refinement time: {(time.time() - start_time) * 1000:.3f} ms"
            )

        vis_img = None
        if self.config.pub_detections:
            labels_np = labels.cpu().numpy()
            masks_np = masks.cpu().numpy()
            xyxy_np = boxes.cpu().numpy()
            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confs,
                class_id=labels_np,
                mask=masks_np,
            )
            vis_img, _ = vis_result_fast(
                rgb_img, curr_det, self.segmenter.config.get_labels(), color=colors
            )

        img = torch.from_numpy(rgb_img).to(self.device)
        start_time = time.time()
        res = self.encode(
            img, depth, masks, boxes, labels, feature_image, panoptic_image
        )
        if self.config.segmentation.verbose:
            rospy.loginfo(
                f"[Open vocabulary node] Total CLIP inference time: {(time.time() - start_time) * 1000:.3f} ms"
            )

        return (
            res,
            vis_img,
        )
