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
"""Model to segment an image and encode segments with CLIP embeddings."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import supervision as sv
import torch
import torch.nn.functional as F
import yaml
from spark_config import Config, config_field
from supervision.draw.color import ColorPalette
from torch import nn

from semantic_inference_python.models.detection_visualizer import DetectionVisualizer
from semantic_inference_python.models.mask_functions import ConstantMask
from semantic_inference_python.models.patch_extractor import (
    PatchExtractor,
    center_crop,
    default_normalization_parameters,
    get_image_preprocessor,
)
from semantic_inference_python.models.segment_refinement import SegmentRefinement

# from semantic_inference_python.misc import Logger


def _default_extractor():
    return PatchExtractor.Config(crop_padding=4)


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
class LabeledResults:
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
        return LabeledResults(
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
        return LabeledResults(
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
class OpensetSegmenterLabeledConfig(Config):
    """Main config for openset segmenter."""

    clip_model: Any = config_field("clip", default="open_clip")
    segmentation: Any = config_field("labeled_segmentation", default="yoloe")
    pixelwise_clip_model: Any = config_field("pixelwise_clip", default="lseg")
    use_pixelwise_embeddings: bool = False
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
    object_labels_path: str = field(default_factory=str)
    cuda: bool = True
    max_depth: float = 8
    min_depth: float = 0.05


class OpensetSegmenterLabeled(nn.Module):
    """Module to segment and encode an image."""

    def __init__(
        self,
        config: OpensetSegmenterLabeledConfig,
        colors: ColorPalette = ColorPalette.DEFAULT,
    ) -> None:
        """Construct an openset segmenter.
        :param config: Configuration for the openset segmenter
        :param colors: Color palette for visualization
        """
        super().__init__()
        # for detecting model device
        self._canary_param = nn.Parameter(torch.empty(0))

        self.config = config
        self.segmenter = self.config.segmentation.create()
        self.segment_refinement = SegmentRefinement(config.refinement)
        self.encoder = self.config.clip_model.create()
        self.pixelwise_encoder = self.config.pixelwise_clip_model.create()

        # Load object labels
        with open(self.config.object_labels_path) as f:
            self.object_labels = yaml.safe_load(f)["object_labels"]

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

        self.visualizer = DetectionVisualizer(
            colors, self.segmenter.config.get_labels()
        )

    def to_device(self, device):
        """Move model to device."""
        self._canary_param = nn.Parameter(torch.empty(0).to(device))
        self.segmenter.to_device(device)
        self.encoder.to_device(device)
        self.pixelwise_encoder.to_device(device)
        self.segment_refinement.to(device)
        self.patch_extractor.to(device)
        self.preprocess.to(device)

        if self.dense_encoder is not None:
            self.dense_encoder.to(device)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = OpensetSegmenterLabeledConfig()
        config.update(kwargs)
        return cls(config)

    @torch.no_grad()
    def segment(
        self,
        rgb_img: np.ndarray,
        depth_img: np.ndarray = None,
        is_rgb_order: bool = True,
    ) -> LabeledResults:
        """
        Segment image and compute language embeddings for each mask.
        :param rgb_img: uint8 image of shape (R, C, 3) in rgb order
        :param depth_img: float32 image of shape (R, C) with depth in meters
        :param is_rgb_order: Whether the input image is in RGB order
        :return: Encoded image
        """
        img = rgb_img if is_rgb_order else rgb_img[:, :, ::-1].copy()
        return self(img, depth_img)

    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device

    def _get_image_embedding(self, img: torch.Tensor) -> torch.Tensor:
        """Get image embedding from the encoder.
        :param img: Image tensor
        :return: Image embedding
        """
        pimg = self.preprocess(img)
        clip_img = center_crop(pimg, self.encoder.input_size).unsqueeze(0)
        return torch.squeeze(self.encoder(clip_img))

    def _get_pixelwise_embeddings(self, img: torch.Tensor) -> torch.Tensor:
        """Get pixelwise embeddings from the pixelwise encoder.
        :param img: Image tensor
        :return: Pixelwise embeddings
        """
        return (
            self.pixelwise_encoder.encode(img.cpu().numpy())
            .squeeze(0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
        )

    def encode(
        self,
        img: torch.Tensor,
        depth: np.ndarray,
        masks: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        panoptic_image: torch.Tensor,
    ) -> LabeledResults:
        """Compute language embeddings for each segment.
        :param img: Image tensor
        :param depth: Depth image
        :param masks: Masks for segments
        :param boxes: Bounding boxes for segments
        :param labels: Labels for segments
        :param feature_image: Feature image
        :param panoptic_image: Panoptic segmentation image
        :return: LabeledResults with encoded features
        """
        pimg = img.permute((2, 0, 1))

        # Get only masks with labels associated to objects and within depth range
        object_masks = []
        object_boxes = []
        object_labels = []

        use_depth = np.any(depth > 0)
        for i, label in enumerate(labels):
            if use_depth:
                mean_depth = depth[masks[i].cpu().numpy()].mean()
                if not (
                    mean_depth > self.config.min_depth
                    and mean_depth < self.config.max_depth
                ):
                    continue
            if label.item() in self.object_labels:
                object_masks.append(masks[i])
                object_boxes.append(boxes[i])
                object_labels.append(label)
        if len(object_masks) == 0:
            return LabeledResults(
                masks=None,
                panoptic_image=panoptic_image.to(self.device),
                boxes=None,
                features=torch.empty(0),
                boxed_patches=None,
                masked_patches=torch.empty(0),
                image_embedding=self._get_image_embedding(pimg).detach().cpu(),
                labels=None,
                panoptic_ids=None,
                feature_image=self._get_pixelwise_embeddings(img)
                if self.config.use_pixelwise_embeddings
                else None,
            )

        object_boxes = torch.stack(object_boxes)
        object_masks = torch.stack(object_masks)

        # Extract patches
        masks_to_use = object_masks if self.dense_encoder is None else None
        patch_boxes, patch_masks = self.patch_extractor(
            pimg, bboxes=object_boxes, masks=masks_to_use
        )
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

        # Encode patches
        features = torch.zeros(num_patches, self.encoder.output_dim_visual).to(
            self.device
        )
        pimg = self.preprocess(pimg)

        # dense clip doesn't use center-crop, so we have to apply it ourselves
        clip_img = center_crop(pimg, self.encoder.input_size)
        if self.dense_encoder is None:
            assert patch_masks is not None
            result = self.encoder(torch.cat([patch_boxes, clip_img.unsqueeze(0)]))
            features += result[:num_patches] * self.config.box_embeddings_weight
            img_embedding = torch.squeeze(result[-1])
            if self.config.text_embeddings:
                class_names = [
                    self.segmenter.config.get_labels()[label.item()]
                    for label in object_labels
                ]
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
            dense_embeddings = self.dense_encoder(pimg)
            dense_features, valid = pool_masked_features(
                dense_embeddings,
                object_masks,
                use_area=self.config.use_dense_area_interpolation,
            )
            ratios = self.config.dense_ratio * valid
            features = (1.0 - ratios) * self.encoder(
                patch_boxes
            ) + ratios * dense_features

        # Get panoptic ids
        panoptic_ids = (
            torch.Tensor(
                [panoptic_image[mask.to(torch.bool)][0] for mask in object_masks]
            )
            .to(torch.int)
            .to(self.device)
        )
        # Keep only panoptic_ids in panoptic_image
        mask = torch.isin(panoptic_image, panoptic_ids)
        panoptic_image = panoptic_image * mask

        # Feature image
        feature_image = None
        if self.config.use_pixelwise_embeddings:
            feature_image = self._get_pixelwise_embeddings(img)

        res = LabeledResults(
            masks=object_masks.detach().cpu(),
            panoptic_image=panoptic_image.detach().cpu(),
            boxes=object_boxes.detach().cpu(),
            features=features.detach().cpu(),
            boxed_patches=patch_boxes.detach().cpu(),
            masked_patches=patch_masks.detach().cpu(),
            image_embedding=img_embedding.detach().cpu(),
            labels=object_labels.detach().cpu() if labels is not None else None,
            panoptic_ids=panoptic_ids.detach().cpu(),
            feature_image=feature_image if feature_image is not None else None,
        )
        return res

    def forward(
        self, rgb_img: np.ndarray, depth_img: np.ndarray = None
    ) -> LabeledResults:
        """
        Segment image and compute language embeddings for each mask.
        :param rgb_img: uint8 image of shape (R, C, 3) in rgb order
        :param depth_img: float32 image of shape (R, C) with depth in meters
        :return: Encoded image
        """
        masks, boxes, labels, panoptic_image, confs = self.segmenter(
            rgb_img, device=self.device
        )
        if masks is None:
            return (
                LabeledResults(
                    masks=None,
                    panoptic_image=torch.zeros((rgb_img.shape[0], rgb_img.shape[1])).to(
                        int
                    ),
                    boxes=None,
                    features=torch.empty(0),
                    boxed_patches=None,
                    masked_patches=torch.empty(0),
                    image_embedding=self._get_image_embedding(
                        torch.from_numpy(rgb_img).to(self.device).permute((2, 0, 1))
                    )
                    .detach()
                    .cpu(),
                    labels=None,
                    panoptic_ids=None,
                    feature_image=self._get_pixelwise_embeddings(
                        torch.from_numpy(rgb_img).to(self.device)
                    )
                    if self.config.use_pixelwise_embeddings
                    else None,
                ),
                None,
            )

        (
            masks,
            boxes,
            labels,
            panoptic_image,
            confs,
        ) = self.segment_refinement(masks, boxes, labels, confs)

        # Visualization image
        curr_det = sv.Detections(
            xyxy=boxes.cpu().numpy(),
            mask=masks.cpu().numpy(),
            class_id=labels.cpu().numpy(),
            confidence=confs,
        )
        det_image = self.visualizer.annotate(rgb_img, curr_det)
        # Encode masks
        img = torch.from_numpy(rgb_img).to(self.device)
        res = self.encode(img, depth_img, masks, boxes, labels, panoptic_image)
        return res, det_image
