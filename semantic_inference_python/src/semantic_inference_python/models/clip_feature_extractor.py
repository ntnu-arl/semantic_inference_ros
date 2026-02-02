"""Model to extract CLIP embeddings from an image and its segmentation."""

from semantic_inference_python.config import Config, config_field
from semantic_inference_python.models.mask_functions import ConstantMask
from semantic_inference_python.models.patch_extractor import PatchExtractor
from semantic_inference_python.models.openset_segmenter import (
    Results,
    _default_extractor,
    pool_masked_features,
)
from semantic_inference_python.models.patch_extractor import get_image_preprocessor
from semantic_inference_python.models.patch_extractor import (
    default_normalization_parameters,
)
from semantic_inference_python.models.patch_extractor import center_crop

import torch
from torch import nn
import torchvision
import numpy as np

from typing import Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class CLIPFeaturesExtractorConfig(Config):
    """Main configuration for CLIP feature extractor."""

    clip_model: Any = config_field("clip", default="clip")
    use_dense: bool = False
    dense_ratio: float = 0.9
    max_batch: int = 10
    box_embeddings: bool = True
    text_embeddings: bool = False
    mask_embeddings: bool = True
    use_dense_area_interpolation: bool = False
    min_area: int = 10
    min_depth: float = 0.05
    dense_representation_radius_m: float = 10
    patches: PatchExtractor.Config = field(default_factory=_default_extractor)
    object_labels: List[int] = field(default_factory=list)
    cuda: bool = True


class CLIPFeaturesExtractor(nn.Module):
    """Module to extract CLIP embeddings from an image and its segmentation."""

    def __init__(self, config, colors_to_labels=None, id_to_name=None):
        """Construct a CLIP feature extractor."""
        super(CLIPFeaturesExtractor, self).__init__()
        # for detecting model device
        self._canary_param = nn.Parameter(torch.empty(0))

        self.config = config
        self.colors_to_labels = colors_to_labels
        self.id_to_name = id_to_name
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

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = CLIPFeaturesExtractorConfig()
        config.update(kwargs)
        return cls(config)

    @property
    def device(self):
        """Get device of model."""
        return self._canary_param.device

    def encode(self, img, masks, boxes, labels, panoptic_img, depth_img):
        """Compute language embeddings for each segment."""
        img = img.permute((2, 0, 1))

        # Get only masks with labels associates to objects
        object_masks = []
        object_boxes = []
        object_labels = []
        use_depth = np.any(depth_img > 0)
        areas = torchvision.ops.box_area(boxes)
        for i, label in enumerate(labels):
            if use_depth:
                mean_depth = depth_img[masks[i].cpu().numpy()].mean()
                if not (
                    mean_depth > self.config.min_depth
                    and mean_depth < self.config.dense_representation_radius_m
                ):
                    continue
            if (
                label.item() in self.config.object_labels
                and areas[i] > self.config.min_area
            ):
                object_masks.append(masks[i])
                object_boxes.append(boxes[i])
                object_labels.append(label)

        if len(object_masks) == 0:
            return Results(
                None,
                torch.from_numpy(panoptic_img).to(self.device),
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
        masks_to_use = object_masks if self.dense_encoder is None else None
        patches = self.patch_extractor(img, bboxes=object_boxes, masks=masks_to_use)
        choice = np.arange(patches[0].shape[0])
        if patches[0].shape[0] > self.config.max_batch:
            choice = np.random.choice(
                patches[0].shape[0], self.config.max_batch, replace=False
            )
        patches = [p[choice] for p in patches]
        object_labels = (
            torch.stack(object_labels)[choice] if object_labels is not None else None
        )
        features = torch.zeros(
            patches[0].shape[0], self.encoder.model.visual.output_dim
        ).to(self.device)

        img = self.preprocess(img)

        # dense clip doesn't use center-crop, so we have to apply it ourselves
        clip_img = center_crop(img, self.encoder.input_size)
        img_embedding = torch.squeeze(self.encoder(clip_img.unsqueeze(0)))

        if self.dense_encoder is None:
            assert patches[1] is not None
            if self.config.text_embeddings:
                class_names = [self.id_to_name[label.item()] for label in object_labels]
                features += self.encoder.embed_text(class_names)
            if self.config.box_embeddings:
                features += self.encoder(patches[0])
            if self.config.mask_embeddings:
                features += self.encoder(patches[1])
            features /= (
                int(self.config.box_embeddings)
                + int(self.config.mask_embeddings)
                + int(self.config.text_embeddings)
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
                patches[0]
            ) + ratios * dense_features

        object_masks = object_masks[choice]

        # Get chosen panoptic ids using the masks
        panoptic_img = torch.from_numpy(panoptic_img).to(self.device)
        panoptic_ids = torch.Tensor(
            [panoptic_img[mask.to(torch.bool)][0] for mask in object_masks]
        ).to(torch.int)

        return Results(
            object_masks,
            panoptic_img,
            object_boxes[choice],
            features,
            patches[0],
            patches[1],
            img_embedding,
            object_labels if labels is not None else None,
            panoptic_ids,
        )

    def get_masks_from_segmentation(
        self, seg_img: np.ndarray, panoptic_img: np.ndarray
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Get masks from panoptic segmentation and their
        corresponding colors from the segmentation image.
        :param seg_img: Segmentation image
        :param panoptic_img: Panoptic image
        :return: Tuple of masks and colors
        """
        # Get unique panptic ids
        unique_ids, indices = np.unique(panoptic_img, return_index=True)

        # Get masks using unique ids
        masks = torch.from_numpy(panoptic_img)[:, :, None] == torch.from_numpy(
            unique_ids
        )

        return (
            masks.permute(2, 0, 1).to(torch.bool).to(self.device),
            np.reshape(seg_img, (-1, 3))[indices],
        )

    def forward(
        self,
        img: np.ndarray,
        seg_img: np.ndarray,
        panoptic_img: np.ndarray,
        depth_img: np.ndarray,
    ) -> Results:
        """
        Extract masks, boxes and labels and encode them.

        Args:
            img (np.array): Image
            seg_img (np.array): Segmentation image
            panoptic_img (np.array): Panoptic image
            depth_img (np.array): Depth

        Returns:
            Results: Encoded image
        """
        masks, colors = self.get_masks_from_segmentation(seg_img, panoptic_img)
        bboxes = torchvision.ops.masks_to_boxes(masks).to(torch.int32)
        labels = None
        if self.colors_to_labels:
            labels = torch.tensor(
                [self.colors_to_labels[tuple(c)] for c in colors], device=masks.device
            )
        return self.encode(
            torch.from_numpy(img).to(self.device),
            masks,
            bboxes,
            labels,
            panoptic_img,
            depth_img,
        )
