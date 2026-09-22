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
#
"""Model wrappers for image segmentation."""

import dataclasses
import os

import einops
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import yaml
from spark_config import Config, register_config

import semantic_inference_python.models.utils as model_utils
from semantic_inference_python import root_path
from semantic_inference_python.misc import Logger
from semantic_inference_python.models.lseg import LSegEncNet
from semantic_inference_python.models.maskclip import MaskClip


def models_path():
    """Get path to pre-trained weight storage."""
    return root_path().parent.parent / "models"


class YOLOESegmentation(nn.Module):
    """YOLOe wrapper."""

    def __init__(self, config) -> None:
        """Load YOLOe.
        :param config: Configuration for YOLOe
        """
        super().__init__()
        from ultralytics import YOLOE

        self.config = config
        self.config.parse_labels()
        self.yoloe = YOLOE(config.model_name)
        if len(self.config.get_labels()) > 0:
            self.set_classes()

    def set_classes(self) -> None:
        """Set class names for YOLOe model."""
        self.yoloe.set_classes(
            self.config.get_labels(), self.yoloe.get_text_pe(self.config.get_labels())
        )

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = YOLOEConfig()
        config.update(kwargs)
        return cls(config)

    def to_device(self, device: torch.device) -> None:
        """Move the model to a device.
        :param device: Device to move the model to
        """
        self._canary_param = nn.Parameter(torch.empty(0, device=device))
        self.yoloe.to(device)

    def train(self, mode):
        """Don't pass train to underlying model."""
        pass

    def forward(
        self, img: np.ndarray, device: torch.device = None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Semantically segment image.
        :param img: Image to segment
        :param device: Device to run the model on
        :return: Masks, bounding boxes, labels,
                 panoptic image, and confidence scores
        """
        if len(self.config.get_labels()) == 0:
            Logger.info("No labels set for YOLOe model.")
            return (
                None,
                None,
                None,
                None,
                None,
            )
        # Get bounding boxes and labels from YOLO
        results_yolo = self.yoloe.predict(
            img,
            device=device,
            conf=self.config.confidence,
            imgsz=self.config.output_size,
            verbose=False,
        )
        labels = results_yolo[0].boxes.cls.to(torch.int)
        xyxy_tensor = results_yolo[0].boxes.xyxy
        if len(xyxy_tensor) == 0:
            return (
                None,
                None,
                None,
                None,
                None,
            )
        masks_tensor = (
            torch.from_numpy(
                model_utils.scale_masks(
                    results_yolo[0].masks.data.permute(1, 2, 0).cpu().numpy(), img.shape
                )
            ).permute(2, 0, 1)
            > 0.5
        )
        masks_tensor = masks_tensor.to(device)

        return (
            masks_tensor.to(torch.bool),
            xyxy_tensor.to(torch.int32),
            labels,
            model_utils.panoptic_image(masks_tensor, labels),
            results_yolo[0].boxes.conf,
        )


@register_config("labeled_segmentation", name="yoloe", constructor=YOLOESegmentation)
@dataclasses.dataclass
class YOLOEConfig(Config):
    """Configuration for YOLOe"""

    model_name: str = "yoloe-11l-seg.pt"
    confidence: float = 0.55
    iou: float = 0.85
    output_size: int = 736
    labels_path: str = ""
    labels: list[str] = dataclasses.field(default_factory=list)
    num_max_labels: int = 0

    def parse_labels(self) -> None:
        """Convert labels from dict to GroupInfo."""
        with open(self.labels_path) as f:
            label_data = yaml.safe_load(f)
        for label in label_data["label_names"]:
            if label["name"] not in ["free", "unknown"]:
                self.labels.append(label["name"])
        self.num_max_labels = label_data["total_semantic_labels"]

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)

    def add_labels(self, new_labels: list[str]) -> bool:
        """
        Add new labels to the model configuration.
        :param new_labels: List of new labels to add
        """
        if len(self.labels) + len(new_labels) > self.num_max_labels:
            Logger.info(
                f"Cannot add {len(new_labels)} new labels. "
                f"Max labels: {self.num_max_labels}, Current labels: {len(self.labels)}"
            )
            return False
        for label in new_labels:
            if label not in self.labels:
                self.labels.append(label)
        return True

    def get_labels(self) -> list[str]:
        """
        Retruns a list of labels for the model.
        :return: list of labels
        """
        return self.labels


class FastSAMSegmentation(nn.Module):
    """Fast SAM wrapper."""

    def __init__(self, config, verbose=False):
        """Load Fast SAM."""
        super().__init__()
        from ultralytics import FastSAM

        self.config = config
        self.verbose = verbose
        self.sam = FastSAM(config.model_name)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = FastSAMConfig()
        config.update(kwargs)
        return cls(config)

    def train(self, mode):
        """Don't pass train to underlying model."""
        pass

    def forward(self, img, device=None):
        """Segment image."""
        # TODO(nathan) resize?
        results = self.sam(
            source=img,
            device=device,
            retina_masks=True,
            imgsz=self.config.output_size,
            conf=self.config.confidence,
            iou=self.config.iou,
            verbose=self.verbose,
        )

        return results[0].masks.data.to(torch.bool), results[0].boxes.xyxy.to(
            torch.int32
        )


@register_config("segmentation", name="fastsam", constructor=FastSAMSegmentation)
@dataclasses.dataclass
class FastSAMConfig(Config):
    """Configuration for FastSAM."""

    model_name: str = "FastSAM-x.pt"
    confidence: float = 0.55
    iou: float = 0.85
    output_size: int = 736

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class SAMSegmentation(nn.Module):
    """SAM wrapper."""

    def __init__(self, config):
        """Load SAM."""
        super().__init__()
        import segment_anything as sam

        self.config = config

        weight_path = models_path() / config.model_name
        model = sam.sam_model_registry["vit_h"](checkpoint=str(weight_path))
        self.sam = sam.SamAutomaticMaskGenerator(
            model=model,
            points_per_side=self.config.points_per_side,
            points_per_batch=self.config.points_per_batch,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            crop_n_layers=self.config.crop_n_layers,
            min_mask_region_area=self.config.min_mask_region_area,
        )

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = SAMConfig()
        config.update(kwargs)
        return cls(config)

    def forward(self, img):
        """
        Segment image.

        Args:
            img (np.ndarray): uint8 image in [H, W, C] order

        Returns
            tuple[torch.Tensor, torch.Tensor]: Masks and bounding boxes
        """
        results = self.sam.generate(img)
        N = len(results)
        masks = torch.zeros((N, img.shape[0], img.shape[1]), dtype=torch.bool)
        b_xywh = torch.zeros((N, 4), dtype=torch.float32)
        for idx, r in enumerate(results):
            masks[idx] = torch.from_numpy(r["segmentation"])
            b_xywh[idx] = torch.tensor(r["bbox"])

        return masks, torchvision.ops.box_convert(b_xywh, "xywh", "xyxy")


@register_config("segmentation", name="sam", constructor=SAMSegmentation)
@dataclasses.dataclass
class SAMConfig(Config):
    """Configuration for FastSAM."""

    model_name = "sam_vit_h_4b8939.pth"
    points_per_side: int = 12
    points_per_batch: int = 144
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    crop_n_layers: int = 0
    min_mask_region_area: int = 100

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class DenseFeatures(nn.Module):
    """Module to compute dense features per mask."""

    def __init__(self, model_name):
        """Load f3rm module."""
        super().__init__()
        from f3rm.features.clip import clip as f3rm_clip

        self.model_name = model_name
        self.model, self.preprocess = f3rm_clip.load(model_name)
        # print(self.preprocess)

    def get_output_dims(self, h_in, w_in):
        """Compute output dimensions."""
        # from https://github.com/f3rm/f3rm/blob/main/f3rm/features/clip_extract.py
        if self.model_name.startswith("ViT"):
            h_out = h_in // self.model.visual.patch_size
            w_out = w_in // self.model.visual.patch_size
            return h_out, w_out

        if self.model_name.startswith("RN"):
            h_out = max(h_in / w_in, 1.0) * self.model.visual.attnpool.spacial_dim
            w_out = max(w_in / h_in, 1.0) * self.model.visual.attnpool.spacial_dim
            return int(h_out), int(w_out)

        raise ValueError(f"unknown clip model: {self.model_name}")

    def forward(self, img):
        """Compute dense clip embeddings for image."""
        embeddings = self.model.get_patch_encodings(img.unsqueeze(0))
        h_in, w_in = img.size()[-2:]
        h_out, w_out = self.get_output_dims(h_in, w_in)
        return einops.rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)


class ClipWrapper(nn.Module):
    """Quick wrapper around clip to simplifiy interface for encoding images."""

    def __init__(self, config):
        """Load the visual encoder for CLIP."""
        super().__init__()
        import clip

        self.config = config
        self.model, self._transform = clip.load(
            config.model_name,
            device="cpu",
            download_root=os.environ.get("CLIP_CACHE_DIR"),
        )
        self._canary_param = nn.Parameter(torch.empty(0))
        self._tokenize = clip.tokenize

    def to_device(self, device: torch.device) -> None:
        """Move the model to a device.
        :param device: Device to move the model to
        """
        self._canary_param = nn.Parameter(torch.empty(0, device=device))
        # Correctly create a Parameter on the desired device
        self.model.to(device)
        self._transform = self._transform.to(device)

    @torch.no_grad()
    def forward(self, imgs):
        """Encode multiple images (without transformation)."""
        # TODO(nathan) think about validation
        return self.model.visual(imgs.to(self.model.dtype))

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = ClipConfig()
        config.update(kwargs)
        return cls(config)

    @property
    def model_name(self):
        """Get current model name."""
        return self.config.model_name

    @property
    def input_size(self):
        """Get input patch size for clip."""
        return self.model.visual.input_resolution

    @property
    def output_dim_visual(self):
        """Get output dimension for visual encoder."""
        return self.model.visual.output_dim

    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device

    def get_dense_encoder(self):
        """Get corresponding dense encoder."""
        return DenseFeatures(self.model_name)

    @torch.no_grad()
    def embed_text(self, text):
        """Encode text."""
        tokens = self._tokenize(text).to(self.device)
        return self.model.encode_text(tokens)


@register_config("clip", name="clip", constructor=ClipWrapper)
@dataclasses.dataclass
class ClipConfig(Config):
    """Configuration for OpenCLIP."""

    model_name: str = "ViT-B/32"

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class OpenClipWrapper(nn.Module):
    """Quick wrapper around openclip to simplifiy image encoding interface."""

    def __init__(self, config) -> None:
        """Load the visual encoder for OpenCLIP.
        :param config: Configuration for OpenCLIP
        """
        super().__init__()
        self.config = config
        self.model, _, self._transform = open_clip.create_model_and_transforms(
            config.model_name, pretrained=config.pretrained, cache_dir=config.cache_dir
        )
        self._canary_param = nn.Parameter(torch.empty(0))
        self._tokenize = open_clip.get_tokenizer(config.tokenizer_name)

    def to_device(self, device):
        """Move the model to a device."""
        self._canary_param = nn.Parameter(torch.empty(0, device=device))
        self.model.to(device)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = OpenClipConfig()
        config.update(kwargs)
        return cls(config)

    def forward(self, imgs):
        """Encode multiple images (without transformation)."""
        return self.model.visual(imgs)

    @property
    def input_size(self):
        """Get input patch size for clip."""
        return self.model.visual.image_size[0]

    @property
    def output_dim_visual(self):
        """Get output dimension for visual encoder."""
        return self.model.visual.output_dim

    @property
    def model_name(self):
        """Get current model name."""
        return self.config.model_name

    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device

    def get_dense_encoder(self):
        """Get corresponding dense encoder."""
        return None

    @torch.no_grad()
    def embed_text(self, text):
        """Encode text."""
        tokens = self._tokenize(text).to(self.device)
        return self.model.encode_text(tokens)


@register_config("clip", name="open_clip", constructor=OpenClipWrapper)
@dataclasses.dataclass
class OpenClipConfig(Config):
    """Configuration for OpenCLIP."""

    model_name: str = "ViT-B-32"
    tokenizer_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    cache_dir: str = ""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class LSegClipWrapper(nn.Module):
    """LSeg CLIP wrapper."""

    def __init__(self, config) -> None:
        """Load the visual encoder for LSeg CLIP.
        :param config: Configuration for LSeg CLIP
        """
        super().__init__()
        self.config = config
        self.model = LSegEncNet(
            arch_option=config.arch_option,
            block_depth=config.block_depth,
            activation=config.activation,
            crop_size=config.crop_size,
        )
        pretrained_state_dict = torch.load(
            config.model_path, map_location="cpu", weights_only=False
        )
        pretrained_state_dict = {
            k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()
        }
        model_state_dict = self.model.state_dict()
        model_state_dict.update(pretrained_state_dict)
        self.model.load_state_dict(pretrained_state_dict)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.config.norm_mean, self.config.norm_std),
            ]
        )
        self._canary_param = nn.Parameter(torch.empty(0))
        stride_rate = 2.0 / 3.0
        self.stride = int(self.config.crop_size * stride_rate)

    def to_device(self, device: torch.device) -> None:
        """
        Move the model to a device.
        :param device: Device to move the model to
        """
        self._canary_param = nn.Parameter(torch.empty(0, device=device))
        # Correctly create a Parameter on the desired device
        self.model.to(device)

    @classmethod
    def construct(cls, **kwargs):
        """
        Load model from configuration dictionary.
        :return: LSegClipWrapper instance
        """
        config = LSegClipConfig()
        config.update(kwargs)
        return cls(config)

    def forward(
        self, img: np.ndarray, orig_h: int = None, orig_w: int = None
    ) -> torch.Tensor:
        """
        Encode multiple images (without transformation).
        :param img: Input image array
        :param orig_h: Original height of the image
        :param orig_w: Original width of the image
        :return: Encoded image tensor
        """
        return self.encode(img, orig_h, orig_w)

    @torch.inference_mode()
    def encode(
        self, img: np.ndarray, orig_h: int = None, orig_w: int = None
    ) -> torch.Tensor:
        """
        Encode multiple images (without transformation).
        :param img: Input image array
        :param orig_h: Original height of the image
        :param orig_w: Original width of the image
        :return: Encoded image tensor
        """

        image = self.transform(img).unsqueeze(0).to(self._canary_param.device)
        image = F.interpolate(
            image,
            size=(self.config.crop_size, self.config.crop_size),
            mode="bilinear",
            align_corners=True,
        )
        # batch, _, h, w = image.size()
        # if h > w:
        #     height = self.config.base_size
        #     width = int(1.0 * w * self.config.base_size / h + 0.5)
        #     short_size = width
        # else:
        #     width = self.config.base_size
        #     height = int(1.0 * h * self.config.base_size / w + 0.5)
        #     short_size = height

        # cur_img = F.interpolate(
        #     image, size=(height, width), mode="bilinear", align_corners=True
        # )

        # if self.config.base_size <= self.config.crop_size:
        #     pad_img = pad_image(
        #         cur_img,
        #         self.config.norm_mean,
        #         self.config.norm_std,
        #         self.config.crop_size,
        #     )
        #     outputs, _ = self.model(pad_img)
        #     outputs = crop_image(outputs, 0, height, 0, width)
        # else:
        #     if short_size < self.config.crop_size:
        #         pad_img = pad_image(
        #             cur_img,
        #             self.config.norm_mean,
        #             self.config.norm_std,
        #             self.config.crop_size,
        #         )
        #     else:
        #         pad_img = cur_img
        #     _, _, ph, pw = pad_img.shape
        #     assert ph >= height and pw >= width
        #     h_grids = (
        #         int(math.ceil(1.0 * (ph - self.config.crop_size) / self.stride)) + 1
        #     )
        #     w_grids = (
        #         int(math.ceil(1.0 * (pw - self.config.crop_size) / self.stride)) + 1
        #     )
        #     outputs = (
        #         image.new()
        #         .resize_(batch, self.model.out_c, ph, pw)
        #         .zero_()
        #         .to(self._canary_param.device)
        #     )
        #     count_norm = (
        #         image.new()
        #         .resize_(batch, 1, ph, pw)
        #         .zero_()
        #         .to(self._canary_param.device)
        #     )
        #     for idh in range(h_grids):
        #         for idw in range(w_grids):
        #             h0 = idh * self.stride
        #             w0 = idw * self.stride
        #             h1 = min(h0 + self.config.crop_size, ph)
        #             w1 = min(w0 + self.config.crop_size, pw)
        #             crop_img = crop_image(pad_img, h0, h1, w0, w1)
        #             # pad if needed
        #             pad_crop_img = pad_image(
        #                 crop_img,
        #                 self.config.norm_mean,
        #                 self.config.norm_std,
        #                 self.config.crop_size,
        #             )
        #             with torch.no_grad():
        #                 output, _ = self.model(pad_crop_img)
        #             cropped = crop_image(output, 0, h1 - h0, 0, w1 - w0)
        #             outputs[:, :, h0:h1, w0:w1] += cropped
        #             count_norm[:, :, h0:h1, w0:w1] += 1.0
        #     assert (count_norm == 0).sum() == 0
        #     outputs = outputs / count_norm
        #     outputs = outputs[:, :, :height, :width]
        # # Interpolate back to original size
        with torch.no_grad():
            outputs, _ = self.model(image, refine=self.config.refine)
        if self.config.interpolate and orig_h is not None and orig_w is not None:
            h, w = orig_h, orig_w
            outputs = F.interpolate(
                outputs, size=(h, w), mode="bilinear", align_corners=True
            )
        return outputs


@register_config("pixelwise_clip", name="lseg", constructor=LSegClipWrapper)
@dataclasses.dataclass
class LSegClipConfig(Config):
    """Configuration for LSeg CLIP."""

    arch_option: int = 0
    block_depth: int = 0
    activation: str = "lrelu"
    crop_size: int = 480
    base_size: int = 520
    norm_mean: list[float] = dataclasses.field(default_factory=lambda: [0.5, 0.5, 0.5])
    norm_std: list[float] = dataclasses.field(default_factory=lambda: [0.5, 0.5, 0.5])
    model_path: str = ""
    interpolate: bool = True
    refine: bool = False

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class MaskClipWrapper(nn.Module):
    """MaskClip wrapper."""

    def __init__(self, config) -> None:
        """
        Load MaskClip model.
        :param config: Configuration for MaskClip
        """
        super().__init__()
        self.config = config
        self.model = MaskClip(
            clip_model=config.model_name, pretrained=self.config.pretrained
        )
        self._canary_param = nn.Parameter(torch.empty(0))

    def to_device(self, device: torch.device) -> None:
        """
        Move the model to a device.
        :param device: Device to move the model to
        """
        self._canary_param = nn.Parameter(torch.empty(0, device=device))
        # Correctly create a Parameter on the desired device
        self.model.to(device)

    @classmethod
    def construct(cls, **kwargs):
        """
        Load model from configuration dictionary.
        :return: MaskClipWrapper instance
        """
        config = MaskClipConfig()
        config.update(kwargs)
        return cls(config)

    def forward(
        self, img: np.ndarray, orig_h: int = None, orig_w: int = None
    ) -> torch.Tensor:
        """
        Encode multiple images (without transformation).
        :param img: Input image array
        :param orig_h: Original height of the image
        :param orig_w: Original width of the image
        :return: Encoded image tensor
        """
        return self.encode(img, orig_h, orig_w)

    @torch.inference_mode()
    def encode(
        self, img: np.ndarray, orig_h: int = None, orig_w: int = None
    ) -> torch.Tensor:
        """
        Encode images (without transformation).
        :param img: Input image array
        :param orig_h: Original height of the image
        :param orig_w: Original width of the image
        :return: Encoded image tensor
        """
        torch_img = torch.from_numpy(img).to(self._canary_param.device) / 255.0
        torch_img = torch_img.unsqueeze(0).permute(0, 3, 1, 2)  # B,C,H,W
        dense_feats = self.model.forward(torch_img, interpolate=self.config.interpolate)
        return dense_feats


@register_config("pixelwise_clip", name="maskclip", constructor=MaskClipWrapper)
@dataclasses.dataclass
class MaskClipConfig(Config):
    """Configuration for MaskClip."""

    model_name: str = "ViT-B/32"
    pretrained: str = "openai"
    interpolate: bool = True

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class F3RMWrapper(DenseFeatures):
    """Module to compute dense features per mask."""

    def __init__(self, config):
        """Load f3rm module."""
        self.config = config
        super().__init__(config.model_name)
        self._canary_param = nn.Parameter(torch.empty(0))

    def to_device(self, device: torch.device) -> None:
        """
        Move the model to a device.
        :param device: Device to move the model to
        """
        self._canary_param = nn.Parameter(torch.empty(0, device=device))
        self.to(device)

    @classmethod
    def construct(cls, **kwargs):
        """
        Load model from configuration dictionary.
        :return: F3RMWrapper instance
        """
        config = F3RMConfig()
        config.update(kwargs)
        return cls(config)

    @torch.inference_mode()
    def encode(self, img: np.ndarray, orig_h: int, orig_w: int) -> torch.Tensor:
        """
        Encode multiple images (without transformation).
        :param img: Input image array
        :param orig_h: Original height of the image
        :param orig_w: Original width of the image
        :return: Encoded image tensor
        """
        dense_feats = self(torch.from_numpy(img).to(self._canary_param.device))

        # Return interpolated to original size
        return F.interpolate(
            dense_feats.permute(0, 3, 2, 1),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True,
        )


@register_config("pixelwise_clip", name="f3rm", constructor=F3RMWrapper)
@dataclasses.dataclass
class F3RMConfig(Config):
    """Configuration for F3RM."""

    model_name: str = "ViT-B/32"

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
