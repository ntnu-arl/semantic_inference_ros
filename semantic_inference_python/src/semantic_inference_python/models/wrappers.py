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
"""Model wrappers for image segmentation."""

import dataclasses
from typing import List, Union, Tuple

import json
import clip
import open_clip
import os
import time
import rospy
from pathlib import Path
import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torchvision
import math
from PIL import Image
from transformers import InstructBlipProcessor
from semantic_inference_python import root_path
from semantic_inference_python.config import Config, register_config
from semantic_inference_python.models.instruct_blip import (
    InstructBlipForConditionalGeneration,
)
from semantic_inference_python.models.deepseek.deepseek_vl2.models import (
    DeepseekVLV2ForCausalLM,
    DeepseekVLV2Processor,
    select_best_resolution,
    VisionTransformer,
    MlpProjector,
    VisionEncoderConfig,
    MlpProjectorConfig,
)
import supervision as sv
from supervision.draw.color import Color, ColorPalette
import ultralytics.utils.ops

try:
    import tensorrt as trt
    import pycuda.driver as cuda

    cuda.init()
    device = cuda.Device(0)
    CTX = device.make_context()
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)
except ImportError:
    print("TensorRT or PyCUDA not found. TRTInference classes will not work.")
# import pycuda.autoinit


DUMMY_IMG_PATH = Path("/home/arl/jetson_ssd/hydra_bags/image.png")


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


def safe_label_positions(
    detections: sv.Detections, labels: List[str], image_shape: Tuple[int, int]
) -> List[str]:
    """
    Adjust label positions to avoid drawing them too close to the top edge of the image.
    This function estimates where the label would be drawn based on the bounding box
    coordinates and adjusts the label text if it would be drawn too close to the top edge.
    :param detections: Detections object containing bounding box coordinates.
    :param labels: List of label texts corresponding to the detections.
    :param image_shape: Shape of the image (height, width).
    :return: List of adjusted label texts.
    """
    safe_labels = []
    height, width = image_shape[:2]
    for i, (xyxy, label_text) in enumerate(zip(detections.xyxy, labels)):
        x1, y1, x2, y2 = xyxy
        # Estimate where label would be drawn (usually near top-left of bbox)
        estimated_label_y = y1 - 10  # assuming upward text offset

        # Adjust label if too close to top edge
        if estimated_label_y < 0:
            label_text = " " + label_text  # you could even tag or reposition here

        safe_labels.append(label_text)
    return safe_labels


def annotate_detections(
    image: np.ndarray,
    color_palette: ColorPalette,
    boxes: np.ndarray,
    masks: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    class_names: List[str],
) -> np.ndarray:
    """
    Annotate an image with bounding boxes, segmentation masks, and labels.

    Args:
        image: Input image (H, W, 3) as numpy array (BGR).
        color_palette: dict or list mapping class_id -> hex color string.
        boxes: (N, 4) array of [x_min, y_min, x_max, y_max].
        masks: (N, H, W) array of binary masks.
        class_ids: (N,) array of class ids.
        confidences: (N,) array of detection confidences.
        class_names: dict or list mapping class_id -> class name.

    Returns:
        Annotated image as numpy array (BGR).
    """

    detections = sv.Detections(
        xyxy=boxes, mask=masks, class_id=class_ids, confidence=confidences
    )

    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    def label_lookup(detections: sv.Detections) -> List[str]:
        labels = []
        for class_id, conf in zip(detections.class_id, detections.confidence):
            name = class_names.get(class_id, "unknown")
            labels.append(f"{name} {conf:.2f}")
        return labels

    # Annotators
    mask_annotator = sv.MaskAnnotator(color=color_palette, opacity=0.4)
    box_annotator = sv.BoxAnnotator(color=color_palette, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=color_palette,
        text_color=sv.Color.WHITE,
        text_scale=1.1,
        text_thickness=3,
        text_padding=2,
    )

    # Apply annotations in order
    annotated = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=label_lookup(detections)
    )

    return annotated


def vis_result_fast(
    image: np.ndarray,
    detections: sv.Detections,
    classes: List[str],
    color: Union[Color, ColorPalette] = ColorPalette.default(),
    instance_random_color: bool = False,
    draw_bbox: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Annotate the image with the detection results.

    This is fast but of the same resolution of the input image, thus can be blurry.
    """
    box_annotator = sv.BoxAnnotator(
        color=color,
        text_scale=0.3,
        text_thickness=1,
        text_padding=2,
    )
    mask_annotator = sv.MaskAnnotator(color=color)

    labels = []

    if hasattr(detections, "confidence") and hasattr(detections, "class_id"):
        confidences = detections.confidence
        class_ids = detections.class_id
        if confidences is not None:
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for confidence, class_id in zip(confidences, class_ids)
            ]
        else:
            labels = [f"{classes[class_id]}" for class_id in class_ids]
    else:
        print("Detections object missing 'confidence' or 'class_id'.")
        return image, []  # safe fallback

    if instance_random_color:
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

    if draw_bbox:
        labels = safe_label_positions(detections, labels, image.shape)
        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections
        )

        # Draw labels safely
        for (x1, y1, x2, y2), label_text in zip(detections.xyxy, labels):
            x1, y1 = int(x1), int(y1)
            text_pos_y = max(y1 - 5, 10)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )

            # Draw background
            cv2.rectangle(
                annotated_image,
                (x1, text_pos_y - text_h - 4),
                (x1 + text_w, text_pos_y),
                (0, 0, 0),
                thickness=-1,
            )

            # Draw label
            cv2.putText(
                annotated_image,
                label_text,
                (x1, text_pos_y - 2),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

    return annotated_image, labels  # ✅ outside the loop


def models_path():
    """Get path to pre-trained weight storage."""
    return root_path().parent.parent / "models"


class DeepSeekVL2(nn.Module):
    """DeepSeekVL2 wrapper."""

    def __init__(self, config, verbose=False):
        """Load DeepSeekVL2."""
        super(DeepSeekVL2, self).__init__()
        self.config: DeepSeekVL2Config = config
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
            config.model_name
        )
        self.tokenizer = self.processor.tokenizer
        self.model: DeepseekVLV2ForCausalLM = (
            DeepseekVLV2ForCausalLM.from_pretrained(
                config.model_name, trust_remote_code=True
            )
            .cpu()
            .eval()
        )
        self.model = self.model.to(torch.bfloat16).cpu().eval()
        self.verbose = verbose
        self._canary_param = nn.Parameter(torch.empty(0))
        if self.config.cropping:
            self.best_width, self.best_height = select_best_resolution(
                (self.config.height, self.config.width),
                self.processor.candidate_resolutions,
            )
        else:
            self.best_width, self.best_height = (
                self.processor.image_size,
                self.processor.image_size,
            )
        num_width_tiles, num_height_tiles = (
            self.best_width // self.processor.image_size,
            self.best_height // self.processor.image_size,
        )
        h = w = math.ceil(
            (self.processor.image_size // self.processor.patch_size)
            / self.processor.downsample_ratio
        )
        self.images_spatial_crop = torch.tensor(
            [[num_width_tiles, num_height_tiles]], dtype=torch.long
        )
        self.num_image_tokens = [
            int(h * (w + 1) + 1 + (num_height_tiles * h) * (num_width_tiles * w + 1))
        ]
        self.feature_channels = 1 + math.ceil(
            self.best_height / self.processor.image_size
        ) * math.ceil(self.best_width / self.processor.image_size)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = DeepSeekVL2Config()
        config.update(kwargs)
        return cls(config)

    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device

    def move_to(self, device):
        """Move the model to a device."""
        pass

    def forward(self, img, text):
        """Forward pass."""
        pass

    def encode_images(self, images: Union[List[np.ndarray], List[torch.Tensor]]):
        """Encode images."""
        # Transform to PIL images
        if isinstance(images[0], torch.Tensor):
            pil_images = [
                [torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))]
                for img in images
            ]
        elif isinstance(images[0], np.ndarray):
            pil_images = [[Image.fromarray(img)] for img in images]
        else:
            raise ValueError(
                "Images must be either a list of np.ndarray or a list of torch.Tensor"
            )
        projected_features = []
        for pil_image in pil_images:
            _, images_list, _, _, _, _, _ = self.processor.encode_images(
                pil_image, self.config.cropping
            )
            with torch.inference_mode():
                feature = self.model.encode_images(
                    images_list[None, ...],
                    self.images_spatial_crop[None, ...],
                    self.config.use_cuda,
                    self.config.cuda_batch_size,
                )
            projected_features.append(
                feature.reshape(
                    self.feature_channels * feature.shape[1], feature.shape[2]
                )
            )

        return torch.stack(projected_features, dim=0)

    def generate_caption(self, image_embeds, text):
        """Generate image caption."""
        image_embeds = image_embeds.to(torch.bfloat16)
        answers = []
        for image_embed, prompt in zip(image_embeds, text):
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"This is the image: <image>\n {prompt}",
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            prepare_inputs = self.processor.encode_conversations_and_features(
                conversation,
                self.best_width,
                self.best_height,
                image_embed.reshape(
                    self.feature_channels,
                    image_embed.shape[0] // self.feature_channels,
                    image_embed.shape[1],
                ),
                self.images_spatial_crop,
                self.num_image_tokens,
            )
            inputs_embeds = self.model.prepare_input_embeds_from_feats(**prepare_inputs)
            outputs = self.model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True,
            )
            answers.append(
                self.tokenizer.decode(
                    outputs[0].cpu().tolist(), skip_special_tokens=True
                )
            )

        return answers


class TRTInference:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self._load_engine()

    def _load_engine(self):
        pass

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        pass


class TRTInferenceVision(TRTInference):
    def __init__(self, engine_path):
        super().__init__(engine_path)

    def _load_engine(self):
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.input_host = None
        self.input_device = None
        self.output_host = None
        self.output_device = None
        self.input_shape = None
        self.output_shape = None  # dynamic shape will be read from context

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        cuda.Context.push(CTX)
        try:
            if self.input_shape != input_tensor.shape:
                self.input_shape = input_tensor.shape
                input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
                input_size = int(np.prod(self.input_shape))
                self.input_host = cuda.pagelocked_empty(input_size, input_dtype)
                self.input_device = cuda.mem_alloc(self.input_host.nbytes)

                # Set input shape for dynamic input
                self.context.set_input_shape(self.input_name, self.input_shape)

                # Query output shape after setting input shape
                self.output_shape = tuple(
                    self.context.get_tensor_shape(self.output_name)
                )
                output_dtype = trt.nptype(
                    self.engine.get_tensor_dtype(self.output_name)
                )
                output_size = int(np.prod(self.output_shape))
                self.output_host = cuda.pagelocked_empty(output_size, output_dtype)
                self.output_device = cuda.mem_alloc(self.output_host.nbytes)

            np.copyto(self.input_host, input_tensor.ravel())
            cuda.memcpy_htod(self.input_device, self.input_host)

            bindings = [int(self.input_device), int(self.output_device)]
            self.context.execute_v2(bindings=bindings)

            cuda.memcpy_dtoh(self.output_host, self.output_device)

            return self.output_host.reshape(self.output_shape)
        finally:
            cuda.Context.pop()


class TRTInferenceProjector(TRTInference):
    def __init__(self, engine_path):
        super().__init__(engine_path)

    def _load_engine(self):
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.input_host = None
        self.input_device = None
        self.output_host = None
        self.output_device = None
        self.input_shape = None
        self.output_shape = None

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        cuda.Context.push(CTX)
        try:
            if self.input_shape != input_tensor.shape:
                self.input_shape = input_tensor.shape
                input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
                input_size = int(np.prod(self.input_shape))
                self.input_host = cuda.pagelocked_empty(input_size, input_dtype)
                self.input_device = cuda.mem_alloc(self.input_host.nbytes)

                self.context.set_input_shape(self.input_name, self.input_shape)

                self.output_shape = tuple(
                    self.context.get_tensor_shape(self.output_name)
                )
                output_dtype = trt.nptype(
                    self.engine.get_tensor_dtype(self.output_name)
                )
                output_size = int(np.prod(self.output_shape))
                self.output_host = cuda.pagelocked_empty(output_size, output_dtype)
                self.output_device = cuda.mem_alloc(self.output_host.nbytes)

            np.copyto(self.input_host, input_tensor.ravel())
            cuda.memcpy_htod(self.input_device, self.input_host)

            bindings = [int(self.input_device), int(self.output_device)]
            self.context.execute_v2(bindings=bindings)

            cuda.memcpy_dtoh(self.output_host, self.output_device)

            return self.output_host.reshape(self.output_shape)
        finally:
            cuda.Context.pop()


class DeepSeekVL2Visual(nn.Module):
    """DeepSeekVL2 Visual Encoder wrapper."""

    def __init__(self, config, verbose=False):
        super(DeepSeekVL2Visual, self).__init__()
        self.config = config
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
            config.processor_path
        )
        base_path = Path(self.config.model_path)
        if (base_path / "config.json").exists():
            with (base_path / "config.json").open() as f:
                models_config = json.load(f)
            self.tokenizer = self.processor.tokenizer
            vision_config = VisionEncoderConfig(**models_config["vision_config"])
            projector_config = MlpProjectorConfig(**models_config["projector_config"])
            self.vision = VisionTransformer(
                img_size=vision_config.image_size,
                patch_size=vision_config.patch_size,
                embed_dim=vision_config.width,
                depth=vision_config.layers,
                num_heads=vision_config.heads,
                mlp_ratio=vision_config.mlp_ratio,
                class_token=vision_config.class_token,
                global_pool=vision_config.global_pool,
                ignore_head=vision_config.ignore_head,
                weight_init=vision_config.weight_init,
                num_classes=0,
                deterministic=vision_config.deterministic,
                num_recomputing_layers=vision_config.num_recomputing_layers,
            )
            self.projector = MlpProjector(projector_config)

            self.vision.load_state_dict(
                torch.load(base_path / "vision.pt", map_location="cpu")
            )
            self.projector.load_state_dict(
                torch.load(base_path / "projector.pt", map_location="cpu")
            )
            self.vision.to(torch.bfloat16).eval()
            self.projector.to(torch.bfloat16).eval()
        elif (
            base_path / f"vision_dyn{self.config.cuda_batch_size}.engine"
        ).exists() and (
            base_path / f"projector_dyn{self.config.cuda_batch_size}.engine"
        ).exists():
            self.vision = TRTInferenceVision(
                base_path / f"vision_dyn{self.config.cuda_batch_size}.engine"
            )
            self.projector = TRTInferenceProjector(
                base_path / f"projector_dyn{self.config.cuda_batch_size}.engine"
            )
        else:
            raise FileNotFoundError("No model files found in the specified path.")

        self.verbose = verbose
        self._canary_param = nn.Parameter(torch.empty(0))

        if self.config.cropping:
            self.best_width, self.best_height = select_best_resolution(
                (self.config.height, self.config.width),
                self.processor.candidate_resolutions,
            )
        else:
            self.best_width, self.best_height = (
                self.processor.image_size,
                self.processor.image_size,
            )
        num_width_tiles, num_height_tiles = (
            self.best_width // self.processor.image_size,
            self.best_height // self.processor.image_size,
        )

        self.images_spatial_crop = torch.tensor(
            [[num_width_tiles, num_height_tiles]], dtype=torch.long
        )

        self.feature_channels = 1 + math.ceil(
            self.best_height / self.processor.image_size
        ) * math.ceil(self.best_width / self.processor.image_size)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = DeepSeekVL2VisualConfig()
        config.update(kwargs)
        return cls(config)

    def move_to(self, device):
        """Move the model to a device."""
        if isinstance(self.vision, TRTInferenceVision):
            # Correctly create a Parameter on the desired device
            self._canary_param = nn.Parameter(torch.empty(0, device=device))
        else:
            self.to(device)

    def forward(self, img):
        return self.encode_images(img)

    def encode_images(self, images: List[np.ndarray]):
        """Encode images."""
        # Transform to PIL images
        if isinstance(images[0], torch.Tensor):
            pil_images = [Image.fromarray(img.cpu().numpy()) for img in images]
        elif isinstance(images[0], np.ndarray):
            pil_images = [Image.fromarray(img) for img in images]
        else:
            raise ValueError(
                "Images must be either a list of np.ndarray or a list of torch.Tensor"
            )
        _, images_list, _, _, _, _, _ = self.processor.encode_images(
            pil_images, self.config.cropping
        )
        with torch.inference_mode():
            feature = self.inference(
                images_list.reshape(
                    len(images),
                    images_list.shape[0] // len(images),
                    images_list.shape[1],
                    images_list.shape[2],
                    images_list.shape[3],
                ),
                self.images_spatial_crop[None, ...].expand(len(images), -1, -1),
                self.device.type == "cuda",
                self.config.cuda_batch_size,
            )

        return feature.reshape(
            len(images), self.feature_channels * feature.shape[1], feature.shape[2]
        )

    def inference(
        self,
        images: torch.FloatTensor,
        images_spatial_crop: torch.LongTensor,
        use_cuda: bool = True,
        cuda_batch_size: int = 3,
    ):
        bs, max_n_images, _ = images_spatial_crop.shape
        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += 1 + num_width_tiles * num_height_tiles

            total_tiles.append(images[idx, : batch_num_tiles[idx]])

        # [batch_all_tiles, 3, height, width]
        total_tiles = torch.cat(total_tiles, dim=0)
        assert total_tiles.shape[0] == sum(batch_num_tiles)
        assert total_tiles.shape[0] > 0

        if not use_cuda:
            # [batch_all_tiles, vit_seq_len, c]
            if isinstance(self.vision, TRTInferenceVision):
                images_feature = self.vision.infer(total_tiles.float().cpu().numpy())

                # [batch_all_tiles, hw, D]
                return torch.from_numpy(self.projector.infer(images_feature)).to(
                    self.device
                )
            else:
                images_feature = self.vision(total_tiles)
                return self.projector(images_feature)
        else:
            projected_features = []
            for idx in range(0, total_tiles.shape[0], cuda_batch_size):
                if isinstance(self.vision, TRTInferenceVision):
                    images_feature = self.vision.infer(
                        total_tiles[idx : idx + cuda_batch_size].float().cpu().numpy()
                    )
                    projected_features.append(
                        torch.from_numpy(self.projector.infer(images_feature)).to(
                            self.device
                        )
                    )
                else:
                    projected_features.append(
                        self.projector(
                            self.vision(total_tiles[idx : idx + cuda_batch_size].cuda())
                        )
                    )
            return torch.cat(projected_features, dim=0)

    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device


@register_config("vlm", name="deepseek_visual_encoder", constructor=DeepSeekVL2Visual)
@dataclasses.dataclass
class DeepSeekVL2VisualConfig(Config):
    model_path: str = ""
    processor_path: str = ""
    cropping: bool = False
    height: int = 512
    width: int = 512
    cuda_batch_size: int = 3

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


@register_config("vlm", name="deepseek", constructor=DeepSeekVL2)
@dataclasses.dataclass
class DeepSeekVL2Config(Config):
    """Configuration for Deepseek."""

    model_name: str = "deepseek-ai/deepseek-vl2-tiny"
    cropping: bool = False
    height: int = 512
    width: int = 512
    use_cuda: bool = True
    cuda_batch_size: int = 3

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class InstructBLIP(nn.Module):
    """InstructBLIP wrapper."""

    def __init__(self, config, verbose=False) -> None:
        """Load Instruct BLIP."""
        super(InstructBLIP, self).__init__()
        self.config = config
        if os.path.exists(self.config.model_path):
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                config.model_path
            )
            self.processor = InstructBlipProcessor.from_pretrained(config.model_path)
        else:
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                config.model_name
            )
            self.processor = InstructBlipProcessor.from_pretrained(config.model_name)
        self.verbose = verbose
        self._canary_param = nn.Parameter(torch.empty(0))

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = InstructBLIPConfig()
        config.update(kwargs)
        return cls(config)

    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device

    def forward(self, img, text):
        """Forward pass."""
        image_embeds = self.encode_image(img)
        return self.generate_caption(image_embeds, text)

    def encode_image(self, img):
        """Encode image."""
        image_input = self.processor(images=img, text=None, return_tensors="pt").to(
            self.device
        )
        return self.model.embedd_image(image_input["pixel_values"], self.device)

    def generate_caption(self, image_embeds, text, max_batch_size=1):
        """Generate image caption in batches."""
        all_results = []

        # Ensure both image_embeds and text are the same length
        assert len(image_embeds) == len(
            text
        ), "Mismatched image_embeds and text lengths"

        for i in range(0, len(text), max_batch_size):
            # Create batch slices
            image_batch = image_embeds[i : i + max_batch_size]
            text_batch = text[i : i + max_batch_size]

            # Process the batch
            text_input = self.processor(
                images=None,
                text=text_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            output = self.model.generate_caption(
                image_embeds=image_batch,
                **text_input,
                max_length=self.config.max_length,
                do_sample=self.config.do_sample,
                num_beams=self.config.num_beams,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                length_penalty=self.config.length_penalty,
                temperature=self.config.temperature,
            )

            decoded = self.processor.batch_decode(output, skip_special_tokens=True)
            cleaned = [
                result.replace(text_batch[j], "").strip()
                for j, result in enumerate(decoded)
            ]
            all_results.extend(cleaned)

        return all_results


@register_config("vlm", name="instruct_blip", constructor=InstructBLIP)
@dataclasses.dataclass
class InstructBLIPConfig(Config):
    """Configuration for InstructBLIP."""

    model_name: str = "Salesforce/instructblip-vicuna-7b"
    model_path: str = ""
    max_length: int = 512
    do_sample: bool = False
    num_beams: int = 5
    top_p: float = 0.9
    repetition_penalty: float = 1.5
    length_penalty: float = 1.0
    temperature: float = 1.0

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class TRTInferenceBLIP(TRTInference):
    def __init__(self, engine_path):
        super().__init__(engine_path)

    def _load_engine(self):
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            size = (
                trt.volume(self.engine.get_binding_shape(binding))
                * self.engine.max_batch_size
            )
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            binding_info = {
                "name": binding,
                "host_mem": host_mem,
                "device_mem": device_mem,
            }
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append(binding_info)
            else:
                self.outputs.append(binding_info)

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        cuda.Context.push(CTX)
        try:
            input_binding = self.inputs[0]
            output_binding = self.outputs[0]

            np.copyto(input_binding["host_mem"], input_tensor.ravel())
            cuda.memcpy_htod(input_binding["device_mem"], input_binding["host_mem"])

            self.context.execute_v2(bindings=self.bindings)

            cuda.memcpy_dtoh(output_binding["host_mem"], output_binding["device_mem"])

            return output_binding["host_mem"].reshape(
                self.engine.get_binding_shape(self.outputs[0]["name"])
            )
        finally:
            cuda.Context.pop()


class InstructBLIPVisualEncoder(nn.Module):
    """Visual BLIP encoder."""

    def __init__(self, config, verbose=False) -> None:
        """Load visual instruct BLIP."""
        super(InstructBLIPVisualEncoder, self).__init__()
        self.config = config

        if os.path.exists(config.processor_path):
            self.processor = InstructBlipProcessor.from_pretrained(
                config.processor_path
            )
        else:
            self.processor = InstructBlipProcessor.from_pretrained(config.model_name)

        if os.path.exists(self.config.model_path):
            self.model = TRTInferenceBLIP(config.model_path)
        else:
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                config.model_name
            ).vision_model
        self.verbose = verbose
        self._canary_param = nn.Parameter(torch.empty(0))

        # if DUMMY_IMG_PATH.exists():
        #     img = cv2.imread(DUMMY_IMG_PATH)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     pixel_values = self.processor(images=img, text=None, return_tensors="pt")["pixel_values"]
        #     if isinstance(self.model, TRTInferenceBLIP):
        #         self.model.infer(pixel_values.cpu().numpy())
        #     else:
        #         self.model(pixel_values.to(self.device), return_dict=True)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = InstructBLIPVisualConfig()
        config.update(kwargs)
        return cls(config)

    def move_to(self, device):
        """Move the model to a device."""
        if isinstance(self.model, TRTInferenceBLIP):
            # Correctly create a Parameter on the desired device
            self._canary_param = nn.Parameter(torch.empty(0, device=device))
        else:
            self.to(device)

    def forward(self, img):
        """Encode image."""
        pixel_values = self.processor(images=img, text=None, return_tensors="pt")[
            "pixel_values"
        ]
        if isinstance(self.model, TRTInferenceBLIP):
            return torch.from_numpy(self.model.infer(pixel_values.cpu().numpy())).to(
                self.device
            )

        return self.model(
            pixel_values.to(self.device), return_dict=True
        ).last_hidden_state

    def encode_images(self, images: List[np.ndarray]):
        """Encode images."""
        return self.forward(images)

    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device


@register_config(
    "vlm",
    name="instruct_blip_visual",
    constructor=InstructBLIPVisualEncoder,
)
@dataclasses.dataclass
class InstructBLIPVisualConfig(Config):
    """Configuration for InstructBLIP."""

    model_name: str = "Salesforce/instructblip-vicuna-7b"
    model_path: str = ""
    processor_path: str = ""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class YOLOESegmentation(nn.Module):
    """YOLOe wrapper."""

    def __init__(self, config, verbose=False):
        super(YOLOESegmentation, self).__init__()
        from ultralytics import YOLOE, YOLO

        self.config = config
        self.config.parse_groups()
        if (
            config.yolo_model_name[-3:] == ".pt"
            and "anymal" not in config.yolo_model_name
        ):
            self.yoloe = YOLOE(config.yolo_model_name)
            self.set_classes()
        else:
            self.yoloe = YOLO(config.yolo_model_name)

        # if DUMMY_IMG_PATH.exists():
        #     img = cv2.imread(DUMMY_IMG_PATH)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     self.yoloe.predict(img, device="cuda", verbose=False)

    def set_classes(self) -> None:
        self.yoloe.set_classes(
            self.config.get_labels(), self.yoloe.get_text_pe(self.config.get_labels())
        )

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = YOLOEConfig()
        config.update(kwargs)
        return cls(config)

    def to_device(self, device):
        """Move the model to a device."""
        self._canary_param = nn.Parameter(torch.empty(0, device=device))
        if self.config.yolo_model_name[-3:] == ".pt":
            # Correctly create a Parameter on the desired device
            self.yoloe.to(device)

    def train(self, mode):
        """Don't pass train to underlying model."""
        pass

    def forward(self, img, device=None):
        """Semantically segment image."""
        # Get bounding boxes and labels from YOLO
        start_time = time.time()
        results_yolo = self.yoloe.predict(
            img,
            device=device,
            conf=self.config.confidence,
            imgsz=self.config.output_size,
            verbose=self.config.verbose,
        )
        if self.config.verbose:
            rospy.loginfo(
                f"YOLOE inference time: {(time.time() - start_time) * 1000:.3f} ms"
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
                None,
            )
        start_time = time.time()
        masks_tensor = (
            torch.from_numpy(
                ultralytics.utils.ops.scale_image(
                    results_yolo[0].masks.data.permute(1, 2, 0).cpu().numpy(), img.shape
                )
            ).permute(2, 0, 1)
            > 0.5
        )
        masks_tensor = masks_tensor.to(device)
        if self.config.verbose:
            rospy.loginfo(
                f"Scale image time: {(time.time() - start_time) * 1000:.3f} ms"
            )

        return (
            masks_tensor.to(torch.bool),
            xyxy_tensor.to(torch.int32),
            labels,
            None,
            panoptic_image(masks_tensor, labels),
            results_yolo[0].boxes.conf,
        )


class YOLOSAMSegmentation(nn.Module):
    """YOLO + FastSAM wrapper."""

    def __init__(self, config, verbose=False):
        """Load Fast SAM."""
        super(YOLOSAMSegmentation, self).__init__()
        from ultralytics import SAM, YOLO

        self.config = config
        self.config.parse_groups()
        self.yolo = YOLO(config.yolo_model_name)
        self.sam = SAM(config.model_name)
        self.set_classes()

    def set_classes(self, batch=80) -> None:
        """Set classes for YOLO."""
        classes = self.config.get_labels()
        # self.yolo.set_classes(classes)
        self.yolo.model.clip_model = clip.load(
            "ViT-B/32", device="cpu", download_root=os.environ.get("CLIP_CACHE_DIR")
        )[0]
        device = next(self.yolo.model.clip_model.parameters()).device
        text_token = clip.tokenize(classes).to(device)
        txt_feats = [
            self.yolo.model.clip_model.encode_text(token).detach()
            for token in text_token.split(batch)
        ]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.yolo.model.txt_feats = txt_feats.reshape(
            -1, len(classes), txt_feats.shape[-1]
        )
        self.yolo.model.model[-1].nc = len(classes)
        background = " "
        if background in classes:
            classes.remove(background)
        self.yolo.model.names = classes
        if self.yolo.predictor:
            self.yolo.predictor.model.names = classes

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = YOLOSAMConfig()
        config.update(kwargs)
        return cls(config)

    def train(self, mode):
        """Don't pass train to underlying model."""
        pass

    def _resolve_overlapping_masks(
        self, masks: torch.Tensor, confidences: torch.Tensor, labels: torch.Tensor
    ):
        """
        Keeps only the highest-confidence mask for each pixel.

        Args:
            masks (torch.Tensor): Boolean tensor of shape (N, H, W) -> (N masks, Height, Width).
            confidences (torch.Tensor): Confidence scores tensor of shape (N,).
            labels (torch.Tensor): Class labels tensor of shape (N,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch,Tensor]: Cleaned masks (N, H, W) with no overlaps, filtered labels, filtered confidences.
        """
        N, H, W = masks.shape

        # Convert boolean masks to float (0. or 1.) for processing
        masks = masks.float()

        # Stack confidences along new axis for broadcasting (N, 1, 1) -> allows per-pixel comparison
        conf_map = (
            confidences.view(N, 1, 1) * masks
        )  # Each mask has its confidence multiplied

        # Get index of highest confidence mask per pixel
        best_mask_idx = conf_map.argmax(
            dim=0
        )  # Shape: (H, W), each pixel gets assigned a mask index

        # Create output masks: Keep only the highest-confidence mask per pixel
        output_masks = []
        output_labels = []
        output_confidences = []
        for i in range(N):

            output_mask = (
                best_mask_idx == i
            )  # Only keep pixels where this mask was the highest
            if torch.any(output_mask):
                output_masks.append(output_mask)
                output_labels.append(labels[i])
                output_confidences.append(confidences[i])

        return (
            torch.stack(output_masks),
            torch.tensor(output_labels),
            torch.tensor(output_confidences),
        )

    def forward(self, img, device=None):
        """Semantically segment image."""
        # Get bounding boxes and labels from YOLO
        results_yolo = self.yolo(
            img,
            device=device,
            conf=self.config.confidence,
            imgsz=self.config.output_size,
            verbose=self.config.verbose,
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
                None,
            )
        # Get masks from FastSAM
        results_sam = self.sam(
            img,
            bboxes=xyxy_tensor,
            verbose=self.config.verbose,
            imgsz=self.config.output_size,
        )
        masks_tensor = results_sam[0].masks.data
        # Apply NMS to masks
        # masks_tensor, labels, confidences = self._resolve_overlapping_masks(results_sam[0].masks.data, results_yolo[0].boxes.conf, labels)
        # masks_tensor = masks_tensor.reshape(-1, img.shape[0], img.shape[1])
        xyxy_tensor = torchvision.ops.masks_to_boxes(masks_tensor)

        # Annotations for visualization
        # labels_np = labels.cpu().numpy()
        # masks_np = masks_tensor.cpu().numpy()
        # xyxy_np = xyxy_tensor.cpu().numpy()
        # curr_det = sv.Detections(
        # xyxy=xyxy_np, confidence=confidences.cpu().numpy(), class_id=labels_np, mask=masks_np
        # )
        # vis_img, _ = vis_result_fast(img, curr_det, self.config.get_labels())

        return (
            masks_tensor.to(torch.bool),
            xyxy_tensor.to(torch.int32),
            labels,
            None,
            panoptic_image(masks_tensor, labels),
            results_yolo[0].boxes.conf,
        )


@register_config("yolofastsam", name="group_info")
@dataclasses.dataclass
class GroupInfo:
    """Structure for labels and names."""

    name: str
    labels: List[int]


@register_config("segmentation", name="yolosam", constructor=YOLOSAMSegmentation)
@dataclasses.dataclass
class YOLOSAMConfig(Config):
    """Configuration for YOLO + FastSAM."""

    model_name: str = "FastSAM-x.pt"
    yolo_model_name: str = "yolov8l-world.pt"
    confidence: float = 0.55
    iou: float = 0.85
    output_size: int = 736
    groups: List[GroupInfo] = dataclasses.field(default_factory=list)
    image_features: bool = False
    verbose: bool = False

    def parse_groups(self) -> None:
        """Convert groupd from dict tp GroupInfo."""
        self.groups = [GroupInfo(**group) for group in self.groups]

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)

    def get_labels(self) -> List[str]:
        """
        Retruns a list of labels for the model.

        :return: List of labels
        """
        return [group.name for group in self.groups]


@register_config("segmentation", name="yoloe", constructor=YOLOESegmentation)
@dataclasses.dataclass
class YOLOEConfig(Config):
    """Configuration for YOLOe"""

    yolo_model_name: str = "yoloe-11l-seg.pt"
    confidence: float = 0.55
    iou: float = 0.85
    output_size: int = 736
    groups: List[GroupInfo] = dataclasses.field(default_factory=list)
    image_features: bool = False
    verbose: bool = False

    def parse_groups(self) -> None:
        """Convert groupd from dict tp GroupInfo."""
        self.groups = [GroupInfo(**group) for group in self.groups]

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)

    def get_labels(self) -> List[str]:
        """
        Retruns a list of labels for the model.

        :return: List of labels
        """
        return [group.name for group in self.groups]


class FastSAMSegmentation(nn.Module):
    """Fast SAM wrapper."""

    def __init__(self, config, verbose=False):
        """Load Fast SAM."""
        super(FastSAMSegmentation, self).__init__()
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
        masks = results[0].masks.data.to(torch.bool)
        return (
            masks,
            results[0].boxes.xyxy.to(torch.int32),
            None,
            None,
            panoptic_image(masks, torch.arange(masks.shape[0]) + 1),
        )


@register_config("segmentation", name="fastsam", constructor=FastSAMSegmentation)
@dataclasses.dataclass
class FastSAMConfig(Config):
    """Configuration for FastSAM."""

    model_name: str = "FastSAM-x.pt"
    confidence: float = 0.55
    iou: float = 0.85
    output_size: int = 736
    image_features: bool = False

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class SAMSegmentation(nn.Module):
    """SAM wrapper."""

    def __init__(self, config):
        """Load SAM."""
        super(SAMSegmentation, self).__init__()
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
            Tuple[torch.Tensor, torch.Tensor]: Masks and bounding boxes
        """
        results = self.sam.generate(img)
        N = len(results)
        masks = torch.zeros((N, img.shape[0], img.shape[1]), dtype=torch.bool)
        b_xywh = torch.zeros((N, 4), dtype=torch.float32)
        for idx, r in enumerate(results):
            masks[idx] = torch.from_numpy(r["segmentation"])
            b_xywh[idx] = torch.tensor(r["bbox"])

        return (
            masks,
            torchvision.ops.box_convert(b_xywh, "xywh", "xyxy"),
            None,
            None,
            panoptic_image(masks, torch.arange(N) + 1),
        )


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
    image_features: bool = False

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class DenseFeatures(nn.Module):
    """Module to compute dense features per mask."""

    def __init__(self, model_name):
        """Load f3rm module."""
        super(DenseFeatures, self).__init__()
        from f3rm.features.clip import clip as f3rm_clip

        self.model_name = model_name
        self.model, self.preprocess = f3rm_clip.load(model_name)
        print(self.preprocess)

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


class TRTInferenceCLIPVision(TRTInference):
    def __init__(self, engine_path):
        super().__init__(engine_path)

    def _load_engine(self):
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        # Get input/output names
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # Do NOT get output shape here, it may be dynamic or invalid at load time
        self.output_shape = None
        if "ViT-L-14" in self.engine_path:
            self.output_shape = (-1, 768)
        elif "ViT-B-32" in self.engine_path:
            self.output_shape = (-1, 512)

        self.input_host = None
        self.input_device = None
        self.output_host = None
        self.output_device = None
        self.input_shape = None

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        cuda.Context.push(CTX)
        try:
            if self.input_shape != input_tensor.shape:
                self.input_shape = input_tensor.shape
                input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
                input_size = int(np.prod(self.input_shape))
                self.input_host = cuda.pagelocked_empty(input_size, input_dtype)
                self.input_device = cuda.mem_alloc(self.input_host.nbytes)

                # Update TRT context with input shape
                self.context.set_input_shape(self.input_name, self.input_shape)

                # Get output shape AFTER setting input shape
                output_shape = tuple(self.context.get_tensor_shape(self.output_name))
                output_dtype = trt.nptype(
                    self.engine.get_tensor_dtype(self.output_name)
                )
                output_size = int(np.prod(output_shape))

                # Allocate output pinned & device memory
                self.output_host = cuda.pagelocked_empty(output_size, output_dtype)
                self.output_device = cuda.mem_alloc(self.output_host.nbytes)
                self.output_shape = output_shape

            # Copy input to device
            np.copyto(self.input_host, input_tensor.ravel())
            cuda.memcpy_htod(self.input_device, self.input_host)

            bindings = [int(self.input_device), int(self.output_device)]
            self.context.execute_v2(bindings=bindings)

            cuda.memcpy_dtoh(self.output_host, self.output_device)

            return self.output_host.reshape(self.output_shape)

        finally:
            cuda.Context.pop()


class ClipWrapper(nn.Module):
    """Quick wrapper around clip to simplifiy interface for encoding images."""

    def __init__(self, config):
        """Load the visual encoder for CLIP."""
        super(ClipWrapper, self).__init__()

        self.config = config
        if os.path.exists(self.config.model_path):
            self.model = TRTInferenceCLIPVision(self.config.model_path)
            self.model_torch, self._transform = clip.load(
                config.model_name,
                device="cpu",
                download_root=os.environ.get("CLIP_CACHE_DIR"),
            )
        else:
            self.model, self._transform = clip.load(
                config.model_name,
                device="cpu",
                download_root=os.environ.get("CLIP_CACHE_DIR"),
            )
        self._canary_param = nn.Parameter(torch.empty(0))
        self._tokenize = clip.tokenize

        # if DUMMY_IMG_PATH.exists():
        #     img = Image.open(DUMMY_IMG_PATH).convert("RGB")
        #     img = self._transform(img)
        #     if isinstance(self.model, TRTInferenceCLIPVision):
        #         self.model.infer(img[None, ...].cpu().numpy())
        #     else:
        #         self.model.visual(img[None, ...].to(self.model.dtype))

    def to_device(self, device):
        """Move the model to a device."""
        self._canary_param = nn.Parameter(torch.empty(0, device=device))
        if not isinstance(self.model, TRTInferenceCLIPVision):
            # Correctly create a Parameter on the desired device
            self.model.to(device)
            self._transform = self._transform.to(device)
        else:
            self.model_torch.to(device)

    @torch.no_grad()
    def forward(self, imgs):
        """Encode multiple images (without transformation)."""
        # TODO(nathan) think about validation
        if isinstance(self.model, TRTInferenceCLIPVision):
            return torch.from_numpy(self.model.infer(imgs.cpu().numpy())).to(
                self.device
            )
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
        return (
            self._transform.transforms[0].size
            if isinstance(self.model, TRTInferenceCLIPVision)
            else self.model.visual.input_resolution
        )

    @property
    def output_dim_visual(self):
        """Get output dimension for visual encoder."""
        return (
            self.model.output_shape[1]
            if isinstance(self.model, TRTInferenceCLIPVision)
            else self.model.visual.output_dim
        )

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
        if isinstance(self.model, TRTInferenceCLIPVision):
            return self.model_torch.encode_text(tokens)
        return self.model.encode_text(tokens)


@register_config("clip", name="clip", constructor=ClipWrapper)
@dataclasses.dataclass
class ClipConfig(Config):
    """Configuration for OpenCLIP."""

    model_name: str = "ViT-L/14"
    model_path: str = ""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class OpenClipWrapper(nn.Module):
    """Quick wrapper around openclip to simplifiy image encoding interface."""

    def __init__(self, config):
        """Load the visual encoder for OpenCLIP."""
        super(OpenClipWrapper, self).__init__()
        self.config = config
        if os.path.exists(self.config.model_path):
            self.model = TRTInferenceCLIPVision(self.config.model_path)
            (
                self.model_torch,
                _,
                self._transform,
            ) = open_clip.create_model_and_transforms(
                config.model_name, pretrained=config.pretrained, device="cpu"
            )
        else:
            self.model, _, self._transform = open_clip.create_model_and_transforms(
                config.model_name, pretrained=config.pretrained
            )
        self._canary_param = nn.Parameter(torch.empty(0))
        # TODO(nathan) load tokenize function
        self._tokenize = open_clip.get_tokenizer(config.tokenizer_name)

    def to_device(self, device):
        """Move the model to a device."""
        self._canary_param = nn.Parameter(torch.empty(0, device=device))
        if not isinstance(self.model, TRTInferenceCLIPVision):
            # Correctly create a Parameter on the desired device
            self.model.to(device)
            # self._transform = self._transform.to(device)
        else:
            self.model_torch.to(device)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = OpenClipConfig()
        config.update(kwargs)
        return cls(config)

    def forward(self, imgs):
        """Encode multiple images (without transformation)."""
        if isinstance(self.model, TRTInferenceCLIPVision):
            return torch.from_numpy(self.model.infer(imgs.cpu().numpy())).to(
                self.device
            )
        return self.model.visual(imgs)

    @property
    def input_size(self):
        """Get input patch size for clip."""
        return (
            self._transform.transforms[0].size
            if isinstance(self.model, TRTInferenceCLIPVision)
            else self.model.visual.image_size[0]
        )

    @property
    def output_dim_visual(self):
        """Get output dimension for visual encoder."""
        return (
            self.model.output_shape[1]
            if isinstance(self.model, TRTInferenceCLIPVision)
            else self.model.visual.output_dim
        )

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
        if isinstance(self.model, TRTInferenceCLIPVision):
            return self.model_torch.encode_text(tokens)
        return self.model.encode_text(tokens)


@register_config("clip", name="open_clip", constructor=OpenClipWrapper)
@dataclasses.dataclass
class OpenClipConfig(Config):
    """Configuration for OpenCLIP."""

    model_name: str = "ViT-L/14"
    tokenizer_name: str = "ViT-L-14"
    pretrained: str = "openai"
    model_path: str = ""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
