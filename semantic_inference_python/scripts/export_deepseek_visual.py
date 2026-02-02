# Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
# Technology All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree

import torch
import torch.nn as nn
from pathlib import Path
import json
import copy

from semantic_inference_python.models.deepseek.deepseek_vl2.models import (
    DeepseekVLV2ForCausalLM,
    DeepseekVLV2Processor,
    select_best_resolution,
    VisionTransformer,
    MlpProjector,
    VisionEncoderConfig,
    MlpProjectorConfig,
)

base_path = Path("/home/arl/jetson_ssd/cache/deepseek-vl2-vision")
with (base_path / "config.json").open() as f:
    models_config = json.load(f)
vision_config = VisionEncoderConfig(**models_config["vision_config"])
projector_config = MlpProjectorConfig(**models_config["projector_config"])
vision = VisionTransformer(
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
projector = MlpProjector(projector_config)


class LayerNormExportFriendly(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            y = y * self.weight + self.bias
        return y


# Recursively replace LayerNorm with export-friendly version
def replace_layernorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(
                module,
                name,
                LayerNormExportFriendly(
                    child.normalized_shape, child.eps, child.elementwise_affine
                ),
            )
        else:
            replace_layernorm(child)


replace_layernorm(vision)


def replace_dropout_with_identity(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Dropout):
            setattr(module, name, torch.nn.Identity())
        else:
            replace_dropout_with_identity(child)


vision.load_state_dict(torch.load(base_path / "vision.pt", map_location="cpu"))
projector.load_state_dict(torch.load(base_path / "projector.pt", map_location="cpu"))
vision.to(torch.float16).eval().to("cuda")
projector.to(torch.float16).eval().to("cuda")

vision_copy = copy.deepcopy(vision)
replace_dropout_with_identity(vision_copy)
vision_copy.eval()


# Paths
onnx_path_vision = base_path / "onnx" / "vision.onnx"
onnx_path_projector = base_path / "onnx" / "projector.onnx"
onnx_path_vision.parent.mkdir(parents=True, exist_ok=True)
onnx_path_projector.parent.mkdir(parents=True, exist_ok=True)

# Dummy input for export
dummy_vision_input = torch.randn(1, 3, 384, 384, device="cuda").to(torch.float16)
dummy_projector_input = torch.randn(1, 729, 1152, device="cuda").to(torch.float16)
# Export to ONNX
from torch.onnx import OperatorExportTypes
import torch.nn.functional as F


def no_dropout(*args, **kwargs):
    # Ignores dropout, just returns input
    return args[0]


# Save original
original_dropout = F.dropout
# Replace with no_dropout
F.dropout = no_dropout

torch.onnx.export(
    vision_copy,
    dummy_vision_input,
    onnx_path_vision.as_posix(),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}},
    operator_export_type=OperatorExportTypes.ONNX,
    training=torch.onnx.TrainingMode.EVAL,
)
print(f"ONNX model saved to {onnx_path_vision}")
torch.onnx.export(
    projector,
    dummy_projector_input,
    onnx_path_projector.as_posix(),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["tokens"],
    output_names=["proj_output"],
    dynamic_axes={"tokens": {0: "batch_size"}, "proj_output": {0: "batch_size"}},
)
print(f"ONNX model saved to {onnx_path_projector}")

vision_out = vision_copy(dummy_vision_input)
print(f"Vision output shape: {vision_out.shape}")
