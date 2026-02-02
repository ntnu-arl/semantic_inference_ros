# Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
# Technology All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree

from semantic_inference_python.models.wrappers import InstructBLIPVisualEncoder
import dataclasses
import torch
import cv2
from pathlib import Path

MODEL_SAVE_PATH = "/home/arl/jetson_ssd/cache/instructblip_vision"
MODEL_ONNX_SAVE_PATH = "/home/arl/jetson_ssd/cache/instructblip_vision/onnx"
Path(MODEL_ONNX_SAVE_PATH).mkdir(exist_ok=True, parents=True)
IMAGE_PATH = "/home/arl/jetson_ssd/hydra_bags/image.png"


@dataclasses.dataclass
class InstructBLIPVisualConfig:
    model_name: str = "Salesforce/instructblip-vicuna-7b"


blip_visual = InstructBLIPVisualEncoder(InstructBLIPVisualConfig())

blip_visual.model.save_pretrained(MODEL_SAVE_PATH)

img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pixel_values = blip_visual.processor(images=img, text=None, return_tensors="pt")[
    "pixel_values"
]
torch.onnx.export(
    blip_visual.model,
    pixel_values,
    MODEL_ONNX_SAVE_PATH + "/instructblip_vision.onnx",
    input_names=["pixel_values"],
    output_names=["last_hidden_state"],
    opset_version=13,
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "last_hidden_state": {0: "batch_size"},
    },
)

# TRT: /usr/src/tensorrt/bin/trtexec --onnx=instructblip_vision.onnx --saveEngine=instructblip_vision.engine --explicitBatch
