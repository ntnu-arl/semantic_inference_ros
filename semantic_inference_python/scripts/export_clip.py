# Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
# Technology All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree

import torch
import open_clip
import os
from PIL import Image
from pathlib import Path
import torch.nn as nn

IMAGE_PATH = "/home/arl/sdcard/hydra_bags/image.png"
MODEL_NAME = "MobileCLIP-S2"  # open_clip uses dashes not slashes
PRETRAINED_NAME = "datacompdr"
EXPORT_ONNX_PATH = f"/home/arl/jetson_ssd/cache/open_clip/{MODEL_NAME}"
Path(EXPORT_ONNX_PATH).mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED_NAME
)
model = model.to(DEVICE).eval()

img = Image.open(IMAGE_PATH).convert("RGB")
dummy_img = preprocess(img)[None, ...].to(DEVICE).to(torch.float32)  # (1,3,224,224)
dummy_text = open_clip.tokenize(["a photo of something"]).to(DEVICE)  # (1,77)


class EncodeImageWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.encode_image(x)


# Image encoder export
torch.onnx.export(
    EncodeImageWrapper(model),
    dummy_img,
    os.path.join(EXPORT_ONNX_PATH, f"{MODEL_NAME}-image.onnx"),
    input_names=["image"],
    output_names=["image_features"],
    dynamic_axes={"image": {0: "batch_size"}, "image_features": {0: "batch_size"}},
    opset_version=14,
    do_constant_folding=True,
)
print("✅ Exported image encoder.")
print(
    f"""
🚀 You can now build TensorRT engines using:
  trtexec --onnx={EXPORT_ONNX_PATH}/{MODEL_NAME}-image.onnx --saveEngine={MODEL_NAME}-image.engine --fp16 --explicitBatch
  trtexec --onnx={EXPORT_ONNX_PATH}/{MODEL_NAME}-text.onnx  --saveEngine={MODEL_NAME}-text.engine  --fp16 --explicitBatch
"""
)
