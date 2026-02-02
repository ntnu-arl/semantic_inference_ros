# Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
# Technology All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree


from semantic_inference_python.models.wrappers import InstructBLIP
from transformers import InstructBlipProcessor
import dataclasses
import torch
import cv2
from pathlib import Path
from torch import nn
from typing import Optional

MODEL_ONNX_SAVE_PATH = "/home/arl/jetson_ssd/cache/instructblip/onnx"
Path(MODEL_ONNX_SAVE_PATH).mkdir(exist_ok=True, parents=True)
IMAGE_PATH = "/home/arl/jetson_ssd/hydra_bags/image.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclasses.dataclass
class InstructBLIPConfig:
    model_name: str = "Salesforce/instructblip-vicuna-7b"
    max_length: int = 512
    do_sample: bool = False
    num_beams: int = 5
    top_p: float = 0.9
    repetition_penalty: float = 1.5
    length_penalty: float = 1.0
    temperature: float = 1.0


# Instantiate model
blip = InstructBLIP(InstructBLIPConfig())
blip = blip.to(DEVICE).eval()

# Load processor
processor = InstructBlipProcessor.from_pretrained(blip.config.model_name)

# Define wrapper
class BLIPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        image_embeds: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        max_length_tensor: Optional[torch.LongTensor] = None,
        do_sample_tensor: Optional[torch.BoolTensor] = None,
        num_beams_tensor: Optional[torch.LongTensor] = None,
        top_p_tensor: Optional[torch.FloatTensor] = None,
        repetition_penalty_tensor: Optional[torch.FloatTensor] = None,
        length_penalty_tensor: Optional[torch.FloatTensor] = None,
        temperature_tensor: Optional[torch.FloatTensor] = None,
    ):

        # Convert tensor inputs to Python scalars with fallback defaults
        max_length = max_length_tensor.item() if max_length_tensor is not None else 512
        do_sample = (
            bool(do_sample_tensor.item()) if do_sample_tensor is not None else False
        )
        num_beams = num_beams_tensor.item() if num_beams_tensor is not None else 5
        top_p = top_p_tensor.item() if top_p_tensor is not None else 0.9
        repetition_penalty = (
            repetition_penalty_tensor.item()
            if repetition_penalty_tensor is not None
            else 1.5
        )
        length_penalty = (
            length_penalty_tensor.item() if length_penalty_tensor is not None else 1.0
        )
        temperature = (
            temperature_tensor.item() if temperature_tensor is not None else 1.0
        )

        output = self.model.generate_caption(
            image_embeds=image_embeds,
            qformer_input_ids=qformer_input_ids,
            qformer_attention_mask=qformer_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        return output


# Preprocess image
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Encode image
pixel_values = blip.encode_image(img).to(DEVICE)  # torch.FloatTensor

# Prepare text inputs
TEXT = ["dummy text"]
text_input = processor(
    images=None, text=TEXT, return_tensors="pt", padding=True, truncation=True
)
text_input = {k: v.to(DEVICE) for k, v in text_input.items()}


max_length_tensor = torch.tensor([blip.config.max_length], dtype=torch.int64)
do_sample_tensor = torch.tensor([blip.config.do_sample], dtype=torch.bool)
num_beams_tensor = torch.tensor([blip.config.num_beams], dtype=torch.int64)
top_p_tensor = torch.tensor([blip.config.top_p], dtype=torch.float32)
repetition_penalty_tensor = torch.tensor(
    [blip.config.repetition_penalty], dtype=torch.float32
)
length_penalty_tensor = torch.tensor([blip.config.length_penalty], dtype=torch.float32)
temperature_tensor = torch.tensor([blip.config.temperature], dtype=torch.float32)

# Export model to ONNX
torch.onnx.export(
    BLIPWrapper(blip),
    (
        pixel_values,
        text_input.get("qformer_input_ids", None),
        text_input.get("qformer_attention_mask", None),
        text_input.get("input_ids", None),
        text_input.get("attention_mask", None),
        max_length_tensor,
        do_sample_tensor,
        num_beams_tensor,
        top_p_tensor,
        repetition_penalty_tensor,
        length_penalty_tensor,
        temperature_tensor,
    ),
    f"{MODEL_ONNX_SAVE_PATH}/instructblip.onnx",
    input_names=[
        "pixel_values",
        "qformer_input_ids",
        "qformer_attention_mask",
        "input_ids",
        "attention_mask",
        "max_length",
        "do_sample",
        "num_beams",
        "top_p",
        "repetition_penalty",
        "length_penalty",
        "temperature",
    ],
    output_names=["generated_caption"],
    opset_version=13,
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "generated_caption": {0: "batch_size", 1: "seq_len"},
    },
)
