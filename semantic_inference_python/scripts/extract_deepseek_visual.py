# Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
# Technology All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree

from logging import config
from semantic_inference_python.models.deepseek.deepseek_vl2.models import (
    DeepseekVLV2ForCausalLM,
)
import torch
import argparse
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Deepseek-VL2 visual model from full VLM."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/deepseek-vl2",
        help="Name of the model to extract.",
        choices=[
            "deepseek-ai/deepseek-vl2",
            "deepseek-ai/deepseek-vl2-small",
            "deepseek-ai/deepseek-vl2-tiny",
        ],
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the extracted visual model.",
    )
    return parser.parse_args()


def main(model_name: str, output_path: str) -> None:
    """Extract Deepseek-VL2 visual model from full VLM.
    :param model_name: Name of the model to extract.
    :param output_path: Path to save the extracted visual model.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model = (
        DeepseekVLV2ForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        .to(torch.bfloat16)
        .eval()
    )

    config = model.config.to_dict()

    with (output_path / "config.json").open("w") as f:
        json.dump(config, f, indent=4)

    torch.save(model.vision.state_dict(), output_path / "vision.pt")
    torch.save(model.projector.state_dict(), output_path / "projector.pt")

    print(f"Extracted visual model saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args.model_name, args.output_path)
