"""Prompt parser methods for open-vocab and reasoning-enhanced navigation."""

from dataclasses import dataclass, field
from typing import Dict, List

import json
import yaml
import numpy as np
from pathlib import Path

from semantic_inference_python.config import Config, config_field, register_config
from semantic_inference_python.models import default_device
from semantic_inference_python.client import OpenAIClient, OpenAIClientConfig


@dataclass
class ObjectsPromptPair:
    """Data structure for object and prompt pair."""

    object: str = ""
    subject: str = ""
    prompt: str = ""


@dataclass
class NavigationPrompterOutput:
    """Output data structure for navigation prompter."""

    objects: List[str] = field(default_factory=list)
    objects_embeddings: List[np.ndarray] = field(default_factory=list)
    room_embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    prompt: str = ""
    room: str = ""
    objects_prompt_pairs: List[ObjectsPromptPair] = field(default_factory=list)
    success: bool = False
    error: str = ""


class HardCodedPrompter:
    """Hard-coded navigation prompter for demonstration purposes."""

    def __init__(self, config) -> None:
        """Construct a navigation prompt service node."""
        self.config = config
        self.clip_model = self.config.clip_model.create().to(
            default_device(self.config.use_cuda)
        )

    @classmethod
    def construct(cls, **kwargs):
        """Construct a HardCodedPrompter instance."""
        config = HardCodedPrompterConfig()
        config.update(kwargs)
        return cls(config)

    def generate(self, prompt: str, room: str) -> NavigationPrompterOutput:
        prompt_key = next((k for k in self.config.prompt_objects if k in prompt), None)
        output = NavigationPrompterOutput()
        if not prompt_key:
            output.error = "Unsupported prompt"
            return output

        prompt_config = self.config.prompt_objects[prompt_key]

        # Handle room ID
        if prompt_config["required_room"]:
            if not room.isdigit():
                output.error = "Room must be an integer"
                return output
            output.room = f"R({room})"

        # Generate object embeddings
        for obj_name in prompt_config["objects"]:
            output.objects.append(obj_name)
            output.objects_embeddings.append(
                self.clip_model.embed_text(obj_name).cpu().numpy()
            )

        # Generate room embedding if applicable
        if prompt_config["room_desc"]:
            output.room_embedding = (
                self.clip_model.embed_text(prompt_config["room_desc"]).cpu().numpy()
            )

        output.prompt = prompt
        output.success = True
        return output


@register_config(
    "navigation_prompter",
    name="hard_coded",
    constructor=HardCodedPrompter,
)
@dataclass
class HardCodedPrompterConfig(Config):
    """Configuration for the hard-coded navigation prompter."""

    clip_model: str = config_field("clip", default="open_clip")
    use_cuda: bool = True
    prompt_objects = {
        "clean": {
            "required_room": True,
            "objects": ["chair"],
            "room_desc": None,
        },
        "prepare": {
            "required_room": False,
            "objects": ["monitor"],
            "room_desc": "a place with monitors or computers",
        },
    }


class OpenAIPrompter:
    def __init__(self, config) -> None:
        """Construct a navigation prompter using OpenAI."""
        self.config = config
        self.clip_model = self.config.clip_model.create().to(
            default_device(self.config.use_cuda)
        )
        self.system_prompt = None
        if Path(self.config.system_prompts_path).exists():
            with open(self.config.system_prompts_path, "r") as f:
                self.system_prompt = f.read().strip()
        if Path(self.config.labels_path).exists():
            with open(self.config.labels_path, "r") as f:
                data = yaml.safe_load(f)
                object_labels = data.get("object_labels", [])
                labels = ", ".join(
                    label["name"]
                    for label in data["label_names"]
                    if label["label"] in object_labels
                )
                self.labels = labels.split(", ")
            self.system_prompt += f"\n[{labels}]\n"
        if Path(self.config.examples_path).exists():
            with open(self.config.examples_path, "r") as f:
                examples = f.read().strip()
            self.system_prompt += f"\n{examples}\n"
        self.client = OpenAIClient(
            config=self.config.client_config, system_prompt=self.system_prompt
        )
        print(f"System prompt: {self.system_prompt}")

    def construct(self, **kwargs):
        """Construct an OpenAIPrompter instance."""
        config = OpenAIPrompterConfig()
        config.update(kwargs)
        return OpenAIPrompter(config)

    @staticmethod
    def _to_json(response: str) -> Dict:
        """Convert a string response to a JSON dictionary."""
        if response.startswith("```json"):
            response = response[7:-3].strip()
        elif response.startswith("```"):
            response = response[3:-3].strip()

        return json.loads(response)

    def generate(self, prompt: str, room: str) -> NavigationPrompterOutput:
        """Generate navigation prompts using OpenAI."""
        output = NavigationPrompterOutput(prompt=prompt)
        if not (room.isdigit() or room == "all" or room == "find"):
            output.error = "Room must be an integer, 'all', or 'find'"
            return output
        if room.isdigit():
            output.room = f"R({room})"
        else:
            output.room = room
        try:
            prompt = f"Task: {prompt}"
            response, success = self.client.generate_response(prompt)
            if not success:
                output.error = response
                return output
        except Exception as e:
            output.error = f"Error generating response: {e}"
            return output

        response = self._to_json(response)

        if "objects" not in response:
            output.error = "Response does not contain 'objects' key"
            return output
        # if 'interactions' not in response:
        #     output.error = "Response does not contain 'interactions' key"
        #     return output

        output.objects = response["objects"]
        embeddings = self.clip_model.embed_text(output.objects).cpu().numpy()
        output.objects_embeddings = [embeddings[i] for i in range(len(output.objects))]
        if "interactions" in response:
            for _, pair in response["interactions"].items():
                if (
                    pair["objects"][0].lower() in pair["prompt"].lower()
                    and pair["objects"][1].lower() in pair["prompt"].lower()
                ):
                    if pair["objects"][0].lower() == "table":
                        pair["objects"][0] = "desk"
                    if pair["objects"][1].lower() == "table":
                        pair["objects"][1] = "desk"
                    if (
                        not pair["objects"][0].lower() in self.labels
                        or not pair["objects"][1].lower() in self.labels
                    ):
                        continue
                    object_prompt_pair = ObjectsPromptPair(
                        object=pair["objects"][0],
                        subject=pair["objects"][1],
                        prompt=pair["prompt"],
                    )
                    output.objects_prompt_pairs.append(object_prompt_pair)
                    if pair["objects"][0].lower() not in output.objects:
                        output.objects.append(pair["objects"][0].lower())
                    if pair["objects"][1].lower() not in output.objects:
                        output.objects.append(pair["objects"][1].lower())

        output.success = True
        return output


@register_config("navigation_prompter", name="openai", constructor=OpenAIPrompter)
@dataclass
class OpenAIPrompterConfig(Config):
    """Configuration for the OpenAI navigation prompter."""

    client_config: OpenAIClientConfig = field(default_factory=OpenAIClientConfig)
    system_prompts_path: str = ""
    examples_path: str = ""
    labels_path: str = ""
    clip_model: str = config_field("clip", default="open_clip")
    use_cuda: bool = True
