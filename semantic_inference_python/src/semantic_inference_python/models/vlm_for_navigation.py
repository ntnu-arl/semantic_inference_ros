# Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
# Technology All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree
"""Model to reason about edges and extract useful information for navigation."""

from typing import Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import copy

import torch
import rospy
from tqdm import tqdm

from semantic_inference_python.config import Config, config_field
from semantic_inference_python.client import (
    FastAPIClientConfig,
    InstructBlipFastAPIClient,
    OpenAIClientConfig,
    OpenAIClient,
)


@dataclass(frozen=True)
class ObjectInput:
    """Input to the object model."""

    id: int
    relation_ids: torch.Tensor
    relation_features: torch.Tensor
    object_labels: List[Tuple[str, str]]
    object_colors: List[Tuple[str, str]]
    num_observations: List[int]
    prompts: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Post initialization checks."""
        if len(self.relation_ids) != len(self.relation_features):
            raise ValueError(
                "relation_ids and relation_features must have the same length."
            )


@dataclass
class VLMForNavigationInput:
    """Input to the VLM for navigation model."""

    general_prompt: str = ""
    object_inputs: List[ObjectInput] = field(default_factory=list)


@dataclass
class VLMForNavigationOutput:
    """Output of the VLM for navigation model."""

    selected_object_ids: List[int] = field(default_factory=list)
    selected_explanations: List[str] = field(default_factory=list)
    selected_prompts: List[str] = field(default_factory=list)
    non_selected_object_ids: List[int] = field(default_factory=list)
    non_selected_explanations: List[str] = field(default_factory=list)
    non_selected_prompts: List[str] = field(default_factory=list)
    all_explanations: List[str] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)


@dataclass
class VLMForNavigationConfig(Config):
    """Main configuration for the VLM for navigation model."""

    vlm: Any = config_field("vlm", default="instruct_blip")
    use_server: bool = False
    vlm_client_config: FastAPIClientConfig = field(default_factory=FastAPIClientConfig)
    use_llm_response_parser: bool = True
    openai_client_config: OpenAIClientConfig = field(default_factory=OpenAIClientConfig)
    llm_response_parser_prompt_path: str = ""
    include_bbs: bool = False
    verbose: bool = False

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class VLMForNavigation:
    """Module to reason about edges and extract useful information for navigation."""

    def __init__(self, config: VLMForNavigationConfig):
        """Construct a VLM for navigation model."""
        super(VLMForNavigation, self).__init__()
        self.config = config
        self._publish_current = False
        self._device = torch.device("cpu")
        if self.config.use_server:
            self.client = InstructBlipFastAPIClient(self.config.vlm_client_config)
        else:
            self.vlm = self.config.vlm.create()

        if self.config.use_llm_response_parser:
            if Path(self.config.llm_response_parser_prompt_path).exists():
                with open(self.config.llm_response_parser_prompt_path, "r") as f:
                    self.llm_response_parser_prompt = f.read().strip()
                self.llm_response_parser_client = OpenAIClient(
                    config=self.config.openai_client_config,
                    system_prompt=self.llm_response_parser_prompt,
                )
            else:
                rospy.logerr(
                    "[VLM for navigation] LLM response parser prompt file not found."
                )
                self.llm_response_parser_prompt = None
                self.llm_response_parser_client = None
        else:
            self.llm_response_parser_prompt = None
            self.llm_response_parser_client = None

    @property
    def device(self):
        """Get the device of the model."""
        return self._device

    @device.setter
    def device(self, device):
        """Set the device of the model."""
        self._device = device

    @property
    def publish_current(self):
        """Get the flag to publish current results."""
        return self._publish_current

    @publish_current.setter
    def publish_current(self, value):
        """Set the flag to publish current results."""
        self._publish_current = value

    def move_to(self, device):
        """Move the model to a device."""
        if not self.config.use_server:
            self.vlm.to(device)
        self.device = device

    def reason_object(self, base_prompt: List[str], object_input: ObjectInput):
        """Reasoning for the VLM for navigation model."""

        all_prompts = list()
        ordered_relation_ids = list()
        i = 0
        for relation_id, object_label in zip(
            object_input.relation_ids, object_input.object_labels
        ):
            object_index = 0 if relation_id[0] == object_input.id else 1
            other_index = 0 if relation_id[1] == object_input.id else 1
            ordered_relation_ids.append(
                [relation_id[object_index], relation_id[other_index]]
            )
            if len(object_input.prompts) > 0:
                prompt = object_input.prompts[i]
                if self.config.include_bbs:
                    prompt += f" Focus only on the objects that have a {object_input.object_colors[i][0]} and "
                    prompt += f"a {object_input.object_colors[i][1]} bounding box."
                all_prompts.append(prompt)
            else:
                # If no prompt is provided, use the base prompt
                all_prompts.append(
                    base_prompt[0]
                    + str(object_label[object_index])
                    + base_prompt[1]
                    + str(object_label[other_index])
                    + base_prompt[2]
                )
            i += 1
        if self.config.use_server:
            # Use the FastAPI client to generate captions
            try:
                job_id = self.client.submit_job(
                    all_prompts, object_input.relation_features.cpu()
                )
                generated_output = self.client.get_result(job_id)
            except:
                rospy.logerr("[vlm_for_navigation] Server error")
                return [], [], [], [], [], []
        else:
            generated_output = self.vlm.generate_caption(
                object_input.relation_features.to(self.vlm.device), all_prompts
            )

        (
            selected_ids,
            selected_explanations,
            selected_prompts,
            non_selected_ids,
            non_selected_explanations,
            non_selected_prompts,
            all_explanations,
        ) = self._parse_llm_response(
            generated_output, all_prompts, ordered_relation_ids
        )

        return (
            selected_ids,
            selected_explanations,
            selected_prompts,
            non_selected_ids,
            non_selected_explanations,
            non_selected_prompts,
            all_prompts,
            all_explanations,
        )

    def _parse_llm_response(
        self,
        generated_output: List[str],
        all_prompts: List[str],
        ordered_relation_ids: List[List[int]],
    ) -> Tuple[
        List[int], List[str], List[str], List[int], List[str], List[str], List[str]
    ]:
        """Parse the LLM response to extract selected and non-selected object IDs and explanations.
        :param generated_output: List of generated outputs from the LLM.
        :param all_prompts: List of prompts used for the LLM.
        :param ordered_relation_ids: List of ordered relation IDs corresponding to the generated output.
        :return: Tuple containing selected IDs, selected explanations, non-selected IDs, non-selected explanations, and all explanations."""
        selected_ids = []
        selected_explanations = []
        non_selected_ids = []
        non_selected_explanations = []
        all_explanations = []
        selected_prompts = []
        non_selected_prompts = []

        for i in range(len(generated_output)):
            output = copy.deepcopy(generated_output[i])
            if self.llm_response_parser_client is not None:
                # Use the LLM response parser to process the generated output
                llm_prompt = (
                    f"\n VLM's prompt: {all_prompts[i]}\n VLM's response: {output}\n"
                )
                output, success = self.llm_response_parser_client.generate_response(
                    llm_prompt, log=self.config.verbose
                )
                if not success:
                    rospy.logerr(
                        "[VLM for navigation] Error parsing LLM response: {}".format(
                            output
                        )
                    )
                    continue

            if "yes" in output.lower():
                selected_ids.append(ordered_relation_ids[i][0])
                selected_ids.append(ordered_relation_ids[i][1])
                selected_explanations.append(generated_output[i])
                selected_prompts.append(all_prompts[i])
            else:
                non_selected_ids.append(ordered_relation_ids[i][0])
                non_selected_ids.append(ordered_relation_ids[i][1])
                non_selected_explanations.append(generated_output[i])
                non_selected_prompts.append(all_prompts[i])
            all_explanations.append(generated_output[i])
        return (
            selected_ids,
            selected_explanations,
            selected_prompts,
            non_selected_ids,
            non_selected_explanations,
            non_selected_prompts,
            all_explanations,
        )

    @torch.inference_mode()
    def reason(self, inputs: VLMForNavigationInput):
        """Forward pass."""
        output = VLMForNavigationOutput()
        if len(inputs.object_inputs) > 0:
            if len(inputs.object_inputs[0].prompts) > 0:
                base_prompt = ""
        else:
            rospy.logerr("[VLM for navigation] No prompts provided.")
            return output

        for object_input in tqdm(inputs.object_inputs):
            (
                selected_ids,
                selected_explanations,
                selected_prompts,
                non_selected_ids,
                non_selected_explanations,
                non_selected_prompts,
                prompts,
                all_explanations,
            ) = self.reason_object(base_prompt, object_input)
            output.selected_object_ids.extend(selected_ids)
            output.selected_explanations.extend(selected_explanations)
            output.selected_prompts.extend(selected_prompts)
            output.non_selected_object_ids.extend(non_selected_ids)
            output.non_selected_explanations.extend(non_selected_explanations)
            output.non_selected_prompts.extend(non_selected_prompts)
            output.prompts.extend(prompts)
            output.all_explanations.extend(all_explanations)
            if self._publish_current:
                self.publish_current = False
                rospy.logwarn(
                    f"Stopping inference early after processing {len(output.selected_object_ids)}/{len(inputs.object_inputs)} objects."
                )
                return output
        return output

    @torch.inference_mode()
    def reason_one(self, object_input: ObjectInput, general_prompt: str):
        """Reasoning for a single object input."""
        output = VLMForNavigationOutput()
        if len(object_input.prompts) > 0:
            base_prompt = ""
        else:
            rospy.logerr("[VLM for navigation] No prompts provided.")
            return output
        (
            selected_ids,
            selected_explanations,
            selected_prompts,
            non_selected_ids,
            non_selected_explanations,
            non_selected_prompts,
            prompts,
            all_explanations,
        ) = self.reason_object(base_prompt, object_input)

        output.selected_object_ids.extend(selected_ids)
        output.selected_explanations.extend(selected_explanations)
        output.selected_prompts.extend(selected_prompts)
        output.non_selected_object_ids.extend(non_selected_ids)
        output.non_selected_explanations.extend(non_selected_explanations)
        output.non_selected_prompts.extend(non_selected_prompts)
        output.prompts.extend(prompts)
        output.all_explanations.extend(all_explanations)

        return output
