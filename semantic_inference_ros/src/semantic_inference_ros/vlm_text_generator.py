"""Generate text using Instruct VLM given a prompt and image encodings."""

import rospy
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import json
from typing import Dict, List
from enum import Enum

from hydra_msgs.msg import VisualRelationshipsEncodings, LabeledRelationships


class VLMGeometry(Enum):
    """Enum for geometry options."""

    NONE = "none"
    BOUNDING_BOX = "bounding_box"
    CENTER = "center"
    CORNERS = "corners"


class VLMTextGenerator:
    """Node implementation."""

    def __init__(self, model, geometry: VLMGeometry = VLMGeometry.NONE) -> None:
        """Construct a text generator node."""
        self._model = model
        self._pub = rospy.Publisher(
            "labeled_relationships", LabeledRelationships, queue_size=1
        )
        self._sub = rospy.Subscriber(
            "visual_relationships_encodings",
            VisualRelationshipsEncodings,
            self._callback,
            queue_size=1,
        )

        self._geometry = geometry

    def _callback(self, msg: VisualRelationshipsEncodings) -> None:
        rospy.loginfo("Received visual relationships encodings.")
        labeled_relationships = LabeledRelationships()
        visual_embeddings = list()
        prompts = list()
        for i, feature in enumerate(msg.features.feature):
            visual_embeddings.append(
                torch.tensor(feature.data)
                .reshape(feature.rows, feature.cols)
                .to(self._model.device)
            )

            prompt = (
                f"{msg.prompt} the {msg.object_classes[int(2 * i)]}"
                f" and the {msg.object_classes[int(2 * i + 1)]}?"
            )
            if self._geometry == VLMGeometry.CORNERS:
                geometry_json = dict()
                geometry_json[msg.object_classes[int(2 * i)]] = self._to_corners(
                    np.array(
                        [
                            msg.object_bounding_boxes.boxes[
                                int(2 * i)
                            ].center.position.x,
                            msg.object_bounding_boxes.boxes[
                                int(2 * i)
                            ].center.position.y,
                            msg.object_bounding_boxes.boxes[
                                int(2 * i)
                            ].center.position.z,
                        ]
                    ),
                    np.array(
                        [
                            msg.object_bounding_boxes.boxes[
                                int(2 * i)
                            ].center.orientation.x,
                            msg.object_bounding_boxes.boxes[
                                int(2 * i)
                            ].center.orientation.y,
                            msg.object_bounding_boxes.boxes[
                                int(2 * i)
                            ].center.orientation.z,
                            msg.object_bounding_boxes.boxes[
                                int(2 * i)
                            ].center.orientation.w,
                        ]
                    ),
                    np.array(
                        [
                            msg.object_bounding_boxes.boxes[int(2 * i)].size.x,
                            msg.object_bounding_boxes.boxes[int(2 * i)].size.y,
                            msg.object_bounding_boxes.boxes[int(2 * i)].size.z,
                        ]
                    ),
                )
                geometry_json[msg.object_classes[int(2 * i)] + 1] = self._to_corners(
                    np.array(
                        [
                            msg.object_bounding_boxes.boxes[
                                int(2 * i + 1)
                            ].center.position.x,
                            msg.object_bounding_boxes.boxes[
                                int(2 * i + 1)
                            ].center.position.y,
                            msg.object_bounding_boxes.boxes[
                                int(2 * i + 1)
                            ].center.position.z,
                        ]
                    ),
                    np.array(
                        [
                            msg.object_bounding_boxes.boxes[
                                int(2 * i + 1)
                            ].center.orientation.x,
                            msg.object_bounding_boxes.boxes[
                                int(2 * i + 1)
                            ].center.orientation.y,
                            msg.object_bounding_boxes.boxes[
                                int(2 * i + 1)
                            ].center.orientation.z,
                            msg.object_bounding_boxes.boxes[
                                int(2 * i + 1)
                            ].center.orientation.w,
                        ]
                    ),
                    np.array(
                        [
                            msg.object_bounding_boxes.boxes[int(2 * i + 1)].size.x,
                            msg.object_bounding_boxes.boxes[int(2 * i + 1)].size.y,
                            msg.object_bounding_boxes.boxes[int(2 * i + 1)].size.z,
                        ]
                    ),
                )
                prompt += (
                    f" The geometries of the objcets are {json.dumps(geometry_json)}."
                )
            elif self._geometry == VLMGeometry.BOUNDING_BOX:
                geometry_json = {
                    msg.object_classes[int(2 * i)]: {"bounding_box": {}},
                    msg.object_classes[int(2 * i) + 1]: {"bounding_box": {}},
                }
                geometry_json[msg.object_classes[int(2 * i)]]["object_center"] = [
                    f"{msg.object_poses[int(2 * i)].position.x:.2f}",
                    f"{msg.object_poses[int(2 * i)].position.y:.2f}",
                    f"{msg.object_poses[int(2 * i)].position.z:.2f}",
                ]
                geometry_json[msg.object_classes[int(2 * i)]]["bounding_box"][
                    "center"
                ] = [
                    f"{msg.object_bounding_boxes.boxes[int(2 * i)].center.position.x:.2f}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i)].center.position.y:.2f}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i)].center.position.z:.2f}",
                ]
                geometry_json[msg.object_classes[int(2 * i)]]["bounding_box"][
                    "quaternion"
                ] = [
                    f"{msg.object_bounding_boxes.boxes[int(2 * i)].center.orientation.x:.2f}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i)].center.orientation.y:.2f}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i)].center.orientation.z:.2f}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i)].center.orientation.w:.2f}",
                ]
                geometry_json[msg.object_classes[int(2 * i)]]["bounding_box"][
                    "axis_size"
                ] = [
                    f"{msg.object_bounding_boxes.boxes[int(2 * i)].size.x:.2f}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i)].size.y:.2f}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i)].size.z:.2f}",
                ]
                geometry_json[msg.object_classes[int(2 * i) + 1]]["object_center"] = [
                    f"{msg.object_poses[int(2 * i + 1)].position.x:.2f}",
                    f"{msg.object_poses[int(2 * i + 1)].position.y:.2f}",
                    f"{msg.object_poses[int(2 * i + 1)].position.z:.2f}",
                ]
                geometry_json[msg.object_classes[int(2 * i) + 1]]["bounding_box"][
                    "center"
                ] = [
                    f"{msg.object_bounding_boxes.boxes[int(2 * i + 1)].center.position.x:.2f}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i + 1)].center.position.y:.2f}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i + 1)].center.position.z:.2f}",
                ]
                geometry_json[msg.object_classes[int(2 * i) + 1]]["bounding_box"][
                    "quaternion"
                ] = [
                    f"{msg.object_bounding_boxes.boxes[int(2 * i + 1)].center.orientation.x}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i + 1)].center.orientation.y}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i + 1)].center.orientation.z}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i + 1)].center.orientation.w}",
                ]
                geometry_json[msg.object_classes[int(2 * i) + 1]]["bounding_box"][
                    "axis_size"
                ] = [
                    f"{msg.object_bounding_boxes.boxes[int(2 * i + 1)].size.x:.2f}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i + 1)].size.y:.2f}",
                    f"{msg.object_bounding_boxes.boxes[int(2 * i + 1)].size.z:.2f}",
                ]
                prompt += (
                    f" The geometries of the objcets are {json.dumps(geometry_json)}."
                )
            elif self._geometry == VLMGeometry.CENTER:
                first_object_center = [
                    f"{msg.object_poses[int(2 * i)].position.x:.2f}",
                    f"{msg.object_poses[int(2 * i)].position.y:.2f}",
                    f"{msg.object_poses[int(2 * i)].position.z:.2f}",
                ]
                second_object_center = [
                    f"{msg.object_poses[int(2 * i + 1)].position.x:.2f}",
                    f"{msg.object_poses[int(2 * i + 1)].position.y:.2f}",
                    f"{msg.object_poses[int(2 * i + 1)].position.z:.2f}",
                ]
                prompt = (
                    f"{msg.prompt} the {msg.object_classes[int(2 * i)]} "
                    f"(spatially locared at {first_object_center} meters) "
                    f" and the {msg.object_classes[int(2 * i + 1)]} "
                    f"(spatially locared at {second_object_center} meters)?"
                )
            prompts.append(prompt)
            labeled_relationships.node_ids.append(msg.features.ids[int(2 * i)])
            labeled_relationships.node_ids.append(msg.features.ids[int(2 * i + 1)])

        labeled_relationships.labels = self._model.generate_caption(
            torch.stack(visual_embeddings), prompts
        )
        labeled_relationships.header.stamp = rospy.Time.now()

        self._pub.publish(labeled_relationships)
        rospy.loginfo("Published labeled relationships.")

    @staticmethod
    def _to_corners(
        translation: np.ndarray, quaternion: np.ndarray, size: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Convert bounding box to 8 corners.

        :param translation: The translation of the bounding box.
        :param quaternion: The quaternion of the bounding box.
        :param size: The size of the bounding box.
        :return: A dictionary containing the 8 corners of the bounding box.
        """
        # Define the 8 corners of a unit cube centered at the origin
        half_size = size / 2.0
        corners = (
            np.array(
                [
                    [-1, -1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [-1, 1, 1],
                    [1, -1, -1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, 1, 1],
                ]
            )
            * half_size
        )  # Scale by half-size to get the bounding box corners

        # Convert quaternion to rotation matrix
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

        # Rotate and translate the corners
        transformed_corners = (rotation_matrix @ corners.T).T + translation

        return {
            f"corner_{i}": f"{element:.2f}"
            for i in range(8)
            for element in transformed_corners[i].tolist()
        }
