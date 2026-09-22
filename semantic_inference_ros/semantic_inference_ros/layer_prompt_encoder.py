# BSD 3-Clause License
#
# Copyright (c) 2021-2024, Massachusetts Institute of Technology.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
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
#
"""Utility for prompt embedding service."""

from dataclasses import dataclass
from typing import Any

from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.task import Future
from spark_config import Config, config_field

from semantic_inference_msgs.msg import FeatureVectorStamped
from semantic_inference_msgs.srv import (
    EncodeFeature,
    EncodeFeatures,
    EncodeLayer,
    LayerEmbedding,
    SetFeatures,
)
from semantic_inference_ros.ros_conversions import Conversions


@dataclass
class LayerPromptEncoderConfig(Config):
    """Configuration for LayerPromptEncoder."""

    layer_model: Any = config_field("clip", default="open_clip")
    feature_model: Any = config_field("clip", default="open_clip")
    layer_service_name: str = "~/encode_layer"
    feature_service_name: str = "~/encode_feature"
    features_service_name: str = "~/encode_features"
    layer_embedding_service_name: str = "~/get_layer_embedding"
    set_features_service_name: str = "~/set_features"


class LayerPromptEncoder:
    """Node implementation."""

    def __init__(self, node, config: LayerPromptEncoderConfig):
        """Construct a feature encoder node."""
        self.config = config
        self._node = node
        self._layer_model = self.config.layer_model.create()
        self._feature_model = self.config.feature_model.create()
        self._node = node
        # Service for encoding prompt at specific layer
        self._srv_layer = node.create_service(
            EncodeLayer, self.config.layer_service_name, self._callback_layer
        )
        # Service for encoding a string
        self._srv_string = node.create_service(
            EncodeFeature,
            self.config.feature_service_name,
            self._callback_encode_feature,
        )

        # Service for encoding multiple features
        self._srv_strings = node.create_service(
            EncodeFeatures,
            self.config.features_service_name,
            self._callback_encode_features,
        )
        # Prompt embedding publisher
        qos = QoSProfile(depth=10)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self._prompt_embedding_pub = node.create_publisher(
            FeatureVectorStamped, "prompt_embedding", qos
        )

        # Subscribe to service for getting layer embedding
        self._layer_embedding_client = node.create_client(
            srv_type=LayerEmbedding, srv_name=self.config.layer_embedding_service_name
        )

        while not self._layer_embedding_client.wait_for_service(timeout_sec=1.0):
            node.get_logger().info(
                f"Service {self._layer_embedding_client.srv_name} "
                "not available, waiting..."
            )

        self._set_features_client = node.create_client(
            srv_type=SetFeatures, srv_name=self.config.set_features_service_name
        )

        self.future: Future = None
        self.future_set: Future = None

        self.layers = {"object": 2, "room": 4}

    def to_device(self, device):
        """Move models to device."""
        self._layer_model.to(device)
        self._feature_model.to(device)

    def _callback_layer(self, request, response):
        # Parse layer id from string to int
        layer_id = self.layers.get(request.layer.lower().rstrip("s"), None)
        if layer_id is None:
            self._node.get_logger().error(
                f"Layer '{request.layer}' not recognized. "
                f"Available layers: {list(self.layers.keys())}"
            )
            return response

        embedding = self._layer_model.embed_text(request.prompt).cpu().numpy().squeeze()
        response.feature.header.stamp = self._node.get_clock().now().to_msg()
        response.feature.feature = Conversions.to_feature(embedding)

        layer_embedding_request = LayerEmbedding.Request()
        layer_embedding_request.layer_id = layer_id
        layer_embedding_request.feature = FeatureVectorStamped()
        layer_embedding_request.feature.header.stamp = (
            self._node.get_clock().now().to_msg()
        )
        layer_embedding_request.feature.feature = response.feature.feature
        # Cancel pending future
        if self.future is not None and not self.future.done():
            self.future.cancel()
            self._node.get_logger().warn(
                "Service Future canceled. The Node took too "
                "long to process the service call."
            )

        self.future = self._layer_embedding_client.call_async(layer_embedding_request)
        self.future.add_done_callback(self.process_response)

        return response

    def process_response(self, future: Future):
        """Callback for the future, that will be called when it is done"""
        response = future.result()
        if response.success:
            self._node.get_logger().info("Successfully retrieved embedding")
        else:
            self._node.get_logger().error("Failed to retrieve embedding")

    def process_set_response(self, future: Future):
        """Callback for the future, that will be called when it is done"""
        response = future.result()
        if response.success:
            self._node.get_logger().info("Successfully set features")
        else:
            self._node.get_logger().error("Failed to set features")

    def _callback_encode_feature(self, request, response):
        embedding = (
            self._feature_model.embed_text(request.prompt).cpu().numpy().squeeze()
        )
        response.feature.header.stamp = self._node.get_clock().now().to_msg()
        response.feature.feature = Conversions.to_feature(embedding)
        self._prompt_embedding_pub.publish(response.feature)
        return response

    def _callback_encode_features(self, request, response):
        embeddings = self._feature_model.embed_text(request.prompts).cpu().numpy()
        set_features_request = SetFeatures.Request()
        for embedding in embeddings:
            set_features_request.data.features.append(Conversions.to_feature(embedding))
        response.features = set_features_request.data
        # Cancel pending future
        if self.future_set is not None and not self.future_set.done():
            self.future_set.cancel()
            self._node.get_logger().warn(
                "Service Future canceled. The Node took too "
                "long to process the service call."
            )
        self.future_set = self._set_features_client.call_async(set_features_request)
        self.future_set.add_done_callback(self.process_set_response)
        return response
