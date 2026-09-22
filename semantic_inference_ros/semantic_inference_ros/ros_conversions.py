# BSD 3-Clause License

# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.

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
#
"""Module containing ROS message conversions."""

import struct

import cv2
import cv_bridge
import numpy as np
import torch
from sensor_msgs.msg import CompressedImage, Image

import semantic_inference_msgs.msg


class Conversions:
    """Conversion namespace."""

    bridge = cv_bridge.CvBridge()

    @staticmethod
    def compressed_depth_to_cv2(msg: CompressedImage, depth_fmt: str):
        # remove header from raw data
        depth_header_size = 12
        raw_data = msg.data[depth_header_size:]

        depth_img = cv2.imdecode(
            np.fromstring(raw_data, np.uint8), cv2.IMREAD_UNCHANGED
        )
        if depth_img is None:
            # probably wrong header size
            raise Exception(
                "Could not decode compressed depth image."
                "You may need to change 'depth_header_size'!"
            )

        if depth_fmt == "32FC1":
            raw_header = msg.data[:depth_header_size]
            # header: int, float, float
            [_, depthQuantA, depthQuantB] = struct.unpack("iff", raw_header)
            depth_img_scaled = depthQuantA / (
                depth_img.astype(np.float32) - depthQuantB
            )
            # filter max values
            depth_img_scaled[depth_img == 0] = 0

            # depth_img_scaled provides distance in meters as f32
            # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
            depth_img = (depth_img_scaled * 1000).astype(np.uint16)

        return (depth_img / 1000.0).astype(np.float32)

    @classmethod
    def to_image(cls, msg):
        """Convert sensor_msgs.Image to numpy array."""
        if isinstance(msg, Image):
            # Color images are expected to be in RGB format
            # Depth images are expected to be in 32FC1 format
            image = cls.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if msg.encoding == "16UC1":
                # Convert depth image from 16-bit unsigned int to float32
                image = image.astype(np.float32) / 1000.0
            elif msg.encoding == "32FC1":
                # Ensure depth image is in meters
                image = image.astype(np.float32)
            elif msg.encoding == "bgr8":
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        elif isinstance(msg, CompressedImage):
            format, _ = msg.format.split(";")
            format = format.strip()
            if format == "rgb8" or format == "bgr8":
                # Convert BGR to RGB
                image = cls.bridge.compressed_imgmsg_to_cv2(
                    msg, desired_encoding="passthrough"
                )
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif format == "32FC1" or format == "16UC1":
                # Convert depth image to float32
                image = Conversions.compressed_depth_to_cv2(msg, format)
            return image
        else:
            raise ValueError(f"Message type '{type(msg)}' not supported!")

    @staticmethod
    def to_feature(feature, choice=None):
        """Create a feature vector message."""
        msg = semantic_inference_msgs.msg.FeatureVector()
        if feature is None:
            return msg
        msg.data = feature.flatten().tolist()
        return msg

    @staticmethod
    def to_stamped_feature(header, feature):
        """
        Create a stamped feature vector message.

        Args:
            header (std_msgs.msg.Header): Original image header
            feature (np.ndarray): Image feature

        """
        msg = semantic_inference_msgs.msg.FeatureVectorStamped()
        msg.header = header
        msg.feature = Conversions.to_feature(feature)
        return msg

    @classmethod
    def to_image_msg(cls, header, img, encoding="passthrough"):
        msg = cls.bridge.cv2_to_imgmsg(img, encoding=encoding)
        msg.header = header
        return msg

    @classmethod
    def to_feature_image(cls, header, results):
        """
        Create a FeatureImage from segmentation results.

        Args:
            header (std_msgs.msg.Header): Original image header
            results: (semantic_inference.SegmentationResults): Segmentation output

        """
        msg = semantic_inference_msgs.msg.FeatureImage()
        msg.header = header
        if results.instances is None:
            return msg
        msg.image = cls.bridge.cv2_to_imgmsg(results.instances, header=header)
        msg.mask_ids = results.get_ids()
        msg.features = [cls.to_feature(x) for x in results.features]
        return msg

    @staticmethod
    def to_feature_image_from_masks_labels(
        header, results, unknown_id
    ) -> semantic_inference_msgs.msg.FeatureImage:
        """
        Create a FeatureImage from masks and labels.
        :param header: std_msgs.msg.Header
        :param results: semantic_inference.SegmentationResults
        :return: semantic_inference_msgs.msg.FeatureImage
        """
        segmentation_img = (
            np.ones(
                (results.panoptic_image.shape[0], results.panoptic_image.shape[1]),
                dtype=np.uint8,
            )
            * unknown_id
        )
        if results.masks is not None and results.labels is not None:
            for mask, label in zip(results.masks, results.labels):
                segmentation_img[mask] = label.item()

        msg = semantic_inference_msgs.msg.FeatureImage()
        msg.header = header
        msg.image = Conversions.bridge.cv2_to_imgmsg(segmentation_img, header=header)
        msg.mask_ids = results.get_ids()
        if len(results.features) == 0:
            return msg
        msg.features = [Conversions.to_feature(x) for x in results.features]
        return msg

    @staticmethod
    def to_full_features(
        header, results, unknown_id
    ) -> semantic_inference_msgs.msg.FullFeatures:
        """
        Create a FullFeatures message from segmentation results.
        :param header: std_msgs.msg.Header
        :param results: semantic_inference.SegmentationResults
        :param unknown_id: ID used for unknown segments
        :return: semantic_inference_msgs.msg.FullFeatures
        """

        msg = semantic_inference_msgs.msg.FullFeatures()
        msg.header = header
        msg.full_image_feature = Conversions.to_stamped_feature(
            header, results.image_embedding
        )
        msg.segmentation_features = semantic_inference_msgs.msg.FeatureImage()
        msg.segmentation_features.header = header
        msg.segmentation_features.image = Conversions.bridge.cv2_to_imgmsg(
            results.panoptic_image.numpy().astype(np.int16), header=header
        )
        if results.labels is not None:
            msg.segmentation_features.mask_ids = results.labels.tolist()
        if results.panoptic_ids is not None:
            msg.segmentation_features.panoptic_ids = results.get_ids()
        msg.segmentation_features.unknown_id = unknown_id
        msg.pixelwise_feature_map = Conversions.to_dense_feature_image_msg(
            header, results.feature_image
        )
        if len(results.features) == 0:
            return msg
        msg.segmentation_features.features = [
            Conversions.to_feature(x) for x in results.features
        ]
        return msg

    @staticmethod
    def to_dense_feature_image_msg(header, feature_image: torch.Tensor | None) -> Image:
        """
        Create a FeatureImage message from a dense feature image.
        :param header: std_msgs.msg.Header
        :param feature_image: Dense feature image as a tensor (HxWxC)
        :return: sensor_msgs.msg.Image
        """

        msg = Image()
        msg.header = header
        if feature_image is None:
            return msg

        msg.height = feature_image.shape[0]
        msg.width = feature_image.shape[1]
        msg.encoding = f"32FC{feature_image.shape[2]}"
        msg.is_bigendian = False
        msg.step = feature_image.shape[1] * feature_image.shape[2] * 4
        msg.data = feature_image.numpy().tobytes()

        return msg
