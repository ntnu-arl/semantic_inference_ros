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
"""Module containing ROS message conversions."""

import cv_bridge
import numpy as np
import torch
import cv2
import struct

import semantic_inference_msgs.msg
from sensor_msgs.msg import Image, CompressedImage


class Conversions:
    """Conversion namespace."""

    bridge = cv_bridge.CvBridge()

    @staticmethod
    def compressed_depth_to_cv2(msg: CompressedImage, depth_fmt: str):
        # 'msg' as type CompressedImage

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
        msg.rows = feature.shape[0] if len(feature.shape) > 0 else 1
        msg.cols = feature.shape[1] if len(feature.shape) > 1 else 1
        msg.data = feature.flatten().tolist()
        if choice is not None:
            msg.choice = choice.tolist()
        return msg

    @staticmethod
    def to_features(features):
        """Create feature vectors message."""
        features_msg = []
        for feature in features:
            msg = semantic_inference_msgs.msg.FeatureVector()
            msg.data = feature.T.flatten().tolist()
            msg.rows = feature.shape[0]
            msg.cols = feature.shape[1]
            features_msg.append(msg)
        return features_msg

    @staticmethod
    def to_stamped_features(header, features, choice=None):
        """
        Create a stamped feature vector message.

        Args:
            header (std_msgs.msg.Header): Original image header
            feature (np.ndarray): Image feature
        """
        msg = semantic_inference_msgs.msg.FeatureVectorsStamped()
        msg.header = header
        msg.feature = Conversions.to_features(features) if features is not None else []
        msg.choice = choice if features is not None and choice is not None else []
        return msg

    @staticmethod
    def to_stamped_feature(header, feature, choice=None):
        """
        Create a stamped feature vector message.

        Args:
            header (std_msgs.msg.Header): Original image header
            feature (np.ndarray): Image feature
        """
        msg = semantic_inference_msgs.msg.FeatureVectorStamped()
        msg.header = header
        msg.feature = Conversions.to_feature(feature, choice)
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

    @classmethod
    def feature_image_tensor_to_image(cls, feature_image_tensor):
        """
        Convert a feature image tensor to a ROS image message.

        Args:
            feature_image_tensor (torch.Tensor): Feature image tensor

        Returns:
            sensor_msgs.msg.Image: ROS image message
        """
        feature_image = feature_image_tensor.cpu().numpy().astype(np.float32)
        height, width = feature_image.shape[:2]
        channels = feature_image.shape[2] if len(feature_image.shape) == 3 else 1

        # Flatten the multi-channel data
        flattened_data = feature_image.reshape(
            -1
        )  # Flatten into a single contiguous array

        # Create sensor_msgs::Image message
        ros_image = Image()
        ros_image.height = height
        ros_image.width = width
        ros_image.encoding = "32FC"  # Generic multi-channel float encoding
        ros_image.is_bigendian = False
        ros_image.step = width * channels * 4  # Step size in bytes (float32 = 4 bytes)
        ros_image.data = flattened_data.tobytes()

        return ros_image

    @classmethod
    def to_colored_feature_image(cls, header, results, img):
        """
        Create a ColoredFeatureImage from segmentation results.

        Args:
            header (std_msgs.msg.Header): Original image header
            results: (semantic_inference.SegmentationResults): Segmentation output
            img (torch.tensor): Original image
        """
        msg = semantic_inference_msgs.msg.FeatureImage()
        msg.header = header
        msg.image = (
            cls.bridge.cv2_to_imgmsg(img.cpu().numpy().astype(np.uint8), header=header)
            if isinstance(img, torch.Tensor)
            else cls.bridge.cv2_to_imgmsg(img, header=header)
        )
        if results.features is not None:
            if results.features.shape[0] > 0:
                msg.mask_ids = results.get_ids()
                msg.features = [cls.to_feature(x) for x in results.features]
                if results.feature_image is not None:
                    msg.feature_image = cls.feature_image_tensor_to_image(
                        results.feature_image
                    )

        return msg

    @classmethod
    def to_sensor_image(cls, img, header=None, encoding="passthrough"):
        """
        Create a sensor image from a tensor/array.

        Args:
            img (np.ndarray or torch.Tensor): image
        """
        return (
            cls.bridge.cv2_to_imgmsg(
                img.cpu().numpy().astype(np.uint8), encoding=encoding, header=header
            )
            if isinstance(img, torch.Tensor)
            else cls.bridge.cv2_to_imgmsg(img, encoding=encoding, header=header)
        )
