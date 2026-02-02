// Portions of the following code and their modifications are originally from
// https://github.com/MIT-SPARK/semantic_inference and are licensed under the following
// license:
/* -----------------------------------------------------------------------------
 * BSD 3-Clause License
 *
 * Copyright (c) 2021-2024, Massachusetts Institute of Technology.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * * -------------------------------------------------------------------------- */

// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree

#pragma once
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <semantic_inference/image_recolor.h>
#include <semantic_inference_msgs/FeatureImage.h>
#include <semantic_inference_msgs/FeatureVectorStamped.h>

#include <opencv2/imgproc.hpp>

namespace semantic_inference {

class OutputPublisher {
 public:
  struct Config {
    ImageRecolor::Config recolor;
    bool publish_labels = true;
    bool publish_color = true;
    bool publish_overlay = true;
    bool publish_panoptic = true;
    double overlay_alpha = 0.4;
    bool open_vocab = false;
  } const config;

  OutputPublisher(const Config& config,
                  image_transport::ImageTransport& transport,
                  ros::NodeHandle& nh);

  void publish(const std_msgs::Header& header,
               const cv::Mat& labels,
               const cv::Mat& color,
               const std::optional<cv::Mat>& panoptic = std::nullopt);

 private:
  void resizePanoptic(const cv::Mat& panoptic) const;

  ImageRecolor image_recolor_;

  image_transport::Publisher label_pub_;
  image_transport::Publisher color_pub_;
  image_transport::Publisher panoptic_pub_;
  image_transport::Publisher overlay_pub_;
  ros::Publisher image_feature_pub_;
  ros::Publisher semantics_with_features_pub_;

  cv_bridge::CvImagePtr label_image_;
  cv_bridge::CvImagePtr color_image_;
  cv_bridge::CvImagePtr panoptic_image_;
  cv_bridge::CvImagePtr overlay_image_;
};

void declare_config(OutputPublisher::Config& config);

}  // namespace semantic_inference
