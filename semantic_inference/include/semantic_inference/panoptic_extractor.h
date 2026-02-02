// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree

#pragma once

#include <opencv2/opencv.hpp>
#include <optional>

namespace semantic_inference {

class PanopticExtractor {
 public:
  PanopticExtractor() = default;

  /**
   * Extract panoptic IDs from a segmentation label image.
   *
   * @param labels cv::Mat containing semantic segmentation labels.
   *               Each pixel value represents a class ID.
   * @param panoptic_ids cv::Mat of type CV_32SC1, where each pixel value is a unique
   * panoptic ID.
   */
  void extract(const cv::Mat& labels, cv::Mat& panoptic_ids) const;
};
}  // namespace semantic_inference
