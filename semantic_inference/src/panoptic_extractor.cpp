// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree
#include "semantic_inference/panoptic_extractor.h"

#include <unordered_set>

namespace semantic_inference {

void PanopticExtractor::extract(const cv::Mat& labels, cv::Mat& panoptic_ids) const {
  // Check input validity
  if (labels.empty()) {
    throw std::invalid_argument("Input segmentation image is empty");
  }
  if (labels.type() != CV_32S) {
    throw std::invalid_argument("Input labels must be of type CV_32S");
  }

  // Prepare the output panoptic IDs matrix
  panoptic_ids = cv::Mat::zeros(labels.size(), CV_16U);

  // Extract unique class IDs from the labels
  std::unordered_set<int32_t> unique_class_ids(labels.begin<int32_t>(),
                                               labels.end<int32_t>());

  // Unique panoptic ID counter
  size_t panoptic_id_counter = 1;

  // Iterate over each unique class ID
  for (int32_t class_id : unique_class_ids) {
    if (class_id <= 0) {
      continue;  // Skip background or invalid class
    }

    // Create a binary mask for the current class
    cv::Mat class_mask = (labels == class_id);

    // Find connected components in the class mask
    cv::Mat instance_labels;
    int num_instances = cv::connectedComponents(class_mask, instance_labels, 8, CV_16U);

    // Assign unique panoptic IDs for each connected component
    for (int instance_id = 1; instance_id < num_instances;
         ++instance_id) {  // Skip background (0)
      uint16_t unique_panoptic_id =
          static_cast<uint16_t>((class_id << 16) | panoptic_id_counter++);
      panoptic_ids.setTo(unique_panoptic_id, instance_labels == instance_id);
    }
  }
}

}  // namespace semantic_inference
