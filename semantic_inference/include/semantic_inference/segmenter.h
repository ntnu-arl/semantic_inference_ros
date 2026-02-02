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
#include <opencv2/core/mat.hpp>

#include "semantic_inference/model_config.h"
#include "semantic_inference/panoptic_extractor.h"

namespace semantic_inference {

struct SegmentationResult {
  bool valid = false;
  cv::Mat labels;
  cv::Mat panoptic_ids;

  inline operator bool() const { return valid; }
};

class Segmenter {
 public:
  struct Config {
    ModelConfig model;
    bool mask_predictions_with_depth = true;
    DepthLabelMask::Config depth_mask;
  } const config;

  explicit Segmenter(const Config& config);

  virtual ~Segmenter();

  SegmentationResult infer(const cv::Mat& img, const cv::Mat& depth = cv::Mat());

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<PanopticExtractor> panoptic_extractor_;
  DepthLabelMask mask_;
};

void declare_config(Segmenter::Config& config);

}  // namespace semantic_inference
