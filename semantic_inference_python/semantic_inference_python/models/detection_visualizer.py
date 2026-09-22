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
"""Detection visualizer using supervision library."""

from typing import Optional, Union

import cv2
import numpy as np
import supervision as sv
from supervision.annotators.core import BoxAnnotator, MaskAnnotator
from supervision.draw.color import Color, ColorPalette


class DetectionVisualizer:
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        classes: Optional[list[str]] = None,
        instance_random_color: bool = False,
        draw_bbox: bool = True,
    ) -> None:
        self.color = color
        self.classes = classes
        self.instance_random_color = instance_random_color
        self.draw_bbox = draw_bbox

        # Updated for PyPI supervision layout
        self.box_annotator = BoxAnnotator(color=self.color)
        self.mask_annotator = MaskAnnotator(color=self.color)

    @staticmethod
    def safe_label_positions(
        detections: sv.Detections, labels: list[str], image_shape: tuple[int, int]
    ) -> list[str]:
        safe_labels = []
        for xyxy, text in zip(detections.xyxy, labels):
            estimated_y = xyxy[1] - 10
            if estimated_y < 0:
                text = " " + text
            safe_labels.append(text)
        return safe_labels

    def annotate(self, image: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if len(detections) == 0:
            return image

        labels = []
        if hasattr(detections, "confidence") and hasattr(detections, "class_id"):
            conf = detections.confidence
            cls = detections.class_id

            if conf is not None:
                if self.classes is None:
                    labels = [f"{c} {score:.2f}" for score, c in zip(conf, cls)]
                else:
                    labels = [
                        f"{self.classes[c]} {score:.2f}" for score, c in zip(conf, cls)
                    ]
            else:
                if self.classes is None:
                    labels = [str(c) for c in cls]
                else:
                    labels = [self.classes[c] for c in cls]
        else:
            return image

        # Make a copy if we want random colors
        if self.instance_random_color:
            detections = detections.copy()
            detections.tracker_id = np.arange(len(detections))

        annotated = image.copy()

        # masks first
        if detections.mask is not None:
            annotated = self.mask_annotator.annotate(annotated, detections)

        # bbox + label overlay
        if self.draw_bbox:
            annotated = self.box_annotator.annotate(annotated, detections)

            labels = self.safe_label_positions(detections, labels, image.shape)

            for (x1, y1, _, _), label in zip(detections.xyxy, labels):
                x1, y1 = int(x1), int(y1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 1

                text_pos_y = max(y1 - 5, 10)
                (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)

                cv2.rectangle(
                    annotated,
                    (x1, text_pos_y - h - 4),
                    (x1 + w, text_pos_y),
                    (0, 0, 0),
                    thickness=-1,
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1, text_pos_y - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        return annotated
