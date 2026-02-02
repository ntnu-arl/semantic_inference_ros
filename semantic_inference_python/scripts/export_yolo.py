# Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
# Technology All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree

from ultralytics import YOLO, YOLOE
import yaml
import cv2
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/home/arl/jetson_ssd/cache/ultralytics/yoloe-11l-seg.pt"
CUSTOM_MODEL = "/home/arl/jetson_ssd/cache/ultralytics/anymal_yoloe-11l-seg.pt"
EXPORT_MODEL = "/home/arl/jetson_ssd/cache/ultralytics/anymal_yoloe-11l-seg.engine"
CLASSES_PATH = "/home/arl/workspaces/hierarchical_reasoning_ws/src/reasoning/reasoning_hydra/config/label_spaces/anymal_label_space.yaml"
IMAGE_PATH = "/home/arl/jetson_ssd/hydra_bags/image.png"

with open(CLASSES_PATH) as f:
    labels_data = yaml.safe_load(f)["label_names"]
    CLASSES = [data["name"] for data in labels_data]

yoloe = YOLOE(MODEL_PATH)
yoloe.set_classes(CLASSES, yoloe.get_text_pe(CLASSES))
yoloe.save(CUSTOM_MODEL)

yoloe = YOLO(CUSTOM_MODEL)
yoloe.export(format="engine", device="cuda:0", half=True)

img = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
yoloe = YOLO(EXPORT_MODEL)
yoloe.predict(img, verbose=False, imgsz=640, device=DEVICE, conf=0.5)
