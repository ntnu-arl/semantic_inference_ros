# <div align="center">Semantic Segmentation and VLM Reasoning in ROS</div>

![License: MIT](https://img.shields.io/badge/License-BSD-green.svg)
![ROS Version](https://img.shields.io/badge/ROS-Noetic-blue)

This repository extends [semantic_inference](https://github.com/MIT-SPARK/semantic_inference) to provide **closed and open set semantic segmentation** methods. Additionally, it provides methods to extract **CLIP embeddings** of objects and **relational embeddings** using Visual Language Models (VLMs).

---

## Table of Contents

- [Setup](#setup)  
  - [General Requirements](#general-requirements)  
  - [Virtual Environment](#virtual-environment)  
  - [Building](#building)  
- [Usage](#usage)  
  - [Open-set Segmentation](#open-set-segmentation)  
  - [VLM for Object Relationship Embeddings](#vlm-for-object-relationship-embeddings)  
  - [VLM/LLM Reasoning](#vlmllm-reasoning)  
- [Citation](#citation)  
- [License](#license)  
- [Acknowledgements](#acknowledgements)  
- [Contact](#contact)  

---

## Setup

### General Requirements

These instructions assume `ros-noetic-desktop-full` is installed on **Ubuntu 20.04**.  

Install the general dependencies:

```bash
sudo apt install python3-rosdep python3-catkin-tools
```

Clone the repository and initialize submodules:

```bash
git clone git@github.com:ntnu-arl/semantic_inference_ros.git
git submodule init
git submodule update --recursive
```

### Virtual Environment

It is highly recommended to set up a **Python virtual environment** to run ROS Python nodes:

```bash
cd /path/to/catkin_ws/src/semantic_inference/semantic_inference_python
python3.8 -m venv --system-site-packages ros_semantics_env
source ros_semantics_env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Building

Install ROS dependencies:

```bash
cd /path/to/catkin_ws/src
rosdep install --from-paths . --ignore-src -r -y
```

For **closed-set segmentation**, follow the setup instructions (skip Python utilities) in [semantic_inference closed-set docs](https://github.com/MIT-SPARK/semantic_inference/blob/archive/ros_noetic/docs/closed_set.md).

Build the workspace:

```bash
catkin config -DCMAKE_BUILD_TYPE=Release
catkin build
```

---

## Usage

### Open-set Segmentation

Open-set segmentation consumes **RGB-D images** and camera information to perform semantic segmentation and extract **open-vocabulary features** for each object.  

- Launch file: [openset_segmentation.launch](./semantic_inference_ros/launch/openset_segmentation.launch)  
- Configuration: [openset_segmentation.yaml](./semantic_inference_ros/config/openset_segmentation.yaml)  

Supported open-set detectors: [YOLOe](https://docs.ultralytics.com/models/yoloe/) and [YOLOw](https://docs.ultralytics.com/models/yolo-world/). These can detect any list of objects without re-training.

```bash
roslaunch semantic_inference_ros openset_segmentation.launch
```

### VLM for Object Relationship Embeddings

This method takes a segmented image along with its **original RGB-D frame** and computes **visual features** for each pair of detected objects. These features can be used to prompt a VLM for reasoning about relationships.  

- Launch file: [vlm_features_node.launch](./semantic_inference_ros/launch/vlm_features_node.launch)  
- Configuration: [vlm.yaml](./semantic_inference_ros/config/vlm.yaml)  

Supported VLMs: [InstructBLIP](https://huggingface.co/collections/Salesforce/instructblip-models) and [DeepSeek-VL2](https://huggingface.co/deepseek-ai/deepseek-vl2).

To use **DeepSeek-VL2**, first extract the visual encoder as a standalone model (~100GB RAM required):

```bash
python semantic_inference_python/scripts/extract_deepseek_visual.py --model_name <model to use> --output_path <path to store model>
```

Then, set the model path in [vlm.yaml](./semantic_inference_ros/config/vlm.yaml).

Launch the node:

```bash
roslaunch semantic_inference_ros vlm_features.launch
```

### VLM/LLM Reasoning

This section enables reasoning on the [relationship-aware hierarchical scene graph](https://github.com/ntnu-arl/reasoning_hydra).  

- LLMs predict relevant objects and interactions for given tasks  
- VLM responses are parsed by LLMs  
- **OpenAI API key** required, run:  
```bash
export OPENAI_API_KEY=<Your OpenAI API Key>
``` 

VLM reasoning is performed on the cloud. Use [DeepSeek-VL2 server code](https://github.com/ntnu-arl/DeepSeek-VL2/tree/server) to run FastAPI server.

Steps to set up the server:

1. Clone the server repo:

```bash
git clone git@github.com:ntnu-arl/DeepSeek-VL2.git -b server
cd DeepSeek-VL2
```

2. Set up the Python virtual environment:

```bash
bash setup.sh
```

3. Configure server path, port, and API key in `run_server.sh`.

4. Run the server (model download may take time):

```bash
bash run_server.sh
```

Finally, set the **server URL** in [vlm_for_navigation.yaml](./semantic_inference_ros/config/vlm_for_navigation.yaml) and export your FASTAPI_KEY:
```bash
export FASTAPI_API_KEY=<Your server FastAPI Key>
``` 

---

## Citation

```bibtex
@inproceedings{puigjaner2026reasoninggraph,
    title={Relationship-Aware Hierarchical 3D Scene Graph},
    author={Gassol Puigjaner, Albert and Zacharia, Angelos and Alexis, Kostas},
    booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)}, 
    year={2026}
}
```

---

## License

Released under **BSD-3-Clause**.

---

## Acknowledgements

This open-source release is based on work supported by the **European Commission** through:

- **Project SYNERGISE**, under **Horizon Europe Grant Agreement No. 101121321**

---

## Contact

For questions or support, reach out via [GitHub Issues](https://github.com/ntnu-arl/semantic_inference_ros/issues) or contact the authors:

- [Albert Gassol Puigjaner](mailto:albert.g.puigjaner@ntnu.no)  
- [Angelos Zacharia](mailto:angelos.zacharia@ntnu.no)  
- [Kostas Alexis](mailto:konstantinos.alexis@ntnu.no)


---
