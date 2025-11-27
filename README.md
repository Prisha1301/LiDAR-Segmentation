This repository provides a complete LiDAR processing workflow for KITTI-style datasets. It includes model inference, semantic/motion segmentation, bounding box extraction, 3D and 2D visualization, and video generation utilities. The toolkit is built using Open3D, PyTorch, and optional OpenPCDet modules. It is suitable for researchers, students, and engineers working on LiDAR perception, segmentation, and visualization tasks.

The pipeline is organized into the following major stages:

Model Inference – Run PointPillars inference on individual LiDAR frames.

Sampling & Evaluation – Perform segmentation, run evaluation metrics, and collect logs.

Visualization – Generate 3D renderings, 2D projections, and stitched videos.

Bounding Box Extraction – Load KITTI-style sequences, apply transforms, and generate bounding box predictions.

Each module is written to be easily extendable and integrates with standard LiDAR datasets.

Project Workflow Overview
1. Model Inference

The inference stage processes one .bin LiDAR file and generates predictions such as segmentation labels or bounding boxes.

Main script: pred.py

Runs PointPillars inference

Loads a single LiDAR frame

Outputs semantic labels and bounding box predictions

Requires OpenPCDet if PointPillars is used

2. Sampling, Dataset Handling, and Evaluation

This stage processes complete datasets, evaluates segmentation performance, and logs results for visualization.

Core script: sem2lidar.py
Includes the following key functions:

get_parser() – Argument parsing

load_model() – Loads trained segmentation models

run() – Executes segmentation and sampling

save_logs() – Saves outputs for visualization and evaluation

Evaluation script: tester_SemanticKITTTI.py
Provides:

calculate_accuracy()

calculate_miou()

calculate_class_iou()

Precomputed metrics: _results.py

Contains expected results for validation

Function: get_precomputed()

3. Visualization and Video Generation

This section handles rendering and video creation for both raw and segmented LiDAR frames.

3D Visualization: vis.py

load_point_cloud() – Reads KITTI .bin files

load_labels() – Loads segmentation labels

Produces side-by-side renderings of raw vs. segmented point clouds

2D Visualization: rawvis.py

project_point_cloud() – Converts 3D point clouds into 2D grayscale frames

Saves per-frame images and optional videos

Video Utilities:

setup.py – create_videos() merges frames into final videos

setupbb.py – Batch video creation per sequence

ss.py – Lightweight video stitching script

4. Bounding Box Extraction

This module demonstrates reading KITTI-style datasets, preprocessing, and generating bounding box predictions.

Main script: boundingbox.py

Contains DemoDataset class

Handles calibration, transformations, and metadata loading

Entry point: main()

Outputs bounding box visualizations and videos

Prerequisites

Environment Requirements

Python 3.8+

CUDA-enabled GPU recommended

Conda environment recommended

Dependencies

Open3D

OpenCV

PyTorch

numpy

tqdm

scikit-learn

PyYAML

omegaconf

joblib

easydict

pytorch-lightning

Optional

OpenPCDet (for PointPillars inference in pred.py)

Installation
conda create -n open3d-ml python=3.8 -y
conda activate open3d-ml

pip install numpy opencv-python open3d tqdm scikit-learn pyyaml omegaconf joblib easydict pytorch-lightning

pip install torch torchvision   # choose version according to your CUDA setup


To use PointPillars:

git clone https://github.com/open-mmlab/OpenPCDet.git
# Follow OpenPCDet installation instructions

Usage Examples

Single-frame PointPillars inference

python pred.py


Run segmentation and evaluation

python sem2lidar.py --dataset kitti --resume /path/to/checkpoint --eval


Generate videos from log directory

python setup.py
python setupbb.py
python ss.py


Run bounding box extraction

python boundingbox.py --cfg_file path/to/config.yaml \
--data_path path/to/sequences \
--ckpt path/to/model.ckpt
