#!/usr/bin/env python
import argparse
import logging
import os
from os.path import exists, join,normpath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud


def print_usage_and_exit():
    print(
        "Usage: ml-test.py [kitti|semantickitti|paris|toronto|semantic3d|s3dis|custom] path/to/dataset"
    )
    exit(0)


kitti_labels = {
    0: 'unlabeled',
    1: 'car',
    2: 'bicycle',
    3: 'motorcycle',
    4: 'truck',
    5: 'other-vehicle',
    6: 'person',
    7: 'bicyclist',
    8: 'motorcyclist',
    9: 'road',
    10: 'parking',
    11: 'sidewalk',
    12: 'other-ground',
    13: 'building',
    14: 'fence',
    15: 'vegetation',
    16: 'trunk',
    17: 'terrain',
    18: 'pole',
    19: 'traffic-sign'
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Datasets')
    parser.add_argument('dataset_name')
    parser.add_argument('dataset_path')
    parser.add_argument('--model', default='custom')
    return parser.parse_args()


def load_point_cloud(seq_id, path, max_files=100):
    """ Load point cloud and labels from SemanticKITTI .bin and .label files, limited to max_files """
    pc_data = []
    seq_id = str(seq_id).zfill(2)  
    sequence_path = normpath(join(path, seq_id)) 
    velodyne_path = join(sequence_path, 'velodyne')
    label_path = join(sequence_path, 'labels')

    print(f"Looking for data in:\n- {velodyne_path}\n- {label_path}") 

    if not exists(velodyne_path):
        print(f"[ERROR] Velodyne folder not found: {velodyne_path}")
        return []
    if not exists(label_path):
        print(f"[ERROR] Labels folder not found: {label_path}")
        return []

    
    bin_files = sorted([f for f in os.listdir(velodyne_path) if f.endswith('.bin')])[:max_files]

    if not bin_files:
        print(f"[ERROR] No .bin files found in {velodyne_path}")
        return []

    print(f"Loading first {len(bin_files)} frames from sequence {seq_id}")

    for bin_file in bin_files:
        frame_id = bin_file.split('.')[0]
        pc_file = join(velodyne_path, bin_file)
        label_file = join(label_path, f"{frame_id}.label")

        if not exists(pc_file):
            print(f"[ERROR] Missing point cloud file: {pc_file}")
            continue
        if not exists(label_file):
            print(f"[ERROR] Missing label file: {label_file}")
            continue

        points = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)[:, :3]
        labels = np.fromfile(label_file, dtype=np.uint32).reshape(-1) & 0xFFFF

        pc_data.append({
            'point': points,
            'label': labels,
            'frame_id': frame_id
        })

    return pc_data

def visualize_point_cloud(points, labels, title="Point Cloud", view='top'):
    """ Visualize point cloud with top-down view option """

    fig = plt.figure(figsize=(15, 6))
    

    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=labels if labels is not None else 'blue',
        cmap='tab20', s=1, alpha=0.7
    )
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'{title} - 3D View')
    ax1.view_init(elev=30, azim=45)  
    ax2 = fig.add_subplot(122)
    top_down = ax2.scatter(
        points[:, 0], points[:, 1],
        c=labels if labels is not None else 'blue',
        cmap='tab20', s=1, alpha=0.7
    )
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'{title} - Top-Down View')
    ax2.grid(True)
    ax2.axis('equal')  # Keep aspect ratio equal
    
    # Add colorbar
    if labels is not None:
        cbar = fig.colorbar(scatter, ax=[ax1, ax2], shrink=0.6)
        cbar.set_label('Class Labels')
    
    plt.tight_layout()
    plt.show()

def predict_labels(points):
    """ Placeholder function for semantic segmentation (Replace with a real model) """
    return np.random.randint(0, 20, size=len(points))  # Random labels matching KITTI labels


def main():
    args = parse_args()

    dataset_name = args.dataset_name.lower()
    path = args.dataset_path

    valid_datasets = ["kitti", "paris", "s3dis", "semantic3d", "semantickitti", "toronto", "custom"]

    if dataset_name not in valid_datasets:
        print(f"[ERROR] '{dataset_name}' is not a valid dataset")
        print_usage_and_exit()

    # Select sequence (only used for SemanticKITTI)
    if dataset_name == "semantickitti":
        seq_id = input("Enter sequence number to visualize (00-21): ")
    else:
        seq_id = "00"  # Default for other datasets

    # Load point clouds
    pcs = load_point_cloud(seq_id, path, max_files=2)  # Limit to 2 frames for demo

    if not pcs:
        print(f"No point clouds loaded for sequence {seq_id}")
        return

    for pc_data in pcs:
        points = pc_data['point']
        labels = pc_data['label']
        frame_id = pc_data['frame_id']

        print(f"Visualizing frame {frame_id} with {len(points)} points")

        # Visualize ground truth
        visualize_point_cloud(points, labels, title=f"Ground Truth - Frame {frame_id}")

        # Generate and visualize predictions
        pred_labels = predict_labels(points)
        visualize_point_cloud(points, pred_labels, title=f"Predictions - Frame {frame_id}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

    main()