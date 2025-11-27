import os
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define paths
SEQUENCE_PATH = "E:/LiDAR-Diffusion/datasets/semantic_kitti/dataset/sequences/11/"
VELODYNE_PATH = os.path.join(SEQUENCE_PATH, "velodyne")
LABELS_PATH = os.path.join(SEQUENCE_PATH, "predictions")

# Function to load point cloud
def load_point_cloud(bin_path):
    pcd = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pcd[:, :3]  # Only X, Y, Z (discard intensity)

# Function to load labels
def load_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
    return labels & 0xFFFF  # Extract semantic label

# Define label colors (you can modify as per SemanticKITTI)
LABEL_COLORS = {
    10: [0, 255, 0],  # Green for vegetation
    30: [0, 0, 255],  # Red for vehicles
    40: [255, 0, 0],  # Blue for roads
    44: [255, 255, 0],  # Yellow
    48: [0, 255, 255],  # Cyan
    50: [255, 165, 0],  # Orange
    51: [128, 0, 128],  # Purple
    70: [255, 0, 255],  # Magenta
    80: [192, 192, 192],  # Grey
    81: [128, 128, 0],  # Olive
}

# Get first 100 frames
frame_files = sorted(os.listdir(VELODYNE_PATH))[:100]
label_files = sorted(os.listdir(LABELS_PATH))[:100]

# Output video settings
video_width = 1200
video_height = 600
fps = 10
output_video_path = "output_combined.avi"

# Initialize OpenCV video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width * 2, video_height))

# Process each frame
for frame_file, label_file in tqdm(zip(frame_files, label_files), total=100):
    bin_path = os.path.join(VELODYNE_PATH, frame_file)
    label_path = os.path.join(LABELS_PATH, label_file)

    # Load point cloud & labels
    points = load_point_cloud(bin_path)
    labels = load_labels(label_path)

    # Create Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Generate color map for labels
    colors = np.array([LABEL_COLORS.get(label, [255, 255, 255]) for label in labels], dtype=np.float32) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize raw point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=video_width, height=video_height, visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    raw_img = vis.capture_screen_float_buffer(True)
    raw_img = np.array(raw_img) * 255
    raw_img = raw_img.astype(np.uint8)
    vis.destroy_window()

    # Visualize segmented point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=video_width, height=video_height, visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    seg_img = vis.capture_screen_float_buffer(True)
    seg_img = np.array(seg_img) * 255
    seg_img = seg_img.astype(np.uint8)
    vis.destroy_window()

    # Combine both images side by side
    combined_frame = np.hstack((raw_img, seg_img))
    video_writer.write(combined_frame)

# Release video writer
video_writer.release()
print("Combined video saved as", output_video_path)
