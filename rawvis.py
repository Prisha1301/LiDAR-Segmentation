import os
import numpy as np
import cv2
import glob

DATASET_PATH = "E:\\Open3D-ML\\sequences"
SEQUENCE = "21"
OUTPUT_PATH = "E:\\lidardataset\\static\\videos\\raw\\21"

COLOR_MAP = {
    0: (255, 255, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255),
    4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255), 7: (128, 128, 128),
    8: (255, 165, 0), 9: (75, 0, 130), 10: (148, 0, 211)
}

os.makedirs(OUTPUT_PATH, exist_ok=True)
img_folder = os.path.join(OUTPUT_PATH, f"seq_{SEQUENCE}_images")
os.makedirs(img_folder, exist_ok=True)

def load_lidar_data(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

def load_labels(label_path):
    return np.fromfile(label_path, dtype=np.uint32) if os.path.exists(label_path) else None

def project_point_cloud(points, labels=None):
    img_size = (1024, 512, 3)
    img = np.zeros(img_size, dtype=np.uint8)
    
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    x_scaled = ((points[:, 0] - x_min) / (x_max - x_min)) * img_size[1]
    y_scaled = ((points[:, 1] - y_min) / (y_max - y_min)) * img_size[0]

    for i, (x, y) in enumerate(zip(x_scaled.astype(int), y_scaled.astype(int))):
        # Use grayscale intensity based on the label or a default value
        intensity = int(255 if labels is None or i >= len(labels) else (labels[i] % 256))
        color = (intensity, intensity, intensity)  # Grayscale color as a tuple of integers
        cv2.circle(img, (x, y), 2, color, -1)

    return img

bin_files = sorted(glob.glob(os.path.join(DATASET_PATH, SEQUENCE, "velodyne", "*.bin")))
label_files = sorted(glob.glob(os.path.join(DATASET_PATH, SEQUENCE, "labels", "*.label")))

for i, bin_file in enumerate(bin_files[:200]):
    points = load_lidar_data(bin_file)
    labels = load_labels(label_files[i]) if i < len(label_files) else None
    img = project_point_cloud(points, labels)
    
    img_path = os.path.join(img_folder, f"frame_{i:03d}.png")
    cv2.imwrite(img_path, img)
    print(f"Saved: {img_path}")

video_path = os.path.join(OUTPUT_PATH, f"seq_{SEQUENCE}_color_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*"avc1") 
video = cv2.VideoWriter(video_path, fourcc, 10, (1024, 512))

for i in range(200):
    img_path = os.path.join(img_folder, f"frame_{i:03d}.png")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        video.write(img)

video.release()
print(f" Video saved at: {video_path}")
