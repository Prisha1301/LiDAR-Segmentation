import cv2
import os
import glob

# Define input and output paths
input_base = r"E:\Open3D-ML\train\tracking_outputs"
output_base = r"E:\Open3D-ML\train\tracking_videos"
os.makedirs(output_base, exist_ok=True)

# Video settings
fps = 15
max_frames = 300  
sequences = sorted([d for d in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, d))])

for seq in sequences:
    img_folder = os.path.join(input_base, seq, "lidar_final")
    output_video_path = os.path.join(output_base, f"{seq}.mp4")

    # Get all images (PNG & JPG), sorted correctly
    images = sorted(glob.glob(os.path.join(img_folder, "*.png")) + glob.glob(os.path.join(img_folder, "*.jpg")), 
                    key=lambda x: int(os.path.basename(x).split('.')[0]))  # Sort numerically

    if not images:
        print(f"No images found for sequence {seq}, skipping...")
        continue

    # Limit to first 200 frames
    images = images[:max_frames]

    # Read first frame to get dimensions
    frame = cv2.imread(images[0])
    if frame is None:
        print(f"Error reading first frame for sequence {seq}, skipping...")
        continue

    h, w, _ = frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    print(f"Creating video for sequence {seq} with {len(images)} frames...")

    # Write all frames to the video
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Skipping frame {img_path} due to read error")
            continue
        out.write(frame)

    out.release()
    print(f" Saved: {output_video_path}")

print(" All videos created successfully!")
