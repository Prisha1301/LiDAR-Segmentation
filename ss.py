import cv2
import os
import glob

base_path = r"E:\Open3D-ML\train\tracking_outputs"
output_path = r"E:\Open3D-ML\train\tracking_videos"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

sequences = ["0000", "0001"]

for seq in sequences:
    img_folder = os.path.join(base_path, seq, "lidar_final")
    images = sorted(glob.glob(os.path.join(img_folder, "*.png")) + glob.glob(os.path.join(img_folder, "*.jpg")))

    # Limit to first 200 frames
    images = images[:200]

    if not images:
        print(f"No images found for sequence {seq}")
        continue

    # Read the first image to get the frame size
    first_frame = cv2.imread(images[0])
    height, width, layers = first_frame.shape
    video_path = os.path.join(output_path, f"{seq}.mp4")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' also works
    out = cv2.VideoWriter(video_path, fourcc, 15, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Skipping unreadable image: {img_path}")
            continue
        out.write(frame)

    out.release()
    print(f" Video saved: {video_path}")

print(" Video processing completed.")
