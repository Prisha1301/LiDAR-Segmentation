import cv2
import os
import numpy as np

LOG_DIR = "log/img"
OUTPUT_DIR = "groundtruth"
FRAMES_PER_VIDEO = 200
VIDEO_FPS = 15  # Reduced FPS for slower playback
MAX_VIDEOS = 10  # Limit to 10 videos

def create_videos():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(LOG_DIR):
        print(f"Error: The log directory '{LOG_DIR}' does not exist.")
        return

    conditioning_files = sorted([f for f in os.listdir(LOG_DIR) if f.startswith("conditioning_") and f.endswith(".png")])
    total_frames = len(conditioning_files)

    if total_frames == 0:
        print("No conditioning images found in the log directory!")
        return

    sample_img_path = os.path.join(LOG_DIR, conditioning_files[0])
    sample_img = cv2.imread(sample_img_path)
    if sample_img is None:
        print(f"Failed to read sample image: {sample_img_path}")
        return

    height, width = sample_img.shape[:2]
    total_height = height * 2  # Stack two images vertically
    num_videos = min((total_frames + FRAMES_PER_VIDEO - 1) // FRAMES_PER_VIDEO, MAX_VIDEOS)

    print(f"Found {total_frames} frames. Creating {num_videos} videos...")

    for video_num in range(num_videos):
        start_frame = video_num * FRAMES_PER_VIDEO
        end_frame = min((video_num + 1) * FRAMES_PER_VIDEO, total_frames)

        video_name = os.path.join(OUTPUT_DIR, f"combined_output_{video_num:03d}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(video_name, fourcc, VIDEO_FPS, (width, total_height))

        if not video.isOpened():
            print(f"Error: Could not open video writer for {video_name}")
            continue

        print(f"Creating video {video_num + 1}/{num_videos} for frames {start_frame} to {end_frame - 1}...")

        for i in range(start_frame, end_frame):
            frame_num = f"{i:06d}"
            cond_path = os.path.join(LOG_DIR, f"conditioning_{frame_num}.png")
            input_path = os.path.join(LOG_DIR, f"inputs_{frame_num}.png")

            # Ensure both images exist
            if not os.path.exists(cond_path) or not os.path.exists(input_path):
                print(f"Warning: Skipping frame {frame_num} due to missing files.")
                continue

            cond_img = cv2.imread(cond_path)
            input_img = cv2.imread(input_path)

            if cond_img is None or input_img is None:
                print(f"Warning: Skipping frame {frame_num} due to unreadable images.")
                continue

            # Resize images to maintain uniformity
            cond_img = cv2.resize(cond_img, (width, height))
            input_img = cv2.resize(input_img, (width, height))

            # Stack images vertically
            combined = np.vstack((cond_img, input_img))

            # Add text labels to each section
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, "Conditioning", (10, height // 2), font, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, "Input", (10, height + height // 2), font, 0.8, (255, 255, 255), 2)

            # Add frame number at the bottom
            cv2.putText(combined, f"Frame: {i}", (width - 150, total_height - 20), font, 0.6, (255, 255, 255), 1)

            video.write(combined)

        video.release()
        print(f"Saved video: {video_name}")

    print("All videos created successfully!")

if __name__ == "__main__":
    create_videos()
