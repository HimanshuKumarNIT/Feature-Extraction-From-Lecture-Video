 
# Step 8: Generate summarized videos
import os
import cv2
import numpy as np

def generate_video(project_root, video_name):
    final_frames_folder = os.path.join(project_root, "final_representative_frames")
    output_video_folder = os.path.join(project_root, "summarized_videos")
    os.makedirs(output_video_folder, exist_ok=True)

    sec_per_frame = 2.0
    fps = 1 / sec_per_frame

    print("Generating summarized videos...\n")

    for video_folder in sorted(os.listdir(final_frames_folder)):
        video_path = os.path.join(final_frames_folder, video_folder)
        if not os.path.isdir(video_path):
            continue

        frames = sorted([f for f in os.listdir(video_path) if f.lower().endswith((".jpg", ".png"))])
        if not frames:
            print(f"No frames found in {video_folder}, skipping.")
            continue

        first_frame_path = os.path.join(video_path, frames[0])
        first_img = cv2.imread(first_frame_path)
        if first_img is None:
            print(f"Could not read first frame in {video_folder}, skipping.")
            continue
        frame_height, frame_width = first_img.shape[:2]
        frame_size = (frame_width, frame_height)

        output_path = os.path.join(output_video_folder, f"{video_folder}_summary.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        for frame_file in frames:
            frame_path = os.path.join(video_path, frame_file)
            img = cv2.imread(frame_path)
            if img is None:
                continue

            if img.shape[1::-1] != frame_size:
                img = cv2.resize(img, frame_size, interpolation=cv2.INTER_CUBIC)
            out.write(img)

        out.release()
        total_duration = len(frames) * sec_per_frame
        print(f"Saved summarized video: {output_path} | Frames: {len(frames)} | Approx. Duration: {total_duration:.1f} sec")

    print("\nAll videos generated successfully!")
