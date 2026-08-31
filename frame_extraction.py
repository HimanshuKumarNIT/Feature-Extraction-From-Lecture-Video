 
# Step 3: dynamically extract frame #streamlit run app.py
import cv2
import os

def extract_frames(project_root):
    input_folder = os.path.join(project_root, "input_videos")
    output_folder = os.path.join(project_root, "extracted_frames")
    os.makedirs(output_folder, exist_ok=True)

    for video_file in os.listdir(input_folder):
        if video_file.endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(input_folder, video_file)
            cap = cv2.VideoCapture(video_path)  # allow us to read video frame by frame

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = frame_count_total / fps

            if duration_sec <= 600:
                seconds_interval = 1
            elif duration_sec <= 900:
                seconds_interval = 2
            else:
                seconds_interval = 4

            frame_interval = int(fps * seconds_interval)
            video_name = os.path.splitext(video_file)[0]
            video_output = os.path.join(output_folder, video_name)
            os.makedirs(video_output, exist_ok=True)

            frame_count, saved_count = 0, 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    frame_filename = os.path.join(video_output, f"frame_{saved_count:05d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    saved_count += 1
                frame_count += 1

            cap.release()
            print(f" {video_file} → {saved_count} frames saved in {video_output}")

    print("Frame extraction completed for all videos!")
