# Step 4: Preprocess extracted frames
import cv2
import os
from tqdm import tqdm

def preprocess_frames(project_root):
    input_frames_folder = os.path.join(project_root, "extracted_frames")
    output_frames_folder = os.path.join(project_root, "processed_frames")
    os.makedirs(output_frames_folder, exist_ok=True)

    output_frames_large_folder = os.path.join(project_root, "processed_frames_large")
    os.makedirs(output_frames_large_folder, exist_ok=True)

    for video_folder in os.listdir(input_frames_folder):
        video_input_path = os.path.join(input_frames_folder, video_folder)

        video_output_path = os.path.join(output_frames_folder, video_folder)
        os.makedirs(video_output_path, exist_ok=True)

        video_output_large_path = os.path.join(output_frames_large_folder, video_folder)
        os.makedirs(video_output_large_path, exist_ok=True)

        for frame_file in tqdm(os.listdir(video_input_path), desc=f"Processing {video_folder}"):
            frame_path = os.path.join(video_input_path, frame_file)
            img = cv2.imread(frame_path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (128, 128))
            save_path_small = os.path.join(video_output_path, frame_file)
            cv2.imwrite(save_path_small, small)

            large = cv2.resize(gray, (640, 480))
            save_path_large = os.path.join(video_output_large_path, frame_file)
            cv2.imwrite(save_path_large, large)

    print("All frames preprocessed: small (features) and large (video) saved.")
