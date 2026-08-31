from frame_extraction import extract_frames
from preprocess import preprocess_frames
from feature_extraction import extract_features
from clustering import cluster_frames
from frame_selection import select_frames
from video_generation import generate_video
import os

def run_full_pipeline(project_root, video_name, callback=None):
    """
    Runs the full feature extraction & video summarization pipeline
    and sends step-by-step updates to Streamlit UI using callback().
    """

    # Step 1: Extract frames
    if callback: callback("Step 1/6 — Extracting frames from video...")
    extract_frames(project_root)

    # Step 2: Preprocess frames
    if callback: callback("Step 2/6 — Preprocessing extracted frames...")
    preprocess_frames(project_root)

    # Step 3: Extract features
    if callback: callback("Step 3/6 — Extracting LTP, LPQ & OCR features...")
    extract_features(project_root)

    # Step 4: Cluster frames
    if callback: callback("Step 4/6 — Clustering frames based on extracted features...")
    cluster_frames(project_root)

    # Step 5: Select unique & chronological frames
    if callback: callback("Step 5/6 — Selecting keyframes from clusters...")
    select_frames(project_root)

    # Step 6: Generate final summarized video
    if callback: callback("Step 6/6 — Generating final key featured output video...")
    output_video = generate_video(project_root, video_name)

    # Completion
    if callback: callback("All steps completed successfully!")

    return output_video


