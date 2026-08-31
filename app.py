import os
import streamlit as st
from main import run_full_pipeline
from frame_extraction import extract_frames
from preprocess import preprocess_frames
from feature_extraction import extract_features
from clustering import cluster_frames
from frame_selection import select_frames
from video_generation import generate_video

project_root = os.path.dirname(os.path.abspath(__file__))

# PAGE CONFIG
 
st.set_page_config(
    page_title="Feature Extraction From Videos",
    layout="centered"
)
# CSS DESIGN  
 
st.markdown("""
<style>

    .stApp {
        background: linear-gradient(135deg, #ff758c, #ff7eb3, #8EC5FC, #E0C3FC, #a0e0ff);
        background-size: 600% 600%;
        animation: bgAnimation 18s ease infinite;
    }

    @keyframes bgAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main-title {
        font-size: 52px;
        font-weight: 900;
        color: white;
        text-align: center;
        text-shadow: 0 4px 18px rgba(0,0,0,0.3);
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #f7f7f7;
        margin-top: -10px;
        text-shadow: 0 3px 12px rgba(0,0,0,0.3);
    }

    .glass-box {
        background: rgba(255, 255, 255, 0.20);
        padding: 25px;
        border-radius: 18px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        margin-top: 20px;
    }

    .step-box {
        background: rgba(255,255,255,0.22);
        padding: 12px;
        border-radius: 10px;
        margin-top: 8px;
        font-weight: 600;
        color: white;
        border-left: 5px solid #00eaff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    }

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("<h1 class='main-title'>Feature Extraction From Videos</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Convert long lecture videos into meaningful featured summaries instantly.</p>", unsafe_allow_html=True)

st.markdown("<div class='glass-box'>", unsafe_allow_html=True)

 
# FILE UPLOAD
uploaded_file = st.file_uploader("Upload a Lecture Video", type=["mp4", "avi", "mov"])

if uploaded_file:

    st.markdown("<div class='step-box'>Step 0: Video Uploaded Successfully!</div>", unsafe_allow_html=True)

    input_dir = os.path.join(project_root, "input_videos")
    os.makedirs(input_dir, exist_ok=True)

    video_path = os.path.join(input_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.markdown("### Processing...")

    progress_bar = st.progress(0)
    step_text = st.empty()

    # STEP 1  
    step_text.markdown("<div class='step-box'>Step 1: Extracting Frames...</div>", unsafe_allow_html=True)
    extract_frames(project_root)
    progress_bar.progress(1/6)

    # STEP 2  
    step_text.markdown("<div class='step-box'>Step 2: Preprocessing Frames...</div>", unsafe_allow_html=True)
    preprocess_frames(project_root)
    progress_bar.progress(2/6)

    # STEP 3  
    step_text.markdown("<div class='step-box'>Step 3: Extracting LTP + LPQ + OCR Features...</div>", unsafe_allow_html=True)
    extract_features(project_root)
    progress_bar.progress(3/6)

    # STEP 4  
    step_text.markdown("<div class='step-box'>Step 4: Clustering Frames...</div>", unsafe_allow_html=True)
    cluster_frames(project_root)
    progress_bar.progress(4/6)

    # STEP 5  
    step_text.markdown("<div class='step-box'>Step 5: Selecting Key Frames...</div>", unsafe_allow_html=True)
    select_frames(project_root)
    progress_bar.progress(5/6)

    # STEP 6  
    step_text.markdown("<div class='step-box'>Step 6: Generating Final key featured Video...</div>", unsafe_allow_html=True)
    output_video = generate_video(project_root, uploaded_file.name)
    progress_bar.progress(1.0)

    st.success("final key featured video completed!")

    if output_video and os.path.exists(output_video):
        st.markdown("### Final key featured Video")
        st.video(output_video)

        with open(output_video, "rb") as f:
            st.download_button(
                label="⬇ Download final Video",
                data=f,
                file_name="final_video.avi",
                mime="video/avi"
            )

st.markdown("</div>", unsafe_allow_html=True)







