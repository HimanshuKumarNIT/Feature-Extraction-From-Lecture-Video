# Feature Extraction from Lecture Video

Turns a long lecture video into a short, information-dense **summary video** made only of its most visually and textually distinct frames (slide changes, whiteboard writing, diagrams, etc). It does this by extracting frames, computing texture + OCR features on them (LTP, LPQ, and Tesseract OCR text), clustering visually similar frames, picking representative/unique frames in chronological order, and stitching the survivors back into a short video.

## How it works

```
                    ┌─────────────────────┐
                    │   Upload Video       │
                    └──────────┬───────────┘
                               ▼
                 ┌───────────────────────────┐
                 │ 1. Frame Extraction        │
                 │  (frame_extraction.py)     │
                 │  • samples 1 frame every    │
                 │    1-4s (adaptive to video  │
                 │    length)                  │
                 └──────────────┬──────────────┘
                                ▼
                 ┌───────────────────────────┐
                 │ 2. Preprocessing            │
                 │  (preprocess.py)            │
                 │  • grayscale conversion      │
                 │  • 128x128 copies for features│
                 │  • 640x480 copies for the     │
                 │    final summary video        │
                 └──────────────┬──────────────┘
                                ▼
                 ┌───────────────────────────┐
                 │ 3. Feature Extraction        │
                 │  (feature_extraction.py)     │
                 │  • LTP  (Local Ternary        │
                 │    Pattern) texture histogram │
                 │  • LPQ  (Local Phase          │
                 │    Quantization) histogram    │
                 │  • combined into one feature   │
                 │    vector per frame, saved as   │
                 │    CSV + NumPy array             │
                 └──────────────┬──────────────┘
                                ▼
                 ┌───────────────────────────┐
                 │ 4. Clustering                │
                 │  (clustering.py)              │
                 │  • KMeans over frame features   │
                 │  • elbow-method plot saved as    │
                 │    features/elbow_plot.png        │
                 └──────────────┬──────────────┘
                                ▼
                 ┌───────────────────────────┐
                 │ 5. Frame Selection            │
                 │  (frame_selection.py)          │
                 │  • splits video into chronological│
                 │    segments                         │
                 │  • per segment, keeps a frame only   │
                 │    if it's dissimilar enough from     │
                 │    the last kept one (OCR text        │
                 │    overlap + LTP/LPQ cosine similarity)│
                 └──────────────┬──────────────┘
                                ▼
                 ┌───────────────────────────┐
                 │ 6. Video Generation           │
                 │  (video_generation.py)         │
                 │  • stitches the selected frames  │
                 │    into a low-fps .avi summary    │
                 │    video (2 sec/frame)             │
                 └───────────────────────────┘
```

`app.py` is a **Streamlit** front end that wires all six steps together with a progress bar, and lets you preview and download the generated summary video. `main.py` exposes the same pipeline as a plain Python function (`run_full_pipeline`) for running it outside Streamlit — e.g. from a script or notebook.

## Project structure

```
feature-extraction-lecture-video/
├── app.py                 # Streamlit UI, drives the pipeline end-to-end
├── main.py                # Same pipeline as a callable function (non-UI entrypoint)
├── frame_extraction.py    # Step 1: sample frames from uploaded video(s)
├── preprocess.py          # Step 2: grayscale + resize (small for features, large for video)
├── feature_extraction.py  # Step 3: LTP + LPQ texture feature extraction
├── clustering.py          # Step 4: KMeans clustering + elbow plot
├── frame_selection.py     # Step 5: chronological, non-redundant frame selection
├── video_generation.py    # Step 6: reassemble selected frames into a summary video
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

All pipeline stages read/write to folders created automatically under the project root at runtime: `input_videos/`, `extracted_frames/`, `processed_frames/`, `processed_frames_large/`, `features/`, `final_representative_frames/`, `summarized_videos/`. These are git-ignored since they're regenerated on every run.

## Setup

### 1. Install Tesseract OCR (system dependency)

`frame_selection.py` uses `pytesseract`, which is a wrapper around the **Tesseract OCR** binary — it must be installed separately from the Python package.

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# Windows
# Download and run the installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
# (default install path: C:\Program Files\Tesseract-OCR\tesseract.exe)
```

The code auto-detects Tesseract on your `PATH`. If it's installed somewhere non-standard, set the `TESSERACT_CMD` environment variable to the full path of the executable:

```bash
export TESSERACT_CMD="/usr/local/bin/tesseract"      # macOS/Linux
setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"   # Windows
```

### 2. Clone and install Python dependencies

```bash
git clone https://github.com/<your-username>/feature-extraction-lecture-video.git
cd feature-extraction-lecture-video

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`), upload a lecture video, and watch it move through all six steps. When it finishes you can preview and download the summarized `.avi` video.

## Running without Streamlit

You can also call the pipeline directly from Python:

```python
from main import run_full_pipeline
import os

project_root = os.path.dirname(os.path.abspath(__file__))
output_video = run_full_pipeline(
    project_root,
    video_name="my_lecture.mp4",
    callback=print   # optional: prints progress
)
print("Summary saved to:", output_video)
```

Just make sure `my_lecture.mp4` is already placed in `input_videos/` first.

## Pipeline details

| Stage | File | What it does |
|---|---|---|
| Frame extraction | `frame_extraction.py` | Samples one frame every 1–4 seconds depending on video length (shorter videos are sampled more densely). |
| Preprocessing | `preprocess.py` | Converts each frame to grayscale and saves two resized copies: 128×128 (fast feature computation) and 640×480 (used for the output video). |
| Feature extraction | `feature_extraction.py` | Computes Local Ternary Pattern (LTP) and Local Phase Quantization (LPQ) texture histograms for every frame and concatenates them into one feature vector, saved to `features/frame_features.csv` / `.npy`. |
| Clustering | `clustering.py` | Runs KMeans (`k=5` by default) over the feature vectors, plots an elbow curve to `features/elbow_plot.png` to help you pick `k`, and writes cluster labels to `features/clustered_frames.csv`. |
| Frame selection | `frame_selection.py` | Walks through frames chronologically in segments, keeping a frame only when it's different enough (by OCR word overlap **and** LTP/LPQ cosine similarity) from the last kept frame — this avoids picking near-duplicate slides. |
| Video generation | `video_generation.py` | Reassembles the selected frames into a `.avi` file at 0.5 fps (2 seconds per frame) using the XVID codec. |

## Known limitations

- **Performance:** `compute_ltp` in `feature_extraction.py` loops over every pixel in pure Python, which is slow on large frames/long videos. For long lectures, expect Step 3 to be the slowest stage. Consider vectorizing with NumPy or swapping in `skimage.feature.local_binary_pattern` (already used in `frame_selection.py`) if you need more speed.
- **Output format:** The summary video is written as `.avi` (XVID codec), which not all browsers preview well — Streamlit's `st.video()` may not render it inline on every platform. Re-encode to `.mp4` (H.264) if you need broad browser compatibility (e.g. with `ffmpeg -i input.avi -c:v libx264 output.mp4`).
- **Clustering `k`:** `clustering.py` currently hardcodes `final_k = 5`. Review `features/elbow_plot.png` after a run and adjust `final_k` manually if your lecture has more/fewer visually distinct segments.
- **OCR accuracy:** Frame selection quality depends on Tesseract's OCR output, which can be noisy on handwriting, low-contrast whiteboards, or stylized slide fonts.
- Session/data folders (`extracted_frames/`, `processed_frames/`, etc.) are not automatically cleaned up between runs — delete them manually if you want a clean re-run on a new video.

## Roadmap ideas

- [ ] Vectorize `compute_ltp` for large speedups
- [ ] Auto re-encode output to `.mp4` for universal browser playback
- [ ] Auto-select `k` for clustering using the elbow/silhouette score instead of a hardcoded value
- [ ] Add a cleanup button in the Streamlit UI (like the session cleanup pattern used in similar pipelines)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
