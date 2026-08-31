
# Step 7: Simplified Unique & Chronological Frame Selection
import os
import cv2
import pytesseract
import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from skimage.feature import local_binary_pattern
import pytesseract
import shutil
import platform

# Locate the Tesseract binary automatically. On Windows it usually isn't on
# PATH, so fall back to the default install location. On Linux/macOS it is
# normally already on PATH after `apt install tesseract-ocr` / `brew install
# tesseract`. Override by setting the TESSERACT_CMD environment variable.
_tess_cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
if not _tess_cmd and platform.system() == "Windows":
    _default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_default_win_path):
        _tess_cmd = _default_win_path
if _tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd


def select_frames(project_root):
    # Paths
    processed_folder = os.path.join(project_root, "processed_frames")
    original_frames_folder = os.path.join(project_root, "extracted_frames")
    final_frames_folder = os.path.join(project_root, "final_representative_frames")
    os.makedirs(final_frames_folder, exist_ok=True)

    # Helper functions
    def compute_ltp(img, thresh=5):
        lbp = local_binary_pattern(img, P=8, R=1, method="default")
        ltp = np.where(img >= lbp + thresh, 1, np.where(img <= lbp - thresh, -1, 0))
        return ltp

    def compute_lpq(img, win_size=7):
        img = np.float32(img)
        fft_res = np.fft.fft2(img, (win_size, win_size))
        return np.abs(fft_res)

    def extract_features(img):
        text = pytesseract.image_to_string(img).strip().replace("\n", " ")
        ltp = compute_ltp(img)
        hist_ltp, _ = np.histogram(ltp.ravel(), bins=256, range=(-128, 128))
        hist_ltp = hist_ltp / (hist_ltp.sum() + 1e-7)
        lpq = compute_lpq(img)
        hist_lpq, _ = np.histogram(lpq.ravel(), bins=256, range=(0, 256))
        hist_lpq = hist_lpq / (hist_lpq.sum() + 1e-7)
        return text, np.hstack([hist_ltp, hist_lpq])

    def similarity_score(text1, text2, feat1, feat2):
        words1, words2 = set(text1.split()), set(text2.split())
        ocr_sim = len(words1 & words2) / len(words1 | words2) if len(words1 | words2) > 0 else 0
        cos_sim = 1 - pairwise_distances([feat1], [feat2], metric="cosine")[0][0]
        return 0.6 * ocr_sim + 0.4 * cos_sim

    sim_threshold = 0.7
    min_frames = 70
    for video_name in tqdm(os.listdir(processed_folder), desc="Selecting frames"):
        video_path = os.path.join(processed_folder, video_name)
        if not os.path.isdir(video_path):
            continue

        frames = sorted(os.listdir(video_path))
        if not frames:
            continue

        total_frames = len(frames)
        frames_needed = max(min_frames, int(total_frames * 0.3))
        num_segments = min(max(10, total_frames // 200), 20)
        segment_size = total_frames // num_segments

        selected_frames, selected_features, selected_texts = [], [], []

        for seg_idx in range(num_segments):
            start = seg_idx * segment_size
            end = (seg_idx + 1) * segment_size if seg_idx < num_segments - 1 else total_frames
            segment_frames = frames[start:end]
            if not segment_frames:
                continue

            img_first = cv2.imread(os.path.join(video_path, segment_frames[0]), cv2.IMREAD_GRAYSCALE)
            if img_first is not None:
                text, feat = extract_features(img_first)
                if not selected_features or similarity_score(selected_texts[-1], text, selected_features[-1], feat) < sim_threshold:
                    selected_frames.append(segment_frames[0])
                    selected_features.append(feat)
                    selected_texts.append(text)

            mid_idx = len(segment_frames) // 2
            img_mid = cv2.imread(os.path.join(video_path, segment_frames[mid_idx]), cv2.IMREAD_GRAYSCALE)
            if img_mid is not None:
                text_mid, feat_mid = extract_features(img_mid)
                if similarity_score(selected_texts[-1], text_mid, selected_features[-1], feat_mid) < sim_threshold:
                    selected_frames.append(segment_frames[mid_idx])
                    selected_features.append(feat_mid)
                    selected_texts.append(text_mid)

            if len(selected_frames) < frames_needed:
                img_last = cv2.imread(os.path.join(video_path, segment_frames[-1]), cv2.IMREAD_GRAYSCALE)
                if img_last is not None:
                    text_last, feat_last = extract_features(img_last)
                    if similarity_score(selected_texts[-1], text_last, selected_features[-1], feat_last) < sim_threshold:
                        selected_frames.append(segment_frames[-1])
                        selected_features.append(feat_last)
                        selected_texts.append(text_last)

        while len(selected_frames) < frames_needed:
            selected_frames += selected_frames[:frames_needed - len(selected_frames)]

        save_dir = os.path.join(final_frames_folder, video_name)
        os.makedirs(save_dir, exist_ok=True)
        for f in selected_frames:
            src = os.path.join(original_frames_folder, video_name, f)
            dst = os.path.join(save_dir, f)
            img = cv2.imread(src)
            if img is not None:
                cv2.imwrite(dst, img)

        print(f"{video_name}: {len(selected_frames)} frames selected across {num_segments} segments (chronological)")
