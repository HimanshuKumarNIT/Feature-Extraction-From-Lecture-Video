import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import local_binary_pattern

def extract_features(project_root):
    """
    Extract LTP + LPQ features from processed frames and save CSV/NPY
    """

    def compute_ltp(image, radius=3, n_points=8, threshold=5):
        rows, cols = image.shape
        ltp = np.zeros_like(image, dtype=np.int32)

        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                code = 0
                for p in range(n_points):
                    theta = 2 * np.pi * p / n_points
                    y = int(i + radius * np.sin(theta))
                    x = int(j + radius * np.cos(theta))
                    diff = int(image[y, x]) - int(center)
                    if diff > threshold:
                        bit = 1
                    elif diff < -threshold:
                        bit = -1
                    else:
                        bit = 0
                    code += bit
                ltp[i, j] = code
        return ltp

    def compute_lpq(image, win_size=7):
        rho = 0.90
        STFTalpha = 1.0 / win_size

        x = np.arange(-(win_size - 1) / 2, (win_size - 1) / 2 + 1)
        [Xp, Yp] = np.meshgrid(x, x)
        r = np.sqrt(Xp**2 + Yp**2)
        w0 = (r <= win_size / 2).astype(float)

        w1 = np.exp(-2 * np.pi * 1j * Xp * STFTalpha) * w0
        w2 = np.exp(-2 * np.pi * 1j * Yp * STFTalpha) * w0
        w3 = np.exp(-2 * np.pi * 1j * (Xp + Yp) * STFTalpha) * w0
        w4 = np.exp(-2 * np.pi * 1j * (Xp - Yp) * STFTalpha) * w0

        f1 = cv2.filter2D(image.astype(float), -1, np.real(w1)) + 1j * cv2.filter2D(image.astype(float), -1, np.imag(w1))
        f2 = cv2.filter2D(image.astype(float), -1, np.real(w2)) + 1j * cv2.filter2D(image.astype(float), -1, np.imag(w2))
        f3 = cv2.filter2D(image.astype(float), -1, np.real(w3)) + 1j * cv2.filter2D(image.astype(float), -1, np.imag(w3))
        f4 = cv2.filter2D(image.astype(float), -1, np.real(w4)) + 1j * cv2.filter2D(image.astype(float), -1, np.imag(w4))

        lpq = (np.real(f1) > 0).astype(int) + \
              ((np.imag(f1) > 0).astype(int) << 1) + \
              ((np.real(f2) > 0).astype(int) << 2) + \
              ((np.imag(f2) > 0).astype(int) << 3) + \
              ((np.real(f3) > 0).astype(int) << 4) + \
              ((np.imag(f3) > 0).astype(int) << 5) + \
              ((np.real(f4) > 0).astype(int) << 6) + \
              ((np.imag(f4) > 0).astype(int) << 7)
        return lpq

    processed_folder = os.path.join(project_root, "processed_frames")
    save_dir = os.path.join(project_root, "features")
    os.makedirs(save_dir, exist_ok=True)

    all_features = []

    for video_folder in os.listdir(processed_folder):
        video_path = os.path.join(processed_folder, video_folder)
        for frame_file in tqdm(os.listdir(video_path), desc=f"Extracting {video_folder}"):
            frame_path = os.path.join(video_path, frame_file)
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            ltp = compute_ltp(img)
            hist_ltp, _ = np.histogram(ltp.ravel(), bins=256, range=(-128, 128))
            hist_ltp = hist_ltp.astype("float")
            hist_ltp /= (hist_ltp.sum() + 1e-7)

            lpq = compute_lpq(img, win_size=7)
            hist_lpq, _ = np.histogram(lpq.ravel(), bins=256, range=(0, 256))
            hist_lpq = hist_lpq.astype("float")
            hist_lpq /= (hist_lpq.sum() + 1e-7)

            combined_features = np.concatenate([hist_ltp, hist_lpq])
            all_features.append([video_folder, frame_file] + combined_features.tolist())

    print("Step 5A done: Extracted LTP + LPQ features for", len(all_features), "frames")

    num_features = len(all_features[0]) - 2
    columns = ["video_name", "frame_name"] + [f"f{i}" for i in range(1, num_features + 1)]

    df = pd.DataFrame(all_features, columns=columns)
    csv_path = os.path.join(save_dir, "frame_features.csv")
    npy_path = os.path.join(save_dir, "frame_features.npy")
    df.to_csv(csv_path, index=False)
    np.save(npy_path, df.drop(columns=["video_name", "frame_name"]).values)

    print(f"Features saved:\nCSV: {csv_path}\nNumPy: {npy_path}")
