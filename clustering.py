import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: safe for Streamlit / headless servers
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def cluster_frames(project_root):
    """
    Step 6: Perform clustering on extracted features
    """

    # Load features CSV
    features_path = os.path.join(project_root, "features", "frame_features.csv")
    df = pd.read_csv(features_path)
    X = df.drop(columns=["video_name", "frame_name"]).values
    print("Features loaded:", X.shape)

    # Elbow method (optional visualization)
    wcss = []
    K_range = range(2, 15)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, wcss, 'bo-')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal k")

    elbow_plot_path = os.path.join(project_root, "features", "elbow_plot.png")
    plt.savefig(elbow_plot_path)
    plt.close()
    print(f"Elbow plot saved to: {elbow_plot_path}")

    # Step 6B: Final clustering
    features_npy_path = os.path.join(project_root, "features", "frame_features.npy")
    features = np.load(features_npy_path)
    df = pd.read_csv(features_path)
    print("Loaded features:", features.shape)

    # Choose k from elbow method manually if needed, here we take k=5 as example
    final_k = 5
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    df["cluster"] = cluster_labels

    clustered_csv = os.path.join(project_root, "features", "clustered_frames.csv")
    df.to_csv(clustered_csv, index=False)
    print(f"Clustering complete! Results saved to: {clustered_csv}")
    print(df.head())
