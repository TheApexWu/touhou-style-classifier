"""Visualize style embeddings with t-SNE."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio import AudioLoader, find_audio_files
from src.features.spectral import SpectralFeatureExtractor


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "tsne_visualization.png"
MAPPING_PATH = PROJECT_ROOT / "data" / "metadata" / "source_track_mapping.json"

CIRCLE_DIRS = {
    "IOSYS": "IOSYS",
    "UNDEAD CORPORATION": "UNDEAD_CORPORATION",
    "暁Records": "Akatsuki_Records",
    "SOUND HOLIC": "SOUND_HOLIC",
    "Liz Triangle": "Liz_Triangle",
}

# Colors for each circle (colorblind-friendly palette)
CIRCLE_COLORS = {
    "IOSYS": "#E69F00",           # Orange - electronic/denpa
    "UNDEAD CORPORATION": "#D55E00",  # Red-orange - death metal
    "暁Records": "#0072B2",        # Blue - rock
    "SOUND HOLIC": "#CC79A7",      # Pink - eurobeat
    "Liz Triangle": "#009E73",     # Green - acoustic
}


def load_source_mapping() -> dict:
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_source_track_name(mapping: dict, relative_path: str) -> str | None:
    entry = mapping.get(relative_path, {})
    sources = entry.get("source_tracks", [])
    if sources:
        return sources[0]["name"]
    return None


def extract_features_for_file(loader: AudioLoader, extractor: SpectralFeatureExtractor,
                               filepath: Path) -> np.ndarray:
    waveform, sr = loader.load(filepath)
    features = extractor.extract_all(waveform)
    return extractor.summarize(features)


def load_all_features():
    """Load and extract features for all tracks."""
    loader = AudioLoader()
    extractor = SpectralFeatureExtractor()
    mapping = load_source_mapping()

    X = []
    labels = []
    track_names = []
    source_tracks = []

    print("Extracting features for visualization...")

    for circle_name, dirname in CIRCLE_DIRS.items():
        circle_dir = DATA_DIR / dirname
        if not circle_dir.exists():
            continue

        files = find_audio_files(circle_dir)

        for filepath in files:
            try:
                features = extract_features_for_file(loader, extractor, filepath)
                X.append(features)
                labels.append(circle_name)
                track_names.append(filepath.stem[:30])

                relative_path = f"{dirname}/{filepath.name}"
                src = get_source_track_name(mapping, relative_path)
                source_tracks.append(src or "unknown")
            except Exception as e:
                print(f"  error: {filepath.name}: {e}")

    print(f"Loaded {len(X)} tracks")
    return np.array(X), labels, track_names, source_tracks


def create_tsne_plot(X: np.ndarray, labels: list, track_names: list,
                     source_tracks: list, perplexity: int = 15):
    """Create t-SNE visualization."""

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run t-SNE
    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000,
        learning_rate='auto',
        init='pca'
    )
    X_embedded = tsne.fit_transform(X_scaled)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each circle
    for circle_name in CIRCLE_DIRS.keys():
        mask = [l == circle_name for l in labels]
        indices = np.where(mask)[0]

        ax.scatter(
            X_embedded[indices, 0],
            X_embedded[indices, 1],
            c=CIRCLE_COLORS[circle_name],
            label=circle_name,
            alpha=0.7,
            s=100,
            edgecolors='white',
            linewidth=0.5
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("Touhou Circle Style Embeddings (t-SNE)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved to: {OUTPUT_PATH}")

    return X_embedded


def print_cluster_analysis(X_embedded: np.ndarray, labels: list):
    """Print basic cluster analysis."""
    print("\nCluster Analysis:")
    print("-" * 40)

    # Calculate centroid for each circle
    centroids = {}
    for circle_name in CIRCLE_DIRS.keys():
        mask = [l == circle_name for l in labels]
        indices = np.where(mask)[0]
        centroids[circle_name] = X_embedded[indices].mean(axis=0)

    # Calculate pairwise distances between centroids
    print("\nCentroid distances (lower = more similar):")
    circles = list(CIRCLE_DIRS.keys())
    for i, c1 in enumerate(circles):
        for c2 in circles[i+1:]:
            dist = np.linalg.norm(centroids[c1] - centroids[c2])
            print(f"  {c1[:15]:15} ↔ {c2[:15]:15}: {dist:.2f}")

    # Intra-cluster spread
    print("\nCluster spread (lower = tighter cluster):")
    for circle_name in circles:
        mask = [l == circle_name for l in labels]
        indices = np.where(mask)[0]
        points = X_embedded[indices]
        centroid = centroids[circle_name]
        spread = np.mean(np.linalg.norm(points - centroid, axis=1))
        print(f"  {circle_name[:20]:20}: {spread:.2f}")


def main():
    X, labels, track_names, source_tracks = load_all_features()

    if len(X) == 0:
        print("No features loaded.")
        return

    X_embedded = create_tsne_plot(X, labels, track_names, source_tracks)
    print_cluster_analysis(X_embedded, labels)


if __name__ == "__main__":
    main()
