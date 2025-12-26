"""
Neural Embeddings Experiment

Compare pretrained audio embeddings vs handcrafted features.
Uses CLAP (Contrastive Language-Audio Pretraining) from Microsoft.

Small-scale test: 20 samples per circle to keep it fast.

Usage:
    python scripts/experiment_embeddings.py --quick     # 10 per class
    python scripts/experiment_embeddings.py             # 20 per class
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from time import time

import numpy as np
import torch
import torchaudio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio import find_audio_files

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MAPPING_PATH = PROJECT_ROOT / "data" / "metadata" / "source_track_mapping.json"

CIRCLES = {
    "IOSYS": "IOSYS",
    "UNDEAD CORPORATION": "UNDEAD_CORPORATION",
    "暁Records": "Akatsuki_Records",
    "SOUND HOLIC": "SOUND_HOLIC",
    "Liz Triangle": "Liz_Triangle",
}

# Poker Facer tracks to exclude
EXCLUDE_FILES = {
    "(01) [lily-an] Ms.Munchikin.flac",
    "(02) [lily-an] レイズ.flac",
    "(03) [lily-an] 星に願いを.flac",
    "(04) [lily-an] ありがとう.flac",
    "(05) [lily-an] ハレーション.リモーション .flac",
    "(06) [lily-an] 深淵の災禍.flac",
    "(07) [lily-an] ネガポジ.flac",
    "(08) [lily-an] contact.flac",
}


def load_audio_for_clap(file_path, target_sr=48000, max_duration=30):
    """Load and preprocess audio for CLAP using librosa."""
    import librosa

    # Load with librosa (handles FLAC well)
    waveform, sr = librosa.load(file_path, sr=target_sr, mono=True, duration=max_duration)

    return torch.from_numpy(waveform).float()


def get_clap_model():
    """Load CLAP model."""
    from transformers import ClapModel, ClapProcessor

    print("Loading CLAP model (first time downloads ~600MB)...")
    model_name = "laion/clap-htsat-unfused"

    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name)
    model.eval()

    # Use MPS if available (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = model.to(device)

    return model, processor, device


def get_mert_model():
    """Load MERT model (music-specific)."""
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    print("Loading MERT model (first time downloads ~330MB)...")
    model_name = "m-a-p/MERT-v1-95M"

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    # Use MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = model.to(device)

    return model, processor, device


def load_audio_for_mert(file_path, target_sr=24000, max_duration=30):
    """Load and preprocess audio for MERT (24kHz)."""
    import librosa
    waveform, sr = librosa.load(file_path, sr=target_sr, mono=True, duration=max_duration)
    return waveform


def extract_mert_embedding(waveform, model, processor, device):
    """Extract MERT embedding for a single audio file."""
    inputs = processor(
        waveform,
        sampling_rate=24000,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use mean of last hidden state as embedding
        hidden_states = outputs.hidden_states[-1]
        embedding = hidden_states.mean(dim=1)

    return embedding.cpu().numpy().flatten()


def load_dataset_mert(model, processor, device, limit_per_circle=20):
    """Load dataset with MERT embeddings."""

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    def get_source_id(dirname, filename):
        rel_path = f"{dirname}/{filename}"
        entry = mapping.get(rel_path, {})
        sources = entry.get("source_tracks", [])
        return sources[0]["id"] if sources else None

    X, y, groups = [], [], []

    print(f"\nExtracting MERT embeddings (limit={limit_per_circle} per circle)...")
    start_time = time()

    for circle_name, dirname in CIRCLES.items():
        circle_dir = DATA_DIR / dirname
        if not circle_dir.exists():
            continue

        files = find_audio_files(circle_dir)
        files = [f for f in files if f.name not in EXCLUDE_FILES]
        files = files[:limit_per_circle]

        circle_count = 0

        for f in files:
            source_id = get_source_id(dirname, f.name)
            if source_id is None:
                source_id = hash(f.name) % 10000

            try:
                waveform = load_audio_for_mert(f)
                embedding = extract_mert_embedding(waveform, model, processor, device)

                X.append(embedding)
                y.append(circle_name)
                groups.append(source_id)
                circle_count += 1

            except Exception as e:
                print(f"    Error: {f.name[:30]}: {e}")

        print(f"  {circle_name}: {circle_count} samples")

    elapsed = time() - start_time
    if len(X) > 0:
        print(f"\nExtraction time: {elapsed:.1f}s ({elapsed/len(X):.2f}s per sample)")
    else:
        print("\nNo samples extracted!")

    return np.array(X) if X else np.array([]), np.array(y), np.array(groups)


def extract_clap_embedding(waveform, model, processor, device):
    """Extract CLAP embedding for a single audio file."""
    # CLAP expects audio at 48kHz
    inputs = processor(
        audios=waveform.numpy(),
        sampling_rate=48000,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.get_audio_features(**inputs)

    return outputs.cpu().numpy().flatten()


def load_dataset_clap(model, processor, device, limit_per_circle=20):
    """Load dataset with CLAP embeddings."""

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    def get_source_id(dirname, filename):
        rel_path = f"{dirname}/{filename}"
        entry = mapping.get(rel_path, {})
        sources = entry.get("source_tracks", [])
        return sources[0]["id"] if sources else None

    X, y, groups = [], [], []

    print(f"\nExtracting CLAP embeddings (limit={limit_per_circle} per circle)...")
    start_time = time()

    for circle_name, dirname in CIRCLES.items():
        circle_dir = DATA_DIR / dirname
        if not circle_dir.exists():
            continue

        files = find_audio_files(circle_dir)

        # Filter excluded files
        files = [f for f in files if f.name not in EXCLUDE_FILES]

        # Limit
        files = files[:limit_per_circle]

        circle_count = 0

        for f in files:
            source_id = get_source_id(dirname, f.name)
            if source_id is None:
                # Use filename hash as fallback group
                source_id = hash(f.name) % 10000

            try:
                waveform = load_audio_for_clap(f)
                embedding = extract_clap_embedding(waveform, model, processor, device)

                X.append(embedding)
                y.append(circle_name)
                groups.append(source_id)
                circle_count += 1

            except Exception as e:
                print(f"    Error: {f.name[:30]}: {e}")

        print(f"  {circle_name}: {circle_count} samples")

    elapsed = time() - start_time
    if len(X) > 0:
        print(f"\nExtraction time: {elapsed:.1f}s ({elapsed/len(X):.2f}s per sample)")
    else:
        print("\nNo samples extracted!")

    return np.array(X) if X else np.array([]), np.array(y), np.array(groups)


def load_dataset_handcrafted(limit_per_circle=20):
    """Load dataset with handcrafted features for comparison."""
    from src.data.audio import AudioLoader
    from src.features.spectral import SpectralFeatureExtractor

    loader = AudioLoader()
    extractor = SpectralFeatureExtractor()

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    def get_source_id(dirname, filename):
        rel_path = f"{dirname}/{filename}"
        entry = mapping.get(rel_path, {})
        sources = entry.get("source_tracks", [])
        return sources[0]["id"] if sources else None

    X, y, groups = [], [], []

    print(f"\nExtracting handcrafted features (limit={limit_per_circle} per circle)...")
    start_time = time()

    for circle_name, dirname in CIRCLES.items():
        circle_dir = DATA_DIR / dirname
        if not circle_dir.exists():
            continue

        files = find_audio_files(circle_dir)
        files = [f for f in files if f.name not in EXCLUDE_FILES]
        files = files[:limit_per_circle]

        circle_count = 0

        for f in files:
            source_id = get_source_id(dirname, f.name)
            if source_id is None:
                source_id = hash(f.name) % 10000

            try:
                waveform, _ = loader.load(f)
                feats = extractor.extract_all(waveform)
                feature_vec = extractor.summarize(feats)

                X.append(feature_vec)
                y.append(circle_name)
                groups.append(source_id)
                circle_count += 1

            except:
                pass

        print(f"  {circle_name}: {circle_count} samples")

    elapsed = time() - start_time
    print(f"\nExtraction time: {elapsed:.1f}s ({elapsed/len(X):.2f}s per sample)")

    return np.array(X), np.array(y), np.array(groups)


def train_and_evaluate(X, y, groups, name):
    """Train classifier and return accuracy."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Use StratifiedKFold for small samples (GroupKFold may fail)
    n_unique_groups = len(set(groups))
    if n_unique_groups >= 5:
        cv = GroupKFold(n_splits=min(5, n_unique_groups))
        y_pred = cross_val_predict(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
            X, y_enc, groups=groups, cv=cv
        )
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
            X, y_enc, cv=cv
        )

    overall_acc = accuracy_score(y_enc, y_pred)

    print(f"\n{name}")
    print("=" * 50)
    print(f"Overall Accuracy: {overall_acc:.1%}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"\nPer-class:")

    class_accs = {}
    for i, cls in enumerate(le.classes_):
        mask = y_enc == i
        acc = (y_pred[mask] == y_enc[mask]).mean()
        class_accs[cls] = acc
        print(f"  {cls:<25}: {acc:.1%}")

    return overall_acc, class_accs


def main():
    parser = argparse.ArgumentParser(description="Neural embeddings experiment")
    parser.add_argument("--quick", action="store_true", help="10 samples per class")
    parser.add_argument("--clap-only", action="store_true", help="Only run CLAP")
    parser.add_argument("--mert-only", action="store_true", help="Only run MERT")
    parser.add_argument("--skip-clap", action="store_true", help="Skip CLAP, run MERT + handcrafted")
    args = parser.parse_args()

    limit = 10 if args.quick else 20

    print("=" * 60)
    print("NEURAL EMBEDDINGS EXPERIMENT")
    print("=" * 60)
    print(f"\nSamples per circle: {limit}")
    print(f"Total samples: ~{limit * 5}")

    results = {}

    # CLAP embeddings
    if not args.skip_clap and not args.mert_only:
        print("\n" + "-" * 60)
        print("PHASE 1: CLAP Embeddings (general audio)")
        print("-" * 60)

        model, processor, device = get_clap_model()
        X_clap, y_clap, groups_clap = load_dataset_clap(model, processor, device, limit_per_circle=limit)

        if len(X_clap) > 0:
            acc_clap, class_accs_clap = train_and_evaluate(X_clap, y_clap, groups_clap, "CLAP EMBEDDINGS")
            results["clap"] = {"overall": acc_clap, "class_accs": class_accs_clap, "dim": X_clap.shape[1]}

        # Free memory
        del model, processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # MERT embeddings (music-specific)
    if not args.clap_only:
        print("\n" + "-" * 60)
        print("PHASE 2: MERT Embeddings (music-specific)")
        print("-" * 60)

        model, processor, device = get_mert_model()
        X_mert, y_mert, groups_mert = load_dataset_mert(model, processor, device, limit_per_circle=limit)

        if len(X_mert) > 0:
            acc_mert, class_accs_mert = train_and_evaluate(X_mert, y_mert, groups_mert, "MERT EMBEDDINGS")
            results["mert"] = {"overall": acc_mert, "class_accs": class_accs_mert, "dim": X_mert.shape[1]}

        # Free memory
        del model, processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not args.clap_only and not args.mert_only:
        # Handcrafted features
        print("\n" + "-" * 60)
        print("PHASE 3: Handcrafted Features (MFCCs + Spectral)")
        print("-" * 60)

        X_hc, y_hc, groups_hc = load_dataset_handcrafted(limit_per_circle=limit)

        if len(X_hc) > 0:
            acc_hc, class_accs_hc = train_and_evaluate(X_hc, y_hc, groups_hc, "HANDCRAFTED FEATURES")
            results["handcrafted"] = {"overall": acc_hc, "class_accs": class_accs_hc, "dim": X_hc.shape[1]}

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    print(f"\n{'Method':<20} {'Overall':>10} {'Liz Triangle':>15} {'Dimensions':>12}")
    print("-" * 60)

    for method, r in results.items():
        liz = r["class_accs"].get("Liz Triangle", 0)
        print(f"{method:<20} {r['overall']:>10.1%} {liz:>15.1%} {r['dim']:>12}")

    # Key question
    if "clap" in results and "handcrafted" in results:
        clap_liz = results["clap"]["class_accs"].get("Liz Triangle", 0)
        hc_liz = results["handcrafted"]["class_accs"].get("Liz Triangle", 0)
        delta = clap_liz - hc_liz

        print(f"\n{'='*60}")
        print("KEY QUESTION: Do neural embeddings help Liz Triangle?")
        if delta > 0.05:
            print(f"YES: +{delta:.1%} improvement with CLAP")
            print("→ Pretrained embeddings capture style better than MFCCs")
        elif delta < -0.05:
            print(f"NO: {delta:.1%} worse with CLAP")
            print("→ Handcrafted features are actually better for this task")
        else:
            print(f"INCONCLUSIVE: {delta:+.1%} difference")
            print("→ Both methods perform similarly")


if __name__ == "__main__":
    main()
