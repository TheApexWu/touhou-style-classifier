"""
Train classifier on source-separated audio.

Compares three approaches:
1. Vocals-only features
2. Instrumental-only features
3. Combined (vocals + instrumental features concatenated)

Run after separate_sources.py has processed the audio files.

Usage:
    python scripts/train_separated.py
    python scripts/train_separated.py --mode vocals      # Vocals only
    python scripts/train_separated.py --mode instrumental  # Instrumental only
    python scripts/train_separated.py --mode combined    # Both (default)
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio import AudioLoader
from src.features.spectral import SpectralFeatureExtractor

PROJECT_ROOT = Path(__file__).parent.parent
SEPARATED_DIR = PROJECT_ROOT / "data" / "separated"
MAPPING_PATH = PROJECT_ROOT / "data" / "metadata" / "source_track_mapping.json"
MODEL_DIR = PROJECT_ROOT / "models"

CIRCLES = {
    "IOSYS": "IOSYS",
    "UNDEAD CORPORATION": "UNDEAD_CORPORATION",
    "暁Records": "Akatsuki_Records",
    "SOUND HOLIC": "SOUND_HOLIC",
    "Liz Triangle": "Liz_Triangle",
}


def load_separated_features(mode: str = "combined"):
    """
    Load features from separated audio files.

    mode: 'vocals', 'instrumental', or 'combined'
    """
    loader = AudioLoader()
    extractor = SpectralFeatureExtractor()

    # Load source track mapping for stratification
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    def get_source_id(dirname, original_name):
        # Map separated filename back to original
        # e.g., "track1_vocals.wav" -> "track1.flac"
        rel_path = f"{dirname}/{original_name}"
        entry = mapping.get(rel_path, {})
        sources = entry.get("source_tracks", [])
        return sources[0]["id"] if sources else None

    X = []
    y = []
    groups = []
    processed = 0

    print(f"Extracting features (mode={mode})...")

    for circle_name, dirname in CIRCLES.items():
        circle_dir = SEPARATED_DIR / dirname
        if not circle_dir.exists():
            print(f"  {circle_name}: no separated files found")
            continue

        # Find all vocal files
        vocal_files = list(circle_dir.glob("*_vocals.wav"))

        circle_count = 0
        for vocal_path in vocal_files:
            # Derive paths
            stem = vocal_path.stem.replace("_vocals", "")
            instrumental_path = circle_dir / f"{stem}_instrumental.wav"

            # Try to find original filename for source track mapping
            original_name = f"{stem}.flac"
            source_id = get_source_id(dirname, original_name)

            if source_id is None:
                continue

            if not instrumental_path.exists() and mode in ["instrumental", "combined"]:
                continue

            try:
                features = []

                if mode in ["vocals", "combined"]:
                    waveform, _ = loader.load(vocal_path)
                    vocal_feats = extractor.extract_all(waveform)
                    features.append(extractor.summarize(vocal_feats))

                if mode in ["instrumental", "combined"]:
                    waveform, _ = loader.load(instrumental_path)
                    inst_feats = extractor.extract_all(waveform)
                    features.append(extractor.summarize(inst_feats))

                # Concatenate if combined mode
                combined_features = np.concatenate(features) if len(features) > 1 else features[0]

                X.append(combined_features)
                y.append(circle_name)
                groups.append(source_id)
                circle_count += 1

            except Exception as e:
                pass

        print(f"  {circle_name}: {circle_count} samples")
        processed += circle_count

    return np.array(X), np.array(y), np.array(groups)


def train_and_evaluate(X, y, groups, model_name: str):
    """Train with GroupKFold and evaluate."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    gkf = GroupKFold(n_splits=5)

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    y_pred = cross_val_predict(clf, X, y_enc, groups=groups, cv=gkf)

    overall_acc = accuracy_score(y_enc, y_pred)

    print(f"\nOverall Accuracy: {overall_acc:.1%}")
    print(f"Feature dimension: {X.shape[1]}")

    print("\nPer-class accuracy:")
    print(f"{'Circle':<25} {'Accuracy':>10} {'Samples':>10}")
    print("-" * 47)

    class_accs = {}
    for i, cls in enumerate(le.classes_):
        mask = y_enc == i
        acc = (y_pred[mask] == y_enc[mask]).mean()
        class_accs[cls] = acc
        print(f"{cls:<25} {acc:>10.1%} {mask.sum():>10}")

    # Train final model
    clf.fit(X, y_enc)

    return clf, le, overall_acc, class_accs


def main():
    parser = argparse.ArgumentParser(description="Train on separated sources")
    parser.add_argument("--mode", type=str, default="combined",
                        choices=["vocals", "instrumental", "combined"],
                        help="Which stems to use for features")
    args = parser.parse_args()

    print("=" * 60)
    print(f"TRAINING ON SEPARATED SOURCES (mode={args.mode})")
    print("=" * 60)

    # Check if separated files exist
    if not SEPARATED_DIR.exists():
        print(f"\nERROR: Separated directory not found: {SEPARATED_DIR}")
        print("Run separate_sources.py first.")
        sys.exit(1)

    X, y, groups = load_separated_features(mode=args.mode)

    if len(X) == 0:
        print("\nERROR: No samples loaded. Run separate_sources.py first.")
        sys.exit(1)

    print(f"\nTotal samples: {len(X)}")
    print(f"Unique source tracks: {len(set(groups))}")

    clf, le, overall_acc, class_accs = train_and_evaluate(X, y, groups, args.mode)

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"rf_separated_{args.mode}.joblib"
    joblib.dump({
        "classifier": clf,
        "label_encoder": le,
        "accuracy": overall_acc,
        "class_accuracies": class_accs,
        "mode": args.mode,
        "n_samples": len(X),
    }, model_path)

    print(f"\nModel saved to: {model_path}")

    # Comparison with baseline
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Original (mixed audio):     89.5%")
    print(f"Separated ({args.mode}):    {overall_acc:.1%}")
    print(f"Delta: {(overall_acc - 0.895) * 100:+.1f}%")

    # Specifically check Liz Triangle (the problem case)
    liz_acc = class_accs.get("Liz Triangle", 0)
    print(f"\nLiz Triangle specifically:")
    print(f"  Original: 67.1%")
    print(f"  Separated: {liz_acc:.1%}")
    print(f"  Delta: {(liz_acc - 0.671) * 100:+.1f}%")


if __name__ == "__main__":
    main()
