"""Train Random Forest baseline classifier on spectral features."""

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio import AudioLoader, find_audio_files
from src.features.spectral import SpectralFeatureExtractor


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
MODEL_DIR = Path(__file__).parent.parent / "models"

CIRCLE_DIRS = {
    "IOSYS": "IOSYS",
    "UNDEAD CORPORATION": "UNDEAD_CORPORATION",
    "暁Records": "Akatsuki_Records",
    "SOUND HOLIC": "SOUND_HOLIC",
    "Liz Triangle": "Liz_Triangle",
}


def extract_features_for_file(loader: AudioLoader, extractor: SpectralFeatureExtractor,
                               filepath: Path) -> np.ndarray:
    """Extract and summarize features for a single audio file."""
    waveform, sr = loader.load(filepath)
    features = extractor.extract_all(waveform)
    return extractor.summarize(features)


def load_dataset():
    """Load all audio files and extract features."""
    loader = AudioLoader()
    extractor = SpectralFeatureExtractor()

    X = []
    y = []
    file_paths = []

    print("Extracting features...")
    print("-" * 50)

    for circle_name, dirname in CIRCLE_DIRS.items():
        circle_dir = DATA_DIR / dirname
        if not circle_dir.exists():
            print(f"  {circle_name}: directory not found, skipping")
            continue

        files = find_audio_files(circle_dir)
        print(f"  {circle_name}: {len(files)} files")

        for i, filepath in enumerate(files):
            try:
                features = extract_features_for_file(loader, extractor, filepath)
                X.append(features)
                y.append(circle_name)
                file_paths.append(filepath)
                if (i + 1) % 5 == 0:
                    print(f"    processed {i + 1}/{len(files)}")
            except Exception as e:
                print(f"    error on {filepath.name}: {e}")

    print("-" * 50)
    print(f"Total: {len(X)} tracks, feature dim: {X[0].shape[0] if X else 0}")

    return np.array(X), np.array(y), file_paths


def train_and_evaluate(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """Train Random Forest and evaluate with holdout set."""

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1
    )

    print("\nCross-validation on training set...")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    print("\nTraining final model...")
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"\nResults:")
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")

    y_pred = clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm, le.classes_)

    print("\nFeature Importances (top 10):")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")

    return clf, le


def print_confusion_matrix(cm: np.ndarray, labels: list):
    """Print confusion matrix with labels."""
    max_label = max(len(str(l)) for l in labels)
    header = " " * (max_label + 2) + "  ".join(f"{l[:8]:>8}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = f"{label:<{max_label}}  " + "  ".join(f"{cm[i, j]:>8}" for j in range(len(labels)))
        print(row)


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    X, y, file_paths = load_dataset()

    if len(X) == 0:
        print("No samples loaded. Check your data directory.")
        return

    clf, le = train_and_evaluate(X, y)

    model_path = MODEL_DIR / "rf_baseline.joblib"
    joblib.dump({"classifier": clf, "label_encoder": le}, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
