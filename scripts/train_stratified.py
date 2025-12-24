"""Train Random Forest with source track stratification (GroupKFold)."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio import AudioLoader, find_audio_files
from src.features.spectral import SpectralFeatureExtractor


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
MAPPING_PATH = PROJECT_ROOT / "data" / "metadata" / "source_track_mapping.json"

CIRCLE_DIRS = {
    "IOSYS": "IOSYS",
    "UNDEAD CORPORATION": "UNDEAD_CORPORATION",
    "暁Records": "Akatsuki_Records",
    "SOUND HOLIC": "SOUND_HOLIC",
    "Liz Triangle": "Liz_Triangle",
}


def load_source_mapping() -> dict:
    """Load source track mapping from JSON."""
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_source_track_id(mapping: dict, relative_path: str) -> int | None:
    """Get source track ID for a file, or None if not mapped."""
    entry = mapping.get(relative_path, {})
    sources = entry.get("source_tracks", [])
    if sources:
        return sources[0]["id"]  # Use first source track
    return None


def extract_features_for_file(loader: AudioLoader, extractor: SpectralFeatureExtractor,
                               filepath: Path) -> np.ndarray:
    """Extract and summarize features for a single audio file."""
    waveform, sr = loader.load(filepath)
    features = extractor.extract_all(waveform)
    return extractor.summarize(features)


def load_dataset_with_groups():
    """Load audio files, extract features, and assign source track groups."""
    loader = AudioLoader()
    extractor = SpectralFeatureExtractor()
    mapping = load_source_mapping()

    X = []
    y = []
    groups = []
    file_paths = []
    skipped = 0

    print("Extracting features with source track grouping...")
    print("-" * 50)

    for circle_name, dirname in CIRCLE_DIRS.items():
        circle_dir = DATA_DIR / dirname
        if not circle_dir.exists():
            continue

        files = find_audio_files(circle_dir)
        print(f"  {circle_name}: {len(files)} files")

        for filepath in files:
            relative_path = f"{dirname}/{filepath.name}"
            source_id = get_source_track_id(mapping, relative_path)

            if source_id is None:
                skipped += 1
                continue

            try:
                features = extract_features_for_file(loader, extractor, filepath)
                X.append(features)
                y.append(circle_name)
                groups.append(source_id)
                file_paths.append(filepath)
            except Exception as e:
                print(f"    error: {filepath.name}: {e}")

    print("-" * 50)
    print(f"Loaded: {len(X)} tracks with source mapping")
    print(f"Skipped: {skipped} tracks (no source mapping)")
    print(f"Unique source tracks (groups): {len(set(groups))}")

    return np.array(X), np.array(y), np.array(groups), file_paths


def train_with_group_kfold(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                            n_splits: int = 5, random_state: int = 42):
    """Train with GroupKFold cross-validation."""

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1
    )

    gkf = GroupKFold(n_splits=n_splits)

    print(f"\nGroupKFold Cross-Validation (k={n_splits})")
    print("=" * 50)

    # Get predictions for each sample using cross_val_predict
    y_pred = cross_val_predict(clf, X, y_encoded, groups=groups, cv=gkf)

    # Overall accuracy
    overall_acc = accuracy_score(y_encoded, y_pred)
    print(f"\nOverall Accuracy: {overall_acc:.3f}")

    # Per-fold accuracies for variance estimate
    fold_accs = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        clf_fold = RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        clf_fold.fit(X_train, y_train)
        fold_acc = clf_fold.score(X_test, y_test)
        fold_accs.append(fold_acc)

        n_groups_test = len(set(groups[test_idx]))
        print(f"  Fold {fold_idx + 1}: {fold_acc:.3f} "
              f"(test: {len(test_idx)} samples, {n_groups_test} source tracks)")

    print(f"\nMean Fold Accuracy: {np.mean(fold_accs):.3f} (+/- {np.std(fold_accs) * 2:.3f})")

    # Classification report
    print("\nClassification Report (aggregated across folds):")
    print(classification_report(y_encoded, y_pred, target_names=le.classes_))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_encoded, y_pred)
    print_confusion_matrix(cm, le.classes_)

    # Train final model on all data for embedding extraction later
    print("\nTraining final model on all data...")
    clf.fit(X, y_encoded)

    return clf, le, overall_acc, fold_accs


def print_confusion_matrix(cm: np.ndarray, labels: list):
    """Print confusion matrix with labels."""
    max_label = max(len(str(l)[:10]) for l in labels)
    header = " " * (max_label + 2) + "  ".join(f"{str(l)[:8]:>8}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = f"{str(label)[:max_label]:<{max_label}}  " + "  ".join(f"{cm[i, j]:>8}" for j in range(len(labels)))
        print(row)


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    X, y, groups, file_paths = load_dataset_with_groups()

    if len(X) == 0:
        print("No samples loaded. Check data and mapping.")
        return

    clf, le, overall_acc, fold_accs = train_with_group_kfold(X, y, groups)

    # Save model
    model_path = MODEL_DIR / "rf_stratified.joblib"
    joblib.dump({
        "classifier": clf,
        "label_encoder": le,
        "accuracy": overall_acc,
        "fold_accuracies": fold_accs
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    # Summary comparison
    print("\n" + "=" * 50)
    print("COMPARISON: Baseline vs Stratified")
    print("=" * 50)
    print(f"Baseline (random split):     76.2% test accuracy")
    print(f"Stratified (GroupKFold):     {overall_acc * 100:.1f}% accuracy")
    print(f"Difference:                  {(overall_acc - 0.762) * 100:+.1f}%")


if __name__ == "__main__":
    main()
