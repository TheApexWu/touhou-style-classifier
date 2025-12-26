"""
Fast Source Separation Experiment using HPSS (Harmonic-Percussive Source Separation)

HPSS is a classical signal processing technique that separates audio into:
- Harmonic: sustained tones (vocals, melodic instruments, synths)
- Percussive: transients (drums, attacks, clicks)

This is NOT as clean as Demucs, but runs in ~1 second per file instead of ~30-60.
Use this for hypothesis testing before committing to expensive neural separation.

Hypothesis: Liz Triangle ↔ IOSYS confusion is driven by shared vocal characteristics
in the harmonic component. Percussive features should better distinguish them.

Usage:
    python scripts/experiment_hpss.py --dry-run
    python scripts/experiment_hpss.py
    python scripts/experiment_hpss.py --circle Liz_Triangle  # Single circle
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from time import time

import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio import AudioLoader, find_audio_files
from src.features.spectral import SpectralFeatureExtractor

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


def hpss_separate(waveform: np.ndarray, margin: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate audio into harmonic and percussive components using median filtering.

    Args:
        waveform: Audio signal
        margin: Separation margin (higher = more aggressive separation)

    Returns:
        (harmonic, percussive) waveforms
    """
    return librosa.effects.hpss(waveform, margin=margin)


def extract_hpss_features(waveform: np.ndarray, extractor: SpectralFeatureExtractor,
                          mode: str = "both") -> np.ndarray:
    """
    Extract features from HPSS-separated components.

    Args:
        waveform: Original audio
        extractor: Feature extractor
        mode: 'harmonic', 'percussive', 'both', or 'original'

    Returns:
        Feature vector
    """
    if mode == "original":
        feats = extractor.extract_all(waveform)
        return extractor.summarize(feats)

    harmonic, percussive = hpss_separate(waveform)

    if mode == "harmonic":
        feats = extractor.extract_all(harmonic)
        return extractor.summarize(feats)

    elif mode == "percussive":
        feats = extractor.extract_all(percussive)
        return extractor.summarize(feats)

    elif mode == "both":
        harm_feats = extractor.extract_all(harmonic)
        perc_feats = extractor.extract_all(percussive)
        return np.concatenate([
            extractor.summarize(harm_feats),
            extractor.summarize(perc_feats)
        ])

    else:
        raise ValueError(f"Unknown mode: {mode}")


def load_dataset_with_hpss(mode: str = "both", limit_per_circle: int = None):
    """Load dataset and extract HPSS-based features."""

    loader = AudioLoader()
    extractor = SpectralFeatureExtractor()

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    def get_source_id(rel_path):
        entry = mapping.get(rel_path, {})
        sources = entry.get("source_tracks", [])
        return sources[0]["id"] if sources else None

    X, y, groups = [], [], []

    print(f"Extracting HPSS features (mode={mode})...")
    start_time = time()

    for circle_name, dirname in CIRCLES.items():
        circle_dir = DATA_DIR / dirname
        if not circle_dir.exists():
            continue

        files = find_audio_files(circle_dir)
        if limit_per_circle:
            files = files[:limit_per_circle]

        circle_count = 0

        for f in files:
            rel_path = f"{dirname}/{f.name}"
            src_id = get_source_id(rel_path)
            if src_id is None:
                continue

            try:
                waveform, _ = loader.load(f)
                features = extract_hpss_features(waveform, extractor, mode=mode)
                X.append(features)
                y.append(circle_name)
                groups.append(src_id)
                circle_count += 1
            except Exception as e:
                pass

        print(f"  {circle_name}: {circle_count} samples")

    elapsed = time() - start_time
    print(f"\nExtraction time: {elapsed:.1f}s ({elapsed/len(X):.2f}s per sample)")

    return np.array(X), np.array(y), np.array(groups)


def train_and_evaluate(X, y, groups, mode: str):
    """Train classifier and return results."""

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

    class_accs = {}
    for i, cls in enumerate(le.classes_):
        mask = y_enc == i
        class_accs[cls] = (y_pred[mask] == y_enc[mask]).mean()

    return overall_acc, class_accs, le.classes_


def main():
    parser = argparse.ArgumentParser(description="HPSS separation experiment")
    parser.add_argument("--dry-run", action="store_true", help="Just count files")
    parser.add_argument("--quick", action="store_true", help="Limit to 50 per circle")
    args = parser.parse_args()

    print("=" * 60)
    print("HPSS SOURCE SEPARATION EXPERIMENT")
    print("=" * 60)
    print()
    print("Comparing feature extraction modes:")
    print("  - original:   Mixed audio (baseline)")
    print("  - harmonic:   Vocals + melodic instruments")
    print("  - percussive: Drums + transients")
    print("  - both:       Harmonic + percussive concatenated")
    print()

    if args.dry_run:
        print("*** DRY RUN ***")
        for circle_name, dirname in CIRCLES.items():
            circle_dir = DATA_DIR / dirname
            if circle_dir.exists():
                count = len(find_audio_files(circle_dir))
                print(f"  {circle_name}: {count} files")
        return

    limit = 50 if args.quick else None
    if limit:
        print(f"*** QUICK MODE: {limit} samples per circle ***\n")

    # Test all modes
    modes = ["original", "harmonic", "percussive", "both"]
    results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()}")
        print("="*60)

        X, y, groups = load_dataset_with_hpss(mode=mode, limit_per_circle=limit)

        if len(X) == 0:
            print("No samples loaded!")
            continue

        print(f"\nSamples: {len(X)}, Features: {X.shape[1]}")

        overall_acc, class_accs, classes = train_and_evaluate(X, y, groups, mode)

        results[mode] = {
            "overall": overall_acc,
            "class_accs": class_accs,
            "n_features": X.shape[1]
        }

        print(f"\nOverall accuracy: {overall_acc:.1%}")
        print(f"\nPer-class:")
        for cls in classes:
            print(f"  {cls:<20}: {class_accs[cls]:.1%}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print()
    print(f"{'Mode':<15} {'Overall':>10} {'Liz Triangle':>15} {'IOSYS':>10}")
    print("-" * 52)

    for mode in modes:
        if mode in results:
            r = results[mode]
            liz = r["class_accs"].get("Liz Triangle", 0)
            iosys = r["class_accs"].get("IOSYS", 0)
            print(f"{mode:<15} {r['overall']:>10.1%} {liz:>15.1%} {iosys:>10.1%}")

    # Key insight
    print("\n" + "-" * 52)
    print("KEY QUESTION: Does percussive mode improve Liz Triangle?")

    if "original" in results and "percussive" in results:
        orig_liz = results["original"]["class_accs"].get("Liz Triangle", 0)
        perc_liz = results["percussive"]["class_accs"].get("Liz Triangle", 0)
        delta = perc_liz - orig_liz

        if delta > 0.05:
            print(f"YES: +{delta:.1%} improvement. Vocals were confusing the classifier.")
            print("→ Consider full Demucs separation for cleaner stems.")
        elif delta < -0.05:
            print(f"NO: {delta:.1%} worse. The harmonic content is actually informative.")
            print("→ Don't bother with Demucs, vocals aren't the problem.")
        else:
            print(f"INCONCLUSIVE: {delta:+.1%} change. Need more data or different approach.")


if __name__ == "__main__":
    main()
