"""
Misclassification Deep Dive Analysis

Analyzes which tracks get confused with which, focusing on:
1. Liz Triangle ↔ IOSYS confusion (the main problem)
2. Album/year patterns in misclassifications
3. Shared ZUN source tracks
4. Feature differences between correct vs misclassified

Usage:
    python scripts/analyze_misclassifications.py
    python scripts/analyze_misclassifications.py --focus "Liz Triangle"
    python scripts/analyze_misclassifications.py --export  # Save detailed CSV
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
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


def load_dataset_with_metadata():
    """Load dataset with full metadata for analysis."""

    loader = AudioLoader()
    extractor = SpectralFeatureExtractor()

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    samples = []
    X = []

    print("Loading dataset with metadata...")

    for circle_name, dirname in CIRCLES.items():
        circle_dir = DATA_DIR / dirname
        if not circle_dir.exists():
            continue

        files = find_audio_files(circle_dir)
        circle_count = 0

        for f in files:
            rel_path = f"{dirname}/{f.name}"
            entry = mapping.get(rel_path, {})
            sources = entry.get("source_tracks", [])

            if not sources:
                continue

            source = sources[0]

            try:
                waveform, _ = loader.load(f)
                feats = extractor.extract_all(waveform)
                feature_vec = extractor.summarize(feats)

                X.append(feature_vec)
                samples.append({
                    "file_path": str(f),
                    "file_name": f.name,
                    "circle": circle_name,
                    "dirname": dirname,
                    "source_id": source["id"],
                    "source_name": source.get("name", "Unknown"),
                    "source_game": source.get("game", "Unknown"),
                    "rel_path": rel_path,
                })
                circle_count += 1

            except Exception as e:
                pass

        print(f"  {circle_name}: {circle_count} samples")

    return np.array(X), samples


def get_predictions(X, samples):
    """Train classifier and get cross-validated predictions."""

    y = np.array([s["circle"] for s in samples])
    groups = np.array([s["source_id"] for s in samples])

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
    y_pred_labels = le.inverse_transform(y_pred)

    # Get prediction probabilities for confidence analysis
    y_proba = cross_val_predict(clf, X, y_enc, groups=groups, cv=gkf, method='predict_proba')

    return y_pred_labels, y_proba, le.classes_


def analyze_confusion_pair(samples, y_true, y_pred, circle_a, circle_b):
    """Detailed analysis of confusion between two circles."""

    results = {
        "a_to_b": [],  # circle_a misclassified as circle_b
        "b_to_a": [],  # circle_b misclassified as circle_a
    }

    for i, sample in enumerate(samples):
        true_label = y_true[i]
        pred_label = y_pred[i]

        if true_label == circle_a and pred_label == circle_b:
            results["a_to_b"].append(sample)
        elif true_label == circle_b and pred_label == circle_a:
            results["b_to_a"].append(sample)

    return results


def analyze_source_track_patterns(misclassified_samples):
    """Check if certain ZUN source tracks are more confusing."""

    source_counts = defaultdict(list)

    for sample in misclassified_samples:
        key = (sample["source_id"], sample["source_name"], sample["source_game"])
        source_counts[key].append(sample)

    # Sort by frequency
    sorted_sources = sorted(source_counts.items(), key=lambda x: -len(x[1]))

    return sorted_sources


def print_detailed_analysis(samples, y_pred, focus_circle=None):
    """Print comprehensive misclassification analysis."""

    y_true = [s["circle"] for s in samples]

    print("\n" + "=" * 70)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 70)

    # Overall confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    correct = defaultdict(int)
    total = defaultdict(int)

    misclassified_samples = []

    for i, sample in enumerate(samples):
        true_label = y_true[i]
        pred_label = y_pred[i]
        total[true_label] += 1

        if true_label == pred_label:
            correct[true_label] += 1
        else:
            confusion[true_label][pred_label] += 1
            misclassified_samples.append({
                **sample,
                "predicted": pred_label,
            })

    # Print confusion summary
    print("\n1. CONFUSION MATRIX SUMMARY")
    print("-" * 70)
    print(f"{'True Label':<25} {'Predicted As':<25} {'Count':>10}")
    print("-" * 70)

    for true_label in CIRCLES.keys():
        if true_label in confusion:
            for pred_label, count in sorted(confusion[true_label].items(), key=lambda x: -x[1]):
                print(f"{true_label:<25} {pred_label:<25} {count:>10}")

    # Focus analysis (Liz Triangle ↔ IOSYS by default)
    focus_pairs = [("Liz Triangle", "IOSYS")]
    if focus_circle:
        focus_pairs = [(focus_circle, c) for c in CIRCLES.keys() if c != focus_circle]

    for circle_a, circle_b in focus_pairs:
        pair_results = analyze_confusion_pair(samples, y_true, y_pred, circle_a, circle_b)

        a_to_b = pair_results["a_to_b"]
        b_to_a = pair_results["b_to_a"]

        if not a_to_b and not b_to_a:
            continue

        print(f"\n\n2. DETAILED: {circle_a} ↔ {circle_b}")
        print("=" * 70)

        if a_to_b:
            print(f"\n{circle_a} misclassified as {circle_b}: {len(a_to_b)} tracks")
            print("-" * 50)

            # Source track patterns
            source_patterns = analyze_source_track_patterns(a_to_b)

            print("\nSource tracks (ZUN originals) that cause confusion:")
            for (src_id, src_name, src_game), tracks in source_patterns[:10]:
                print(f"  • {src_name}")
                print(f"    Game: {src_game}")
                print(f"    Confused tracks: {len(tracks)}")
                for t in tracks[:3]:
                    print(f"      - {t['file_name'][:50]}")
                if len(tracks) > 3:
                    print(f"      ... and {len(tracks) - 3} more")
                print()

        if b_to_a:
            print(f"\n{circle_b} misclassified as {circle_a}: {len(b_to_a)} tracks")
            print("-" * 50)

            source_patterns = analyze_source_track_patterns(b_to_a)

            print("\nSource tracks (ZUN originals) that cause confusion:")
            for (src_id, src_name, src_game), tracks in source_patterns[:10]:
                print(f"  • {src_name}")
                print(f"    Game: {src_game}")
                print(f"    Confused tracks: {len(tracks)}")
                for t in tracks[:3]:
                    print(f"      - {t['file_name'][:50]}")
                if len(tracks) > 3:
                    print(f"      ... and {len(tracks) - 3} more")
                print()

    # All misclassified tracks list
    print("\n\n3. ALL MISCLASSIFIED TRACKS")
    print("=" * 70)

    by_circle = defaultdict(list)
    for m in misclassified_samples:
        by_circle[m["circle"]].append(m)

    for circle in CIRCLES.keys():
        if circle in by_circle:
            tracks = by_circle[circle]
            acc = correct[circle] / total[circle] if total[circle] > 0 else 0
            print(f"\n{circle} ({len(tracks)} misclassified, {acc:.1%} accuracy)")
            print("-" * 50)

            # Group by what they were predicted as
            by_pred = defaultdict(list)
            for t in tracks:
                by_pred[t["predicted"]].append(t)

            for pred, pred_tracks in sorted(by_pred.items(), key=lambda x: -len(x[1])):
                print(f"\n  → Predicted as {pred} ({len(pred_tracks)} tracks):")
                for t in pred_tracks[:5]:
                    print(f"     • {t['file_name'][:60]}")
                    print(f"       Source: {t['source_name'][:50]}")
                if len(pred_tracks) > 5:
                    print(f"     ... and {len(pred_tracks) - 5} more")

    return misclassified_samples


def export_misclassifications(misclassified_samples, output_path):
    """Export misclassifications to CSV for manual review."""

    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'file_name', 'true_circle', 'predicted_circle',
            'source_name', 'source_game', 'file_path'
        ])
        writer.writeheader()

        for m in misclassified_samples:
            writer.writerow({
                'file_name': m['file_name'],
                'true_circle': m['circle'],
                'predicted_circle': m['predicted'],
                'source_name': m['source_name'],
                'source_game': m['source_game'],
                'file_path': m['file_path'],
            })

    print(f"\nExported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze misclassifications")
    parser.add_argument("--focus", type=str, help="Focus on specific circle's errors")
    parser.add_argument("--export", action="store_true", help="Export to CSV")
    args = parser.parse_args()

    X, samples = load_dataset_with_metadata()

    print(f"\nTotal samples: {len(samples)}")
    print("Getting cross-validated predictions...")

    y_pred, y_proba, classes = get_predictions(X, samples)

    misclassified = print_detailed_analysis(samples, y_pred, focus_circle=args.focus)

    if args.export:
        output_path = PROJECT_ROOT / "data" / "metadata" / "misclassifications.csv"
        export_misclassifications(misclassified, output_path)

    # Key insights summary
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    # Count Liz Triangle → IOSYS specifically
    liz_to_iosys = sum(1 for m in misclassified
                       if m["circle"] == "Liz Triangle" and m["predicted"] == "IOSYS")
    iosys_to_liz = sum(1 for m in misclassified
                       if m["circle"] == "IOSYS" and m["predicted"] == "Liz Triangle")

    print(f"\nLiz Triangle → IOSYS: {liz_to_iosys} tracks")
    print(f"IOSYS → Liz Triangle: {iosys_to_liz} tracks")

    if liz_to_iosys > 0:
        liz_misclassified = [m for m in misclassified
                            if m["circle"] == "Liz Triangle" and m["predicted"] == "IOSYS"]

        # Check for source track patterns
        source_counts = defaultdict(int)
        for m in liz_misclassified:
            source_counts[m["source_name"]] += 1

        if source_counts:
            top_source = max(source_counts.items(), key=lambda x: x[1])
            print(f"\nMost confusing source track: '{top_source[0]}' ({top_source[1]} instances)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
