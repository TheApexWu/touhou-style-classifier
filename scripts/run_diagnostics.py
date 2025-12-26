"""
Dataset Diagnostics for Touhou Style Classifier

Run before training to validate data quality and inform dataset strategy.
Outputs justify decisions on class balancing, weighting, and sampling.

Usage:
    python scripts/run_diagnostics.py
    python scripts/run_diagnostics.py --full  # Don't sample, check all files
"""

import argparse
import json
import random
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio import AudioLoader
from src.features.spectral import SpectralFeatureExtractor


# === Configuration ===

TLMC_DIR = Path("/Users/amadeuswoo/Downloads/TLMC temp ver")
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MAPPING_PATH = PROJECT_ROOT / "data" / "metadata" / "source_track_mapping.json"

TLMC_CIRCLES = {
    "IOSYS": "[IOSYS] イオシス",
    "UNDEAD CORPORATION": "[UNDEAD CORPORATION]",
    "暁Records": "[暁Records]",
    "SOUND HOLIC": "[SOUND HOLIC]",
    "Liz Triangle": "[Liz triangle] りすとら",
}

LOCAL_CIRCLES = {
    "IOSYS": "IOSYS",
    "UNDEAD CORPORATION": "UNDEAD_CORPORATION",
    "暁Records": "Akatsuki_Records",
    "SOUND HOLIC": "SOUND_HOLIC",
    "Liz Triangle": "Liz_Triangle",
}


@dataclass
class DiagnosticResult:
    """Container for diagnostic results."""
    circle: str
    total_files: int
    checked_files: int
    corrupted: list
    short_files: list  # < 30s
    long_files: list   # > 10min
    variance: float
    spread: float


def diagnostic_1_data_quality(circles: dict[str, str], base_path: Path,
                               check_all: bool = False) -> dict[str, dict]:
    """
    Check data quality: corrupted files, unusual durations.

    Returns dict of {circle_name: {corrupted, short, long, total}}
    """
    import librosa

    print("=" * 60)
    print("DIAGNOSTIC 1: Data Quality Check")
    print("=" * 60)
    print()

    results = {}

    for name, folder in circles.items():
        path = base_path / folder
        if not path.exists():
            print(f"{name}: folder not found, skipping")
            continue

        flacs = list(path.rglob("*.flac"))

        errors = []
        short_files = []
        long_files = []

        check_list = flacs if check_all else flacs[:100]  # Sample for speed

        print(f"{name}: checking {len(check_list)}/{len(flacs)} files...")

        for f in check_list:
            try:
                duration = librosa.get_duration(path=f)
                if duration < 30:
                    short_files.append((f.name, duration))
                elif duration > 600:
                    long_files.append((f.name, duration))
            except Exception as e:
                errors.append((f.name, str(e)[:50]))

        results[name] = {
            "total": len(flacs),
            "checked": len(check_list),
            "corrupted": errors,
            "short": short_files,
            "long": long_files,
        }

        print(f"  Total files: {len(flacs)}")
        print(f"  Corrupted/unreadable: {len(errors)}")
        if errors:
            for fname, err in errors[:3]:
                print(f"    - {fname[:40]}: {err}")
        print(f"  Very short (<30s): {len(short_files)}")
        if short_files:
            for fname, dur in short_files[:3]:
                print(f"    - {fname[:40]}: {dur:.1f}s")
        print(f"  Very long (>10min): {len(long_files)}")
        if long_files:
            for fname, dur in long_files[:3]:
                print(f"    - {fname[:40]}: {dur/60:.1f}min")
        print()

    return results


def diagnostic_2_class_variance(circles: dict[str, str], base_path: Path,
                                 samples_per_class: int = 50) -> dict[str, dict]:
    """
    Compute within-class variance to measure stylistic consistency.

    Higher variance/spread = more internally diverse style.
    """
    print("=" * 60)
    print("DIAGNOSTIC 2: Within-Class Variance")
    print("=" * 60)
    print(f"Sampling {samples_per_class} tracks per circle\n")

    loader = AudioLoader()
    extractor = SpectralFeatureExtractor()

    results = {}

    for name, folder in circles.items():
        path = base_path / folder
        if not path.exists():
            continue

        flacs = list(path.rglob("*.flac"))
        sample_size = min(samples_per_class, len(flacs))
        sampled = random.sample(flacs, sample_size)

        features_list = []
        errors = 0

        print(f"{name}: extracting from {sample_size}/{len(flacs)} files...", end=" ", flush=True)

        for f in sampled:
            try:
                waveform, _ = loader.load(f, duration=60)
                feats = extractor.extract_all(waveform)
                summary = extractor.summarize(feats)
                features_list.append(summary)
            except:
                errors += 1

        if features_list:
            X = np.array(features_list)
            variance = X.var(axis=0).mean()
            centroid = X.mean(axis=0)
            spread = np.mean(np.linalg.norm(X - centroid, axis=1))

            results[name] = {
                "n_samples": len(features_list),
                "n_available": len(flacs),
                "variance": float(variance),
                "spread": float(spread),
                "errors": errors,
            }
            print(f"spread={spread:.1f}, variance={variance:.1f}")
        else:
            print("FAILED")

    print("\n" + "-" * 40)
    print("Ranking by internal diversity (spread):\n")

    sorted_results = sorted(results.items(), key=lambda x: x[1]["spread"], reverse=True)
    for name, r in sorted_results:
        print(f"  {name:20}: spread={r['spread']:.1f} (n={r['n_available']})")

    print("\nInterpretation:")
    print("  High spread = varied styles within circle (harder to learn)")
    print("  Low spread = consistent sound signature (easier to learn)")
    print()

    return results


def diagnostic_3_weighting_effect() -> dict:
    """
    Compare classifier performance with and without class weighting.
    Uses current local dataset (data/raw/).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from src.data.audio import find_audio_files

    print("=" * 60)
    print("DIAGNOSTIC 3: Class Weighting Effect")
    print("=" * 60)
    print()

    if not MAPPING_PATH.exists():
        print("Source track mapping not found. Run match_source_tracks.py first.")
        return {}

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    def get_source_id(rel_path):
        entry = mapping.get(rel_path, {})
        sources = entry.get("source_tracks", [])
        return sources[0]["id"] if sources else None

    loader = AudioLoader()
    extractor = SpectralFeatureExtractor()

    X, y, groups = [], [], []

    print("Loading features from local dataset...")

    for circle_name, dirname in LOCAL_CIRCLES.items():
        circle_dir = DATA_DIR / dirname
        if not circle_dir.exists():
            continue
        for f in find_audio_files(circle_dir):
            rel_path = f"{dirname}/{f.name}"
            src_id = get_source_id(rel_path)
            if src_id is None:
                continue
            try:
                waveform, _ = loader.load(f)
                feats = extractor.extract_all(waveform)
                X.append(extractor.summarize(feats))
                y.append(circle_name)
                groups.append(src_id)
            except:
                pass

    if not X:
        print("No samples loaded. Check data/raw/ directory.")
        return {}

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    gkf = GroupKFold(n_splits=5)

    # Without weighting
    clf_no = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    y_pred_no = cross_val_predict(clf_no, X, y_enc, groups=groups, cv=gkf)

    # With weighting
    clf_yes = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                      class_weight='balanced')
    y_pred_yes = cross_val_predict(clf_yes, X, y_enc, groups=groups, cv=gkf)

    print(f"\nSamples: {len(X)}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    results = {"no_weight": {}, "weighted": {}}

    print(f"{'Circle':<20} {'No Weight':>10} {'Weighted':>10} {'Delta':>10}")
    print("-" * 52)

    for i, cls in enumerate(le.classes_):
        mask = y_enc == i
        acc_no = (y_pred_no[mask] == y_enc[mask]).mean()
        acc_yes = (y_pred_yes[mask] == y_enc[mask]).mean()
        delta = acc_yes - acc_no

        results["no_weight"][cls] = float(acc_no)
        results["weighted"][cls] = float(acc_yes)

        print(f"{cls[:20]:<20} {acc_no:>10.1%} {acc_yes:>10.1%} {delta:>+10.1%}")

    overall_no = accuracy_score(y_enc, y_pred_no)
    overall_yes = accuracy_score(y_enc, y_pred_yes)

    print("-" * 52)
    print(f"{'OVERALL':<20} {overall_no:>10.1%} {overall_yes:>10.1%} {overall_yes - overall_no:>+10.1%}")

    results["overall_no_weight"] = float(overall_no)
    results["overall_weighted"] = float(overall_yes)

    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run dataset diagnostics")
    parser.add_argument("--full", action="store_true", help="Check all files (slower)")
    parser.add_argument("--samples", type=int, default=50, help="Samples per class for variance")
    parser.add_argument("--skip-quality", action="store_true", help="Skip data quality check")
    parser.add_argument("--skip-variance", action="store_true", help="Skip variance analysis")
    parser.add_argument("--skip-weighting", action="store_true", help="Skip weighting simulation")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TOUHOU CLASSIFIER - DATASET DIAGNOSTICS")
    print("=" * 60 + "\n")

    all_results = {}

    if not args.skip_quality:
        all_results["data_quality"] = diagnostic_1_data_quality(
            TLMC_CIRCLES, TLMC_DIR, check_all=args.full
        )

    if not args.skip_variance:
        all_results["class_variance"] = diagnostic_2_class_variance(
            TLMC_CIRCLES, TLMC_DIR, samples_per_class=args.samples
        )

    if not args.skip_weighting:
        all_results["weighting_effect"] = diagnostic_3_weighting_effect()

    # Save results
    output_path = PROJECT_ROOT / "data" / "metadata" / "diagnostic_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
