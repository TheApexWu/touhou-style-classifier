"""
Visualize spectrograms and features for audio files.

Usage:
    python runtest.py path/to/audio.mp3
    python runtest.py path/to/audio.flac --save output.png
    python runtest.py path/to/folder/  # Process all audio files
"""

import argparse
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.features.spectral import SpectralConfig, SpectralFeatureExtractor


def plot_features(audio_path: Path, save_path: Path = None, duration: float = 30.0):
    """Generate visualization of all spectral features for an audio file."""

    print(f"Loading: {audio_path}")
    waveform, sr = librosa.load(audio_path, sr=22050, duration=duration)
    print(f"  Duration: {len(waveform)/sr:.1f}s, Sample rate: {sr}")

    config = SpectralConfig(sample_rate=sr)
    extractor = SpectralFeatureExtractor(config)
    print(f"  Config: {config.describe()}")

    features = extractor.extract_all(waveform)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"{audio_path.name}", fontsize=12)

    time_axis = np.linspace(0, len(waveform) / sr, features["mel_spectrogram"].shape[1])

    # Mel spectrogram
    ax = axes[0, 0]
    img = ax.imshow(
        features["mel_spectrogram"],
        aspect="auto",
        origin="lower",
        extent=[0, time_axis[-1], 0, config.n_mels],
        cmap="magma"
    )
    ax.set_ylabel("Mel bin")
    ax.set_xlabel("Time (s)")
    ax.set_title("Mel Spectrogram")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")

    # MFCCs
    ax = axes[0, 1]
    mfcc_display = features["mfcc"][:20]  # Just first 20 (no deltas)
    img = ax.imshow(mfcc_display, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_ylabel("MFCC coefficient")
    ax.set_xlabel("Frame")
    ax.set_title("MFCCs")
    plt.colorbar(img, ax=ax)

    # Chroma
    ax = axes[1, 0]
    pitch_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    img = ax.imshow(features["chroma"], aspect="auto", origin="lower", cmap="Blues")
    ax.set_yticks(range(12))
    ax.set_yticklabels(pitch_labels)
    ax.set_xlabel("Frame")
    ax.set_title("Chroma (pitch classes)")
    plt.colorbar(img, ax=ax)

    # Spectral contrast
    ax = axes[1, 1]
    img = ax.imshow(features["spectral_contrast"], aspect="auto", origin="lower", cmap="viridis")
    ax.set_ylabel("Frequency band")
    ax.set_xlabel("Frame")
    ax.set_title("Spectral Contrast")
    plt.colorbar(img, ax=ax)

    # Spectral statistics over time
    ax = axes[2, 0]
    stats = features["spectral_stats"]
    ax.plot(stats["centroid"] / 1000, label="Centroid (kHz)", alpha=0.8)
    ax.plot(stats["rolloff"] / 1000, label="Rolloff (kHz)", alpha=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("Spectral Centroid & Rolloff")
    ax.legend(loc="upper right")

    # Summary stats
    ax = axes[2, 1]
    summary = extractor.summarize(features)
    ax.text(0.1, 0.8, f"Tempo: {features['tempo']:.1f} BPM", transform=ax.transAxes, fontsize=11)
    ax.text(0.1, 0.65, f"Avg Centroid: {stats['centroid'].mean():.0f} Hz", transform=ax.transAxes)
    ax.text(0.1, 0.5, f"Avg Flatness: {stats['flatness'].mean():.4f}", transform=ax.transAxes)
    ax.text(0.1, 0.35, f"Feature vector dim: {len(summary)}", transform=ax.transAxes)
    ax.text(0.1, 0.15, f"Mel spec shape: {features['mel_spectrogram'].shape}", transform=ax.transAxes, fontsize=9)
    ax.axis("off")
    ax.set_title("Summary")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def find_audio_files(path: Path) -> list[Path]:
    """Find audio files in a directory."""
    extensions = {".mp3", ".flac", ".wav", ".ogg", ".m4a"}
    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in extensions)


def main():
    parser = argparse.ArgumentParser(description="Visualize audio spectrograms")
    parser.add_argument("path", type=Path, help="Audio file or directory")
    parser.add_argument("--save", type=Path, help="Save plot to file instead of showing")
    parser.add_argument("--duration", type=float, default=30.0, help="Max duration to analyze (seconds)")
    parser.add_argument("--output-dir", type=Path, help="Output directory for batch processing")
    args = parser.parse_args()

    files = find_audio_files(args.path)
    if not files:
        print(f"No audio files found at: {args.path}")
        return

    print(f"Found {len(files)} audio file(s)")

    for audio_path in files:
        if args.output_dir:
            save_path = args.output_dir / f"{audio_path.stem}_features.png"
            args.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_path = args.save

        try:
            plot_features(audio_path, save_path, args.duration)
        except Exception as e:
            print(f"  Error processing {audio_path}: {e}")


if __name__ == "__main__":
    main()
