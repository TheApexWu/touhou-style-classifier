"""Validate test audio files and show dataset summary."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio import AudioLoader, find_audio_files
from src.utils.config import TARGET_CIRCLES

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

CIRCLE_DIRS = {
    "IOSYS": "IOSYS",
    "UNDEAD CORPORATION": "UNDEAD_CORPORATION",
    "暁Records": "Akatsuki_Records",
    "SOUND HOLIC": "SOUND_HOLIC",
    "Liz Triangle": "Liz_Triangle",
}


def main():
    loader = AudioLoader()

    print("Test Set Summary")
    print("=" * 50)

    total_files = 0
    total_duration = 0

    for circle, dirname in CIRCLE_DIRS.items():
        circle_dir = DATA_DIR / dirname

        if not circle_dir.exists():
            print(f"\n{circle}: directory not found")
            continue

        files = find_audio_files(circle_dir)

        if not files:
            print(f"\n{circle}: no audio files")
            continue

        durations = []
        errors = []

        for f in files:
            try:
                dur = loader.get_duration(f)
                durations.append(dur)
            except Exception as e:
                errors.append((f.name, str(e)))

        total_files += len(files)
        circle_duration = sum(durations)
        total_duration += circle_duration

        print(f"\n{circle}")
        print(f"  Files: {len(files)}")
        print(f"  Duration: {circle_duration/60:.1f} min")
        if durations:
            print(f"  Avg track: {sum(durations)/len(durations):.1f}s")
        if errors:
            print(f"  Errors: {len(errors)}")
            for name, err in errors[:3]:
                print(f"    - {name}: {err}")

    print("\n" + "=" * 50)
    print(f"Total: {total_files} files, {total_duration/60:.1f} min")

    if total_files < 50:
        print("\nRecommendation: aim for 10-20 files per circle (50-100 total)")


if __name__ == "__main__":
    main()
