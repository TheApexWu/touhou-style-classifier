"""
Source Separation Pipeline using Demucs

Separates audio into stems (vocals, drums, bass, other) to enable
vocal-independent feature extraction for better circle classification.

Hypothesis: Liz Triangle vs IOSYS confusion is driven by shared vocal
characteristics. Separating vocals from instrumentals may improve
discrimination.

Requirements:
    pip install demucs torch torchaudio

Usage:
    python scripts/separate_sources.py --dry-run     # Preview without processing
    python scripts/separate_sources.py               # Run separation
    python scripts/separate_sources.py --circle IOSYS  # Process single circle

Output structure:
    data/separated/
    ├── IOSYS/
    │   ├── track1_vocals.wav
    │   ├── track1_instrumental.wav  # drums + bass + other combined
    │   └── ...
    ├── Liz_Triangle/
    │   └── ...
    └── ...
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "separated"

CIRCLES = {
    "IOSYS": "IOSYS",
    "UNDEAD_CORPORATION": "UNDEAD_CORPORATION",
    "Akatsuki_Records": "Akatsuki_Records",
    "SOUND_HOLIC": "SOUND_HOLIC",
    "Liz_Triangle": "Liz_Triangle",
}


def check_demucs_installed() -> bool:
    """Check if demucs is available."""
    try:
        result = subprocess.run(
            ["python3", "-c", "import demucs"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False


def separate_single_file(input_path: Path, output_dir: Path, model: str = "htdemucs") -> dict:
    """
    Separate a single audio file using Demucs.

    Returns dict with status and output paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    stem_name = input_path.stem
    vocals_path = output_dir / f"{stem_name}_vocals.wav"
    instrumental_path = output_dir / f"{stem_name}_instrumental.wav"

    if vocals_path.exists() and instrumental_path.exists():
        return {
            "status": "skipped",
            "reason": "already exists",
            "vocals": vocals_path,
            "instrumental": instrumental_path,
        }

    try:
        # Run demucs
        # Output goes to: output_dir/htdemucs/track_name/{vocals,drums,bass,other}.wav
        cmd = [
            "python3", "-m", "demucs",
            "-n", model,
            "-o", str(output_dir / "temp"),
            str(input_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout per track
        )

        if result.returncode != 0:
            return {
                "status": "error",
                "reason": result.stderr[:200],
            }

        # Demucs outputs to: temp/htdemucs/track_name/
        demucs_output = output_dir / "temp" / model / input_path.stem

        # Move vocals
        vocals_src = demucs_output / "vocals.wav"
        if vocals_src.exists():
            vocals_src.rename(vocals_path)

        # Combine drums + bass + other into instrumental
        drums_src = demucs_output / "drums.wav"
        bass_src = demucs_output / "bass.wav"
        other_src = demucs_output / "other.wav"

        if drums_src.exists() and bass_src.exists() and other_src.exists():
            # Use ffmpeg to mix the three stems
            mix_cmd = [
                "ffmpeg", "-y",
                "-i", str(drums_src),
                "-i", str(bass_src),
                "-i", str(other_src),
                "-filter_complex", "amix=inputs=3:duration=longest",
                str(instrumental_path)
            ]
            subprocess.run(mix_cmd, capture_output=True, timeout=60)

        # Cleanup temp directory
        import shutil
        temp_dir = output_dir / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        return {
            "status": "success",
            "vocals": vocals_path,
            "instrumental": instrumental_path,
        }

    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": "timeout"}
    except Exception as e:
        return {"status": "error", "reason": str(e)[:200]}


def process_circle(circle_dir: str, dry_run: bool = False) -> dict:
    """Process all files in a circle directory."""
    input_dir = DATA_DIR / circle_dir
    output_dir = OUTPUT_DIR / circle_dir

    if not input_dir.exists():
        return {"error": f"Directory not found: {input_dir}"}

    audio_files = list(input_dir.glob("*.flac")) + list(input_dir.glob("*.mp3"))

    results = {
        "total": len(audio_files),
        "success": 0,
        "skipped": 0,
        "errors": [],
    }

    for f in audio_files:
        if dry_run:
            vocals_path = output_dir / f"{f.stem}_vocals.wav"
            if vocals_path.exists():
                results["skipped"] += 1
            else:
                results["success"] += 1  # Would be processed
        else:
            result = separate_single_file(f, output_dir)
            if result["status"] == "success":
                results["success"] += 1
            elif result["status"] == "skipped":
                results["skipped"] += 1
            else:
                results["errors"].append((f.name, result.get("reason", "unknown")))

    return results


def main():
    parser = argparse.ArgumentParser(description="Separate audio sources using Demucs")
    parser.add_argument("--dry-run", action="store_true", help="Preview without processing")
    parser.add_argument("--circle", type=str, help="Process single circle only")
    parser.add_argument("--model", type=str, default="htdemucs",
                        help="Demucs model (htdemucs, htdemucs_ft, mdx_extra)")
    args = parser.parse_args()

    print("=" * 60)
    print("SOURCE SEPARATION PIPELINE")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN - No files will be processed ***\n")

    # Check dependencies
    if not args.dry_run:
        if not check_demucs_installed():
            print("ERROR: Demucs not installed.")
            print("Run: pip install demucs torch torchaudio")
            sys.exit(1)
        print(f"Using model: {args.model}\n")

    # Determine which circles to process
    circles_to_process = CIRCLES
    if args.circle:
        if args.circle in CIRCLES:
            circles_to_process = {args.circle: CIRCLES[args.circle]}
        else:
            print(f"ERROR: Unknown circle '{args.circle}'")
            print(f"Available: {list(CIRCLES.keys())}")
            sys.exit(1)

    # Process each circle
    total_stats = {"total": 0, "success": 0, "skipped": 0, "errors": 0}

    for circle_name, circle_dir in circles_to_process.items():
        print(f"\n{circle_name}")
        print("-" * 40)

        results = process_circle(circle_dir, dry_run=args.dry_run)

        if "error" in results:
            print(f"  ERROR: {results['error']}")
            continue

        print(f"  Total files: {results['total']}")
        print(f"  To process: {results['success']}")
        print(f"  Already done: {results['skipped']}")

        if results["errors"]:
            print(f"  Errors: {len(results['errors'])}")
            for fname, reason in results["errors"][:3]:
                print(f"    - {fname[:30]}: {reason[:40]}")

        total_stats["total"] += results["total"]
        total_stats["success"] += results["success"]
        total_stats["skipped"] += results["skipped"]
        total_stats["errors"] += len(results.get("errors", []))

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_stats['total']} files")
    print(f"  To process: {total_stats['success']}")
    print(f"  Already done: {total_stats['skipped']}")
    print(f"  Errors: {total_stats['errors']}")
    print("=" * 60)

    if args.dry_run:
        print("\n*** Run without --dry-run to execute ***")
        print("\nEstimated time: ~30-60 seconds per track")
        print(f"Total estimate: {total_stats['success'] * 45 // 60} - {total_stats['success'] * 60 // 60} minutes")
    else:
        print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
