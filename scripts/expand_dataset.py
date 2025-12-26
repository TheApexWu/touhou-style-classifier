"""
Expand training dataset from TLMC with smart sampling.

Strategy based on diagnostics:
- UNDEAD CORPORATION: all 61 (consistent, high-quality signal)
- Liz Triangle: all 82 (clean, moderate variance)
- 暁Records: all 284 (good consistency)
- IOSYS: all 331 (diverse but manageable)
- SOUND HOLIC: capped at ~200 (most noisy, sample across albums/dates)

Usage:
    python scripts/expand_dataset.py --dry-run  # Preview without copying
    python scripts/expand_dataset.py            # Execute copy
"""

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

TLMC_DIR = Path("/Users/amadeuswoo/Downloads/TLMC temp ver")
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

CIRCLES = {
    "IOSYS": {
        "tlmc_folder": "[IOSYS] イオシス",
        "local_folder": "IOSYS",
        "cap": None,  # Use all
    },
    "UNDEAD CORPORATION": {
        "tlmc_folder": "[UNDEAD CORPORATION]",
        "local_folder": "UNDEAD_CORPORATION",
        "cap": None,
    },
    "暁Records": {
        "tlmc_folder": "[暁Records]",
        "local_folder": "Akatsuki_Records",
        "cap": None,
    },
    "SOUND HOLIC": {
        "tlmc_folder": "[SOUND HOLIC]",
        "local_folder": "SOUND_HOLIC",
        "cap": 200,  # Cap based on diagnostic findings (high internal variance)
    },
    "Liz Triangle": {
        "tlmc_folder": "[Liz triangle] りすとら",
        "local_folder": "Liz_Triangle",
        "cap": None,
    },
}


def extract_album_date(album_path: Path) -> str:
    """Extract date from album folder name (e.g., '2017.08.11 [SDHC-0070]...')"""
    name = album_path.name
    match = re.match(r"(\d{4}\.\d{2}\.\d{2})", name)
    if match:
        return match.group(1)
    return "unknown"


def extract_album_info(file_path: Path) -> dict:
    """Extract album metadata from file path."""
    # Path structure: TLMC/[Circle]/AlbumFolder/track.flac
    album_folder = file_path.parent
    return {
        "album_name": album_folder.name,
        "album_date": extract_album_date(album_folder),
        "file_name": file_path.name,
        "full_path": file_path,
    }


def diverse_sample(files: list[Path], n: int, seed: int = 42) -> list[Path]:
    """
    Sample n files ensuring diversity across albums and dates.

    Strategy:
    1. Group files by album
    2. Sort albums by date
    3. Round-robin sample from albums (oldest to newest)
    """
    random.seed(seed)

    # Group by album
    albums = defaultdict(list)
    for f in files:
        info = extract_album_info(f)
        albums[info["album_name"]].append(f)

    # Sort albums by date
    sorted_albums = sorted(albums.keys(), key=lambda x: extract_album_date(Path(x)))

    # Shuffle files within each album
    for album in albums:
        random.shuffle(albums[album])

    # Round-robin sample
    selected = []
    album_indices = {album: 0 for album in sorted_albums}

    while len(selected) < n:
        added_this_round = False
        for album in sorted_albums:
            if len(selected) >= n:
                break
            idx = album_indices[album]
            if idx < len(albums[album]):
                selected.append(albums[album][idx])
                album_indices[album] += 1
                added_this_round = True

        if not added_this_round:
            break  # All albums exhausted

    return selected


def get_files_to_copy(circle_name: str, config: dict) -> tuple[list[Path], dict]:
    """Get list of files to copy for a circle, with metadata."""
    tlmc_path = TLMC_DIR / config["tlmc_folder"]

    if not tlmc_path.exists():
        return [], {"error": f"TLMC folder not found: {tlmc_path}"}

    all_flacs = list(tlmc_path.rglob("*.flac"))

    # Filter out single-file album rips (usually > 200MB, entire CD in one file)
    individual_tracks = [f for f in all_flacs if f.stat().st_size < 200 * 1024 * 1024]

    cap = config.get("cap")

    if cap and len(individual_tracks) > cap:
        selected = diverse_sample(individual_tracks, cap)
        sampling_method = f"diverse_sample (cap={cap})"
    else:
        selected = individual_tracks
        sampling_method = "all"

    # Gather album distribution stats
    albums = defaultdict(int)
    for f in selected:
        album = f.parent.name
        albums[album] += 1

    metadata = {
        "total_available": len(all_flacs),
        "individual_tracks": len(individual_tracks),
        "selected": len(selected),
        "sampling_method": sampling_method,
        "albums_represented": len(albums),
        "album_distribution": dict(albums),
    }

    return selected, metadata


def main():
    parser = argparse.ArgumentParser(description="Expand dataset from TLMC")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying")
    parser.add_argument("--clean", action="store_true", help="Remove existing files first")
    args = parser.parse_args()

    print("=" * 60)
    print("DATASET EXPANSION")
    print("=" * 60)
    print()

    if args.dry_run:
        print("*** DRY RUN - No files will be copied ***\n")

    total_files = 0
    all_metadata = {}

    for circle_name, config in CIRCLES.items():
        print(f"\n{circle_name}")
        print("-" * 40)

        files, metadata = get_files_to_copy(circle_name, config)
        all_metadata[circle_name] = metadata

        if "error" in metadata:
            print(f"  ERROR: {metadata['error']}")
            continue

        print(f"  Available: {metadata['total_available']} files")
        print(f"  Individual tracks: {metadata['individual_tracks']}")
        print(f"  Selected: {metadata['selected']} ({metadata['sampling_method']})")
        print(f"  Albums represented: {metadata['albums_represented']}")

        if not args.dry_run:
            dest_dir = DATA_DIR / config["local_folder"]

            if args.clean and dest_dir.exists():
                shutil.rmtree(dest_dir)

            dest_dir.mkdir(parents=True, exist_ok=True)

            copied = 0
            for f in files:
                dest_file = dest_dir / f.name
                if not dest_file.exists():
                    shutil.copy2(f, dest_file)
                    copied += 1

            print(f"  Copied: {copied} new files")

        total_files += metadata["selected"]

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_files} files")
    print("=" * 60)

    # Summary table
    print("\nDataset composition:")
    print(f"{'Circle':<25} {'Selected':>10} {'Method':<20}")
    print("-" * 55)
    for circle_name, meta in all_metadata.items():
        if "error" not in meta:
            print(f"{circle_name:<25} {meta['selected']:>10} {meta['sampling_method']:<20}")

    if args.dry_run:
        print("\n*** Run without --dry-run to execute copy ***")
    else:
        # Save metadata
        import json
        meta_path = PROJECT_ROOT / "data" / "metadata" / "dataset_expansion.json"
        with open(meta_path, "w") as f:
            json.dump(all_metadata, f, indent=2, default=str)
        print(f"\nMetadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
