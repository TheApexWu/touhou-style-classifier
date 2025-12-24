"""Match audio files to their ZUN source tracks via TLMC paths and database."""

import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TLMC_DIR = Path("/Users/amadeuswoo/Downloads/TLMC temp ver")
DB_PATH = PROJECT_ROOT / "data" / "metadata" / "touhou-music.db"
OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata" / "source_track_mapping.json"

CIRCLE_MAPPING = {
    "IOSYS": "[IOSYS] イオシス",
    "UNDEAD_CORPORATION": "[UNDEAD CORPORATION]",
    "Akatsuki_Records": "[暁Records]",
    "SOUND_HOLIC": "[SOUND HOLIC]",
    "Liz_Triangle": "[Liz triangle] りすとら",
}

CIRCLE_DB_NAMES = {
    "IOSYS": "IOSYS",
    "UNDEAD_CORPORATION": "UNDEAD CORPORATION",
    "Akatsuki_Records": "暁Records",
    "SOUND_HOLIC": "SOUND HOLIC",
    "Liz_Triangle": "Liz Triangle",
}


def find_file_in_tlmc(filename: str, circle_dir: str) -> tuple[Path, str] | None:
    """Find a file in TLMC and return (full_path, album_folder_name)."""
    tlmc_circle = TLMC_DIR / circle_dir
    if not tlmc_circle.exists():
        return None

    for album_dir in tlmc_circle.iterdir():
        if not album_dir.is_dir():
            continue
        for f in album_dir.iterdir():
            if f.name == filename:
                return f, album_dir.name
    return None


def extract_track_number(filename: str) -> int | None:
    """Extract track number from filename."""
    patterns = [
        r"^\((\d+)\)",      # (01) ...
        r"^(\d+)\.",        # 01. ...
        r"^(\d+)\s",        # 01 ...
    ]
    for pat in patterns:
        m = re.match(pat, filename)
        if m:
            return int(m.group(1))
    return None


def clean_track_name(filename: str) -> str:
    """Extract clean track name from filename."""
    name = Path(filename).stem
    # Remove track number prefixes
    name = re.sub(r"^\(\d+\)\s*", "", name)
    name = re.sub(r"^\d+\.\s*", "", name)
    name = re.sub(r"^\d+\s+", "", name)
    # Remove artist prefixes like [lily-an] or "lily-an -"
    name = re.sub(r"^\[.*?\]\s*", "", name)
    name = re.sub(r"^[\w\-]+\s*[-—–]\s*", "", name)
    return name.strip()


def fuzzy_match(s1: str, s2: str) -> float:
    """Return similarity ratio between two strings."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def query_db(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    """Execute query and return list of dicts."""
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(sql, params)
    return [dict(row) for row in cursor.fetchall()]


def find_album_in_db(conn: sqlite3.Connection, album_folder: str) -> dict | None:
    """Find album in database by matching tlmc_path."""
    # The tlmc_path in DB might have the circle prefix too
    results = query_db(conn, """
        SELECT id, album_name, tlmc_path
        FROM albums_index
        WHERE tlmc_path LIKE ?
    """, (f"%{album_folder}%",))

    if results:
        return results[0]
    return None


def find_track_by_name_and_circle(conn: sqlite3.Connection, track_name: str,
                                   circle_name: str) -> dict | None:
    """Find track by fuzzy name match within a circle."""
    # Search for tracks by this circle
    tracks = query_db(conn, """
        SELECT t.id, t.name, t.track_number, a.album_name
        FROM tracks t
        JOIN release_circle_index r ON t.release_circle_id = r.id
        LEFT JOIN albums_index a ON t.album_id = a.id
        WHERE r.name = ?
    """, (circle_name,))

    if not tracks:
        return None

    # Fuzzy match
    best_match = None
    best_score = 0.0
    for t in tracks:
        score = fuzzy_match(track_name, t["name"])
        if score > best_score and score > 0.6:
            best_score = score
            best_match = t

    return best_match


def find_track_in_album(conn: sqlite3.Connection, album_id: int,
                        track_num: int | None, track_name: str) -> dict | None:
    """Find track in album by track number or fuzzy name match."""
    tracks = query_db(conn, """
        SELECT id, name, track_number
        FROM tracks
        WHERE album_id = ?
    """, (album_id,))

    if not tracks:
        return None

    # Try exact track number match first
    if track_num:
        for t in tracks:
            if t["track_number"] == track_num:
                return t

    # Fall back to fuzzy name matching
    best_match = None
    best_score = 0.0
    for t in tracks:
        score = fuzzy_match(track_name, t["name"])
        if score > best_score and score > 0.5:
            best_score = score
            best_match = t

    return best_match


def get_source_tracks(conn: sqlite3.Connection, track_id: int) -> list[dict]:
    """Get source tracks for a given arrangement."""
    return query_db(conn, """
        SELECT s.id, s.name
        FROM source_tracks s
        JOIN track_vs_source_index tvs ON s.id = tvs.source_track_id
        WHERE tvs.track_id = ?
    """, (track_id,))


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)

    results = {}
    stats = {"matched": 0, "unmatched": 0, "no_source": 0}

    print("Matching audio files to source tracks...")
    print("=" * 60)

    for circle_dir in DATA_DIR.iterdir():
        if not circle_dir.is_dir():
            continue

        circle_name = circle_dir.name
        tlmc_circle_dir = CIRCLE_MAPPING.get(circle_name)
        db_circle_name = CIRCLE_DB_NAMES.get(circle_name)

        if not tlmc_circle_dir:
            print(f"\n{circle_name}: no TLMC mapping, skipping")
            continue

        print(f"\n{circle_name}:")

        for audio_file in sorted(circle_dir.iterdir()):
            if audio_file.suffix.lower() not in {".flac", ".mp3", ".wav"}:
                continue

            filename = audio_file.name
            relative_path = f"{circle_name}/{filename}"

            # Find in TLMC
            tlmc_result = find_file_in_tlmc(filename, tlmc_circle_dir)
            if not tlmc_result:
                print(f"  ✗ {filename[:40]}: not found in TLMC")
                stats["unmatched"] += 1
                results[relative_path] = {"error": "not found in TLMC"}
                continue

            _, album_folder = tlmc_result

            # Find album in DB
            track_num = extract_track_number(filename)
            track_name = clean_track_name(filename)
            track = None
            album = find_album_in_db(conn, album_folder)

            if album:
                # Find track within album
                track = find_track_in_album(conn, album["id"], track_num, track_name)

            # Fallback: search by track name + circle directly
            if not track:
                track = find_track_by_name_and_circle(conn, track_name, db_circle_name)
                if track:
                    album = {"album_name": track.get("album_name", "unknown")}

            if not track:
                print(f"  ✗ {filename[:40]}: track not found")
                stats["unmatched"] += 1
                results[relative_path] = {
                    "album_folder": album_folder,
                    "error": "track not in database"
                }
                continue

            # Get source tracks
            sources = get_source_tracks(conn, track["id"])
            album_name = album["album_name"] if album else "unknown"

            if not sources:
                print(f"  ~ {filename[:40]}: no source track linked")
                stats["no_source"] += 1
                results[relative_path] = {
                    "album_folder": album_folder,
                    "album_name": album_name,
                    "track_id": track["id"],
                    "track_name": track["name"],
                    "source_tracks": []
                }
                continue

            # Success!
            source_names = [s["name"] for s in sources]
            print(f"  ✓ {filename[:40]}: {source_names[0][:20]}")
            stats["matched"] += 1

            results[relative_path] = {
                "album_folder": album_folder,
                "album_name": album_name,
                "track_id": track["id"],
                "track_name": track["name"],
                "source_tracks": sources
            }

    conn.close()

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Matched:    {stats['matched']}")
    print(f"No source:  {stats['no_source']} (track found but no ZUN original linked)")
    print(f"Unmatched:  {stats['unmatched']}")
    print(f"\nSaved to: {OUTPUT_PATH}")

    # Summary of unique source tracks
    unique_sources = set()
    for entry in results.values():
        if "source_tracks" in entry:
            for s in entry["source_tracks"]:
                unique_sources.add(s["name"])

    print(f"\nUnique ZUN source tracks: {len(unique_sources)}")
    if unique_sources:
        print("Examples:", list(unique_sources)[:5])


if __name__ == "__main__":
    main()
