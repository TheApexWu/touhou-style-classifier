"""SQLite utilities for querying the Touhou Music Database."""

import sqlite3
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


class TouhouMusicDB:
    """Interface to the Touhou Music Database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self._schema_cache: dict[str, list[str]] = {}
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            for (table,) in tables:
                cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
                self._schema_cache[table] = [c[1] for c in cols]

    @property
    def tables(self) -> list[str]:
        return list(self._schema_cache.keys())

    def schema(self) -> dict[str, list[str]]:
        return self._schema_cache.copy()

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_all_circles(self) -> pd.DataFrame:
        try:
            return self.query("SELECT * FROM circle ORDER BY name")
        except Exception:
            return self.query(
                "SELECT DISTINCT circle_id, circle_name as name FROM release_circle"
            )

    def get_circle_stats(self, min_tracks: int = 0) -> pd.DataFrame:
        sql = """
            SELECT c.id as circle_id, c.name,
                   COUNT(DISTINCT t.id) as track_count,
                   COUNT(DISTINCT a.id) as album_count
            FROM circle c
            LEFT JOIN album_circle ac ON c.id = ac.circle_id
            LEFT JOIN album a ON ac.album_id = a.id
            LEFT JOIN track t ON a.id = t.album_id
            GROUP BY c.id, c.name
            HAVING track_count >= ?
            ORDER BY track_count DESC
        """
        try:
            return self.query(sql, (min_tracks,))
        except Exception as e:
            print(f"Query failed: {e}. Tables: {self.tables}")
            return pd.DataFrame()

    def get_tracks_by_circle(self, circle_name: str) -> pd.DataFrame:
        sql = """
            SELECT t.id as track_id, t.name as title,
                   a.id as album_id, a.name as album_title,
                   c.name as circle_name
            FROM track t
            JOIN album a ON t.album_id = a.id
            JOIN album_circle ac ON a.id = ac.album_id
            JOIN circle c ON ac.circle_id = c.id
            WHERE c.name = ?
            ORDER BY a.id, t.id
        """
        try:
            return self.query(sql, (circle_name,))
        except Exception as e:
            print(f"Query failed: {e}")
            return pd.DataFrame()

    def get_source_mappings(self) -> pd.DataFrame:
        """Get mappings from arrangements to original ZUN compositions."""
        sql = """
            SELECT t.id as track_id, t.name as track_title,
                   st.name as source_title, st.id as source_id
            FROM track t
            LEFT JOIN track_source ts ON t.id = ts.track_id
            LEFT JOIN source st ON ts.source_id = st.id
            WHERE st.id IS NOT NULL
        """
        try:
            return self.query(sql)
        except Exception as e:
            print(f"Query failed: {e}")
            return pd.DataFrame()

    def get_tracks_for_training(
        self,
        circles: Optional[list[str]] = None,
        min_tracks: int = 50
    ) -> pd.DataFrame:
        """Get tracks filtered by circle names or minimum track count."""
        if circles:
            placeholders = ",".join("?" * len(circles))
            sql = f"""
                SELECT t.id as track_id, t.name as title,
                       a.id as album_id, a.name as album_title,
                       c.id as circle_id, c.name as circle_name
                FROM track t
                JOIN album a ON t.album_id = a.id
                JOIN album_circle ac ON a.id = ac.album_id
                JOIN circle c ON ac.circle_id = c.id
                WHERE c.name IN ({placeholders})
                ORDER BY c.name, a.id, t.id
            """
            return self.query(sql, tuple(circles))
        else:
            sql = """
                WITH counts AS (
                    SELECT c.id as cid, COUNT(DISTINCT t.id) as cnt
                    FROM circle c
                    JOIN album_circle ac ON c.id = ac.circle_id
                    JOIN album a ON ac.album_id = a.id
                    JOIN track t ON a.id = t.album_id
                    GROUP BY c.id HAVING cnt >= ?
                )
                SELECT t.id as track_id, t.name as title,
                       a.id as album_id, a.name as album_title,
                       c.id as circle_id, c.name as circle_name
                FROM track t
                JOIN album a ON t.album_id = a.id
                JOIN album_circle ac ON a.id = ac.album_id
                JOIN circle c ON ac.circle_id = c.id
                JOIN counts ON c.id = counts.cid
                ORDER BY c.name, a.id, t.id
            """
            return self.query(sql, (min_tracks,))

    def search_circles(self, query: str) -> pd.DataFrame:
        return self.query(
            "SELECT * FROM circle WHERE name LIKE ? ORDER BY name",
            (f"%{query}%",)
        )


def download_database(target_path: str | Path) -> Path:
    """Download the Touhou Music Database from GitHub."""
    url = "https://github.com/solaasan/Touhou-Music-Database/raw/main/touhou-music.db"
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to {target}...")
    urllib.request.urlretrieve(url, target)
    print(f"Done ({target.stat().st_size / 1024 / 1024:.1f} MB)")
    return target


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python database.py <path_to_db>")
        print("       python database.py download [target_path]")
        sys.exit(1)

    if sys.argv[1] == "download":
        target = sys.argv[2] if len(sys.argv) > 2 else "data/metadata/touhou-music.db"
        download_database(target)
    else:
        db = TouhouMusicDB(sys.argv[1])
        print(f"Tables: {db.tables}")
        for table, cols in db.schema().items():
            print(f"  {table}: {cols}")
