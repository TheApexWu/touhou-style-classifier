# Touhou Music Database Schema

Source: https://github.com/solaasan/Touhou-Music-Database

## Entity Relationship

```
┌─────────────────────────┐      ┌──────────────────────┐
│ tracks                  │      │ albums_index         │
├─────────────────────────┤      ├──────────────────────┤
│ id                      │──┐   │ id                   │
│ name                    │  │   │ album_name           │
│ track_number            │  └──>│ tlmc_path  ←── file location
│ album_id ───────────────┼─────>│ genre                │
│ release_circle_id ──────┼──┐   │ disc_number          │
│ songtrack_artist_id     │  │   │ url_links            │
└─────────────────────────┘  │   └──────────────────────┘
                             │
                             │   ┌──────────────────────┐
                             └──>│ release_circle_index │
                                 ├──────────────────────┤
                                 │ id                   │
                                 │ name  ←── circle name
                                 └──────────────────────┘

┌─────────────────────────┐      ┌──────────────────────┐
│ track_vs_source_index   │      │ source_tracks        │
├─────────────────────────┤      ├──────────────────────┤
│ track_id ───────────────┼─┐    │ id                   │
│ source_track_id ────────┼─┼───>│ name  ←── ZUN original
└─────────────────────────┘ │    └──────────────────────┘
                            │
                            └── links arrangement → original composition

┌─────────────────────────┐
│ songtrack_artist_index  │
├─────────────────────────┤
│ id                      │
│ name  ←── individual artist/vocalist
└─────────────────────────┘
```

## Table Definitions

```sql
CREATE TABLE albums_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    album_name TEXT NOT NULL,
    url_links TEXT,
    tlmc_path TEXT,
    genre TEXT,
    disc_number INTEGER
);

CREATE TABLE release_circle_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
);

CREATE TABLE songtrack_artist_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
);

CREATE TABLE source_tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
);

CREATE TABLE tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    track_number INTEGER,
    album_id INTEGER,
    release_circle_id INTEGER,
    songtrack_artist_id INTEGER,
    FOREIGN KEY (album_id) REFERENCES albums_index (id),
    FOREIGN KEY (release_circle_id) REFERENCES release_circle_index (id),
    FOREIGN KEY (songtrack_artist_id) REFERENCES songtrack_artist_index (id)
);

CREATE TABLE track_vs_source_index (
    track_id INTEGER,
    source_track_id INTEGER,
    PRIMARY KEY (track_id, source_track_id),
    FOREIGN KEY (track_id) REFERENCES tracks (id),
    FOREIGN KEY (source_track_id) REFERENCES source_tracks (id)
);
```

## Key Statistics

| Metric | Count |
|--------|-------|
| Total tracks | 163,667 |
| Albums | 19,000 |
| Circles | 3,834 |
| ZUN originals | 1,528 |
| Source mapping coverage | 99.9% |

## Most Arranged ZUN Compositions

| Original | Arrangements |
|----------|-------------|
| U.N.オーエンは彼女なのか？ | 4,217 |
| 亡き王女の為のセプテット | 3,722 |
| 恋色マスタースパーク | 3,099 |
| ネクロファンタジア | 2,653 |
| おてんば恋娘 | 2,662 |

## TLMC Path Format

Paths are relative to TLMC root:
```
[Circle Name]/YYYY.MM.DD [Catalog#] Album Name [Event]/
```

Example:
```
[IOSYS] イオシス/2014.11.24 [IO-0275] Across Phantasm [秋]/
```

## Useful Queries

### Get tracks for a circle
```sql
SELECT t.name, a.album_name, a.tlmc_path
FROM tracks t
JOIN albums_index a ON t.album_id = a.id
JOIN release_circle_index r ON t.release_circle_id = r.id
WHERE r.name = 'IOSYS';
```

### Get source track for an arrangement
```sql
SELECT t.name as arrangement, s.name as original
FROM tracks t
JOIN track_vs_source_index tvs ON t.id = tvs.track_id
JOIN source_tracks s ON tvs.source_track_id = s.id
WHERE t.id = 12345;
```

### Circle track counts
```sql
SELECT r.name, COUNT(*) as tracks
FROM tracks t
JOIN release_circle_index r ON t.release_circle_id = r.id
GROUP BY r.id
ORDER BY tracks DESC;
```

## Target Circles for Classification

| Circle | DB Name | Tracks | Genre |
|--------|---------|--------|-------|
| IOSYS | IOSYS | 1,550 | Electronic, denpa |
| SOUND HOLIC | SOUND HOLIC | 1,271 | Eurobeat, trance |
| Akatsuki Records | 暁Records | 449 | Rock, vocal |
| UNDEAD CORPORATION | UNDEAD CORPORATION | 403 | Death metal |
| Liz Triangle | Liz Triangle | 238 | Acoustic, folk |
