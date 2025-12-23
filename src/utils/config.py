"""Project configuration and paths."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()


@dataclass
class Config:
    """Project configuration."""
    project_root: Path = field(default_factory=_get_project_root)
    sample_rate: int = 22050
    chunk_duration: float = 5.0
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 50
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

    def __post_init__(self):
        self.project_root = Path(self.project_root)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def db_path(self) -> Path:
        return self.metadata_dir / "touhou-music.db"

    def ensure_dirs(self) -> None:
        for d in [self.raw_dir, self.processed_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Target circles for classification
TARGET_CIRCLES = [
    "IOSYS",              # Electronic, denpa
    "UNDEAD CORPORATION", # Death metal
    "暁Records",          # Rock, vocal (Akatsuki Records)
    "SOUND HOLIC",        # Eurobeat, trance
    "Liz Triangle",       # Acoustic, folk
]
