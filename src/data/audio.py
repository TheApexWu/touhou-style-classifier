"""Audio loading and preprocessing utilities."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import librosa
import numpy as np


@dataclass
class AudioConfig:
    """Audio loading configuration."""
    sample_rate: int = 22050
    mono: bool = True
    normalize: bool = True
    chunk_duration: float = 5.0
    chunk_overlap: float = 0.0


@dataclass
class AudioChunk:
    """A chunk of audio with metadata."""
    waveform: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    source_path: Optional[str] = None
    chunk_index: int = 0


class AudioLoader:
    """Load and preprocess audio files."""

    SUPPORTED_EXTENSIONS = {".flac", ".mp3", ".wav", ".ogg", ".m4a"}

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        self.config = config or AudioConfig()

    def load(
        self,
        path: Union[str, Path],
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> tuple[np.ndarray, int]:
        """Load audio file, return (waveform, sample_rate)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, sr = librosa.load(
                path,
                sr=self.config.sample_rate,
                mono=self.config.mono,
                duration=duration,
                offset=offset
            )

        if self.config.normalize:
            peak = np.abs(waveform).max()
            if peak > 0:
                waveform = waveform / peak

        return waveform, self.config.sample_rate

    def get_duration(self, path: Union[str, Path]) -> float:
        """Get duration in seconds without loading full file."""
        return librosa.get_duration(path=path)

    def chunk_audio(
        self,
        waveform: np.ndarray,
        chunk_duration: Optional[float] = None,
        overlap: Optional[float] = None,
        source_path: Optional[str] = None
    ) -> list[AudioChunk]:
        """Split audio into fixed-length chunks."""
        chunk_duration = chunk_duration or self.config.chunk_duration
        overlap = overlap or self.config.chunk_overlap

        sr = self.config.sample_rate
        chunk_samples = int(chunk_duration * sr)
        hop_samples = int((chunk_duration - overlap) * sr)

        chunks = []
        for i, start in enumerate(range(0, len(waveform) - chunk_samples + 1, hop_samples)):
            end = start + chunk_samples
            chunks.append(AudioChunk(
                waveform=waveform[start:end],
                sample_rate=sr,
                start_time=start / sr,
                end_time=end / sr,
                source_path=source_path,
                chunk_index=i
            ))
        return chunks

    def iter_chunks(
        self,
        path: Union[str, Path],
        chunk_duration: Optional[float] = None,
        overlap: Optional[float] = None
    ) -> Iterator[AudioChunk]:
        """Iterate over chunks of an audio file."""
        waveform, _ = self.load(path)
        yield from self.chunk_audio(
            waveform, chunk_duration, overlap, source_path=str(path)
        )


class AudioAugmenter:
    """Audio augmentation for training."""

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate

    def time_stretch(self, waveform: np.ndarray, rate: float = 1.0) -> np.ndarray:
        return librosa.effects.time_stretch(waveform, rate=rate)

    def pitch_shift(self, waveform: np.ndarray, n_steps: float = 0.0) -> np.ndarray:
        return librosa.effects.pitch_shift(waveform, sr=self.sample_rate, n_steps=n_steps)

    def add_noise(self, waveform: np.ndarray, level: float = 0.005) -> np.ndarray:
        return waveform + np.random.randn(len(waveform)) * level

    def random_gain(self, waveform: np.ndarray, min_g: float = 0.8, max_g: float = 1.2) -> np.ndarray:
        return waveform * np.random.uniform(min_g, max_g)

    def random_augment(self, waveform: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        result = waveform.copy()
        if np.random.random() < 0.3:
            result = self.time_stretch(result, np.random.uniform(0.95, 1.05))
        if np.random.random() < 0.3:
            result = self.pitch_shift(result, np.random.uniform(-1, 1))
        if np.random.random() < 0.2:
            result = self.add_noise(result, 0.002)
        return self.random_gain(result)


def find_audio_files(directory: Union[str, Path], recursive: bool = True) -> list[Path]:
    """Find all audio files in directory."""
    directory = Path(directory)
    extensions = AudioLoader.SUPPORTED_EXTENSIONS
    pattern = "**/*" if recursive else "*"

    files = []
    for ext in extensions:
        files.extend(directory.glob(f"{pattern}{ext}"))
        files.extend(directory.glob(f"{pattern}{ext.upper()}"))
    return sorted(files)
