"""Spectral feature extraction for music style classification."""

from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np


@dataclass
class SpectralConfig:
    """
    Feature extraction parameters derived from sample rate and
    desired time resolution.
    """
    sample_rate: int = 22050
    frame_duration_ms: float = 23.0  # ~43 fps
    fft_multiplier: int = 4  # 75% overlap
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 8000.0
    n_mfcc: int = 20

    @property
    def hop_length(self) -> int:
        return int(self.sample_rate * self.frame_duration_ms / 1000)

    @property
    def n_fft(self) -> int:
        raw = self.hop_length * self.fft_multiplier
        return int(2 ** np.ceil(np.log2(raw)))

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_length

    def describe(self) -> str:
        return (
            f"sample_rate={self.sample_rate}, hop={self.hop_length}, "
            f"n_fft={self.n_fft}, n_mels={self.n_mels}, "
            f"fps={self.frames_per_second:.1f}"
        )


class SpectralFeatureExtractor:
    """Extract spectral features from audio waveforms."""

    def __init__(self, config: Optional[SpectralConfig] = None) -> None:
        self.config = config or SpectralConfig()

    def mel_spectrogram(
        self,
        waveform: np.ndarray,
        to_db: bool = True,
        normalize: bool = False
    ) -> np.ndarray:
        cfg = self.config
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
        )

        if to_db:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        if normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        return mel_spec

    def mfcc(self, waveform: np.ndarray, include_deltas: bool = True) -> np.ndarray:
        cfg = self.config
        mfccs = librosa.feature.mfcc(
            y=waveform,
            sr=cfg.sample_rate,
            n_mfcc=cfg.n_mfcc,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )

        if include_deltas:
            delta = librosa.feature.delta(mfccs, order=1)
            delta2 = librosa.feature.delta(mfccs, order=2)
            mfccs = np.vstack([mfccs, delta, delta2])

        return mfccs

    def chroma(self, waveform: np.ndarray) -> np.ndarray:
        cfg = self.config
        return librosa.feature.chroma_stft(
            y=waveform,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )

    def spectral_contrast(self, waveform: np.ndarray) -> np.ndarray:
        cfg = self.config
        return librosa.feature.spectral_contrast(
            y=waveform,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )

    def spectral_stats(self, waveform: np.ndarray) -> dict[str, np.ndarray]:
        cfg = self.config
        return {
            "centroid": librosa.feature.spectral_centroid(
                y=waveform, sr=cfg.sample_rate, hop_length=cfg.hop_length
            ).squeeze(),
            "bandwidth": librosa.feature.spectral_bandwidth(
                y=waveform, sr=cfg.sample_rate, hop_length=cfg.hop_length
            ).squeeze(),
            "rolloff": librosa.feature.spectral_rolloff(
                y=waveform, sr=cfg.sample_rate, hop_length=cfg.hop_length
            ).squeeze(),
            "flatness": librosa.feature.spectral_flatness(
                y=waveform, hop_length=cfg.hop_length
            ).squeeze(),
        }

    def tempo(self, waveform: np.ndarray) -> float:
        cfg = self.config
        tempo, _ = librosa.beat.beat_track(
            y=waveform, sr=cfg.sample_rate, hop_length=cfg.hop_length
        )
        return float(tempo) if np.isscalar(tempo) else float(tempo[0])

    def extract_all(self, waveform: np.ndarray) -> dict:
        return {
            "mel_spectrogram": self.mel_spectrogram(waveform),
            "mfcc": self.mfcc(waveform),
            "chroma": self.chroma(waveform),
            "spectral_contrast": self.spectral_contrast(waveform),
            "spectral_stats": self.spectral_stats(waveform),
            "tempo": self.tempo(waveform),
        }

    def summarize(self, features: dict) -> np.ndarray:
        parts = []
        for name, feat in features.items():
            if name == "tempo":
                parts.append([feat])
            elif isinstance(feat, dict):
                for arr in feat.values():
                    parts.append([arr.mean(), arr.std(), arr.min(), arr.max()])
            elif feat.ndim == 2:
                parts.append(feat.mean(axis=1))
                parts.append(feat.std(axis=1))
            elif feat.ndim == 1:
                parts.append([feat.mean(), feat.std(), feat.min(), feat.max()])

        return np.concatenate([np.atleast_1d(p) for p in parts]).astype(np.float32)
