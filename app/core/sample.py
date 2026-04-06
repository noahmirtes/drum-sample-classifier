from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .config import AppConfig


DEFAULT_SILENCE_THRESHOLD = 1e-4


@dataclass
class Sample:
    path: str
    filename: str
    extension: str
    label_raw: str | None = None
    label: str | None = None
    group_id: str | None = None
    duration: float | None = None
    frames: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    excluded: bool = False
    exclusion_reasons: list[str] = field(default_factory=list)
    _audio: np.ndarray | None = field(default=None, init=False, repr=False)
    _loaded_sample_rate: int | None = field(default=None, init=False, repr=False)
    _model_audio_cache: dict[tuple[int, bool], np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        label_raw: str | None = None,
        label: str | None = None,
        group_id: str | None = None,
    ) -> "Sample":
        # Build a sample directly from audio metadata without reading the full waveform.
        sample_path = Path(path)
        info = sf.info(sample_path)
        return cls(
            path=str(sample_path),
            filename=sample_path.name,
            extension=sample_path.suffix.casefold(),
            label_raw=label_raw,
            label=label,
            group_id=group_id,
            duration=float(info.duration),
            frames=int(info.frames),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
        )

    @classmethod
    def from_db_row(cls, row) -> "Sample":
        # Rehydrate a Sample from a SQLite row while keeping audio loading lazy.
        reasons = _parse_exclusion_reasons(row["exclusion_reasons"])
        return cls(
            path=row["path"],
            filename=row["filename"],
            extension=row["extension"],
            label_raw=row["label_raw"],
            label=row["label"],
            group_id=row["group_id"],
            duration=row["duration"],
            frames=row["frames"],
            sample_rate=row["sample_rate"],
            channels=row["channels"],
            excluded=bool(row["is_included"] == 0) if row["is_included"] is not None else False,
            exclusion_reasons=reasons,
        )

    def load_audio(self) -> tuple[np.ndarray, int]:
        # Read the waveform only once, then keep it cached for downstream steps.
        if self._audio is None or self._loaded_sample_rate is None:
            audio, sample_rate = sf.read(self.path, always_2d=False)
            self._audio = np.asarray(audio, dtype=np.float32)
            self._loaded_sample_rate = int(sample_rate)
        return self._audio.copy(), self._loaded_sample_rate

    def get_model_audio(self, config: AppConfig) -> tuple[np.ndarray, int]:
        # Apply the audio prep path: downmix, trim silence, then resample.
        cache_key = (config.audio.target_sample_rate, config.audio.trim_silence)
        if cache_key in self._model_audio_cache:
            return self._model_audio_cache[cache_key].copy(), config.audio.target_sample_rate

        audio, sample_rate = self.load_audio()
        if config.audio.mono:
            audio = self.downmix(audio)
        if config.audio.trim_silence:
            audio = self.trim_silence(audio)
        if sample_rate != config.audio.target_sample_rate:
            audio = self.resample(audio, sample_rate, config.audio.target_sample_rate)
            sample_rate = config.audio.target_sample_rate

        self._model_audio_cache[cache_key] = audio.astype(np.float32, copy=False)
        return self._model_audio_cache[cache_key].copy(), sample_rate

    def get_windowed_audio(self, config: AppConfig) -> tuple[np.ndarray, int]:
        # Pad or crop the prepared signal to the class window used by the model.
        audio, sample_rate = self.get_model_audio(config)
        target_length = int(round(config.get_window_sec(self.label or "default") * sample_rate))
        if target_length <= 0:
            return audio, sample_rate
        if len(audio) > target_length:
            return audio[:target_length], sample_rate
        if len(audio) < target_length:
            padded = np.zeros(target_length, dtype=np.float32)
            padded[: len(audio)] = audio
            return padded, sample_rate
        return audio, sample_rate

    def add_exclusion_reason(self, reason: str) -> None:
        # Collect curation decisions without duplicating the same flag.
        if reason not in self.exclusion_reasons:
            self.exclusion_reasons.append(reason)
        self.excluded = True

    @staticmethod
    def downmix(audio: np.ndarray) -> np.ndarray:
        # Collapse multichannel audio to mono by averaging the channel axis.
        if audio.ndim == 1:
            return audio.astype(np.float32, copy=False)
        return np.mean(audio, axis=1, dtype=np.float32)

    @staticmethod
    def trim_silence(audio: np.ndarray, threshold: float = DEFAULT_SILENCE_THRESHOLD) -> np.ndarray:
        # Remove leading and trailing near-silent regions while preserving internal gaps.
        if audio.ndim != 1:
            raise ValueError("trim_silence expects mono audio")
        if audio.size == 0:
            return audio.astype(np.float32, copy=False)

        active_indices = np.flatnonzero(np.abs(audio) > threshold)
        if active_indices.size == 0:
            return audio.astype(np.float32, copy=False)
        start = int(active_indices[0])
        end = int(active_indices[-1]) + 1
        return audio[start:end].astype(np.float32, copy=False)

    @staticmethod
    def resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        # Resample with a polyphase filter to avoid pulling in heavier preprocessing.
        if source_sr == target_sr:
            return audio.astype(np.float32, copy=False)
        gcd = np.gcd(source_sr, target_sr)
        up = target_sr // gcd
        down = source_sr // gcd
        resampled = signal.resample_poly(audio, up=up, down=down)
        return np.asarray(resampled, dtype=np.float32)


def _parse_exclusion_reasons(raw_value: str | None) -> list[str]:
    # Normalize the DB representation into a list for the Sample object.
    if not raw_value:
        return []
    return [part for part in raw_value.split("|") if part]
