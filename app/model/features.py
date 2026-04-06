from __future__ import annotations

import sqlite3

import numpy as np
from scipy import fftpack, signal

from sample_library_cleaner.core.config import AppConfig
from sample_library_cleaner.core.sample import Sample


def load_samples_for_split(
    connection: sqlite3.Connection,
    split_name: str | None = None,
) -> list[Sample]:
    # Load included samples from SQLite and optionally filter down to a single split.
    params: list[str] = []
    sql = """
    SELECT
        path,
        filename,
        extension,
        label_raw,
        label,
        group_id,
        duration,
        frames,
        sample_rate,
        channels,
        is_included,
        exclusion_reasons
    FROM sample_metadata
    WHERE is_included = 1
      AND label IS NOT NULL
    """
    if split_name is not None:
        sql += " AND split = ?"
        params.append(split_name)
    sql += " ORDER BY id"

    rows = connection.execute(sql, params).fetchall()
    return [Sample.from_db_row(row) for row in rows]


def build_feature_matrix(
    samples: list[Sample],
    config: AppConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Convert a list of Samples into feature rows, labels, and source paths.
    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []

    for sample in samples:
        feature_rows.append(extract_sample_features(sample, config))
        labels.append(sample.label or "")
        paths.append(sample.path)

    if not feature_rows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=object), []

    features = np.vstack(feature_rows).astype(np.float32, copy=False)
    return features, np.asarray(labels, dtype=object), paths


def extract_sample_features(sample: Sample, config: AppConfig) -> np.ndarray:
    # Summarize the model-ready waveform into a compact baseline feature vector.
    audio, sample_rate = sample.get_windowed_audio(config)
    if audio.size == 0:
        return np.zeros(_feature_width(), dtype=np.float32)

    # Build frame-wise statistics from a lightweight STFT-based representation.
    frames = _frame_audio(audio)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    zero_crossing_rate = np.mean(frames[:, 1:] * frames[:, :-1] < 0, axis=1)

    _, freqs, magnitude = _compute_magnitude_spectrogram(audio, sample_rate)
    spectral_centroid = _spectral_centroid(magnitude, freqs)
    spectral_bandwidth = _spectral_bandwidth(magnitude, freqs, spectral_centroid)
    spectral_rolloff = _spectral_rolloff(magnitude, freqs, roll_percent=0.85)
    spectral_flatness = _spectral_flatness(magnitude)
    band_energy = _band_energy_features(magnitude, band_count=10)
    cepstral = _cepstral_features(band_energy, coeff_count=8)

    features = [
        float(sample.duration or 0.0),
        float(np.max(np.abs(audio))) if audio.size else 0.0,
        *_summary_stats(rms),
        *_summary_stats(zero_crossing_rate),
        *_summary_stats(spectral_centroid),
        *_summary_stats(spectral_bandwidth),
        *_summary_stats(spectral_rolloff),
        *_summary_stats(spectral_flatness),
    ]

    # Reduce coarse spectral bands and cepstral coefficients to compact summary stats.
    for band_values in band_energy:
        features.extend(_summary_stats(band_values))
    for coeff in cepstral:
        features.extend(_summary_stats(coeff))

    return np.asarray(features, dtype=np.float32)


def _summary_stats(values: np.ndarray) -> tuple[float, float]:
    # Reduce a frame-wise feature into stable aggregate statistics.
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _feature_width() -> int:
    # Keep the zero-vector fallback in sync with the engineered feature layout.
    base_feature_count = 2
    descriptor_count = 6 * 2
    band_energy_count = 10 * 2
    cepstral_count = 8 * 2
    return base_feature_count + descriptor_count + band_energy_count + cepstral_count


def _frame_audio(audio: np.ndarray, frame_length: int = 1024, hop_length: int = 512) -> np.ndarray:
    # Slice the waveform into overlapping frames for time-domain statistics.
    if audio.size == 0:
        return np.zeros((1, frame_length), dtype=np.float32)
    if audio.size < frame_length:
        padded = np.zeros(frame_length, dtype=np.float32)
        padded[: audio.size] = audio
        return padded[np.newaxis, :]

    frame_count = 1 + max(0, (audio.size - frame_length) // hop_length)
    shape = (frame_count, frame_length)
    strides = (audio.strides[0] * hop_length, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    return np.array(frames, dtype=np.float32, copy=True)


def _compute_magnitude_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    nperseg: int = 1024,
    noverlap: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Generate a compact magnitude spectrogram used by the spectral descriptors.
    freqs, times, stft = signal.stft(
        audio,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
    )
    magnitude = np.abs(stft).astype(np.float32, copy=False)
    if magnitude.size == 0:
        magnitude = np.zeros((nperseg // 2 + 1, 1), dtype=np.float32)
        freqs = np.linspace(0.0, sample_rate / 2.0, magnitude.shape[0], dtype=np.float32)
        times = np.zeros(1, dtype=np.float32)
    return times, freqs.astype(np.float32, copy=False), magnitude


def _spectral_centroid(magnitude: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    # Measure the center of mass of each magnitude spectrum.
    power = np.maximum(magnitude, 1e-10)
    numerator = np.sum(power * freqs[:, np.newaxis], axis=0)
    denominator = np.sum(power, axis=0)
    return numerator / np.maximum(denominator, 1e-10)


def _spectral_bandwidth(
    magnitude: np.ndarray,
    freqs: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    # Measure how spread out each magnitude spectrum is around its centroid.
    power = np.maximum(magnitude, 1e-10)
    deviation = np.abs(freqs[:, np.newaxis] - centroid[np.newaxis, :])
    numerator = np.sum(power * deviation, axis=0)
    denominator = np.sum(power, axis=0)
    return numerator / np.maximum(denominator, 1e-10)


def _spectral_rolloff(
    magnitude: np.ndarray,
    freqs: np.ndarray,
    roll_percent: float,
) -> np.ndarray:
    # Find the frequency below which a fixed amount of spectral energy accumulates.
    power = np.maximum(magnitude, 1e-10)
    cumulative = np.cumsum(power, axis=0)
    threshold = cumulative[-1, :] * roll_percent
    indices = np.argmax(cumulative >= threshold[np.newaxis, :], axis=0)
    return freqs[indices]


def _spectral_flatness(magnitude: np.ndarray) -> np.ndarray:
    # Compare geometric and arithmetic means to separate tonal vs noisy signals.
    power = np.maximum(magnitude, 1e-10)
    geometric = np.exp(np.mean(np.log(power), axis=0))
    arithmetic = np.mean(power, axis=0)
    return geometric / np.maximum(arithmetic, 1e-10)


def _band_energy_features(magnitude: np.ndarray, band_count: int) -> np.ndarray:
    # Pool the spectrum into coarse frequency bands for a compact energy profile.
    if magnitude.shape[0] < band_count:
        padded = np.zeros((band_count, magnitude.shape[1]), dtype=np.float32)
        padded[: magnitude.shape[0], :] = magnitude
        magnitude = padded

    band_edges = np.linspace(0, magnitude.shape[0], num=band_count + 1, dtype=int)
    bands: list[np.ndarray] = []
    for start, end in zip(band_edges[:-1], band_edges[1:]):
        band_slice = magnitude[start:max(end, start + 1), :]
        band_power = np.log1p(np.mean(np.square(band_slice), axis=0))
        bands.append(np.asarray(band_power, dtype=np.float32))
    return np.vstack(bands)


def _cepstral_features(band_energy: np.ndarray, coeff_count: int) -> np.ndarray:
    # Apply a DCT over coarse band energies to get lightweight cepstral features.
    cepstrum = fftpack.dct(band_energy, type=2, axis=0, norm="ortho")
    return np.asarray(cepstrum[:coeff_count], dtype=np.float32)
