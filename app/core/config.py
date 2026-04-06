from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "config_v1.json"
DEFAULT_CONFIG_V2_PATH = CONFIGS_DIR / "config_v2.json"
DEFAULT_CONFIG_V3_PATH = CONFIGS_DIR / "config_v3.json"


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int
    mono: bool
    trim_silence: bool


@dataclass(frozen=True)
class InferenceConfig:
    top_k: int
    confidence_threshold: float


@dataclass(frozen=True)
class AppConfig:
    labels: dict[str, list[str]]
    allowed_extensions: tuple[str, ...]
    duration_limits_sec: dict[str, float]
    exclude_filename_tokens: tuple[str, ...]
    audio: AudioConfig
    windows_sec: dict[str, float]
    inference: InferenceConfig

    @property
    def canonical_labels(self) -> tuple[str, ...]:
        return tuple(self.labels.keys())

    @property
    def label_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for canonical_label, raw_aliases in self.labels.items():
            aliases[canonical_label.casefold()] = canonical_label
            for alias in raw_aliases:
                aliases[alias.casefold()] = canonical_label
        return aliases

    def normalize_label(self, raw_label: str) -> str | None:
        return self.label_aliases.get(raw_label.strip().casefold())

    def get_window_sec(self, label: str) -> float:
        return self.windows_sec.get(label, self.windows_sec["default"])


def _validate_required_keys(data: dict) -> None:
    required_keys = {
        "labels",
        "allowed_extensions",
        "duration_limits_sec",
        "exclude_filename_tokens",
        "audio",
        "windows_sec",
        "inference",
    }
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Config is missing required keys: {sorted(missing)}")


def _validate_labels(labels: dict[str, list[str]], duration_limits: dict[str, float]) -> None:
    missing_limits = set(labels) - set(duration_limits)
    if missing_limits:
        raise ValueError(f"Missing duration limits for labels: {sorted(missing_limits)}")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    # Load a versioned JSON config and normalize the primitive values for runtime use.
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    _validate_required_keys(raw_data)
    _validate_labels(raw_data["labels"], raw_data["duration_limits_sec"])

    audio = raw_data["audio"]
    inference = raw_data["inference"]

    return AppConfig(
        labels=raw_data["labels"],
        allowed_extensions=tuple(ext.casefold() for ext in raw_data["allowed_extensions"]),
        duration_limits_sec=raw_data["duration_limits_sec"],
        exclude_filename_tokens=tuple(
            token.casefold() for token in raw_data["exclude_filename_tokens"]
        ),
        audio=AudioConfig(
            target_sample_rate=int(audio["target_sample_rate"]),
            mono=bool(audio["mono"]),
            trim_silence=bool(audio["trim_silence"]),
        ),
        windows_sec={key: float(value) for key, value in raw_data["windows_sec"].items()},
        inference=InferenceConfig(
            top_k=int(inference["top_k"]),
            confidence_threshold=float(inference["confidence_threshold"]),
        ),
    )
