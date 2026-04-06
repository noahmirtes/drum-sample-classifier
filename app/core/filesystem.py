from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf


def get_item_paths_recursive(path: str | Path, target_extensions: tuple[str, ...]) -> list[str]:
    # Walk a directory tree and return files whose suffix matches the allowed set.
    data = []
    normalized_extensions = {extension.casefold() for extension in target_extensions}
    for root, dirs, files in os.walk(path):
        for file in files:
            if Path(file).suffix.casefold() not in normalized_extensions:
                continue
            # Build the absolute path for each matching file.
            full_path = os.path.join(root, file)
            data.append(full_path)

    return data


def load_audio(path: str | Path):
    # Keep the low-level audio reader available for quick debugging scripts.
    audio, sr = sf.read(path)
    return audio, sr
