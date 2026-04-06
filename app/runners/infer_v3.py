from __future__ import annotations

from pathlib import Path

from sample_library_cleaner.core.config import DEFAULT_CONFIG_V3_PATH
from sample_library_cleaner.model.infer import run_sort_inference
from sample_library_cleaner.runners.train_v3 import DEFAULT_MODEL_V3_PATH


TARGET_PATHS = [
    "",
]
TOP_K = 4


def main() -> None:
    # Edit the target paths directly when running v3 sort inference by hand.
    for raw_path in TARGET_PATHS:
        if not raw_path:
            continue
        run_sort_inference(
            target_path=Path(raw_path),
            model_path=DEFAULT_MODEL_V3_PATH,
            config_path=DEFAULT_CONFIG_V3_PATH,
            top_k=TOP_K,
        )


if __name__ == "__main__":
    main()
