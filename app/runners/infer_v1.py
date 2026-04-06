from __future__ import annotations

import json
from pathlib import Path

from sample_library_cleaner.core.config import DEFAULT_CONFIG_PATH
from sample_library_cleaner.model.infer import run_inference
from sample_library_cleaner.model.train import DEFAULT_MODEL_PATH


TARGET_PATHS = [
    "",
]
TOP_K = 3


def main() -> None:
    # Edit the target paths directly when running v1 inference by hand.
    for raw_path in TARGET_PATHS:
        if not raw_path:
            continue
        results = run_inference(
            target_path=Path(raw_path),
            model_path=DEFAULT_MODEL_PATH,
            config_path=DEFAULT_CONFIG_PATH,
            top_k=TOP_K,
        )
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
