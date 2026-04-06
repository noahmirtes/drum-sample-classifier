from __future__ import annotations

import json

from sample_library_cleaner.core.config import DEFAULT_CONFIG_V2_PATH
from sample_library_cleaner.core.db import DEFAULT_DB_PATH, get_connection
from sample_library_cleaner.model.train import DEFAULT_ARTIFACT_DIR, train_baseline


DEFAULT_MODEL_V2_PATH = DEFAULT_ARTIFACT_DIR / "baseline_model_v2_percussion.pkl"
DEFAULT_METRICS_V2_PATH = DEFAULT_ARTIFACT_DIR / "baseline_metrics_v2_percussion.json"


def main() -> None:
    # Train the percussion-enabled v2 baseline without overwriting v1 artifacts.
    connection = get_connection(DEFAULT_DB_PATH)
    try:
        metrics = train_baseline(
            connection,
            config_path=DEFAULT_CONFIG_V2_PATH,
            model_path=DEFAULT_MODEL_V2_PATH,
            metrics_path=DEFAULT_METRICS_V2_PATH,
        )
    finally:
        connection.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
