from __future__ import annotations

import json

from sample_library_cleaner.core.config import DEFAULT_CONFIG_PATH
from sample_library_cleaner.core.db import DEFAULT_DB_PATH, get_connection
from sample_library_cleaner.model.train import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, train_baseline


def main() -> None:
    # Train the baseline v1 model without changing its existing artifact names.
    connection = get_connection(DEFAULT_DB_PATH)
    try:
        metrics = train_baseline(
            connection,
            config_path=DEFAULT_CONFIG_PATH,
            model_path=DEFAULT_MODEL_PATH,
            metrics_path=DEFAULT_METRICS_PATH,
        )
    finally:
        connection.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
