from __future__ import annotations

from pathlib import Path

from sample_library_cleaner.core.config import DEFAULT_CONFIG_PATH, load_config
from sample_library_cleaner.core.db import DEFAULT_DB_PATH, ensure_schema, get_connection, reset_sample_metadata
from sample_library_cleaner.data.dataset import index_sample_roots


def rebuild_metadata(
    sample_root_paths: list[str],
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    # Rebuild the metadata database from the provided sample roots and config.
    config = load_config(config_path)
    db = get_connection(db_path)
    ensure_schema(db)
    reset_sample_metadata(db)

    index_sample_roots(db, sample_root_paths, config)
    db.commit()
    db.close()
