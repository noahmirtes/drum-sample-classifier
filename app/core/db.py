from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import PROJECT_ROOT


DEFAULT_DB_PATH = PROJECT_ROOT / "sample_metadata.db"


TABLE_DEFINITION = """
CREATE TABLE IF NOT EXISTS sample_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    filename TEXT NOT NULL,
    extension TEXT NOT NULL,
    label_raw TEXT,
    label TEXT,
    group_id TEXT,
    duration REAL,
    frames INTEGER,
    sample_rate INTEGER,
    channels INTEGER,
    is_included INTEGER,
    exclusion_reasons TEXT,
    split TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


EXPECTED_COLUMNS: dict[str, str] = {
    "id": "INTEGER",
    "path": "TEXT",
    "filename": "TEXT",
    "extension": "TEXT",
    "label_raw": "TEXT",
    "label": "TEXT",
    "group_id": "TEXT",
    "duration": "REAL",
    "frames": "INTEGER",
    "sample_rate": "INTEGER",
    "channels": "INTEGER",
    "is_included": "INTEGER",
    "exclusion_reasons": "TEXT",
    "split": "TEXT",
    "created_at": "TEXT",
    "updated_at": "TEXT",
}


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    # Open SQLite with row access by column name for downstream tools.
    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    # Create the table if it does not exist, then migrate older shapes in place.
    connection.execute(TABLE_DEFINITION)

    existing_columns = _get_existing_columns(connection)
    if "id" not in existing_columns:
        _migrate_legacy_sample_metadata(connection, existing_columns)
        existing_columns = _get_existing_columns(connection)

    for column_name, column_type in EXPECTED_COLUMNS.items():
        if column_name in existing_columns:
            continue
        connection.execute(f"ALTER TABLE sample_metadata ADD COLUMN {column_name} {column_type}")

    # Repair rows that were indexed before raw and normalized labels were split apart.
    _repair_legacy_label_fields(connection)

    # Add the indexes after the table shape is finalized.
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_sample_metadata_path ON sample_metadata(path)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_sample_metadata_label ON sample_metadata(label)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_sample_metadata_group_id ON sample_metadata(group_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_sample_metadata_is_included "
        "ON sample_metadata(is_included)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_sample_metadata_split ON sample_metadata(split)"
    )
    connection.commit()


def reset_sample_metadata(connection: sqlite3.Connection) -> None:
    # Clear indexed rows when rebuilding the metadata database from scratch.
    connection.execute("DELETE FROM sample_metadata")
    connection.commit()


def _get_existing_columns(connection: sqlite3.Connection) -> set[str]:
    # Introspect the current table shape so older databases can be migrated in place.
    rows = connection.execute("PRAGMA table_info(sample_metadata)").fetchall()
    return {row["name"] for row in rows}


def _migrate_legacy_sample_metadata(
    connection: sqlite3.Connection,
    existing_columns: set[str],
) -> None:
    # Rebuild legacy databases so the table has the full current schema, including `id`.
    connection.execute(
        """
        CREATE TABLE sample_metadata__new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            filename TEXT NOT NULL,
            extension TEXT NOT NULL,
            label_raw TEXT,
            label TEXT,
            group_id TEXT,
            duration REAL,
            frames INTEGER,
            sample_rate INTEGER,
            channels INTEGER,
            is_included INTEGER,
            exclusion_reasons TEXT,
            split TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    filename_expr = "filename" if "filename" in existing_columns else "path"
    label_raw_expr = "label_raw" if "label_raw" in existing_columns else "label"
    label_expr = "label" if "label" in existing_columns and "label_raw" in existing_columns else "NULL"
    extension_expr = """
        CASE
            WHEN lower(path) LIKE '%.aiff' THEN '.aiff'
            WHEN lower(path) LIKE '%.aif' THEN '.aif'
            WHEN lower(path) LIKE '%.wav' THEN '.wav'
            ELSE ''
        END
    """

    # Carry forward legacy rows into the new schema and leave new curation fields unset.
    connection.execute(
        f"""
        INSERT INTO sample_metadata__new (
            path,
            filename,
            extension,
            label_raw,
            label,
            duration,
            frames,
            sample_rate,
            channels
        )
        SELECT
            path,
            {filename_expr},
            {extension_expr},
            {label_raw_expr},
            {label_expr},
            duration,
            frames,
            sample_rate,
            channels
        FROM sample_metadata
        """
    )

    # Swap the migrated table into place once the legacy rows have been copied.
    connection.execute("DROP TABLE sample_metadata")
    connection.execute("ALTER TABLE sample_metadata__new RENAME TO sample_metadata")


def _repair_legacy_label_fields(connection: sqlite3.Connection) -> None:
    # Move legacy folder labels into `label_raw` and clear `label` until re-indexed.
    connection.execute(
        """
        UPDATE sample_metadata
        SET
            label_raw = label,
            label = NULL
        WHERE label_raw IS NULL
          AND label IS NOT NULL
        """
    )
