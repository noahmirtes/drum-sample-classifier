from __future__ import annotations

from pathlib import Path
import sqlite3

from sample_library_cleaner.core.config import AppConfig
from sample_library_cleaner.core.filesystem import get_item_paths_recursive
from sample_library_cleaner.core.sample import Sample


def index_sample_roots(
    connection: sqlite3.Connection,
    sample_root_paths: list[str],
    config: AppConfig,
) -> None:
    # Scan the labeled roots, derive curation state, and persist one row per sample.
    sql = """
    INSERT INTO sample_metadata (
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
        exclusion_reasons,
        updated_at
    )
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
    ON CONFLICT(path) DO UPDATE SET
        filename = excluded.filename,
        extension = excluded.extension,
        label_raw = excluded.label_raw,
        label = excluded.label,
        group_id = excluded.group_id,
        duration = excluded.duration,
        frames = excluded.frames,
        sample_rate = excluded.sample_rate,
        channels = excluded.channels,
        is_included = excluded.is_included,
        exclusion_reasons = excluded.exclusion_reasons,
        updated_at = CURRENT_TIMESTAMP
    """

    for sample_root_path in sample_root_paths:
        for sample in scan_sample_root(sample_root_path, config):
            # Persist both the raw indexed metadata and the initial curation verdict.
            connection.execute(sql, sample_to_db_row(sample))

    connection.commit()


def scan_sample_root(sample_root_path: str, config: AppConfig) -> list[Sample]:
    # Build Sample objects from a single labeled root folder.
    sample_root = Path(sample_root_path)
    raw_label = sample_root.name
    label = config.normalize_label(raw_label)
    sample_paths = get_item_paths_recursive(sample_root, config.allowed_extensions)

    samples: list[Sample] = []
    for sample_path in sample_paths:
        try:
            sample = Sample.from_path(
                sample_path,
                label_raw=raw_label,
                label=label,
                group_id=infer_group_id(sample_path, sample_root),
            )
            apply_exclusion_rules(sample, config)
            samples.append(sample)
        except:
            pass
    return samples


def infer_group_id(sample_path: str | Path, sample_root_path: str | Path) -> str:
    # Use the first folder below the label root as the pack id when available.
    sample_root = Path(sample_root_path)
    relative_parts = Path(sample_path).relative_to(sample_root).parts
    if len(relative_parts) >= 2:
        return relative_parts[0]
    return sample_root.name


def apply_exclusion_rules(sample: Sample, config: AppConfig) -> None:
    # Apply the label, extension, token, and duration gates in a fixed order.
    sample.excluded = False
    sample.exclusion_reasons.clear()

    if sample.extension.casefold() not in config.allowed_extensions:
        sample.add_exclusion_reason("unsupported_extension")

    if sample.label is None:
        sample.add_exclusion_reason("unknown_label")
    else:
        max_duration = config.duration_limits_sec[sample.label]
        if sample.duration is not None and sample.duration > max_duration:
            sample.add_exclusion_reason("duration_limit")

    searchable_text = f"{sample.path} {sample.filename}".casefold()
    for token in config.exclude_filename_tokens:
        if token in searchable_text:
            sample.add_exclusion_reason(f"filename_token:{token}")


def sample_to_db_row(sample: Sample) -> tuple:
    # Flatten a Sample into the storage shape used by SQLite.
    return (
        sample.path,
        sample.filename,
        sample.extension,
        sample.label_raw,
        sample.label,
        sample.group_id,
        sample.duration,
        sample.frames,
        sample.sample_rate,
        sample.channels,
        0 if sample.excluded else 1,
        serialize_exclusion_reasons(sample.exclusion_reasons),
    )


def serialize_exclusion_reasons(reasons: list[str]) -> str | None:
    # Store multiple curation flags in a compact single-column representation.
    if not reasons:
        return None
    return "|".join(reasons)
