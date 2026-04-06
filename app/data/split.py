from __future__ import annotations

import random
import sqlite3
from collections import defaultdict


DEFAULT_SPLIT_RATIOS: dict[str, float] = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}


def assign_splits(
    connection: sqlite3.Connection,
    split_ratios: dict[str, float] | None = None,
    seed: int = 7,
) -> dict[str, int]:
    # Assign a single split per group_id so related samples cannot leak across sets.
    ratios = split_ratios or DEFAULT_SPLIT_RATIOS
    _validate_split_ratios(ratios)

    rows = connection.execute(
        """
        SELECT id, group_id, label
        FROM sample_metadata
        WHERE is_included = 1
        ORDER BY id
        """
    ).fetchall()
    if not rows:
        return {"train": 0, "val": 0, "test": 0}

    groups = _collect_groups(rows)
    group_labels = _collect_group_labels(rows)
    assignments = _split_groups_with_label_coverage(groups, group_labels, ratios, seed)

    # Clear stale split values before applying the current assignment pass.
    connection.execute("UPDATE sample_metadata SET split = NULL")
    for split_name, group_ids in assignments.items():
        if not group_ids:
            continue
        placeholders = ",".join("?" for _ in group_ids)
        connection.execute(
            f"""
            UPDATE sample_metadata
            SET split = ?, updated_at = CURRENT_TIMESTAMP
            WHERE is_included = 1
              AND group_id IN ({placeholders})
            """,
            (split_name, *group_ids),
        )

    connection.commit()
    return {
        split_name: sum(len(groups[group_id]) for group_id in group_ids)
        for split_name, group_ids in assignments.items()
    }


def load_split_counts(connection: sqlite3.Connection) -> dict[str, int]:
    # Return the current number of included rows assigned to each split.
    rows = connection.execute(
        """
        SELECT split, COUNT(*) AS sample_count
        FROM sample_metadata
        WHERE is_included = 1
          AND split IS NOT NULL
        GROUP BY split
        """
    ).fetchall()
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        counts[row["split"]] = int(row["sample_count"])
    return counts


def _collect_groups(rows) -> dict[str, list[int]]:
    # Bucket included sample ids under their group boundary for split assignment.
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        group_id = row["group_id"] or "__ungrouped__"
        groups[group_id].append(int(row["id"]))
    return dict(groups)


def _collect_group_labels(rows) -> dict[str, set[str]]:
    # Track the label membership of each group so split assignment can avoid starving classes.
    group_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_id = row["group_id"] or "__ungrouped__"
        group_labels[group_id].add(str(row["label"]))
    return dict(group_labels)


def _split_groups_with_label_coverage(
    groups: dict[str, list[int]],
    group_labels: dict[str, set[str]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    # Prefer per-label group assignment when groups belong to a single label.
    if any(len(labels) != 1 for labels in group_labels.values()):
        return _split_group_ids(sorted(groups), ratios, seed)

    label_groups: dict[str, list[str]] = defaultdict(list)
    for group_id, labels in group_labels.items():
        (label,) = tuple(labels)
        label_groups[label].append(group_id)

    assignments = {"train": [], "val": [], "test": []}
    for label in sorted(label_groups):
        label_assignment = _split_group_ids(sorted(label_groups[label]), ratios, seed)
        for split_name, group_ids in label_assignment.items():
            assignments[split_name].extend(group_ids)
    return assignments


def _split_group_ids(
    group_ids: list[str],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    # Shuffle once and then slice the ordered groups into train/val/test buckets.
    shuffled_group_ids = list(group_ids)
    random.Random(seed).shuffle(shuffled_group_ids)

    total_groups = len(shuffled_group_ids)
    train_cutoff = round(total_groups * ratios["train"])
    val_cutoff = train_cutoff + round(total_groups * ratios["val"])

    if total_groups >= 3:
        train_cutoff = min(max(train_cutoff, 1), total_groups - 2)
        val_cutoff = min(max(val_cutoff, train_cutoff + 1), total_groups - 1)
    else:
        train_cutoff = min(max(train_cutoff, 1), total_groups)
        val_cutoff = min(max(val_cutoff, train_cutoff), total_groups)

    return {
        "train": shuffled_group_ids[:train_cutoff],
        "val": shuffled_group_ids[train_cutoff:val_cutoff],
        "test": shuffled_group_ids[val_cutoff:],
    }


def _validate_split_ratios(ratios: dict[str, float]) -> None:
    # Guard against malformed split configs before mutating the DB.
    expected_keys = {"train", "val", "test"}
    if set(ratios) != expected_keys:
        raise ValueError(f"Split ratios must contain exactly {sorted(expected_keys)}")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
