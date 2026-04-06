from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from sample_library_cleaner.core.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config


MOVE_LOG_DIR = PROJECT_ROOT / "results"


def run_cleanup(
    packs_root_path: str | Path,
    target_group_roots: dict[str, str],
    manual_folder_aliases: dict[str, str],
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    dry_run: bool = True,
    scan_recursively: bool = False,
    move_log_dir: str | Path = MOVE_LOG_DIR,
) -> tuple[list[dict], Path | None]:
    # Run the folder cleanup workflow and optionally persist a move log.
    packs_root_raw = str(packs_root_path).strip()
    if not packs_root_raw:
        raise ValueError("Set a packs root path before running cleanup.")
    packs_root = Path(packs_root_raw)
    if not packs_root.exists() or not packs_root.is_dir():
        raise ValueError(f"PACKS_ROOT_PATH must be an existing directory: {packs_root}")

    label_aliases = build_label_aliases(config_path, manual_folder_aliases)
    target_roots = build_target_roots(target_group_roots)

    operations = process_pack_root(
        packs_root=packs_root,
        label_aliases=label_aliases,
        target_roots=target_roots,
        dry_run=dry_run,
        scan_recursively=scan_recursively,
    )
    operations.extend(prune_empty_dirs(packs_root, dry_run=dry_run))

    move_log_path = None
    if not dry_run:
        move_log_path = save_move_log(packs_root, operations, move_log_dir=move_log_dir)

    return operations, move_log_path


def build_label_aliases(
    config_path: str | Path,
    manual_folder_aliases: dict[str, str],
) -> dict[str, str]:
    # Merge the project config aliases with cleanup-specific folder aliases.
    config = load_config(config_path)
    aliases: dict[str, str] = {}

    for raw_alias, label in config.label_aliases.items():
        aliases[normalize_text(raw_alias)] = label
    for raw_alias, label in manual_folder_aliases.items():
        aliases[normalize_text(raw_alias)] = label

    return aliases


def build_target_roots(target_group_roots: dict[str, str]) -> dict[str, Path]:
    # Convert the editable destination mapping into validated Path objects.
    target_roots: dict[str, Path] = {}
    for label, raw_path in target_group_roots.items():
        if not raw_path:
            continue
        target_roots[label] = Path(raw_path)
    return target_roots


def process_pack_root(
    packs_root: Path,
    label_aliases: dict[str, str],
    target_roots: dict[str, Path],
    dry_run: bool,
    scan_recursively: bool,
) -> list[dict]:
    # Process each immediate pack folder under the selected root.
    operations: list[dict] = []

    for pack_dir in sorted(child for child in packs_root.iterdir() if child.is_dir()):
        operations.extend(
            process_pack_dir(
                pack_dir=pack_dir,
                label_aliases=label_aliases,
                target_roots=target_roots,
                dry_run=dry_run,
                scan_recursively=scan_recursively,
            )
        )

    return operations


def process_pack_dir(
    pack_dir: Path,
    label_aliases: dict[str, str],
    target_roots: dict[str, Path],
    dry_run: bool,
    scan_recursively: bool,
) -> list[dict]:
    # Rename and move any recognized sample-group folders inside one pack.
    operations: list[dict] = []

    for source_dir in iter_candidate_dirs(pack_dir, scan_recursively):
        label = classify_folder(source_dir.name, label_aliases)
        if label is None:
            operations.append(
                {
                    "action": "skip_unmatched_folder",
                    "pack": pack_dir.name,
                    "source": str(source_dir),
                }
            )
            continue

        target_root = target_roots.get(label)
        if target_root is None:
            operations.append(
                {
                    "action": "skip_missing_target_root",
                    "pack": pack_dir.name,
                    "label": label,
                    "source": str(source_dir),
                }
            )
            continue

        prefixed_name = prefix_folder_name(pack_dir.name, source_dir.name)
        renamed_source = source_dir if source_dir.name == prefixed_name else source_dir.with_name(prefixed_name)
        destination_dir = resolve_destination_path(target_root / prefixed_name)

        if not dry_run and source_dir != renamed_source:
            source_dir.rename(renamed_source)
        active_source = renamed_source if (not dry_run or source_dir == renamed_source) else source_dir

        if not dry_run:
            target_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(active_source), str(destination_dir))

        operations.append(
            {
                "action": "move_folder",
                "pack": pack_dir.name,
                "label": label,
                "source": str(source_dir),
                "renamed_source": str(renamed_source),
                "destination": str(destination_dir),
                "renamed": source_dir.name != prefixed_name,
            }
        )

    return operations


def iter_candidate_dirs(pack_dir: Path, scan_recursively: bool) -> list[Path]:
    # Yield either immediate child folders or nested folders under a pack.
    if not scan_recursively:
        return sorted(child for child in pack_dir.iterdir() if child.is_dir())
    return sorted(child for child in pack_dir.rglob("*") if child.is_dir())


def classify_folder(folder_name: str, label_aliases: dict[str, str]) -> str | None:
    # Match a folder name against the known label aliases.
    normalized_name = normalize_text(folder_name)
    if normalized_name in label_aliases:
        return label_aliases[normalized_name]

    for alias, label in label_aliases.items():
        if normalized_name == alias:
            return label
        if f" {alias} " in f" {normalized_name} ":
            return label

    return None


def prefix_folder_name(pack_name: str, folder_name: str) -> str:
    # Add the pack prefix once so moved folders remain traceable.
    prefix = f"{pack_name} - "
    if folder_name.startswith(prefix):
        return folder_name
    return f"{prefix}{folder_name}"


def resolve_destination_path(destination_path: Path) -> Path:
    # Avoid overwriting when two folders would land on the same destination path.
    if not destination_path.exists():
        return destination_path

    stem = destination_path.name
    index = 1
    while True:
        candidate = destination_path.with_name(f"{stem} ({index})")
        if not candidate.exists():
            return candidate
        index += 1


def normalize_text(value: str) -> str:
    # Normalize folder names so alias matching is tolerant of punctuation and spacing.
    lowered = value.casefold().replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return " ".join(cleaned.split())


def print_summary(operations: list[dict], dry_run: bool) -> None:
    # Print a concise summary so the script is easy to inspect while tinkering.
    action_label = "DRY RUN" if dry_run else "DONE"
    moved_count = sum(1 for item in operations if item["action"] == "move_folder")
    deleted_count = sum(1 for item in operations if item["action"] == "delete_empty_folder")
    skipped_count = sum(
        1 for item in operations if item["action"] not in {"move_folder", "delete_empty_folder"}
    )

    print(
        f"{action_label}: {moved_count} folder(s) matched, "
        f"{deleted_count} empty folder(s) deleted, {skipped_count} skipped"
    )
    for item in operations:
        if item["action"] == "move_folder":
            print(f"[move] {item['source']} -> {item['destination']}")
        elif item["action"] == "delete_empty_folder":
            print(f"[delete-empty] {item['path']}")
        else:
            print(f"[skip] {item['source']} ({item['action']})")


def save_move_log(
    packs_root: Path,
    operations: list[dict],
    move_log_dir: str | Path = MOVE_LOG_DIR,
) -> Path | None:
    # Persist only the completed moves so the run has a clean revert record.
    move_entries = []
    for item in operations:
        if item["action"] != "move_folder":
            continue
        move_entries.append(
            {
                "pack": item["pack"],
                "label": item["label"],
                "source_path": item["source"],
                "renamed_source_path": item["renamed_source"],
                "destination_path": item["destination"],
                "renamed": item["renamed"],
            }
        )

    if not move_entries:
        return None

    move_log_root = Path(move_log_dir)
    move_log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = move_log_root / f"{packs_root.name} - cleanup_moves_{timestamp}.json"

    payload = {
        "packs_root": str(packs_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "move_count": len(move_entries),
        "moves": move_entries,
    }

    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return log_path


def prune_empty_dirs(root_path: Path, dry_run: bool) -> list[dict]:
    # Delete empty directories from the bottom up so nested empties get removed cleanly.
    operations: list[dict] = []
    scheduled_for_deletion: set[Path] = set()
    candidate_dirs = sorted(
        (path for path in root_path.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )

    for candidate_dir in candidate_dirs:
        remaining_children = [
            child for child in candidate_dir.iterdir() if child not in scheduled_for_deletion
        ]
        if remaining_children:
            continue
        if not dry_run:
            candidate_dir.rmdir()
        scheduled_for_deletion.add(candidate_dir)
        operations.append(
            {
                "action": "delete_empty_folder",
                "path": str(candidate_dir),
            }
        )

    return operations
