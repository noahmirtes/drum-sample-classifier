from __future__ import annotations

import csv
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from sample_library_cleaner.core.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    load_config,
)
from sample_library_cleaner.core.filesystem import get_item_paths_recursive
from sample_library_cleaner.core.sample import Sample
from sample_library_cleaner.model.train import (
    DEFAULT_MODEL_PATH,
)


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
LABEL_DISPLAY_NAMES = {
    "808": "808",
    "ambience": "Ambience",
    "bass": "Bass",
    "clap": "Clap",
    "closed_hat": "Closed Hat",
    "cymbal": "Cymbal",
    "kick": "Kick",
    "open_hat": "Open Hat",
    "percussion": "Percussion",
    "rimshot": "Rimshot",
    "snap": "Snap",
    "snare": "Snare",
    "textures": "Textures",
    "tom": "Tom",
    "triangle": "Triangle",
    "vocals": "Vocals",
}


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    # Load the persisted baseline classifier for scoring new samples.
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def predict_sample(
    sample_path: str | Path,
    model,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    top_k: int | None = None,
) -> dict:
    # Score one audio file and return the confidence-gated prediction payload.
    config = load_config(config_path)
    sample = Sample.from_path(sample_path)
    features = extract_sample_features_for_inference(sample, config).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    classes = np.asarray(model.classes_)

    requested_top_k = top_k if top_k is not None else config.inference.top_k
    requested_top_k = max(1, min(int(requested_top_k), len(classes)))
    ranked_indices = np.argsort(probabilities)[::-1][:requested_top_k]

    top_predictions = [
        {
            "label": str(classes[index]),
            "confidence": float(probabilities[index]),
        }
        for index in ranked_indices
    ]

    best_prediction = top_predictions[0]
    threshold = config.inference.confidence_threshold
    auto_label = best_prediction["label"] if best_prediction["confidence"] >= threshold else None

    return {
        "path": str(sample.path),
        "top_predictions": top_predictions,
        "predicted_label": auto_label,
        "best_label": best_prediction["label"],
        "best_confidence": best_prediction["confidence"],
        "threshold": threshold,
        "should_auto_label": auto_label is not None,
    }


def predict_directory(
    directory_path: str | Path,
    model,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    top_k: int | None = None,
) -> list[dict]:
    # Score every allowed audio file under a directory tree.
    config = load_config(config_path)
    sample_paths = _get_predictable_paths(directory_path, config.allowed_extensions)
    return [predict_sample(path, model, config_path=config_path, top_k=top_k) for path in sample_paths]


def save_results_json(results: dict | list[dict], output_path: str | Path) -> None:
    # Save the raw inference payload for later inspection or reuse.
    output_target = Path(output_path)
    output_target.parent.mkdir(parents=True, exist_ok=True)
    with output_target.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def save_results_csv(results: dict | list[dict], output_path: str | Path) -> None:
    # Flatten the ranked predictions into a CSV-friendly row format.
    rows = [_flatten_result_row(result) for result in _normalize_results(results)]
    output_target = Path(output_path)
    output_target.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = _collect_csv_fieldnames(rows)

    with output_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_predictions(
    root_path: str | Path,
    results: list[dict],
) -> list[dict]:
    # Move confident predictions into label folders rooted at the selected pack directory.
    root = Path(root_path)
    updated_results: list[dict] = []

    for result in results:
        enriched_result = dict(result)
        enriched_result["moved"] = False
        enriched_result["destination_path"] = None

        if not enriched_result["should_auto_label"]:
            updated_results.append(enriched_result)
            continue

        source_path = Path(enriched_result["path"])
        destination_dir = root / _format_guess_folder_name(root.name, enriched_result["predicted_label"])
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination_path = _resolve_destination_path(destination_dir / source_path.name)
        shutil.move(str(source_path), str(destination_path))

        enriched_result["moved"] = True
        enriched_result["destination_path"] = str(destination_path)
        enriched_result["path"] = str(destination_path)
        updated_results.append(enriched_result)

    return updated_results


def run_inference(
    target_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    top_k: int | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict | list[dict]:
    # Run inference and write paired JSON/CSV outputs for the target file or directory.
    model = load_model(model_path)
    target = Path(target_path)
    if target.is_dir():
        results = predict_directory(target, model, config_path=config_path, top_k=top_k)
    else:
        results = predict_sample(target, model, config_path=config_path, top_k=top_k)

    output_stem = target.name or "results"
    output_root = Path(results_dir)
    save_results_json(results, output_root / f"{output_stem}.json")
    save_results_csv(results, output_root / f"{output_stem}.csv")
    return results


def run_sort_inference(
    target_path: str | Path,
    model_path: str | Path,
    config_path: str | Path,
    top_k: int | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> list[dict]:
    # Run directory inference, sort confident files, and export the move summary.
    target = Path(target_path)
    if not target.is_dir():
        raise ValueError("run_sort_inference requires a directory target")

    model = load_model(model_path)
    results = predict_directory(target, model, config_path=config_path, top_k=top_k)
    sorted_results = sort_predictions(target, results)

    output_stem = f"{target.name} - sorted"
    output_root = Path(results_dir)
    save_results_json(sorted_results, output_root / f"{output_stem}.json")
    save_results_csv(sorted_results, output_root / f"{output_stem}.csv")
    return sorted_results


def extract_sample_features_for_inference(sample: Sample, config):
    # Import lazily to keep the inference module focused on orchestration.
    from sample_library_cleaner.model.features import extract_sample_features

    return extract_sample_features(sample, config)


def _normalize_results(results: dict | list[dict]) -> list[dict]:
    # Treat single-sample and directory inference outputs the same for export.
    if isinstance(results, list):
        return results
    return [results]


def _flatten_result_row(result: dict) -> dict:
    # Expand top-k predictions into separate CSV columns.
    row = {
        "path": result["path"],
        "predicted_label": result["predicted_label"],
        "best_label": result["best_label"],
        "best_confidence": result["best_confidence"],
        "threshold": result["threshold"],
        "should_auto_label": result["should_auto_label"],
        "moved": result.get("moved"),
        "destination_path": result.get("destination_path"),
    }

    for index, prediction in enumerate(result["top_predictions"], start=1):
        row[f"top_{index}_label"] = prediction["label"]
        row[f"top_{index}_confidence"] = prediction["confidence"]

    return row


def _get_predictable_paths(
    directory_path: str | Path,
    allowed_extensions: tuple[str, ...],
) -> list[str]:
    # Skip files that already live inside generated guess folders for this root.
    root = Path(directory_path)
    sample_paths = get_item_paths_recursive(root, allowed_extensions)
    sorted_prefix = f"{root.name} - "
    predictable_paths = []

    for path in sample_paths:
        file_path = Path(path)
        if any(part.startswith(sorted_prefix) for part in file_path.relative_to(root).parts[:-1]):
            continue
        predictable_paths.append(path)

    return predictable_paths


def _collect_csv_fieldnames(rows: list[dict]) -> list[str]:
    # Keep a stable CSV column order while allowing optional move fields.
    base_fieldnames = [
        "path",
        "predicted_label",
        "best_label",
        "best_confidence",
        "threshold",
        "should_auto_label",
        "moved",
        "destination_path",
    ]
    extra_fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key in base_fieldnames or key in extra_fieldnames:
                continue
            extra_fieldnames.append(key)

    return base_fieldnames + extra_fieldnames


def _format_guess_folder_name(root_name: str, predicted_label: str | None) -> str:
    # Build human-readable destination folders like "Pack Name - Kick".
    if predicted_label is None:
        raise ValueError("predicted_label is required for folder naming")
    display_label = LABEL_DISPLAY_NAMES.get(predicted_label, predicted_label.replace("_", " ").title())
    return f"{display_label}"
    #return f"{root_name} - {display_label}" # commented this out so its just the inst name


def _resolve_destination_path(destination_path: Path) -> Path:
    # Avoid overwriting files when multiple packs contain the same filename.
    if not destination_path.exists():
        return destination_path

    stem = destination_path.stem
    suffix = destination_path.suffix
    index = 1
    while True:
        candidate = destination_path.with_name(f"{stem} ({index}){suffix}")
        if not candidate.exists():
            return candidate
        index += 1
