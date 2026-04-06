from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sample_library_cleaner.core.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from sample_library_cleaner.model.features import build_feature_matrix, load_samples_for_split
from sample_library_cleaner.model.metrics import evaluate_split


DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "baseline_model.pkl"
DEFAULT_METRICS_PATH = DEFAULT_ARTIFACT_DIR / "baseline_metrics.json"


def train_baseline(
    connection: sqlite3.Connection,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> dict:
    # Load config and split-specific datasets before fitting the baseline classifier.
    config = load_config(config_path)
    datasets = _load_split_datasets(connection, config)
    _validate_training_data(datasets)

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=7,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    classifier.fit(datasets["train"]["X"], datasets["train"]["y"])

    # Evaluate the fitted model on each available split with shared metrics logic.
    metrics = {
        "classes": classifier.classes_.tolist(),
        "threshold": config.inference.confidence_threshold,
        "splits": {},
    }
    for split_name, split_data in datasets.items():
        metrics["splits"][split_name] = evaluate_split(
            classifier=classifier,
            X=split_data["X"],
            y_true=split_data["y"],
            labels=classifier.classes_,
            threshold=config.inference.confidence_threshold,
            top_k=config.inference.top_k,
        )

    _save_artifacts(
        classifier=classifier,
        model_path=model_path,
        metrics=metrics,
        metrics_path=metrics_path,
    )
    return metrics


def _load_split_datasets(connection: sqlite3.Connection, config) -> dict[str, dict[str, np.ndarray]]:
    # Build feature matrices for each split from the curated rows in SQLite.
    datasets: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("train", "val", "test"):
        samples = load_samples_for_split(connection, split_name)
        features, labels, _ = build_feature_matrix(samples, config)
        datasets[split_name] = {
            "X": features,
            "y": labels,
        }
    return datasets


def _validate_training_data(datasets: dict[str, dict[str, np.ndarray]]) -> None:
    # Catch empty or incompatible splits before model fitting starts.
    train_labels = set(datasets["train"]["y"].tolist())
    if datasets["train"]["X"].size == 0 or not train_labels:
        raise ValueError("Training split is empty. Index, curate, and assign splits before training.")

    for split_name in ("val", "test"):
        split_labels = set(datasets[split_name]["y"].tolist())
        missing_labels = sorted(split_labels - train_labels)
        if missing_labels:
            raise ValueError(
                f"{split_name} split contains labels not present in train: {missing_labels}"
            )


def _save_artifacts(
    classifier,
    model_path: str | Path,
    metrics: dict,
    metrics_path: str | Path,
) -> None:
    # Persist the trained model and metrics summary for later inference and inspection.
    model_target = Path(model_path)
    metrics_target = Path(metrics_path)
    model_target.parent.mkdir(parents=True, exist_ok=True)
    metrics_target.parent.mkdir(parents=True, exist_ok=True)

    with model_target.open("wb") as handle:
        pickle.dump(classifier, handle)
    with metrics_target.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
