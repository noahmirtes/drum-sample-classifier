from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_split(
    classifier,
    X: np.ndarray,
    y_true: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    top_k: int,
) -> dict:
    # Compute accuracy, confidence-gated coverage, and a confusion matrix for one split.
    if X.size == 0 or y_true.size == 0:
        return {
            "sample_count": 0,
            "accuracy": None,
            "top_k_accuracy": None,
            "coverage_at_threshold": None,
            "precision_at_threshold": None,
            "confusion_matrix": [],
            "classification_report": {},
        }

    probabilities = classifier.predict_proba(X)
    predicted_indices = np.argmax(probabilities, axis=1)
    predicted_labels = labels[predicted_indices]
    confidences = probabilities[np.arange(len(predicted_indices)), predicted_indices]

    top_k = max(1, min(top_k, probabilities.shape[1]))
    top_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :top_k]
    top_k_matches = [
        y_true[index] in labels[top_indices[index]]
        for index in range(len(y_true))
    ]

    high_confidence_mask = confidences >= threshold
    high_confidence_correct = predicted_labels[high_confidence_mask] == y_true[high_confidence_mask]

    return {
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "top_k_accuracy": float(np.mean(top_k_matches)),
        "coverage_at_threshold": float(np.mean(high_confidence_mask)),
        "precision_at_threshold": (
            float(np.mean(high_confidence_correct))
            if np.any(high_confidence_mask)
            else None
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predicted_labels,
            labels=labels,
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predicted_labels,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
