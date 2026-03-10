"""Evaluation metrics, results saving, and confusion matrix generation."""

import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FIGURES_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute a full suite of classification metrics.

    Returns a dict with: accuracy, precision (per-class + macro),
    recall (per-class + macro), F1 (per-class + macro), AUC-ROC,
    and Cohen's Kappa.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_left": float(precision_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)),
        "precision_right": float(precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_left": float(recall_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)),
        "recall_right": float(recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_left": float(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)),
        "f1_right": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    return metrics


def save_results(
    metrics: dict,
    model_name: str,
    subject_id: int,
    output_dir: Path | None = None,
) -> Path:
    """Save metrics dict to a JSON file in the results directory."""
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name}_subject{subject_id:03d}.json"
    filepath = output_dir / filename

    # Make sure all values are JSON-serializable
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        else:
            serializable[k] = v

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info("Results saved to %s", filepath)
    return filepath


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    subject_id: int,
    output_dir: Path | None = None,
) -> Path:
    """Generate and save a confusion matrix heatmap."""
    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Left", "Right"],
        yticklabels=["Left", "Right"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name} (Subject {subject_id})")

    filepath = output_dir / f"confusion_{model_name}_subject{subject_id:03d}.png"
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Confusion matrix saved to %s", filepath)
    return filepath


def print_metrics_table(metrics: dict, model_name: str, subject_id: int) -> None:
    """Print a formatted summary table of all metrics to console."""
    print(f"\n{'='*55}")
    print(f"  {model_name} — Subject {subject_id} — Evaluation Metrics")
    print(f"{'='*55}")
    print(f"  {'Metric':<25s} {'Value':>10s}")
    print(f"  {'-'*25} {'-'*10}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<25s} {value:>10.4f}")
    print(f"{'='*55}\n")
