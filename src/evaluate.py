"""Evaluation metrics, results saving, and comprehensive report generation."""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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

from src.config import FIGURES_DIR, RESULTS_DIR, SUBJECTS, RANDOM_SEED

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


def save_results_csv(
    all_results: list[dict],
    output_dir: Path | None = None,
) -> Path:
    """Export full metrics for all subjects/models to CSV."""
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / "full_metrics.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(filepath, index=False)
    logger.info("Full metrics CSV saved to %s", filepath)
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


# ---------------------------------------------------------------------------
# 3.3  Comprehensive Evaluation Report
# ---------------------------------------------------------------------------

def compute_aggregate_stats(results_df: pd.DataFrame) -> dict:
    """Compute aggregate statistics across subjects for each model.

    Returns a dict mapping model names to their aggregate stats
    (mean, std, min, max, median accuracy).
    """
    model_cols = {
        "LR_PSD": "lr_psd_accuracy",
        "LR_CSP": "lr_csp_accuracy",
        "EEGNet_Raw": "eegnet_accuracy",
    }

    stats = {}
    for model_name, col in model_cols.items():
        if col not in results_df.columns:
            continue
        accs = results_df[col].dropna()
        if len(accs) == 0:
            continue
        stats[model_name] = {
            "mean": float(accs.mean()),
            "std": float(accs.std()),
            "min": float(accs.min()),
            "max": float(accs.max()),
            "median": float(accs.median()),
            "n_subjects": int(len(accs)),
        }

    return stats


def generate_per_subject_breakdown(results_df: pd.DataFrame) -> str:
    """Generate a text table of per-subject accuracy breakdown."""
    lines = []
    lines.append("Per-Subject Accuracy Breakdown")
    lines.append("=" * 75)

    header_parts = [f"{'Subject':>8s}"]
    model_cols = {
        "LR+PSD": "lr_psd_accuracy",
        "LR+CSP": "lr_csp_accuracy",
        "EEGNet": "eegnet_accuracy",
    }
    available = {k: v for k, v in model_cols.items() if v in results_df.columns}
    for label in available:
        header_parts.append(f"{label:>12s}")
    header_parts.append(f"{'Best':>12s}")
    lines.append("  ".join(header_parts))
    lines.append("-" * 75)

    for _, row in results_df.iterrows():
        parts = [f"{int(row['subject']):>8d}"]
        accs = {}
        for label, col in available.items():
            val = row.get(col, float("nan"))
            parts.append(f"{val:>12.4f}")
            accs[label] = val
        if accs:
            best = max(accs, key=accs.get)
            parts.append(f"{best:>12s}")
        lines.append("  ".join(parts))

    return "\n".join(lines)


def generate_evaluation_report(
    results_df: pd.DataFrame,
    comparison: dict | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Auto-generate a comprehensive text-based evaluation report.

    Includes per-subject breakdown, aggregate statistics, model comparison,
    and statistical test results.
    """
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 75)
    lines.append("  EEG BCI PIPELINE — COMPREHENSIVE EVALUATION REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 75)
    lines.append("")

    # 1. Per-subject breakdown
    lines.append(generate_per_subject_breakdown(results_df))
    lines.append("")

    # 2. Aggregate statistics
    agg_stats = compute_aggregate_stats(results_df)
    lines.append("")
    lines.append("Aggregate Statistics Across Subjects")
    lines.append("=" * 75)
    lines.append(f"  {'Model':<15s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Median':>8s} {'N':>5s}")
    lines.append(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
    for model_name, s in agg_stats.items():
        lines.append(
            f"  {model_name:<15s} {s['mean']:>8.4f} {s['std']:>8.4f} "
            f"{s['min']:>8.4f} {s['max']:>8.4f} {s['median']:>8.4f} {s['n_subjects']:>5d}"
        )
    lines.append("")

    # 3. F1 and AUC summary
    f1_auc_cols = {
        "LR_PSD": ("lr_psd_f1", "lr_psd_auc"),
        "LR_CSP": ("lr_csp_f1", "lr_csp_auc"),
        "EEGNet_Raw": ("eegnet_f1", "eegnet_auc"),
    }
    lines.append("F1 (Macro) and AUC-ROC Summary")
    lines.append("=" * 75)
    lines.append(f"  {'Model':<15s} {'Mean F1':>10s} {'Mean AUC':>10s}")
    lines.append(f"  {'-'*15} {'-'*10} {'-'*10}")
    for model_name, (f1_col, auc_col) in f1_auc_cols.items():
        if f1_col in results_df.columns and auc_col in results_df.columns:
            mean_f1 = results_df[f1_col].mean()
            mean_auc = results_df[auc_col].mean()
            lines.append(f"  {model_name:<15s} {mean_f1:>10.4f} {mean_auc:>10.4f}")
    lines.append("")

    # 4. Statistical comparisons
    if comparison and "pairwise_tests" in comparison:
        lines.append("Pairwise Statistical Tests (paired t-test)")
        lines.append("=" * 75)
        for test in comparison["pairwise_tests"]:
            sig_str = "SIGNIFICANT" if test["significant"] else "not significant"
            lines.append(
                f"  {test['model_a']} vs {test['model_b']}: "
                f"t={test['t_statistic']:.4f}, p={test['p_value']:.4f} — {sig_str}"
            )
        lines.append("")

    # 5. Best model summary
    if agg_stats:
        best_model = max(agg_stats, key=lambda m: agg_stats[m]["mean"])
        lines.append(f"Best model by mean accuracy: {best_model} ({agg_stats[best_model]['mean']:.4f})")
        lines.append("")

    lines.append("=" * 75)
    lines.append("  END OF REPORT")
    lines.append("=" * 75)

    report_text = "\n".join(lines)

    # Save to file
    filepath = output_dir / "evaluation_report.txt"
    with open(filepath, "w") as f:
        f.write(report_text)

    # Also save aggregate stats as JSON
    stats_path = output_dir / "aggregate_stats.json"
    with open(stats_path, "w") as f:
        json.dump(agg_stats, f, indent=2)

    logger.info("Evaluation report saved to %s", filepath)
    logger.info("Aggregate stats saved to %s", stats_path)

    # Print to console
    print(report_text)

    return filepath


# ---------------------------------------------------------------------------
# 4.2  Leave-One-Subject-Out (LOSO) Cross-Validation
# ---------------------------------------------------------------------------

def run_loso_cv(
    subject_data: dict[int, dict],
    model_type: str = "LR_PSD",
) -> dict:
    """Run Leave-One-Subject-Out cross-validation.

    Parameters
    ----------
    subject_data : dict mapping subject_id -> {"X": ndarray, "y": ndarray}
        Feature matrices and labels for each subject.
    model_type : str
        One of "LR_PSD", "LR_CSP", "EEGNet_Raw".

    Returns
    -------
    dict with per-subject accuracies, mean/std, and predictions.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from src.config import LR_SOLVER, LR_CLASS_WEIGHT, LR_MAX_ITER, RANDOM_SEED

    subject_ids = sorted(subject_data.keys())
    per_subject_acc = {}
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for test_subj in subject_ids:
        # Train on all other subjects
        train_X_parts = []
        train_y_parts = []
        for s in subject_ids:
            if s == test_subj:
                continue
            train_X_parts.append(subject_data[s]["X"])
            train_y_parts.append(subject_data[s]["y"])

        train_X = np.concatenate(train_X_parts, axis=0)
        train_y = np.concatenate(train_y_parts, axis=0)
        test_X = subject_data[test_subj]["X"]
        test_y = subject_data[test_subj]["y"]

        if model_type in ("LR_PSD", "LR_CSP"):
            clf = LogisticRegression(
                solver=LR_SOLVER,
                class_weight=LR_CLASS_WEIGHT,
                max_iter=LR_MAX_ITER,
                random_state=RANDOM_SEED,
            )
            clf.fit(train_X, train_y)
            y_pred = clf.predict(test_X)
            y_prob = clf.predict_proba(test_X)[:, 1]
        elif model_type == "EEGNet_Raw":
            from src.models.eegnet import train_eegnet
            result = train_eegnet(train_X, train_y, val_X=test_X, val_y=test_y)
            import torch
            model = result["model"]
            model.eval()
            device = next(model.parameters()).device
            test_tensor = torch.FloatTensor(test_X[:, np.newaxis, :, :]).to(device)
            with torch.no_grad():
                outputs = model(test_tensor)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
            y_prob = probs[:, 1].cpu().numpy()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        acc = accuracy_score(test_y, y_pred)
        per_subject_acc[test_subj] = acc
        all_y_true.extend(test_y)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        logger.info("LOSO — test subject %03d: accuracy=%.4f", test_subj, acc)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    mean_acc = np.mean(list(per_subject_acc.values()))
    std_acc = np.std(list(per_subject_acc.values()))
    macro_f1 = f1_score(all_y_true, all_y_pred, average="macro")
    auc_roc = roc_auc_score(all_y_true, all_y_prob)

    logger.info(
        "LOSO %s — Accuracy: %.4f +/- %.4f  F1: %.4f  AUC: %.4f",
        model_type, mean_acc, std_acc, macro_f1, auc_roc,
    )

    return {
        "model_type": model_type,
        "cv_type": "LOSO",
        "per_subject_accuracy": per_subject_acc,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "macro_f1": float(macro_f1),
        "auc_roc": float(auc_roc),
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_prob": all_y_prob,
    }


def compare_within_vs_cross_subject(
    within_results: dict,
    loso_results: dict,
    output_dir: Path | None = None,
) -> Path:
    """Generate comparison table of within-subject (10-fold) vs LOSO performance.

    Parameters
    ----------
    within_results : dict
        Keys are model names, values contain 'mean_accuracy', 'std_accuracy'.
    loso_results : dict
        Same structure but from LOSO CV.
    """
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("  Within-Subject (10-Fold) vs Cross-Subject (LOSO) Performance")
    lines.append("=" * 70)
    lines.append(f"  {'Model':<15s} {'Within Acc':>12s} {'LOSO Acc':>12s} {'Drop':>8s}")
    lines.append(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*8}")

    for model_name in within_results:
        w_acc = within_results[model_name].get("mean_accuracy", 0)
        l_acc = loso_results.get(model_name, {}).get("mean_accuracy", 0)
        drop = w_acc - l_acc
        lines.append(
            f"  {model_name:<15s} {w_acc:>12.4f} {l_acc:>12.4f} {drop:>+8.4f}"
        )

    lines.append("")
    lines.append("  Note: Cross-subject generalization typically shows 10-20% accuracy drop.")
    lines.append("  This is expected due to inter-subject variability in EEG signals.")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    filepath = output_dir / "within_vs_loso_comparison.txt"
    with open(filepath, "w") as f:
        f.write(report)

    logger.info("Within vs LOSO comparison saved to %s", filepath)
    return filepath
