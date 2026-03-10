"""Main training entrypoint for the EEG BCI pipeline.

Orchestrates: load data -> preprocess -> extract features -> train -> evaluate -> save.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SUBJECTS, get_config
from src.data_loader import download_data, load_raw
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    print_metrics_table,
    save_results,
)
from src.features import extract_psd_features
from src.models.logistic import train_logistic
from src.preprocessing import apply_filters, extract_epochs

logger = logging.getLogger(__name__)


def run_subject(subject_id: int) -> dict | None:
    """Run the full pipeline for a single subject.

    Returns a results dict or None if the subject fails.
    """
    logger.info("=" * 60)
    logger.info("Processing Subject %03d", subject_id)
    logger.info("=" * 60)

    try:
        # 1. Load data
        download_data(subjects=[subject_id])
        raw = load_raw(subject=subject_id)

        # 2. Preprocess
        filtered = apply_filters(raw)
        epochs = extract_epochs(filtered)

        if len(epochs) == 0:
            logger.warning("No epochs survived for Subject %d — skipping.", subject_id)
            return None

        # 3. Extract features
        X, y = extract_psd_features(epochs)

        # 4. Train logistic regression
        lr_results = train_logistic(X, y)

        # 5. Evaluate
        metrics = compute_metrics(
            lr_results["y_true"],
            lr_results["y_pred"],
            lr_results["y_prob"],
        )

        # Add cross-validation stats to metrics
        metrics["cv_mean_accuracy"] = lr_results["mean_accuracy"]
        metrics["cv_std_accuracy"] = lr_results["std_accuracy"]

        # 6. Print, save, and plot
        print_metrics_table(metrics, "LogisticRegression", subject_id)
        save_results(metrics, "LogisticRegression", subject_id)
        plot_confusion_matrix(
            lr_results["y_true"],
            lr_results["y_pred"],
            "LogisticRegression",
            subject_id,
        )

        return {
            "subject": subject_id,
            "accuracy": metrics["accuracy"],
            "cv_mean_accuracy": lr_results["mean_accuracy"],
            "cv_std_accuracy": lr_results["std_accuracy"],
            "f1_macro": metrics["f1_macro"],
            "auc_roc": metrics["auc_roc"],
            "cohens_kappa": metrics["cohens_kappa"],
            "n_epochs": len(epochs),
        }

    except Exception as exc:
        logger.error("Subject %d failed: %s", subject_id, exc, exc_info=True)
        return None


def main(subjects: list[int] | None = None) -> None:
    """Run the full pipeline for all configured subjects."""
    subjects = subjects or SUBJECTS

    # Log configuration
    cfg = get_config()
    logger.info("Pipeline configuration: %s", cfg)
    logger.info("Processing %d subjects: %s", len(subjects), subjects)

    all_results = []

    for subj in subjects:
        result = run_subject(subj)
        if result is not None:
            all_results.append(result)

    if not all_results:
        logger.error("No subjects completed successfully.")
        return

    # Aggregate results into a summary DataFrame
    df = pd.DataFrame(all_results)

    # Save summary CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "summary_logistic_regression.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Summary CSV saved to %s", csv_path)

    # Print final summary
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — Per-Subject Results")
    print("=" * 70)
    print(
        df[["subject", "cv_mean_accuracy", "cv_std_accuracy", "f1_macro", "auc_roc"]]
        .to_string(index=False, float_format="%.4f")
    )
    print("-" * 70)
    print(f"  Overall mean accuracy: {df['cv_mean_accuracy'].mean():.4f}")
    print(f"  Overall std accuracy:  {df['cv_mean_accuracy'].std():.4f}")
    print(f"  Overall mean F1:       {df['f1_macro'].mean():.4f}")
    print(f"  Overall mean AUC:      {df['auc_roc'].mean():.4f}")
    print(f"  Subjects completed:    {len(df)}/{len(subjects)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="EEG BCI Training Pipeline")
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Subject IDs to process (default: config.SUBJECTS)",
    )
    args = parser.parse_args()

    main(subjects=args.subjects)
