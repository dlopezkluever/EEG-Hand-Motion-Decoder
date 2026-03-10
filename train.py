"""Main training entrypoint for the EEG BCI pipeline.

Orchestrates: load data -> preprocess -> extract features -> train -> evaluate -> save.
Supports three model-feature combinations:
  - LR + PSD (Logistic Regression on PSD band-power features)
  - LR + CSP (Logistic Regression on CSP features)
  - EEGNet + Raw (EEGNet CNN on normalized raw epochs)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import RESULTS_DIR, SUBJECTS, get_config
from src.data_loader import download_data, load_raw
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    print_metrics_table,
    save_results,
)
from src.features import extract_csp_features, extract_psd_features, extract_raw_features
from src.models.eegnet import train_eegnet_cv
from src.models.logistic import train_logistic
from src.preprocessing import apply_filters, extract_epochs

logger = logging.getLogger(__name__)


def run_subject(subject_id: int) -> dict | None:
    """Run the full pipeline for a single subject across all model-feature combos.

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

        subject_results = {"subject": subject_id, "n_epochs": len(epochs)}

        # ---- Model 1: LR + PSD ----
        logger.info("--- LR + PSD ---")
        X_psd, y_psd = extract_psd_features(epochs)
        lr_psd = train_logistic(X_psd, y_psd)

        metrics_lr_psd = compute_metrics(lr_psd["y_true"], lr_psd["y_pred"], lr_psd["y_prob"])
        metrics_lr_psd["cv_mean_accuracy"] = lr_psd["mean_accuracy"]
        metrics_lr_psd["cv_std_accuracy"] = lr_psd["std_accuracy"]

        print_metrics_table(metrics_lr_psd, "LR_PSD", subject_id)
        save_results(metrics_lr_psd, "LR_PSD", subject_id)
        plot_confusion_matrix(lr_psd["y_true"], lr_psd["y_pred"], "LR_PSD", subject_id)

        subject_results["lr_psd_accuracy"] = lr_psd["mean_accuracy"]
        subject_results["lr_psd_std"] = lr_psd["std_accuracy"]
        subject_results["lr_psd_f1"] = metrics_lr_psd["f1_macro"]
        subject_results["lr_psd_auc"] = metrics_lr_psd["auc_roc"]
        subject_results["lr_psd_fold_accs"] = lr_psd["fold_accuracies"]

        # ---- Model 2: LR + CSP ----
        logger.info("--- LR + CSP ---")
        X_csp, y_csp, csp_obj = extract_csp_features(epochs)
        lr_csp = train_logistic(X_csp, y_csp)

        metrics_lr_csp = compute_metrics(lr_csp["y_true"], lr_csp["y_pred"], lr_csp["y_prob"])
        metrics_lr_csp["cv_mean_accuracy"] = lr_csp["mean_accuracy"]
        metrics_lr_csp["cv_std_accuracy"] = lr_csp["std_accuracy"]

        print_metrics_table(metrics_lr_csp, "LR_CSP", subject_id)
        save_results(metrics_lr_csp, "LR_CSP", subject_id)
        plot_confusion_matrix(lr_csp["y_true"], lr_csp["y_pred"], "LR_CSP", subject_id)

        subject_results["lr_csp_accuracy"] = lr_csp["mean_accuracy"]
        subject_results["lr_csp_std"] = lr_csp["std_accuracy"]
        subject_results["lr_csp_f1"] = metrics_lr_csp["f1_macro"]
        subject_results["lr_csp_auc"] = metrics_lr_csp["auc_roc"]
        subject_results["lr_csp_fold_accs"] = lr_csp["fold_accuracies"]

        # ---- Model 3: EEGNet + Raw ----
        logger.info("--- EEGNet + Raw ---")
        X_raw, y_raw = extract_raw_features(epochs)
        eegnet_results = train_eegnet_cv(X_raw, y_raw)

        metrics_eegnet = compute_metrics(
            eegnet_results["y_true"], eegnet_results["y_pred"], eegnet_results["y_prob"]
        )
        metrics_eegnet["cv_mean_accuracy"] = eegnet_results["mean_accuracy"]
        metrics_eegnet["cv_std_accuracy"] = eegnet_results["std_accuracy"]

        print_metrics_table(metrics_eegnet, "EEGNet_Raw", subject_id)
        save_results(metrics_eegnet, "EEGNet_Raw", subject_id)
        plot_confusion_matrix(
            eegnet_results["y_true"], eegnet_results["y_pred"], "EEGNet_Raw", subject_id
        )

        subject_results["eegnet_accuracy"] = eegnet_results["mean_accuracy"]
        subject_results["eegnet_std"] = eegnet_results["std_accuracy"]
        subject_results["eegnet_f1"] = metrics_eegnet["f1_macro"]
        subject_results["eegnet_auc"] = metrics_eegnet["auc_roc"]
        subject_results["eegnet_fold_accs"] = eegnet_results["fold_accuracies"]

        return subject_results

    except Exception as exc:
        logger.error("Subject %d failed: %s", subject_id, exc, exc_info=True)
        return None


def run_model_comparison(all_results: list[dict]) -> dict:
    """Run paired t-tests and build a comparison summary across models.

    Returns a comparison dict suitable for JSON export.
    """
    models = [
        ("LR_PSD", "lr_psd_fold_accs"),
        ("LR_CSP", "lr_csp_fold_accs"),
        ("EEGNet_Raw", "eegnet_fold_accs"),
    ]

    # Collect per-subject mean accuracies
    comparison = {"models": {}, "pairwise_tests": []}

    for model_name, fold_key in models:
        accs = [r[fold_key] for r in all_results if fold_key in r]
        mean_accs = [np.mean(a) for a in accs]
        comparison["models"][model_name] = {
            "mean_accuracy": float(np.mean(mean_accs)),
            "std_accuracy": float(np.std(mean_accs)),
            "n_subjects": len(mean_accs),
            "per_subject_accuracy": [float(a) for a in mean_accs],
        }

    # Pairwise t-tests between all model combinations
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            name_a, key_a = models[i]
            name_b, key_b = models[j]

            accs_a = [np.mean(r[key_a]) for r in all_results if key_a in r and key_b in r]
            accs_b = [np.mean(r[key_b]) for r in all_results if key_a in r and key_b in r]

            if len(accs_a) >= 2:
                t_stat, p_value = stats.ttest_rel(accs_a, accs_b)
                significant = p_value < 0.05
            else:
                t_stat, p_value, significant = float("nan"), float("nan"), False

            comparison["pairwise_tests"].append({
                "model_a": name_a,
                "model_b": name_b,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(significant),
            })

    return comparison


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

    # Build comparison DataFrame (exclude fold_accs lists for CSV)
    rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if not k.endswith("_fold_accs")}
        rows.append(row)
    df = pd.DataFrame(rows)

    # Save unified comparison CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Comparison CSV saved to %s", csv_path)

    # Run statistical comparison
    comparison = run_model_comparison(all_results)
    comparison_path = RESULTS_DIR / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparison JSON saved to %s", comparison_path)

    # Print side-by-side comparison table
    print("\n" + "=" * 80)
    print("  PIPELINE COMPLETE — Model Comparison")
    print("=" * 80)

    # Per-subject table
    acc_cols = ["subject", "lr_psd_accuracy", "lr_csp_accuracy", "eegnet_accuracy"]
    available_cols = [c for c in acc_cols if c in df.columns]
    print("\n  Per-Subject Accuracy:")
    print(df[available_cols].to_string(index=False, float_format="%.4f"))

    # Summary statistics
    print("\n" + "-" * 80)
    print("  Model Summary:")
    print(f"  {'Model':<15s} {'Mean Acc':>10s} {'Std Acc':>10s} {'Mean F1':>10s} {'Mean AUC':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    model_configs = [
        ("LR + PSD", "lr_psd_accuracy", "lr_psd_f1", "lr_psd_auc"),
        ("LR + CSP", "lr_csp_accuracy", "lr_csp_f1", "lr_csp_auc"),
        ("EEGNet + Raw", "eegnet_accuracy", "eegnet_f1", "eegnet_auc"),
    ]

    for name, acc_col, f1_col, auc_col in model_configs:
        if acc_col in df.columns:
            print(
                f"  {name:<15s} {df[acc_col].mean():>10.4f} {df[acc_col].std():>10.4f} "
                f"{df[f1_col].mean():>10.4f} {df[auc_col].mean():>10.4f}"
            )

    # Statistical significance
    print("\n  Pairwise t-tests (paired, p < 0.05):")
    for test in comparison["pairwise_tests"]:
        sig = "YES *" if test["significant"] else "no"
        print(
            f"  {test['model_a']} vs {test['model_b']}: "
            f"t={test['t_statistic']:.3f}, p={test['p_value']:.4f} — significant: {sig}"
        )

    print(f"\n  Subjects completed: {len(df)}/{len(subjects)}")
    print("=" * 80 + "\n")


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
