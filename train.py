"""Main training entrypoint for the EEG BCI pipeline.

Orchestrates: load data -> preprocess -> extract features -> train -> evaluate -> save.
Supports three model-feature combinations:
  - LR + PSD (Logistic Regression on PSD band-power features)
  - LR + CSP (Logistic Regression on CSP features)
  - EEGNet + Raw (EEGNet CNN on normalized raw epochs)

Phase 4 additions:
  - Full 109-subject scaling with tqdm progress bars
  - Subject-level result caching for resumable runs
  - LOSO cross-validation mode
  - LR hyperparameter tuning via GridSearchCV
  - EEGNet data augmentation (Gaussian noise + temporal jitter)
  - Global random seed control for reproducibility
  - Extended CLI: --models, --cv-folds, --output-dir, --no-cache, --loso, --tune, --augment

Phase 5 additions:
  - Optional ICA artifact removal (--ica)
  - Optional motor cortex ROI channel selection (--roi)
  - MLflow experiment tracking integration (--mlflow)
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        for i, item in enumerate(iterable):
            if total:
                print(f"\r  {desc} [{i+1}/{total}]", end="", flush=True)
            yield item
        if total:
            print()

from src.config import (
    CACHE_RESULTS,
    FIGURES_DIR,
    FREQ_BANDS,
    RANDOM_SEED,
    RESULTS_DIR,
    SUBJECTS,
    USE_ICA,
    USE_ROI_CHANNELS,
    get_config,
)
from src.data_loader import download_data, load_raw
from src.evaluate import (
    compare_within_vs_cross_subject,
    compute_metrics,
    generate_evaluation_report,
    plot_confusion_matrix,
    print_metrics_table,
    run_loso_cv,
    save_results,
    save_results_csv,
)
from src.features import extract_csp_features, extract_psd_features, extract_raw_features
from src.models.eegnet import train_eegnet_cv
from src.models.logistic import train_logistic, train_logistic_tuned
from src.preprocessing import apply_filters, apply_ica, extract_epochs, pick_roi_channels
from src.tracking import is_tracking_enabled, log_artifact, log_metrics, log_params, start_run
from src.visualize import (
    generate_all_signal_figures,
    plot_feature_importance,
    plot_multi_roc,
    plot_roc_curve,
    plot_subject_accuracy_bar,
    plot_training_curves,
    plot_confusion_matrix as viz_plot_confusion_matrix,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 4.5  Reproducibility — Global Seed Control
# ---------------------------------------------------------------------------

def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for NumPy, PyTorch, and Python random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info("Global random seeds set to %d", seed)


# ---------------------------------------------------------------------------
# 4.1  Result Caching
# ---------------------------------------------------------------------------

def _subject_result_exists(subject_id: int, output_dir: Path | None = None) -> bool:
    """Check if results already exist for a subject (all three models)."""
    output_dir = output_dir or RESULTS_DIR
    models = ["LR_PSD", "LR_CSP", "EEGNet_Raw"]
    return all(
        (output_dir / f"{model}_subject{subject_id:03d}.json").exists()
        for model in models
    )


def _load_cached_result(subject_id: int, output_dir: Path | None = None) -> dict | None:
    """Load cached results for a subject from JSON files."""
    output_dir = output_dir or RESULTS_DIR
    models_map = {
        "LR_PSD": ("lr_psd_accuracy", "lr_psd_f1", "lr_psd_auc"),
        "LR_CSP": ("lr_csp_accuracy", "lr_csp_f1", "lr_csp_auc"),
        "EEGNet_Raw": ("eegnet_accuracy", "eegnet_f1", "eegnet_auc"),
    }

    result = {"subject": subject_id}

    for model_name, (acc_key, f1_key, auc_key) in models_map.items():
        filepath = output_dir / f"{model_name}_subject{subject_id:03d}.json"
        if not filepath.exists():
            return None
        with open(filepath) as f:
            metrics = json.load(f)
        result[acc_key] = metrics.get("cv_mean_accuracy", metrics.get("accuracy", 0))
        result[f1_key] = metrics.get("f1_macro", 0)
        result[auc_key] = metrics.get("auc_roc", 0)

    return result


# ---------------------------------------------------------------------------
# Per-Subject Pipeline
# ---------------------------------------------------------------------------

def run_subject(
    subject_id: int,
    tune: bool = False,
    augment: bool = False,
    use_ica: bool = False,
    use_roi: bool = False,
) -> dict | None:
    """Run the full pipeline for a single subject across all model-feature combos.

    Parameters
    ----------
    subject_id : int
        Subject number (1-109).
    tune : bool
        If True, also run LR with GridSearchCV hyperparameter tuning.
    augment : bool
        If True, apply data augmentation for EEGNet training.
    use_ica : bool
        If True, apply ICA artifact removal before epoching.
    use_roi : bool
        If True, restrict channels to motor cortex ROI after epoching.

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

        # 2b. Optional ICA artifact removal (Phase 5.1)
        ica_info = None
        if use_ica:
            filtered, ica_info = apply_ica(filtered)
            logger.info("ICA applied: excluded %d components", ica_info["n_excluded"])

        epochs = extract_epochs(filtered)

        # 2c. Optional ROI channel selection (Phase 5.2)
        if use_roi:
            epochs = pick_roi_channels(epochs)

        if len(epochs) == 0:
            logger.warning("No epochs survived for Subject %d — skipping.", subject_id)
            return None

        subject_results = {
            "subject": subject_id,
            "n_epochs": len(epochs),
            "n_channels": len(epochs.ch_names),
            "ica_applied": use_ica,
            "roi_applied": use_roi,
        }
        if ica_info:
            subject_results["ica_n_excluded"] = ica_info["n_excluded"]

        # ---- Phase 3: EEG Signal Visualizations ----
        logger.info("--- Signal Visualizations ---")
        generate_all_signal_figures(epochs, subject_id)

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
        plot_roc_curve(lr_psd["y_true"], lr_psd["y_prob"], "LR_PSD", subject_id)

        # Feature importance for LR+PSD
        from sklearn.linear_model import LogisticRegression
        from src.config import LR_SOLVER, LR_CLASS_WEIGHT, LR_MAX_ITER
        lr_model = LogisticRegression(
            solver=LR_SOLVER, class_weight=LR_CLASS_WEIGHT,
            max_iter=LR_MAX_ITER, random_state=RANDOM_SEED,
        )
        lr_model.fit(X_psd, y_psd)
        bands = list(FREQ_BANDS.keys())
        ch_names = epochs.ch_names
        feat_names = [f"{ch}_{band}" for ch in ch_names for band in bands]
        plot_feature_importance(lr_model.coef_[0], feat_names, "LR_PSD", subject_id)

        subject_results["lr_psd_accuracy"] = lr_psd["mean_accuracy"]
        subject_results["lr_psd_std"] = lr_psd["std_accuracy"]
        subject_results["lr_psd_f1"] = metrics_lr_psd["f1_macro"]
        subject_results["lr_psd_auc"] = metrics_lr_psd["auc_roc"]
        subject_results["lr_psd_fold_accs"] = lr_psd["fold_accuracies"]

        # ---- Phase 4.3: LR + PSD (Tuned) ----
        if tune:
            logger.info("--- LR + PSD (Tuned) ---")
            lr_psd_tuned = train_logistic_tuned(X_psd, y_psd)
            metrics_tuned = compute_metrics(
                lr_psd_tuned["y_true"], lr_psd_tuned["y_pred"], lr_psd_tuned["y_prob"]
            )
            metrics_tuned["cv_mean_accuracy"] = lr_psd_tuned["mean_accuracy"]
            metrics_tuned["cv_std_accuracy"] = lr_psd_tuned["std_accuracy"]
            metrics_tuned["best_C"] = lr_psd_tuned["overall_best_c"]

            print_metrics_table(metrics_tuned, "LR_PSD_Tuned", subject_id)
            save_results(metrics_tuned, "LR_PSD_Tuned", subject_id)

            # Save grid search details
            grid_path = RESULTS_DIR / f"grid_search_subject{subject_id:03d}.json"
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(grid_path, "w") as f:
                json.dump(lr_psd_tuned["grid_search_details"], f, indent=2)

            subject_results["lr_psd_tuned_accuracy"] = lr_psd_tuned["mean_accuracy"]
            subject_results["lr_psd_tuned_best_c"] = lr_psd_tuned["overall_best_c"]

            # Log tuned vs default comparison
            default_acc = lr_psd["mean_accuracy"]
            tuned_acc = lr_psd_tuned["mean_accuracy"]
            logger.info(
                "  Tuned vs Default: %.4f vs %.4f (delta=%+.4f)",
                tuned_acc, default_acc, tuned_acc - default_acc,
            )

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
        plot_roc_curve(lr_csp["y_true"], lr_csp["y_prob"], "LR_CSP", subject_id)

        subject_results["lr_csp_accuracy"] = lr_csp["mean_accuracy"]
        subject_results["lr_csp_std"] = lr_csp["std_accuracy"]
        subject_results["lr_csp_f1"] = metrics_lr_csp["f1_macro"]
        subject_results["lr_csp_auc"] = metrics_lr_csp["auc_roc"]
        subject_results["lr_csp_fold_accs"] = lr_csp["fold_accuracies"]

        # ---- Model 3: EEGNet + Raw ----
        logger.info("--- EEGNet + Raw ---")
        X_raw, y_raw = extract_raw_features(epochs)

        # Phase 4.4: Data augmentation (training-time only, handled inside CV)
        if augment:
            logger.info("--- EEGNet + Raw (Augmented) ---")
            from src.augmentation import apply_augmentation
            from src.config import AUGMENT_GAUSSIAN_STD, AUGMENT_TEMPORAL_JITTER_MS, SAMPLING_RATE
            jitter_samples = int(AUGMENT_TEMPORAL_JITTER_MS * SAMPLING_RATE / 1000)
            eegnet_results = train_eegnet_cv(
                X_raw, y_raw,
                augment_fn=lambda X_tr, y_tr: apply_augmentation(
                    X_tr, y_tr,
                    gaussian_std=AUGMENT_GAUSSIAN_STD,
                    jitter_samples=jitter_samples,
                    seed=RANDOM_SEED,
                ),
            )
        else:
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
        plot_roc_curve(
            eegnet_results["y_true"], eegnet_results["y_prob"], "EEGNet_Raw", subject_id
        )

        # Training curves (use history from last fold — representative)
        if "history" in eegnet_results:
            plot_training_curves(eegnet_results["history"], "EEGNet", subject_id)

        # Multi-model ROC comparison
        plot_multi_roc(
            [
                {"model_name": "LR+PSD", "y_true": lr_psd["y_true"], "y_prob": lr_psd["y_prob"]},
                {"model_name": "LR+CSP", "y_true": lr_csp["y_true"], "y_prob": lr_csp["y_prob"]},
                {"model_name": "EEGNet", "y_true": eegnet_results["y_true"], "y_prob": eegnet_results["y_prob"]},
            ],
            subject_id,
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


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4.2  LOSO Mode
# ---------------------------------------------------------------------------

def run_loso_mode(subjects: list[int], model_types: list[str] | None = None) -> None:
    """Run Leave-One-Subject-Out cross-validation for specified models.

    Loads and preprocesses all subjects' data, then runs LOSO for each model type.
    """
    model_types = model_types or ["LR_PSD"]
    logger.info("LOSO mode: %d subjects, models=%s", len(subjects), model_types)

    # Collect all subject data
    subject_data_psd = {}
    subject_data_csp = {}
    subject_data_raw = {}

    for subj in tqdm(subjects, desc="Loading subjects"):
        try:
            download_data(subjects=[subj])
            raw = load_raw(subject=subj)
            filtered = apply_filters(raw)
            epochs = extract_epochs(filtered)

            if len(epochs) == 0:
                logger.warning("No epochs for Subject %d — skipping in LOSO.", subj)
                continue

            if "LR_PSD" in model_types:
                X_psd, y_psd = extract_psd_features(epochs)
                subject_data_psd[subj] = {"X": X_psd, "y": y_psd}

            if "LR_CSP" in model_types:
                X_csp, y_csp, _ = extract_csp_features(epochs)
                subject_data_csp[subj] = {"X": X_csp, "y": y_csp}

            if "EEGNet_Raw" in model_types:
                X_raw, y_raw = extract_raw_features(epochs)
                subject_data_raw[subj] = {"X": X_raw, "y": y_raw}

        except Exception as exc:
            logger.error("LOSO — failed to load Subject %d: %s", subj, exc)

    # Run LOSO for each model type
    loso_results = {}
    model_data_map = {
        "LR_PSD": subject_data_psd,
        "LR_CSP": subject_data_csp,
        "EEGNet_Raw": subject_data_raw,
    }

    for model_type in model_types:
        data = model_data_map.get(model_type, {})
        if len(data) < 2:
            logger.warning("Not enough subjects for LOSO with %s", model_type)
            continue
        result = run_loso_cv(data, model_type=model_type)
        loso_results[model_type] = result

        # Save per-subject LOSO accuracy bar chart
        loso_accs = result["per_subject_accuracy"]
        loso_df = pd.DataFrame([
            {"subject": s, "loso_accuracy": a}
            for s, a in loso_accs.items()
        ])
        plot_subject_accuracy_bar(
            loso_df.rename(columns={"loso_accuracy": "lr_psd_accuracy"})
            if model_type == "LR_PSD" else loso_df,
        )

    # Save LOSO results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    loso_path = RESULTS_DIR / "loso_results.json"
    serializable = {}
    for model, res in loso_results.items():
        serializable[model] = {
            k: v for k, v in res.items()
            if k not in ("y_true", "y_pred", "y_prob")
        }
        # Convert per_subject_accuracy keys to strings for JSON
        if "per_subject_accuracy" in serializable[model]:
            serializable[model]["per_subject_accuracy"] = {
                str(k): v for k, v in serializable[model]["per_subject_accuracy"].items()
            }
    with open(loso_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("LOSO results saved to %s", loso_path)

    # Print LOSO summary
    print("\n" + "=" * 70)
    print("  LOSO Cross-Validation Results")
    print("=" * 70)
    for model_type, result in loso_results.items():
        print(f"\n  {model_type}:")
        print(f"    Mean accuracy: {result['mean_accuracy']:.4f} +/- {result['std_accuracy']:.4f}")
        print(f"    Macro F1:      {result['macro_f1']:.4f}")
        print(f"    AUC-ROC:       {result['auc_roc']:.4f}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    subjects: list[int] | None = None,
    use_cache: bool = True,
    tune: bool = False,
    augment: bool = False,
    loso: bool = False,
    loso_models: list[str] | None = None,
    use_ica: bool = False,
    use_roi: bool = False,
) -> None:
    """Run the full pipeline for all configured subjects.

    Parameters
    ----------
    subjects : list of int, optional
        Subject IDs to process. Defaults to config.SUBJECTS.
    use_cache : bool
        If True, skip subjects with existing results.
    tune : bool
        If True, run LR hyperparameter tuning via GridSearchCV.
    augment : bool
        If True, apply data augmentation for EEGNet.
    loso : bool
        If True, run LOSO cross-validation instead of within-subject.
    loso_models : list of str, optional
        Model types to run in LOSO mode.
    use_ica : bool
        If True, apply ICA artifact removal (Phase 5.1).
    use_roi : bool
        If True, restrict to motor cortex ROI channels (Phase 5.2).
    """
    # Set global seeds for reproducibility
    set_global_seeds(RANDOM_SEED)

    subjects = subjects or SUBJECTS

    # Log configuration
    cfg = get_config()
    logger.info("Pipeline configuration: %s", cfg)
    logger.info("Processing %d subjects: %s", len(subjects), subjects[:20])
    if len(subjects) > 20:
        logger.info("  ... and %d more", len(subjects) - 20)
    logger.info(
        "Options: cache=%s, tune=%s, augment=%s, loso=%s, ica=%s, roi=%s",
        use_cache, tune, augment, loso, use_ica, use_roi,
    )

    # MLflow experiment tracking (Phase 5.3)
    run_tags = {
        "ica": str(use_ica),
        "roi": str(use_roi),
        "augment": str(augment),
        "tune": str(tune),
        "n_subjects": str(len(subjects)),
        "cv_strategy": "LOSO" if loso else "within-subject",
    }

    # LOSO mode — separate code path
    if loso:
        run_loso_mode(subjects, model_types=loso_models)
        return

    all_results = []
    cached_count = 0

    with start_run(run_name=f"train_{len(subjects)}subj", tags=run_tags):
        log_params({k: str(v)[:500] for k, v in cfg.items()})

        for subj in tqdm(subjects, desc="Subjects", total=len(subjects)):
            # Check cache
            if use_cache and CACHE_RESULTS and _subject_result_exists(subj):
                cached = _load_cached_result(subj)
                if cached is not None:
                    all_results.append(cached)
                    cached_count += 1
                    logger.info("Subject %03d — loaded from cache", subj)
                    continue

            result = run_subject(subj, tune=tune, augment=augment, use_ica=use_ica, use_roi=use_roi)
            if result is not None:
                all_results.append(result)

        # --- remaining processing inside MLflow run context ---
        _finalize_results(
            all_results, subjects, cached_count, tune,
        )


def _finalize_results(
    all_results: list[dict],
    subjects: list[int],
    cached_count: int,
    tune: bool,
) -> None:
    """Post-processing: save CSVs, generate reports, log to MLflow."""
    if cached_count > 0:
        logger.info("Loaded %d subjects from cache, processed %d new",
                     cached_count, len(all_results) - cached_count)

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

    # Run statistical comparison (only for results with fold data)
    results_with_folds = [r for r in all_results if "lr_psd_fold_accs" in r]
    if results_with_folds:
        comparison = run_model_comparison(results_with_folds)
    else:
        comparison = {"models": {}, "pairwise_tests": []}
    comparison_path = RESULTS_DIR / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparison JSON saved to %s", comparison_path)

    # ---- Phase 3: Subject accuracy bar chart ----
    plot_subject_accuracy_bar(df)

    # ---- Phase 3: Full metrics CSV ----
    metric_rows = []
    for r in all_results:
        for model_key, acc_key, f1_key, auc_key in [
            ("LR_PSD", "lr_psd_accuracy", "lr_psd_f1", "lr_psd_auc"),
            ("LR_CSP", "lr_csp_accuracy", "lr_csp_f1", "lr_csp_auc"),
            ("EEGNet_Raw", "eegnet_accuracy", "eegnet_f1", "eegnet_auc"),
        ]:
            if acc_key in r:
                metric_rows.append({
                    "subject": r["subject"],
                    "model": model_key,
                    "accuracy": r[acc_key],
                    "f1_macro": r.get(f1_key),
                    "auc_roc": r.get(auc_key),
                })
    if metric_rows:
        save_results_csv(metric_rows)

    # ---- Phase 3: Comprehensive evaluation report ----
    generate_evaluation_report(df, comparison)

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
    if comparison["pairwise_tests"]:
        print("\n  Pairwise t-tests (paired, p < 0.05):")
        for test in comparison["pairwise_tests"]:
            sig = "YES *" if test["significant"] else "no"
            print(
                f"  {test['model_a']} vs {test['model_b']}: "
                f"t={test['t_statistic']:.3f}, p={test['p_value']:.4f} — significant: {sig}"
            )

    # Phase 4.3: Tuned vs default comparison
    if tune and "lr_psd_tuned_accuracy" in df.columns:
        print("\n" + "-" * 80)
        print("  Hyperparameter Tuning Results (LR + PSD):")
        print(f"  {'Metric':<20s} {'Default':>10s} {'Tuned':>10s} {'Delta':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        default_mean = df["lr_psd_accuracy"].mean()
        tuned_mean = df["lr_psd_tuned_accuracy"].mean()
        print(f"  {'Mean Accuracy':<20s} {default_mean:>10.4f} {tuned_mean:>10.4f} {tuned_mean - default_mean:>+10.4f}")

    print(f"\n  Subjects completed: {len(df)}/{len(subjects)}")
    if cached_count > 0:
        print(f"  (from cache: {cached_count}, newly processed: {len(df) - cached_count})")
    print("=" * 80 + "\n")

    # Phase 5.3: Log aggregate metrics and artifacts to MLflow
    for name, acc_col, f1_col, auc_col in model_configs:
        if acc_col in df.columns:
            log_metrics({
                f"{name}_mean_accuracy": float(df[acc_col].mean()),
                f"{name}_std_accuracy": float(df[acc_col].std()),
                f"{name}_mean_f1": float(df[f1_col].mean()),
                f"{name}_mean_auc": float(df[auc_col].mean()),
            })
    log_artifact(csv_path)
    log_artifact(RESULTS_DIR / "evaluation_report.txt")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="EEG BCI Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --subjects 1 2 3          # Process specific subjects
  python train.py --tune                     # Enable LR hyperparameter tuning
  python train.py --augment                  # Enable EEGNet data augmentation
  python train.py --loso --subjects 1 2 3    # Run LOSO cross-validation
  python train.py --no-cache                 # Force reprocessing all subjects
  python train.py --cv-folds 5               # Use 5-fold instead of 10-fold CV
  python train.py --ica                      # Enable ICA artifact removal
  python train.py --roi                      # Use motor cortex ROI channels only
  python train.py --mlflow                   # Enable MLflow experiment tracking
        """,
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Subject IDs to process (default: all 109 from config)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable result caching — reprocess all subjects",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable LR hyperparameter tuning via GridSearchCV",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation for EEGNet training",
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Run Leave-One-Subject-Out cross-validation mode",
    )
    parser.add_argument(
        "--loso-models",
        type=str,
        nargs="+",
        default=["LR_PSD"],
        choices=["LR_PSD", "LR_CSP", "EEGNet_Raw"],
        help="Models to evaluate in LOSO mode (default: LR_PSD)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Number of CV folds (overrides config.CV_N_FOLDS)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for results and figures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed (default: config.RANDOM_SEED=42)",
    )
    parser.add_argument(
        "--ica",
        action="store_true",
        help="Enable ICA artifact removal (Phase 5.1)",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        help="Restrict to motor cortex ROI channels (Phase 5.2)",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking (Phase 5.3)",
    )
    args = parser.parse_args()

    # Override config values from CLI
    if args.cv_folds is not None:
        import src.config as cfg_module
        cfg_module.CV_N_FOLDS = args.cv_folds
        logger.info("CV folds overridden to %d", args.cv_folds)

    if args.output_dir is not None:
        import src.config as cfg_module
        output = Path(args.output_dir)
        cfg_module.RESULTS_DIR = output / "results"
        cfg_module.FIGURES_DIR = output / "figures"
        cfg_module.MODELS_DIR = output / "models"
        logger.info("Output directory overridden to %s", args.output_dir)

    if args.seed is not None:
        import src.config as cfg_module
        cfg_module.RANDOM_SEED = args.seed

    # Phase 5: ICA and ROI config overrides
    if args.ica:
        import src.config as cfg_module
        cfg_module.USE_ICA = True

    if args.roi:
        import src.config as cfg_module
        cfg_module.USE_ROI_CHANNELS = True

    if args.mlflow:
        import src.config as cfg_module
        cfg_module.USE_MLFLOW = True

    main(
        subjects=args.subjects,
        use_cache=not args.no_cache,
        tune=args.tune,
        augment=args.augment,
        loso=args.loso,
        loso_models=args.loso_models,
        use_ica=args.ica,
        use_roi=args.roi,
    )
