"""Standalone evaluation script — load saved models and evaluate without retraining.

Supports loading both scikit-learn (joblib) and PyTorch (state dict) models,
generating all evaluation figures and metrics for specified subjects.

Usage
-----
    python evaluate.py --model-path outputs/models/eegnet_fold01.pt --data-subjects 1
    python evaluate.py --model-path outputs/models/lr_psd.joblib --data-subjects 1 2 3
    python evaluate.py --results-dir outputs/results/         # regenerate report from saved JSONs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    get_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_sklearn_model(model_path: Path):
    """Load a scikit-learn model from joblib or pickle file."""
    suffix = model_path.suffix.lower()
    if suffix == ".joblib":
        import joblib
        return joblib.load(model_path)
    elif suffix in (".pkl", ".pickle"):
        import pickle
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported sklearn model format: {suffix}")


def load_pytorch_model(model_path: Path, n_channels: int = 64, n_timepoints: int = 721):
    """Load a PyTorch EEGNet model from a state dict .pt file."""
    import torch
    from src.models.eegnet import EEGNet

    model = EEGNet(n_channels=n_channels, n_timepoints=n_timepoints)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _infer_model_type(model_path: Path) -> str:
    """Infer model type from filename."""
    name = model_path.stem.lower()
    if "eegnet" in name:
        return "EEGNet_Raw"
    elif "csp" in name:
        return "LR_CSP"
    else:
        return "LR_PSD"


# ---------------------------------------------------------------------------
# Evaluation with a loaded model
# ---------------------------------------------------------------------------

def evaluate_sklearn(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate a scikit-learn model on features X with labels y."""
    from src.evaluate import compute_metrics
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return compute_metrics(y, y_pred, y_prob)


def evaluate_eegnet(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate a PyTorch EEGNet model on raw epoch data."""
    import torch
    from src.evaluate import compute_metrics

    model.eval()
    device = next(model.parameters()).device
    X_tensor = torch.FloatTensor(X[:, np.newaxis, :, :]).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    y_pred = predicted.cpu().numpy()
    y_prob = probs[:, 1].cpu().numpy()
    return compute_metrics(y, y_pred, y_prob)


# ---------------------------------------------------------------------------
# Regenerate report from saved results
# ---------------------------------------------------------------------------

def regenerate_report(results_dir: Path) -> None:
    """Regenerate evaluation report and figures from saved JSON results."""
    from src.evaluate import generate_evaluation_report

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        logger.error("No JSON result files found in %s", results_dir)
        return

    # Parse per-subject results
    rows = []
    for jf in json_files:
        name = jf.stem
        # Skip non-model result files
        if name in ("model_comparison", "aggregate_stats", "loso_results"):
            continue
        if "_subject" not in name:
            continue

        with open(jf) as f:
            metrics = json.load(f)

        # Parse model name and subject ID
        parts = name.rsplit("_subject", 1)
        if len(parts) != 2:
            continue
        model_name = parts[0]
        try:
            subject_id = int(parts[1])
        except ValueError:
            continue

        rows.append({
            "subject": subject_id,
            "model": model_name,
            "accuracy": metrics.get("cv_mean_accuracy", metrics.get("accuracy", 0)),
            "f1_macro": metrics.get("f1_macro", 0),
            "auc_roc": metrics.get("auc_roc", 0),
        })

    if not rows:
        logger.error("No parseable result files found.")
        return

    df_long = pd.DataFrame(rows)

    # Pivot to wide format for the report
    pivot_rows = []
    for subj, grp in df_long.groupby("subject"):
        row = {"subject": subj}
        for _, r in grp.iterrows():
            model = r["model"]
            if model == "LR_PSD":
                row["lr_psd_accuracy"] = r["accuracy"]
                row["lr_psd_f1"] = r["f1_macro"]
                row["lr_psd_auc"] = r["auc_roc"]
            elif model == "LR_CSP":
                row["lr_csp_accuracy"] = r["accuracy"]
                row["lr_csp_f1"] = r["f1_macro"]
                row["lr_csp_auc"] = r["auc_roc"]
            elif model == "EEGNet_Raw":
                row["eegnet_accuracy"] = r["accuracy"]
                row["eegnet_f1"] = r["f1_macro"]
                row["eegnet_auc"] = r["auc_roc"]
        pivot_rows.append(row)

    results_df = pd.DataFrame(pivot_rows)

    # Load comparison if it exists
    comparison_path = results_dir / "model_comparison.json"
    comparison = None
    if comparison_path.exists():
        with open(comparison_path) as f:
            comparison = json.load(f)

    generate_evaluation_report(results_df, comparison, output_dir=results_dir)
    print(f"\nReport regenerated at {results_dir / 'evaluation_report.txt'}")


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    model_path: Path,
    subjects: list[int],
    model_type: str | None = None,
) -> None:
    """Load a saved model and evaluate it on specified subjects.

    Generates all evaluation figures and metrics without retraining.
    """
    from src.data_loader import download_data, load_raw
    from src.preprocessing import apply_filters, extract_epochs
    from src.features import extract_psd_features, extract_csp_features, extract_raw_features
    from src.evaluate import compute_metrics, save_results, print_metrics_table
    from src.visualize import plot_confusion_matrix, plot_roc_curve

    model_type = model_type or _infer_model_type(model_path)
    logger.info("Evaluating model: %s (type=%s)", model_path, model_type)

    # Load model
    suffix = model_path.suffix.lower()
    if suffix == ".pt":
        # Need to peek at data shape first — load one subject to infer dimensions
        download_data(subjects=[subjects[0]])
        raw = load_raw(subject=subjects[0])
        filtered = apply_filters(raw)
        epochs = extract_epochs(filtered)
        X_sample, _ = extract_raw_features(epochs)
        model = load_pytorch_model(model_path, n_channels=X_sample.shape[1], n_timepoints=X_sample.shape[2])
        is_pytorch = True
    elif suffix in (".joblib", ".pkl", ".pickle"):
        model = load_sklearn_model(model_path)
        is_pytorch = False
    else:
        logger.error("Unsupported model file format: %s", suffix)
        return

    all_metrics = []

    for subj in subjects:
        logger.info("Evaluating on Subject %03d", subj)
        try:
            download_data(subjects=[subj])
            raw = load_raw(subject=subj)
            filtered = apply_filters(raw)
            epochs = extract_epochs(filtered)

            if len(epochs) == 0:
                logger.warning("No epochs for Subject %d — skipping.", subj)
                continue

            # Extract features based on model type
            if model_type == "EEGNet_Raw":
                X, y = extract_raw_features(epochs)
                metrics = evaluate_eegnet(model, X, y)
            elif model_type == "LR_CSP":
                X, y, _ = extract_csp_features(epochs)
                metrics = evaluate_sklearn(model, X, y)
            else:  # LR_PSD
                X, y = extract_psd_features(epochs)
                metrics = evaluate_sklearn(model, X, y)

            print_metrics_table(metrics, model_type, subj)
            save_results(metrics, f"{model_type}_eval", subj)

            # Generate evaluation figures
            y_pred = (metrics["accuracy"] > 0.5)  # placeholder; recompute
            # Re-extract predictions for figures
            if is_pytorch:
                import torch
                model.eval()
                X_tensor = torch.FloatTensor(X[:, np.newaxis, :, :])
                with torch.no_grad():
                    outputs = model(X_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                y_pred = predicted.numpy()
                y_prob = probs[:, 1].numpy()
            else:
                y_pred = model.predict(X)
                y_prob = model.predict_proba(X)[:, 1]

            plot_confusion_matrix(y, y_pred, f"{model_type}_eval", subj)
            plot_roc_curve(y, y_prob, f"{model_type}_eval", subj)

            metrics["subject"] = subj
            all_metrics.append(metrics)

        except Exception as exc:
            logger.error("Subject %d evaluation failed: %s", subj, exc, exc_info=True)

    if all_metrics:
        # Summary
        accs = [m["accuracy"] for m in all_metrics]
        print(f"\n{'='*60}")
        print(f"  Standalone Evaluation — {model_type}")
        print(f"  Model: {model_path}")
        print(f"{'='*60}")
        print(f"  Subjects evaluated: {len(all_metrics)}")
        print(f"  Mean accuracy:      {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  Mean F1 (macro):    {np.mean([m['f1_macro'] for m in all_metrics]):.4f}")
        print(f"  Mean AUC-ROC:       {np.mean([m['auc_roc'] for m in all_metrics]):.4f}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Standalone BCI model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --model-path outputs/models/eegnet_fold01.pt --data-subjects 1
  python evaluate.py --model-path outputs/models/lr_psd.joblib --data-subjects 1 2 3
  python evaluate.py --results-dir outputs/results/
        """,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a saved model file (.pt for PyTorch, .joblib/.pkl for sklearn)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["LR_PSD", "LR_CSP", "EEGNet_Raw"],
        help="Override auto-detected model type",
    )
    parser.add_argument(
        "--data-subjects",
        type=int,
        nargs="+",
        default=[1],
        help="Subject IDs to evaluate on (default: [1])",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Regenerate evaluation report from saved result JSONs in this directory",
    )

    args = parser.parse_args()

    if args.results_dir:
        regenerate_report(Path(args.results_dir))
    elif args.model_path:
        run_evaluation(
            model_path=Path(args.model_path),
            subjects=args.data_subjects,
            model_type=args.model_type,
        )
    else:
        parser.print_help()
        print("\nError: provide either --model-path or --results-dir.")
        sys.exit(1)


if __name__ == "__main__":
    main()
