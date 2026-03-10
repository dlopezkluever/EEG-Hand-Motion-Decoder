"""Logistic regression baseline model with stratified cross-validation."""

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import (
    CV_N_FOLDS,
    LR_CLASS_WEIGHT,
    LR_MAX_ITER,
    LR_SOLVER,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Train logistic regression with stratified k-fold cross-validation.

    Returns a dict containing per-fold accuracies, mean/std accuracy,
    macro F1, AUC-ROC, and the cross-validated predictions.
    """
    clf = LogisticRegression(
        solver=LR_SOLVER,
        class_weight=LR_CLASS_WEIGHT,
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_SEED,
    )

    skf = StratifiedKFold(
        n_splits=CV_N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    fold_accuracies = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        fold_acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_acc)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        logger.info("  Fold %2d/%d — accuracy: %.4f", fold_idx + 1, CV_N_FOLDS, fold_acc)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    macro_f1 = f1_score(all_y_true, all_y_pred, average="macro")
    auc_roc = roc_auc_score(all_y_true, all_y_prob)

    logger.info(
        "Logistic Regression — Accuracy: %.4f +/- %.4f  F1: %.4f  AUC: %.4f",
        mean_acc,
        std_acc,
        macro_f1,
        auc_roc,
    )

    # Print classification summary to console
    print(f"\n{'='*50}")
    print("Logistic Regression — Cross-Validation Results")
    print(f"{'='*50}")
    for i, acc in enumerate(fold_accuracies):
        print(f"  Fold {i+1:2d}: {acc:.4f}")
    print(f"{'-'*50}")
    print(f"  Mean accuracy:  {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Macro F1-score: {macro_f1:.4f}")
    print(f"  AUC-ROC:        {auc_roc:.4f}")
    print(f"{'='*50}\n")

    return {
        "model_name": "LogisticRegression",
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "macro_f1": float(macro_f1),
        "auc_roc": float(auc_roc),
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_prob": all_y_prob,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    from src.data_loader import download_data, load_raw
    from src.preprocessing import apply_filters, extract_epochs
    from src.features import extract_psd_features

    logger.info("=== Logistic Regression — Smoke Test ===")

    download_data(subjects=[1], runs=[3, 7])
    raw = load_raw(subject=1, runs=[3, 7])
    filtered = apply_filters(raw)
    epochs = extract_epochs(filtered)
    X, y = extract_psd_features(epochs)

    results = train_logistic(X, y)
    logger.info("=== Smoke test complete ===")
