"""Logistic regression baseline model with stratified cross-validation."""

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import (
    CV_N_FOLDS,
    LR_C_GRID,
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


def train_logistic_tuned(
    X: np.ndarray,
    y: np.ndarray,
    c_grid: list[float] | None = None,
) -> dict:
    """Train logistic regression with grid search over regularization C.

    Uses nested cross-validation: outer 10-fold for evaluation, inner
    GridSearchCV for hyperparameter selection.

    Parameters
    ----------
    X : ndarray of shape (n_epochs, n_features)
    y : ndarray of shape (n_epochs,)
    c_grid : list of float, optional
        Regularization strengths to search. Defaults to LR_C_GRID from config.

    Returns
    -------
    dict with per-fold accuracies, best C per fold, overall best C,
    and comparison to default C=1.0.
    """
    c_grid = c_grid or LR_C_GRID

    skf = StratifiedKFold(
        n_splits=CV_N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    fold_accuracies = []
    fold_best_c = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    grid_search_details = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV for hyperparameter selection
        inner_clf = LogisticRegression(
            solver=LR_SOLVER,
            class_weight=LR_CLASS_WEIGHT,
            max_iter=LR_MAX_ITER,
            random_state=RANDOM_SEED,
        )
        grid_search = GridSearchCV(
            inner_clf,
            param_grid={"C": c_grid},
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_c = grid_search.best_params_["C"]
        fold_best_c.append(best_c)

        # Evaluate with best C on outer test fold
        y_pred = grid_search.predict(X_test)
        y_prob = grid_search.predict_proba(X_test)[:, 1]

        fold_acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_acc)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        # Log grid search results for this fold
        cv_results = {
            "fold": fold_idx + 1,
            "best_C": best_c,
            "best_score": grid_search.best_score_,
            "all_scores": {
                str(c): float(score)
                for c, score in zip(c_grid, grid_search.cv_results_["mean_test_score"])
            },
        }
        grid_search_details.append(cv_results)

        logger.info(
            "  Fold %2d/%d — accuracy: %.4f  best_C: %.4f",
            fold_idx + 1, CV_N_FOLDS, fold_acc, best_c,
        )

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    macro_f1 = f1_score(all_y_true, all_y_pred, average="macro")
    auc_roc = roc_auc_score(all_y_true, all_y_prob)

    # Overall best C (most frequently selected)
    from collections import Counter
    c_counts = Counter(fold_best_c)
    overall_best_c = c_counts.most_common(1)[0][0]

    logger.info(
        "Tuned LR — Accuracy: %.4f +/- %.4f  F1: %.4f  AUC: %.4f  Best C: %s",
        mean_acc, std_acc, macro_f1, auc_roc, overall_best_c,
    )

    print(f"\n{'='*55}")
    print("Logistic Regression (Tuned) — Cross-Validation Results")
    print(f"{'='*55}")
    for i, (acc, c) in enumerate(zip(fold_accuracies, fold_best_c)):
        print(f"  Fold {i+1:2d}: {acc:.4f}  (C={c})")
    print(f"{'-'*55}")
    print(f"  Mean accuracy:  {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Macro F1-score: {macro_f1:.4f}")
    print(f"  AUC-ROC:        {auc_roc:.4f}")
    print(f"  Overall best C: {overall_best_c}")
    print(f"{'='*55}\n")

    return {
        "model_name": "LogisticRegression_Tuned",
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "macro_f1": float(macro_f1),
        "auc_roc": float(auc_roc),
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_prob": all_y_prob,
        "best_c_per_fold": fold_best_c,
        "overall_best_c": overall_best_c,
        "grid_search_details": grid_search_details,
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
