"""Tests for logistic regression baseline model."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def lr_results():
    from src.data_loader import download_data, load_raw
    from src.preprocessing import apply_filters, extract_epochs
    from src.features import extract_psd_features
    from src.models.logistic import train_logistic

    download_data(subjects=[1], runs=[3, 7])
    raw = load_raw(subject=1, runs=[3, 7])
    filtered = apply_filters(raw)
    epochs = extract_epochs(filtered)
    X, y = extract_psd_features(epochs)
    return train_logistic(X, y)


class TestLogisticRegression:
    def test_above_chance(self, lr_results):
        assert lr_results["mean_accuracy"] > 0.5

    def test_has_fold_accuracies(self, lr_results):
        assert len(lr_results["fold_accuracies"]) == 10

    def test_has_predictions(self, lr_results):
        assert len(lr_results["y_true"]) > 0
        assert len(lr_results["y_pred"]) > 0
        assert len(lr_results["y_prob"]) > 0

    def test_auc_valid(self, lr_results):
        assert 0.0 <= lr_results["auc_roc"] <= 1.0
