"""Tests for evaluation metrics computation."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestComputeMetrics:
    def test_perfect_predictions(self):
        from src.evaluate import compute_metrics

        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.95])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["cohens_kappa"] == 1.0
        assert metrics["auc_roc"] == 1.0

    def test_random_predictions(self):
        from src.evaluate import compute_metrics

        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.random.rand(10)

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_all_keys_present(self):
        from src.evaluate import compute_metrics

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.3, 0.7, 0.6, 0.4])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        expected_keys = [
            "accuracy",
            "precision_macro",
            "precision_left",
            "precision_right",
            "recall_macro",
            "recall_left",
            "recall_right",
            "f1_macro",
            "f1_left",
            "f1_right",
            "auc_roc",
            "cohens_kappa",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_metric_values_are_floats(self):
        from src.evaluate import compute_metrics

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.2, 0.8, 0.6, 0.9])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not a float: {type(value)}"
