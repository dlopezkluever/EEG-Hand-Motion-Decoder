"""Tests for PSD band-power feature extraction."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def feature_data():
    from src.data_loader import download_data, load_raw
    from src.preprocessing import apply_filters, extract_epochs
    from src.features import extract_psd_features

    download_data(subjects=[1], runs=[3, 7])
    raw = load_raw(subject=1, runs=[3, 7])
    filtered = apply_filters(raw)
    epochs = extract_epochs(filtered)
    X, y = extract_psd_features(epochs)
    return X, y, epochs


class TestPSDFeatures:
    def test_feature_shape(self, feature_data):
        X, y, epochs = feature_data
        n_epochs = len(epochs)
        assert X.shape == (n_epochs, 384)  # 64 channels * 6 bands

    def test_labels_shape(self, feature_data):
        X, y, epochs = feature_data
        assert y.shape == (len(epochs),)

    def test_no_nan(self, feature_data):
        X, y, _ = feature_data
        assert not np.any(np.isnan(X))

    def test_no_inf(self, feature_data):
        X, y, _ = feature_data
        assert not np.any(np.isinf(X))

    def test_binary_labels(self, feature_data):
        _, y, _ = feature_data
        assert set(np.unique(y)).issubset({0, 1})
