"""Tests for signal preprocessing — filtering and epoch extraction."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def raw():
    from src.data_loader import download_data, load_raw

    download_data(subjects=[1], runs=[3, 7])
    return load_raw(subject=1, runs=[3, 7])


@pytest.fixture(scope="module")
def filtered(raw):
    from src.preprocessing import apply_filters

    return apply_filters(raw)


@pytest.fixture(scope="module")
def epochs(filtered):
    from src.preprocessing import extract_epochs

    return extract_epochs(filtered)


class TestFiltering:
    def test_shape_unchanged(self, raw, filtered):
        assert raw.get_data().shape == filtered.get_data().shape

    def test_values_differ(self, raw, filtered):
        assert not np.allclose(raw.get_data(), filtered.get_data())

    def test_no_nan(self, filtered):
        assert not np.any(np.isnan(filtered.get_data()))

    def test_channel_count(self, filtered):
        assert len(filtered.ch_names) == 64


class TestEpochs:
    def test_epochs_exist(self, epochs):
        assert len(epochs) > 0

    def test_both_classes(self, epochs):
        assert len(epochs["left"]) > 0
        assert len(epochs["right"]) > 0

    def test_epoch_shape(self, epochs):
        data = epochs.get_data()
        assert data.ndim == 3
        assert data.shape[1] == 64  # channels
        assert data.shape[2] > 0   # timepoints
