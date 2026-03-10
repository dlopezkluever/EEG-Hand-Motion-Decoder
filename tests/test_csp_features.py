"""Tests for CSP feature extraction."""

import numpy as np
import pytest

from src.config import CSP_N_COMPONENTS


@pytest.fixture(scope="module")
def epochs():
    from src.data_loader import download_data, load_raw
    from src.preprocessing import apply_filters, extract_epochs

    download_data(subjects=[1], runs=[3, 7])
    raw = load_raw(subject=1, runs=[3, 7])
    filtered = apply_filters(raw)
    return extract_epochs(filtered)


@pytest.fixture(scope="module")
def csp_result(epochs):
    from src.features import extract_csp_features

    return extract_csp_features(epochs)


def test_csp_feature_shape(csp_result, epochs):
    X, y, csp = csp_result
    assert X.shape == (len(epochs), CSP_N_COMPONENTS)


def test_csp_no_nan(csp_result):
    X, y, csp = csp_result
    assert not np.any(np.isnan(X))
    assert not np.any(np.isinf(X))


def test_csp_labels_binary(csp_result):
    X, y, csp = csp_result
    assert set(np.unique(y)).issubset({0, 1})


def test_csp_has_spatial_filters(csp_result):
    X, y, csp = csp_result
    assert hasattr(csp, "filters_")
    assert csp.filters_.shape[0] >= CSP_N_COMPONENTS
