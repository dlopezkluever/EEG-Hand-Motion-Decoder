"""Feature extraction — PSD band power, CSP, raw epoch flattening."""

import logging

import mne
from mne.decoding import CSP
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import (
    CSP_N_COMPONENTS,
    FREQ_BANDS,
    PSD_FMAX,
    PSD_FMIN,
    PSD_N_FFT,
    PSD_N_OVERLAP,
    PSD_WINDOW,
)

logger = logging.getLogger(__name__)


def extract_psd_features(
    epochs: mne.Epochs,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract PSD band-power features from epochs.

    For each epoch, computes the average power in each of 6 canonical
    frequency bands across all 64 channels, producing a feature vector
    of shape (n_epochs, n_channels * n_bands).

    Returns
    -------
    X : ndarray of shape (n_epochs, 384)
        Scaled feature matrix.
    y : ndarray of shape (n_epochs,)
        Integer labels (0 = left, 1 = right).
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)

    # Compute PSD using Welch's method
    psds, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=epochs.info["sfreq"],
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        n_fft=PSD_N_FFT,
        n_overlap=PSD_N_OVERLAP,
        window=PSD_WINDOW,
        verbose=False,
    )
    # psds shape: (n_epochs, n_channels, n_freqs)

    n_epochs, n_channels, _ = psds.shape
    n_bands = len(FREQ_BANDS)
    band_powers = np.zeros((n_epochs, n_channels, n_bands))

    for b_idx, (band_name, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
        freq_mask = (freqs >= fmin) & (freqs < fmax)
        if not np.any(freq_mask):
            logger.warning("No frequency bins for band %s (%.0f–%.0f Hz)", band_name, fmin, fmax)
            continue
        band_powers[:, :, b_idx] = psds[:, :, freq_mask].mean(axis=2)

    # Flatten to (n_epochs, n_channels * n_bands)
    X = band_powers.reshape(n_epochs, -1)

    assert X.shape == (n_epochs, n_channels * n_bands), (
        f"Expected shape ({n_epochs}, {n_channels * n_bands}), got {X.shape}"
    )

    # Normalize with StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Verify no NaN/Inf
    assert not np.any(np.isnan(X)), "NaN values in feature matrix!"
    assert not np.any(np.isinf(X)), "Inf values in feature matrix!"

    # Extract labels: map event codes to 0/1
    y = epochs.events[:, 2]
    event_id = epochs.event_id
    left_code = event_id["left"]
    right_code = event_id["right"]
    y = np.where(y == left_code, 0, 1)

    logger.info(
        "PSD features: X=%s  y=%s  bands=%d  channels=%d",
        X.shape,
        y.shape,
        n_bands,
        n_channels,
    )

    return X, y


def extract_csp_features(
    epochs: mne.Epochs,
    n_components: int = CSP_N_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, CSP]:
    """Extract CSP features from epochs.

    Uses MNE's CSP implementation with log-variance transformation.

    Returns
    -------
    X : ndarray of shape (n_epochs, n_components)
        CSP feature matrix.
    y : ndarray of shape (n_epochs,)
        Integer labels (0 = left, 1 = right).
    csp : mne.decoding.CSP
        Fitted CSP object (for inspecting spatial filters).
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)

    # Extract labels
    y = epochs.events[:, 2]
    event_id = epochs.event_id
    left_code = event_id["left"]
    y = np.where(y == left_code, 0, 1)

    # Fit CSP with log-variance transformation
    csp = CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False,
    )
    X = csp.fit_transform(data, y)

    assert X.shape == (len(epochs), n_components), (
        f"Expected shape ({len(epochs)}, {n_components}), got {X.shape}"
    )
    assert not np.any(np.isnan(X)), "NaN values in CSP feature matrix!"
    assert not np.any(np.isinf(X)), "Inf values in CSP feature matrix!"

    # Log CSP spatial filter weights
    filters = csp.filters_[:n_components]
    logger.info(
        "CSP features: X=%s  y=%s  components=%d",
        X.shape, y.shape, n_components,
    )
    logger.info(
        "CSP filter weight ranges: %s",
        [(f"comp{i}: [{w.min():.4f}, {w.max():.4f}]") for i, w in enumerate(filters)],
    )

    return X, y, csp


def extract_raw_features(
    epochs: mne.Epochs,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw epoch data with per-channel z-score normalization (Pathway C).

    Returns
    -------
    X : ndarray of shape (n_epochs, n_channels, n_timepoints)
        Normalized raw epoch data suitable for EEGNet input.
    y : ndarray of shape (n_epochs,)
        Integer labels (0 = left, 1 = right).
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)

    # Per-channel z-score normalization across all epochs and timepoints
    # Compute mean and std per channel across all epochs
    mean = data.mean(axis=(0, 2), keepdims=True)  # (1, n_channels, 1)
    std = data.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0  # avoid division by zero
    X = (data - mean) / std

    # Extract labels
    y = epochs.events[:, 2]
    event_id = epochs.event_id
    left_code = event_id["left"]
    y = np.where(y == left_code, 0, 1)

    logger.info(
        "Raw features: X=%s  y=%s  (per-channel z-score normalized)",
        X.shape, y.shape,
    )

    return X, y


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    from src.data_loader import download_data, load_raw
    from src.preprocessing import apply_filters, extract_epochs

    logger.info("=== Feature Extraction — Smoke Test ===")

    download_data(subjects=[1], runs=[3, 7])
    raw = load_raw(subject=1, runs=[3, 7])
    filtered = apply_filters(raw)
    epochs = extract_epochs(filtered)

    X, y = extract_psd_features(epochs)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape:         {y.shape}")
    print(f"Label distribution:   left={np.sum(y == 0)}, right={np.sum(y == 1)}")

    logger.info("=== Smoke test complete ===")
