"""Data augmentation utilities for EEGNet training.

Each augmentation function doubles the dataset by appending augmented copies
to the original data.  The caller is responsible for applying these functions
**only to training data** — never to validation or test splits — to avoid
information leakage.

Expected array shapes
---------------------
X : (n_epochs, n_channels, n_timepoints)
y : (n_epochs,)
"""

import logging

import numpy as np

from src.config import (
    AUGMENT_GAUSSIAN_STD,
    AUGMENT_TEMPORAL_JITTER_MS,
    SAMPLING_RATE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

def augment_gaussian_noise(
    X: np.ndarray,
    y: np.ndarray,
    std: float = AUGMENT_GAUSSIAN_STD,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add zero-mean Gaussian noise to every epoch.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_timepoints)
        EEG epoch array.
    y : ndarray, shape (n_epochs,)
        Label array.
    std : float, optional
        Standard deviation (sigma) of the Gaussian noise.  Default is
        ``AUGMENT_GAUSSIAN_STD`` from config (0.01).
    seed : int or None, optional
        Random seed for reproducibility.  When *None*, results are
        non-deterministic.

    Returns
    -------
    X_out : ndarray, shape (2 * n_epochs, n_channels, n_timepoints)
        Original epochs followed by noisy copies.
    y_out : ndarray, shape (2 * n_epochs,)
        Labels doubled (original + augmented).

    Notes
    -----
    Apply **only** to training data to prevent data leakage.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=std, size=X.shape)
    X_noisy = X + noise

    logger.info(
        "Gaussian-noise augmentation: %d -> %d epochs (std=%.4f)",
        len(X), len(X) * 2, std,
    )

    X_out = np.concatenate([X, X_noisy], axis=0)
    y_out = np.concatenate([y, y], axis=0)
    return X_out, y_out


def augment_temporal_jitter(
    X: np.ndarray,
    y: np.ndarray,
    max_shift_samples: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly shift each epoch along the time axis.

    Each epoch is independently shifted by a random integer number of
    samples drawn uniformly from ``[-max_shift_samples, +max_shift_samples]``.
    Regions that fall outside the original time window are padded with the
    nearest edge value (``numpy.pad`` mode ``"edge"``).

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_timepoints)
        EEG epoch array.
    y : ndarray, shape (n_epochs,)
        Label array.
    max_shift_samples : int
        Maximum absolute shift in samples.  For example, with a 160 Hz
        sampling rate and a 50 ms jitter window the value is
        ``int(50 * 160 / 1000) = 8``.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    X_out : ndarray, shape (2 * n_epochs, n_channels, n_timepoints)
        Original epochs followed by jittered copies.
    y_out : ndarray, shape (2 * n_epochs,)
        Labels doubled (original + augmented).

    Notes
    -----
    Apply **only** to training data to prevent data leakage.
    """
    rng = np.random.default_rng(seed)
    n_epochs, n_channels, n_timepoints = X.shape

    shifts = rng.integers(
        -max_shift_samples, max_shift_samples + 1, size=n_epochs,
    )

    X_jittered = np.empty_like(X)
    for i, shift in enumerate(shifts):
        if shift == 0:
            X_jittered[i] = X[i]
        elif shift > 0:
            # Shift signal to the right -> pad left edge
            X_jittered[i, :, shift:] = X[i, :, :n_timepoints - shift]
            X_jittered[i, :, :shift] = X[i, :, 0:1]  # edge pad
        else:
            # Shift signal to the left -> pad right edge
            abs_shift = -shift
            X_jittered[i, :, :n_timepoints - abs_shift] = X[i, :, abs_shift:]
            X_jittered[i, :, n_timepoints - abs_shift:] = X[i, :, -1:]  # edge pad

    logger.info(
        "Temporal-jitter augmentation: %d -> %d epochs "
        "(max_shift=%d samples)",
        n_epochs, n_epochs * 2, max_shift_samples,
    )

    X_out = np.concatenate([X, X_jittered], axis=0)
    y_out = np.concatenate([y, y], axis=0)
    return X_out, y_out


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def apply_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    gaussian_std: float = AUGMENT_GAUSSIAN_STD,
    jitter_samples: int = int(AUGMENT_TEMPORAL_JITTER_MS * SAMPLING_RATE / 1000),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply all augmentation strategies and return the combined dataset.

    The returned arrays contain the **original** epochs plus one copy from
    each augmentation (Gaussian noise and temporal jitter), tripling the
    effective training set size.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_timepoints)
        EEG epoch array.
    y : ndarray, shape (n_epochs,)
        Label array.
    gaussian_std : float, optional
        Standard deviation for Gaussian-noise augmentation.  Defaults to
        ``AUGMENT_GAUSSIAN_STD`` (0.01).
    jitter_samples : int, optional
        Maximum absolute shift in samples for temporal-jitter augmentation.
        Defaults to ``int(AUGMENT_TEMPORAL_JITTER_MS * SAMPLING_RATE / 1000)``
        which equals 8 samples at 160 Hz / 50 ms.
    seed : int or None, optional
        Base random seed.  Derived seeds are used for each augmentation to
        ensure independent but reproducible noise.

    Returns
    -------
    X_out : ndarray, shape (3 * n_epochs, n_channels, n_timepoints)
        Original + Gaussian-augmented + jitter-augmented epochs.
    y_out : ndarray, shape (3 * n_epochs,)
        Corresponding labels.

    Notes
    -----
    Apply **only** to training data to prevent data leakage.
    """
    n_orig = len(X)

    # Derive independent seeds so augmentations are reproducible yet
    # uncorrelated with each other.
    if seed is not None:
        seed_gaussian = seed
        seed_jitter = seed + 1
    else:
        seed_gaussian = None
        seed_jitter = None

    # --- Gaussian noise (returns original + noisy) ---
    X_gauss_full, _ = augment_gaussian_noise(
        X, y, std=gaussian_std, seed=seed_gaussian,
    )
    X_gauss_aug = X_gauss_full[n_orig:]  # only the augmented portion
    y_gauss_aug = y.copy()

    # --- Temporal jitter (returns original + jittered) ---
    X_jitter_full, _ = augment_temporal_jitter(
        X, y, max_shift_samples=jitter_samples, seed=seed_jitter,
    )
    X_jitter_aug = X_jitter_full[n_orig:]  # only the augmented portion
    y_jitter_aug = y.copy()

    # --- Combine: original + gaussian-augmented + jitter-augmented ---
    X_out = np.concatenate([X, X_gauss_aug, X_jitter_aug], axis=0)
    y_out = np.concatenate([y, y_gauss_aug, y_jitter_aug], axis=0)

    logger.info(
        "Combined augmentation: %d -> %d epochs (x3)",
        n_orig, len(X_out),
    )

    return X_out, y_out
