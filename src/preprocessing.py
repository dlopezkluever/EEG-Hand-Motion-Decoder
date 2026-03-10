"""Signal preprocessing — filtering, epoching, ICA artifact rejection."""

import logging

import mne
import numpy as np

from src.config import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    BASELINE,
    EPOCH_TMAX,
    EPOCH_TMIN,
    EVENT_ID,
    ICA_EOG_THRESHOLD,
    ICA_METHOD,
    ICA_MUSCLE_THRESHOLD,
    ICA_N_COMPONENTS,
    ICA_RANDOM_STATE,
    NOTCH_FREQS,
    REJECT_THRESHOLD,
    REJECT_WARN_RATIO,
    USE_ICA,
)

logger = logging.getLogger(__name__)


def apply_filters(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply bandpass filter, notch filter, and common average reference.

    Operates in-place on *raw* and returns it for chaining convenience.
    """
    raw_copy = raw.copy()

    logger.info(
        "Applying filters: bandpass %.1f–%.1f Hz, notch %s Hz, FIR",
        BANDPASS_LOW,
        BANDPASS_HIGH,
        NOTCH_FREQS,
    )

    shape_before = raw_copy.get_data().shape

    # 1. Bandpass FIR filter
    raw_copy.filter(
        l_freq=BANDPASS_LOW,
        h_freq=BANDPASS_HIGH,
        method="fir",
        verbose=False,
    )

    # 2. Notch filter — only apply frequencies below Nyquist
    nyquist = raw_copy.info["sfreq"] / 2.0
    valid_notch = [f for f in NOTCH_FREQS if f < nyquist]
    if valid_notch:
        raw_copy.notch_filter(
            freqs=valid_notch,
            method="fir",
            verbose=False,
        )
    else:
        logger.info(
            "Skipping notch filter: all notch freqs %s >= Nyquist %.1f Hz",
            NOTCH_FREQS,
            nyquist,
        )

    # 3. Common average reference (CAR)
    raw_copy.set_eeg_reference("average", projection=True, verbose=False)
    raw_copy.apply_proj(verbose=False)

    shape_after = raw_copy.get_data().shape
    data = raw_copy.get_data()

    assert shape_before == shape_after, "Filter changed data shape!"
    assert not np.any(np.isnan(data)), "NaN values found after filtering!"

    logger.info(
        "Filtering complete: shape before=%s  after=%s  channels=%d",
        shape_before,
        shape_after,
        len(raw_copy.ch_names),
    )

    return raw_copy


def apply_ica(
    raw: mne.io.Raw,
    n_components: int = ICA_N_COMPONENTS,
    method: str = ICA_METHOD,
    eog_threshold: float = ICA_EOG_THRESHOLD,
    muscle_threshold: float = ICA_MUSCLE_THRESHOLD,
) -> tuple[mne.io.Raw, dict]:
    """Apply ICA-based artifact removal to filtered raw data.

    Auto-detects and excludes EOG (eye blink) and EMG (muscle) components.
    Operates on a copy and returns the cleaned data.

    Parameters
    ----------
    raw : mne.io.Raw
        Filtered raw EEG data.
    n_components : int
        Number of ICA components to estimate.
    method : str
        ICA decomposition algorithm ('fastica', 'infomax', 'picard').
    eog_threshold : float
        Correlation threshold for identifying EOG components.
    muscle_threshold : float
        Z-score threshold for identifying muscle artifact components.

    Returns
    -------
    raw_clean : mne.io.Raw
        ICA-cleaned raw data.
    ica_info : dict
        Logging info: components excluded, method used, etc.
    """
    raw_copy = raw.copy()

    # Fit ICA
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=ICA_RANDOM_STATE,
        max_iter="auto",
    )
    ica.fit(raw_copy, verbose=False)

    logger.info(
        "ICA fitted: method=%s, n_components=%d, n_iter=%d",
        method, ica.n_components_, getattr(ica, "n_iter_", -1),
    )

    exclude_idx = []
    component_labels = {}

    # --- Detect EOG-like components ---
    # PhysioNet EEGBCI has no dedicated EOG channel, so we use frontal
    # channels (Fp1, Fp2) as surrogate EOG references.
    eog_ch_names = []
    for ch in raw_copy.ch_names:
        clean = ch.replace(".", "").upper()
        if clean in ("FP1", "FP2", "FPZ"):
            eog_ch_names.append(ch)

    if eog_ch_names:
        for eog_ch in eog_ch_names:
            try:
                eog_indices, eog_scores = ica.find_bads_eog(
                    raw_copy, ch_name=eog_ch, threshold=eog_threshold, verbose=False,
                )
                for idx in eog_indices:
                    if idx not in exclude_idx:
                        exclude_idx.append(idx)
                        component_labels[idx] = f"EOG (via {eog_ch})"
            except Exception as exc:
                logger.debug("EOG detection with %s failed: %s", eog_ch, exc)
    else:
        # Fallback: correlate with frontal channels using the raw data
        logger.info("No Fp1/Fp2/Fpz channels found; skipping EOG correlation detection.")

    # --- Detect muscle artifact components ---
    # Use high-frequency power (> 30 Hz) as a proxy for muscle artifacts.
    try:
        muscle_indices, muscle_scores = ica.find_bads_muscle(
            raw_copy, threshold=muscle_threshold, verbose=False,
        )
        for idx in muscle_indices:
            if idx not in exclude_idx:
                exclude_idx.append(idx)
                component_labels[idx] = "Muscle"
    except Exception as exc:
        logger.debug("Muscle artifact detection failed: %s", exc)

    # Apply exclusion
    ica.exclude = exclude_idx
    raw_clean = ica.apply(raw_copy, verbose=False)

    ica_info = {
        "method": method,
        "n_components_fitted": ica.n_components_,
        "n_excluded": len(exclude_idx),
        "excluded_indices": exclude_idx,
        "component_labels": {str(k): v for k, v in component_labels.items()},
    }

    logger.info(
        "ICA artifact removal: excluded %d components %s",
        len(exclude_idx),
        [(idx, component_labels.get(idx, "unknown")) for idx in exclude_idx],
    )

    return raw_clean, ica_info


def extract_epochs(
    raw: mne.io.Raw,
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
    reject_threshold: float = REJECT_THRESHOLD,
) -> mne.Epochs:
    """Extract and return epochs from filtered raw data.

    Reads event annotations, maps T1 -> left (1), T2 -> right (2),
    applies baseline correction and artifact rejection.
    """
    # Extract events from annotations
    events, event_id_full = mne.events_from_annotations(raw, verbose=False)

    # Map annotation labels to our left/right event IDs.
    # PhysioNet uses T1 and T2 annotations. After standardisation the
    # annotations may appear as 'T1'/'T2' or as numeric strings like
    # '1'/'2'/'3'.  We need to figure out which keys in `event_id_full`
    # correspond to T1 (left) and T2 (right).
    mapping = {}
    for key, code in event_id_full.items():
        low = key.strip().lower()
        if low in ("t1", "1"):
            # In PhysioNet EEGBCI runs 3 & 7 T1 = left fist
            continue  # we remap below
        if low in ("t2", "2"):
            continue

    # Build the event_id mapping that MNE expects.
    # After mne.events_from_annotations, the returned event_id_full maps
    # description strings to integer codes. We need to select only the
    # T1 and T2 events.
    target_event_id = {}
    for key, code in event_id_full.items():
        low = key.strip().lower()
        if low in ("t1",):
            target_event_id["left"] = code
        elif low in ("t2",):
            target_event_id["right"] = code

    # PhysioNet annotations after standardize() use 'T0'=rest, 'T1'=left, 'T2'=right.
    # But sometimes the raw annotations come as '1', '2', '3'.
    # Handle that case:
    if not target_event_id:
        # Try numeric keys: typically '2' -> T1 (left), '3' -> T2 (right)
        # in the PhysioNet motor imagery runs.
        for key, code in event_id_full.items():
            if key.strip() == "2":
                target_event_id["left"] = code
            elif key.strip() == "3":
                target_event_id["right"] = code

    if len(target_event_id) < 2:
        raise ValueError(
            f"Could not find left/right events. Available: {event_id_full}"
        )

    logger.info("Event mapping: %s (from raw annotations: %s)", target_event_id, event_id_full)

    # Create rejection dict
    reject = {"eeg": reject_threshold} if reject_threshold else None

    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=BASELINE,
        reject=reject,
        preload=True,
        verbose=False,
    )

    # Log epoch statistics
    n_total = len(events[np.isin(events[:, 2], list(target_event_id.values()))])
    n_kept = len(epochs)
    n_rejected = n_total - n_kept
    reject_ratio = n_rejected / n_total if n_total > 0 else 0.0

    n_left = len(epochs["left"])
    n_right = len(epochs["right"])

    logger.info(
        "Epochs: total=%d  kept=%d  rejected=%d (%.1f%%)  "
        "left=%d  right=%d",
        n_total,
        n_kept,
        n_rejected,
        reject_ratio * 100,
        n_left,
        n_right,
    )

    if reject_ratio > REJECT_WARN_RATIO:
        logger.warning(
            "HIGH REJECTION RATE: %.1f%% of epochs rejected (threshold=%.0f%%)",
            reject_ratio * 100,
            REJECT_WARN_RATIO * 100,
        )

    return epochs


def pick_roi_channels(
    epochs: mne.Epochs,
    roi_channels: list[str] | None = None,
) -> mne.Epochs:
    """Restrict epochs to motor cortex ROI channels.

    Matches channel names case-insensitively and handles dotted names
    (e.g., 'C3.' -> 'C3').

    Parameters
    ----------
    epochs : mne.Epochs
        Full-channel epochs.
    roi_channels : list of str, optional
        Channel names to keep. Defaults to MOTOR_ROI_CHANNELS from config.

    Returns
    -------
    mne.Epochs with only the matched ROI channels.
    """
    from src.config import MOTOR_ROI_CHANNELS
    roi_channels = roi_channels or MOTOR_ROI_CHANNELS

    # Build a mapping from clean name -> actual name in epochs
    roi_upper = {ch.upper() for ch in roi_channels}
    matched = []
    for ch in epochs.ch_names:
        clean = ch.replace(".", "").upper()
        if clean in roi_upper:
            matched.append(ch)

    if not matched:
        logger.warning(
            "No ROI channels matched from %s in %s — returning all channels.",
            roi_channels, epochs.ch_names[:10],
        )
        return epochs

    logger.info(
        "ROI channel selection: %d/%d channels -> %s",
        len(matched), len(epochs.ch_names), matched,
    )

    return epochs.copy().pick(matched)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    from src.data_loader import download_data, load_raw

    logger.info("=== Preprocessing — Smoke Test ===")

    download_data(subjects=[1], runs=[3, 7])
    raw = load_raw(subject=1, runs=[3, 7])

    filtered = apply_filters(raw)
    print(f"Filtered data shape: {filtered.get_data().shape}")

    epochs = extract_epochs(filtered)
    print(f"Epochs shape: {epochs.get_data().shape}")
    print(f"  Left epochs:  {len(epochs['left'])}")
    print(f"  Right epochs: {len(epochs['right'])}")

    logger.info("=== Smoke test complete ===")
