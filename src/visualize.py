"""Visualization — topomaps, PSD plots, confusion matrices, training curves."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

import mne

from src.config import FIGURES_DIR, FREQ_BANDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, filepath_stem: Path) -> list[Path]:
    """Save a figure as both PNG (300 DPI) and PDF. Returns list of paths."""
    filepath_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in (".png", ".pdf"):
        p = filepath_stem.with_suffix(ext)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    logger.info("Saved figure: %s (.png + .pdf)", filepath_stem.name)
    return paths


# ---------------------------------------------------------------------------
# 3.1  EEG Signal Visualizations
# ---------------------------------------------------------------------------

def plot_topomap(
    epochs: mne.Epochs,
    subject_id: int = 0,
    output_dir: Path | None = None,
) -> list[Path]:
    """Generate scalp topographic maps of ERD (left vs right MI).

    Computes the average power in the mu band (8-12 Hz) for each condition
    and plots the difference (right − left) as a topographic map, highlighting
    event-related desynchronisation patterns.
    """
    output_dir = output_dir or FIGURES_DIR

    left_epochs = epochs["left"]
    right_epochs = epochs["right"]

    # Compute PSD for mu band (8-12 Hz)
    mu_low, mu_high = FREQ_BANDS["mu"]

    def _band_power(ep):
        psds, freqs = mne.time_frequency.psd_array_welch(
            ep.get_data(), sfreq=ep.info["sfreq"],
            fmin=mu_low, fmax=mu_high, n_fft=320, verbose=False,
        )
        return psds.mean(axis=2).mean(axis=0)  # (n_channels,)

    power_left = _band_power(left_epochs)
    power_right = _band_power(right_epochs)

    # ERD: relative change from left to right
    erd = (power_right - power_left) / (power_left + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    mne.viz.plot_topomap(power_left, epochs.info, axes=axes[0], show=False)
    axes[0].set_title("Left MI — Mu Power")

    mne.viz.plot_topomap(power_right, epochs.info, axes=axes[1], show=False)
    axes[1].set_title("Right MI — Mu Power")

    mne.viz.plot_topomap(erd, epochs.info, axes=axes[2], show=False)
    axes[2].set_title("ERD (Right − Left)")

    fig.suptitle(f"Topographic Maps — Subject {subject_id:03d}", fontsize=13)
    fig.tight_layout()

    return _save_fig(fig, output_dir / f"topomap_subject{subject_id:03d}")


def plot_psd_comparison(
    epochs: mne.Epochs,
    subject_id: int = 0,
    output_dir: Path | None = None,
) -> list[Path]:
    """Compare left vs right MI power spectra at C3 and C4 electrodes."""
    output_dir = output_dir or FIGURES_DIR
    ch_names = epochs.ch_names

    # Find C3 and C4 indices (case-insensitive, handle dotted names)
    target_channels = {"C3": None, "C4": None}
    for idx, name in enumerate(ch_names):
        clean = name.replace(".", "").upper()
        if clean == "C3":
            target_channels["C3"] = idx
        elif clean == "C4":
            target_channels["C4"] = idx

    if None in target_channels.values():
        logger.warning("C3/C4 channels not found in %s — using first two channels", ch_names[:5])
        target_channels = {"Ch0": 0, "Ch1": 1}

    left_data = epochs["left"].get_data()
    right_data = epochs["right"].get_data()
    sfreq = epochs.info["sfreq"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (ch_label, ch_idx) in zip(axes, target_channels.items()):
        # Compute PSD per condition
        psd_left, freqs = mne.time_frequency.psd_array_welch(
            left_data[:, ch_idx:ch_idx+1, :], sfreq=sfreq,
            fmin=1, fmax=40, n_fft=320, verbose=False,
        )
        psd_right, _ = mne.time_frequency.psd_array_welch(
            right_data[:, ch_idx:ch_idx+1, :], sfreq=sfreq,
            fmin=1, fmax=40, n_fft=320, verbose=False,
        )

        mean_left = psd_left.mean(axis=0).squeeze()
        mean_right = psd_right.mean(axis=0).squeeze()

        ax.semilogy(freqs, mean_left, label="Left MI", color="tab:blue", linewidth=1.5)
        ax.semilogy(freqs, mean_right, label="Right MI", color="tab:red", linewidth=1.5)
        ax.axvspan(8, 12, alpha=0.15, color="green", label="Mu band")
        ax.axvspan(13, 30, alpha=0.10, color="orange", label="Beta band")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density")
        ax.set_title(f"Electrode {ch_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"PSD Comparison — Subject {subject_id:03d}", fontsize=13)
    fig.tight_layout()

    return _save_fig(fig, output_dir / f"psd_comparison_subject{subject_id:03d}")


def plot_butterfly(
    epochs: mne.Epochs,
    subject_id: int = 0,
    output_dir: Path | None = None,
) -> list[Path]:
    """Butterfly plot for epoch quality visual inspection.

    Plots all channels overlaid for each condition (left/right MI).
    """
    output_dir = output_dir or FIGURES_DIR

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    times = epochs.times

    for ax, condition, color in zip(axes, ["left", "right"], ["tab:blue", "tab:red"]):
        data = epochs[condition].get_data()  # (n_epochs, n_ch, n_times)
        # Plot mean across epochs for each channel
        mean_data = data.mean(axis=0)  # (n_ch, n_times)
        for ch in range(mean_data.shape[0]):
            ax.plot(times, mean_data[ch] * 1e6, color=color, alpha=0.15, linewidth=0.5)
        # Plot grand average
        grand_avg = mean_data.mean(axis=0)
        ax.plot(times, grand_avg * 1e6, color="black", linewidth=2, label="Grand avg")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1, label="Onset")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title(f"{condition.title()} MI")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Butterfly Plot — Subject {subject_id:03d}", fontsize=13)
    fig.tight_layout()

    return _save_fig(fig, output_dir / f"butterfly_subject{subject_id:03d}")


def generate_all_signal_figures(
    epochs: mne.Epochs,
    subject_id: int = 0,
    output_dir: Path | None = None,
) -> list[Path]:
    """Convenience function: generate all EEG signal visualizations."""
    all_paths = []
    all_paths.extend(plot_topomap(epochs, subject_id, output_dir))
    all_paths.extend(plot_psd_comparison(epochs, subject_id, output_dir))
    all_paths.extend(plot_butterfly(epochs, subject_id, output_dir))
    logger.info("Generated %d signal figures for Subject %03d", len(all_paths), subject_id)
    return all_paths


# ---------------------------------------------------------------------------
# 3.2  Model Performance Visualizations
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    subject_id: int = 0,
    output_dir: Path | None = None,
) -> list[Path]:
    """Generate and save a confusion matrix heatmap."""
    output_dir = output_dir or FIGURES_DIR

    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Left", "Right"],
        yticklabels=["Left", "Right"],
        ax=ax,
    )
    # Add percentages as secondary annotations
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f"({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=9, color="gray")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name} (Subject {subject_id:03d})")

    return _save_fig(fig, output_dir / f"confusion_{model_name}_subject{subject_id:03d}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    subject_id: int = 0,
    output_dir: Path | None = None,
) -> list[Path]:
    """Plot ROC curve with AUC annotation."""
    output_dir = output_dir or FIGURES_DIR

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="tab:blue", linewidth=2,
            label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name} (Subject {subject_id:03d})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    return _save_fig(fig, output_dir / f"roc_{model_name}_subject{subject_id:03d}")


def plot_training_curves(
    history: dict,
    model_name: str = "EEGNet",
    subject_id: int = 0,
    output_dir: Path | None = None,
) -> list[Path]:
    """Plot training loss, validation loss, and validation accuracy over epochs."""
    output_dir = output_dir or FIGURES_DIR

    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curves
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="tab:blue")
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="tab:red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs_range, history["val_accuracy"], label="Val Accuracy", color="tab:green")
    best_epoch = np.argmax(history["val_accuracy"]) + 1
    best_acc = max(history["val_accuracy"])
    ax2.axhline(best_acc, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(best_epoch, color="gray", linestyle=":", alpha=0.5)
    ax2.scatter([best_epoch], [best_acc], color="tab:red", zorder=5,
                label=f"Best: {best_acc:.3f} @ epoch {best_epoch}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} Training Curves — Subject {subject_id:03d}", fontsize=13)
    fig.tight_layout()

    return _save_fig(fig, output_dir / f"training_curves_{model_name}_subject{subject_id:03d}")


def plot_feature_importance(
    coef: np.ndarray,
    feature_names: list[str] | None = None,
    model_name: str = "LR_PSD",
    subject_id: int = 0,
    n_top: int = 20,
    output_dir: Path | None = None,
) -> list[Path]:
    """Map logistic regression coefficients to electrode/band combinations.

    Parameters
    ----------
    coef : ndarray of shape (n_features,)
        Model coefficients (e.g. from clf.coef_[0]).
    feature_names : list of str, optional
        Names for each feature. If None, auto-generates from bands × channels.
    n_top : int
        Number of top features to display.
    """
    output_dir = output_dir or FIGURES_DIR

    coef = np.asarray(coef).ravel()

    if feature_names is None:
        bands = list(FREQ_BANDS.keys())
        n_bands = len(bands)
        n_channels = len(coef) // n_bands
        feature_names = [f"Ch{ch}_{band}" for ch in range(n_channels) for band in bands]

    # Get top features by absolute coefficient value
    top_idx = np.argsort(np.abs(coef))[-n_top:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_coefs = coef[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:red" if c > 0 else "tab:blue" for c in top_coefs]
    ax.barh(range(len(top_coefs)), top_coefs, color=colors)
    ax.set_yticks(range(len(top_coefs)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Coefficient Value")
    ax.set_title(f"Top {n_top} Feature Importances — {model_name} (Subject {subject_id:03d})")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()

    return _save_fig(fig, output_dir / f"feature_importance_{model_name}_subject{subject_id:03d}")


def plot_subject_accuracy_bar(
    results_df,
    output_dir: Path | None = None,
) -> list[Path]:
    """Per-subject accuracy comparison bar chart across models."""
    output_dir = output_dir or FIGURES_DIR

    import pandas as pd
    df = results_df if isinstance(results_df, pd.DataFrame) else pd.DataFrame(results_df)

    model_cols = {
        "LR + PSD": "lr_psd_accuracy",
        "LR + CSP": "lr_csp_accuracy",
        "EEGNet + Raw": "eegnet_accuracy",
    }
    available = {k: v for k, v in model_cols.items() if v in df.columns}

    if not available:
        logger.warning("No accuracy columns found in results DataFrame")
        return []

    subjects = df["subject"].values
    x = np.arange(len(subjects))
    width = 0.8 / len(available)

    fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 1.2), 6))

    colors = ["tab:blue", "tab:green", "tab:red"]
    for i, (label, col) in enumerate(available.items()):
        offset = (i - len(available) / 2 + 0.5) * width
        ax.bar(x + offset, df[col].values, width, label=label, color=colors[i], alpha=0.85)

    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Subject Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{int(s):03d}" for s in subjects], fontsize=8, rotation=45)
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    return _save_fig(fig, output_dir / "subject_accuracy_comparison")


def plot_multi_roc(
    roc_data: list[dict],
    subject_id: int = 0,
    output_dir: Path | None = None,
) -> list[Path]:
    """Overlay ROC curves for multiple models on one plot.

    Parameters
    ----------
    roc_data : list of dict
        Each dict has keys: model_name, y_true, y_prob.
    """
    output_dir = output_dir or FIGURES_DIR

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["tab:blue", "tab:green", "tab:red", "tab:purple"]

    for i, entry in enumerate(roc_data):
        fpr, tpr, _ = roc_curve(entry["y_true"], entry["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f"{entry['model_name']} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — Subject {subject_id:03d}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    return _save_fig(fig, output_dir / f"roc_comparison_subject{subject_id:03d}")
