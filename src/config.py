"""Central configuration for the EEG BCI pipeline."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "eegbci"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
SUBJECTS = list(range(1, 110))         # Full 109-subject dataset
CACHE_RESULTS = True                   # Skip subjects with existing results
RUNS = [3, 7]                          # Motor imagery left/right fist
SAMPLING_RATE = 160                    # Hz (PhysioNet EEGBCI)
N_CHANNELS = 64

# ---------------------------------------------------------------------------
# Preprocessing — Filtering
# ---------------------------------------------------------------------------
BANDPASS_LOW = 1.0                     # Hz — removes DC drift
BANDPASS_HIGH = 40.0                   # Hz — preserves mu & beta bands
NOTCH_FREQS = [60.0, 120.0]           # US power-line noise + 1st harmonic

# ---------------------------------------------------------------------------
# Preprocessing — Epochs
# ---------------------------------------------------------------------------
EPOCH_TMIN = -0.5                      # seconds before event onset
EPOCH_TMAX = 4.0                       # seconds after event onset
BASELINE = (None, 0)                   # mean-subtract over pre-stimulus window
REJECT_THRESHOLD = None                # Set to 150e-6 when ICA is enabled
REJECT_WARN_RATIO = 0.30              # flag subject if >30% epochs rejected

# ---------------------------------------------------------------------------
# Preprocessing — ICA Artifact Removal (Phase 5)
# ---------------------------------------------------------------------------
USE_ICA = False                        # Toggle ICA artifact removal on/off
ICA_N_COMPONENTS = 20                  # Number of ICA components to fit
ICA_METHOD = "fastica"                 # ICA decomposition method
ICA_RANDOM_STATE = 42                  # Reproducibility for ICA
ICA_EOG_THRESHOLD = 3.0                # Z-score threshold for EOG detection (find_bads_eog)
ICA_MUSCLE_THRESHOLD = 1.0            # Z-score threshold for muscle artifact detection

EVENT_ID = {"left": 1, "right": 2}    # T1 → left, T2 → right

# ---------------------------------------------------------------------------
# Feature Engineering — PSD
# ---------------------------------------------------------------------------
PSD_FMIN = 1.0
PSD_FMAX = 40.0
PSD_N_FFT = 320                       # 2-second window at 160 Hz
PSD_N_OVERLAP = 160                   # 50% overlap
PSD_WINDOW = "hann"

FREQ_BANDS = {
    "delta":     (1, 4),
    "theta":     (4, 8),
    "mu":        (8, 12),
    "low_beta":  (13, 20),
    "high_beta": (20, 30),
    "low_gamma": (30, 40),
}

# Motor cortex ROI channels (optional subset — Phase 5)
USE_ROI_CHANNELS = False               # Toggle ROI channel selection on/off
MOTOR_ROI_CHANNELS = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4"]

# ---------------------------------------------------------------------------
# Feature Engineering — CSP
# ---------------------------------------------------------------------------
CSP_N_COMPONENTS = 4

# ---------------------------------------------------------------------------
# Model — Logistic Regression
# ---------------------------------------------------------------------------
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 1000
LR_CLASS_WEIGHT = "balanced"
LR_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]

# ---------------------------------------------------------------------------
# Model — EEGNet
# ---------------------------------------------------------------------------
EEGNET_F1 = 8
EEGNET_F2 = 16
EEGNET_D = 2
EEGNET_DROPOUT = 0.5
EEGNET_LR = 0.001
EEGNET_BATCH_SIZE = 32
EEGNET_MAX_EPOCHS = 300
EEGNET_PATIENCE = 30

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_N_FOLDS = 10

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Data augmentation (EEGNet)
# ---------------------------------------------------------------------------
AUGMENT_GAUSSIAN_STD = 0.01
AUGMENT_TEMPORAL_JITTER_MS = 50

# ---------------------------------------------------------------------------
# Experiment Tracking (Phase 5)
# ---------------------------------------------------------------------------
USE_MLFLOW = False                     # Toggle MLflow experiment tracking
MLFLOW_EXPERIMENT_NAME = "EEG-BCI-Pipeline"
MLFLOW_TRACKING_URI = "file:./mlruns"  # Local file-based tracking


def get_config() -> dict:
    """Return all configuration values as a flat dictionary for logging."""
    return {
        k: v
        for k, v in globals().items()
        if k.isupper() and not k.startswith("_")
    }
