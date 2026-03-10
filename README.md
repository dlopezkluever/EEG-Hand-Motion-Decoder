# EEG Motor Imagery Brain-Computer Interface

A complete EEG-based Brain-Computer Interface (BCI) pipeline that decodes left vs. right hand motor imagery from the PhysioNet EEGBCI dataset. Implements signal preprocessing, multiple feature extraction strategies, and three classification approaches — evaluated with rigorous cross-validation on up to 109 subjects.

## Architecture

```
Raw EEG (64ch, 160Hz)
  │
  ├─ Bandpass 1-40Hz + Notch 60/120Hz + CAR
  ├─ [Optional] ICA Artifact Removal
  │
  ├─ Epoch Extraction (T1→left, T2→right, -0.5s to 4.0s)
  ├─ [Optional] Motor Cortex ROI Channel Selection (7ch)
  │
  ├─ Feature Pathway A: PSD Band Power (384 features)
  │     └─ Logistic Regression (10-fold CV)
  │
  ├─ Feature Pathway B: CSP (4 components)
  │     └─ Logistic Regression (10-fold CV)
  │
  └─ Feature Pathway C: Raw Epochs (z-scored)
        └─ EEGNet CNN (10-fold CV, early stopping)
```

## Quick Start

### 1. Setup Environment

```bash
# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate eeg-bci

# Option B: pip
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Process 3 subjects (quick test)
python train.py --subjects 1 2 3

# Process with all enhancements
python train.py --subjects 1 2 3 --ica --tune --augment

# Full 109-subject run
python train.py

# LOSO cross-validation
python train.py --loso --subjects 1 2 3 4 5 --loso-models LR_PSD LR_CSP
```

### 3. Evaluate Saved Models

```bash
# Re-evaluate a saved EEGNet model
python evaluate.py --model-path outputs/models/eegnet_fold01.pt --data-subjects 1 2

# Regenerate report from saved results
python evaluate.py --results-dir outputs/results/
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--subjects 1 2 3` | Subject IDs to process (default: all 109) |
| `--tune` | Enable LR hyperparameter tuning via GridSearchCV |
| `--augment` | Enable data augmentation for EEGNet (Gaussian noise + temporal jitter) |
| `--ica` | Enable ICA-based artifact removal (auto-detects EOG/EMG components) |
| `--roi` | Restrict to 7 motor cortex ROI channels (C3, C4, Cz, FC3, FC4, CP3, CP4) |
| `--mlflow` | Enable MLflow experiment tracking |
| `--loso` | Run Leave-One-Subject-Out cross-validation |
| `--loso-models` | Models for LOSO mode: `LR_PSD`, `LR_CSP`, `EEGNet_Raw` |
| `--no-cache` | Force reprocessing (skip result caching) |
| `--cv-folds N` | Override number of CV folds (default: 10) |
| `--output-dir DIR` | Override output directory |
| `--seed N` | Override random seed (default: 42) |

## Project Structure

```
BCI/
├── train.py                 # Main training pipeline entrypoint
├── evaluate.py              # Standalone evaluation script
├── src/
│   ├── config.py            # All configurable parameters
│   ├── data_loader.py       # PhysioNet EEGBCI download & loading
│   ├── preprocessing.py     # Filtering, ICA, epoching, ROI selection
│   ├── features.py          # PSD, CSP, and raw feature extraction
│   ├── evaluate.py          # Metrics, LOSO CV, report generation
│   ├── visualize.py         # All plotting functions
│   ├── augmentation.py      # Data augmentation (Gaussian + jitter)
│   ├── tracking.py          # MLflow experiment tracking integration
│   └── models/
│       ├── logistic.py      # Logistic Regression + GridSearchCV tuning
│       └── eegnet.py        # EEGNet CNN (PyTorch)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   └── 04_model_comparison.ipynb
├── tests/                   # Unit tests (pytest)
├── outputs/
│   ├── figures/             # PNG + PDF plots (300 DPI)
│   ├── models/              # Saved model weights
│   └── results/             # JSON/CSV metrics & reports
├── requirements.txt
└── environment.yml
```

## Models

### Logistic Regression (Baseline)
- Solver: L-BFGS, balanced class weights, max 1000 iterations
- Hyperparameter tuning: Grid search over C = [0.001, 0.01, 0.1, 1.0, 10.0]
- Two feature inputs: PSD band power (384-dim) and CSP (4-dim)

### EEGNet (CNN)
- Architecture from [Lawhern et al., 2018](https://doi.org/10.1088/1741-2552/aace8c)
- Temporal Conv → Depthwise Conv → Separable Conv → Dense
- Parameters: F1=8, F2=16, D=2, dropout=0.5
- Training: Adam (lr=0.001), cross-entropy, early stopping (patience=30)
- Optional data augmentation: Gaussian noise (σ=0.01) + temporal jitter (±50ms)

## Feature Extraction

| Pathway | Method | Output Shape | Used By |
|---------|--------|-------------|---------|
| A | PSD Band Power (Welch, 6 bands × 64 channels) | (n, 384) | LR |
| B | Common Spatial Patterns (4 components, log-variance) | (n, 4) | LR |
| C | Raw epochs (per-channel z-score) | (n, 64, 721) | EEGNet |

**Frequency Bands:** Delta (1-4 Hz), Theta (4-8), Mu (8-12), Low Beta (13-20), High Beta (20-30), Low Gamma (30-40)

## Preprocessing Pipeline

1. **Bandpass filter**: 1-40 Hz (FIR)
2. **Notch filter**: 60 Hz + 120 Hz (power line noise)
3. **Common Average Reference** (CAR)
4. **[Optional] ICA Artifact Removal**: Auto-detects EOG (via Fp1/Fp2 correlation) and EMG (high-frequency power) components
5. **Epoch extraction**: -0.5s to 4.0s around event onset, baseline correction
6. **[Optional] ROI Channel Selection**: Restrict to 7 motor cortex channels

## Evaluation

- **Within-subject**: 10-fold stratified cross-validation
- **Cross-subject**: Leave-One-Subject-Out (LOSO)
- **Metrics**: Accuracy, Precision, Recall, F1 (per-class + macro), AUC-ROC, Cohen's Kappa
- **Statistical tests**: Paired t-tests between model accuracies (p < 0.05)
- **Visualizations**: Confusion matrices, ROC curves, topographic maps, PSD comparisons, training curves, feature importance

For detailed results, figures, and test logs see **[RESULTS.md](RESULTS.md)**.

## Experiment Tracking

MLflow integration for comparing runs across configurations:

```bash
# Enable tracking
python train.py --mlflow --subjects 1 2 3

# View results
mlflow ui
# Open http://localhost:5000
```

Logged artifacts include hyperparameters, per-model metrics, figures, and result CSVs.

## Dataset

**PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI)**
- 109 subjects, 64-channel EEG, 160 Hz sampling rate
- Runs 3 & 7: Motor imagery — left fist (T1) vs. right fist (T2)
- Downloaded automatically via `mne.datasets.eegbci`

Reference: Schalk, G., McFarland, D.J., Hinterberger, T., et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. *IEEE TBME*, 51(6), 1034-1043.

## References

- Lawhern, V.J., Solon, A.J., Waytowich, N.R., et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces. *Journal of Neural Engineering*, 15(5), 056013.
- Blankertz, B., Tomioka, R., Lemm, S., et al. (2008). Optimizing Spatial Filters for Robust EEG Single-Trial Analysis. *IEEE Signal Processing Magazine*, 25(1), 41-56.
- Pfurtscheller, G. & Lopes da Silva, F.H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11), 1842-1857.

## License

This project is for educational and research purposes.
