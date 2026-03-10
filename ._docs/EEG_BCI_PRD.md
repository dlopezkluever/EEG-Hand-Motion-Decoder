# EEG Hand Movement Decoder — Product Requirements Document
**Foundational Brain-Computer Interface Project**
Version 1.0 · March 2026

> Predict left vs. right hand movement intention from raw EEG signals.

---

## 1. Project Overview

### 1.1 Purpose
This document defines the full product requirements for building an EEG-based Hand Movement Decoder — a foundational Brain-Computer Interface (BCI) system that classifies a subject's intent to move their left or right hand from electroencephalography (EEG) signals. It is the canonical starting point for BCI research and provides a complete, reproducible machine learning pipeline from raw neural signal to actionable prediction.

### 1.2 Background & Motivation
Brain-Computer Interfaces allow direct communication between the brain and external devices without relying on peripheral nervous system or muscular pathways. EEG is the dominant non-invasive modality for BCI development due to its high temporal resolution, low cost, portability, and the availability of well-characterized open datasets.

Motor imagery (MI) — the mental simulation of movement without overt action — produces reliable, measurable neural signatures, specifically modulations of sensorimotor rhythms (mu ~8–12 Hz, beta ~13–30 Hz) over the motor cortex. Decoding these rhythms is the archetypal classification problem in BCI and underpins prosthetics control, assistive technology, neuro-feedback, gaming, and rehabilitative medicine.

### 1.3 Strategic Value
- Establishes a rigorous signal processing and ML pipeline applicable to all future BCI work
- Provides hands-on familiarity with MNE-Python, the standard EEG analysis library
- Produces a benchmarkable accuracy score against published literature on EEGBCI (PhysioNet)
- Serves as a portfolio centerpiece for robotics, medtech, and neurotech opportunities
- Lays the groundwork for real-time decoding with live EEG hardware (OpenBCI, g.tec, etc.)

---

## 2. Goals & Success Criteria

### 2.1 Primary Goal
Build an end-to-end, reproducible pipeline that accurately classifies left-hand vs. right-hand motor imagery from resting EEG using the PhysioNet EEG Motor Movement/Imagery Dataset, achieving accuracy meaningfully above chance (50%) and approaching published state-of-the-art baselines.

### 2.2 Success Metrics

| Metric | Minimum | Target | Stretch |
|---|---|---|---|
| Binary classification accuracy | 65% | 75% | 85% |
| F1-score (macro avg) | 0.60 | 0.72 | 0.82 |
| AUC-ROC | 0.65 | 0.78 | 0.88 |
| Cross-validation folds | 5-fold | 10-fold | LOSO (subject-out) |
| Subjects evaluated | 5 | 20 | 109 (full dataset) |

### 2.3 Non-Goals
- Real-time (live) decoding is out of scope for v1 (offline analysis only)
- Multi-class classification beyond left vs. right hand is not required
- Hardware EEG acquisition is out of scope
- Clinical or medical device certification is not applicable
- Deployment as a web service or API is not required for v1

---

## 3. Dataset Specification

### 3.1 Source Dataset

| Field | Value |
|---|---|
| Dataset Name | EEG Motor Movement/Imagery Dataset |
| Provider | PhysioNet — physionet.org/content/eegmmidb/1.0.0/ |
| Subjects | 109 healthy subjects |
| EEG Channels | 64 channels (10-20 system montage) |
| Sampling Rate | 160 Hz |
| Format | EDF (European Data Format), directly supported by MNE-Python |
| Tasks of Interest | Tasks 3 & 7: Motor imagery — open/close left fist vs. right fist |
| License | Open Data Commons Attribution (ODC-By) — freely usable with attribution |
| Download | Automated via `mne.datasets.eegbci.load_data()` or wget from PhysioNet |

### 3.2 Task Structure
Each subject performed 14 experimental runs. Relevant runs for this project:

- **Run 3:** Motor imagery — open/close left fist (T1) vs. right fist (T2) — 1st session
- **Run 7:** Motor imagery — open/close left fist (T1) vs. right fist (T2) — 2nd session
- **Runs 4 & 8 (optional):** Both fists / both feet — used for multi-class extension

Each run contains 30-second segments with event annotations. Event codes: T0 = rest baseline, T1 = left hand MI, T2 = right hand MI. Target label mapping: T1 → class 0 (left), T2 → class 1 (right).

---

## 4. System Architecture & Pipeline

The system follows a classical offline BCI pipeline with six discrete, testable stages. Each stage has defined inputs, outputs, and validation criteria.

| Stage | Name | Description | Input | Output |
|---|---|---|---|---|
| 1 | Data Acquisition | Download EDF files from PhysioNet via MNE API or direct HTTP | PhysioNet URL | Raw .edf files |
| 2 | Preprocessing | Bandpass filter, notch filter, epoch extraction, artifact rejection | Raw .edf files | Cleaned epochs array |
| 3 | Feature Engineering | PSD extraction, band power, CSP, optionally raw epoch flattening | Epochs array | Feature matrix X, labels y |
| 4 | Model Training | Train logistic regression and/or CNN with cross-validation | Feature matrix | Trained model + metrics |
| 5 | Evaluation | Accuracy, F1, AUC-ROC, confusion matrix, ROC curve | Model + test set | Evaluation report |
| 6 | Visualization | Topographic maps, PSD plots, feature importance, training curves | Raw + model data | Figures + report |

---

## 5. Detailed Functional Requirements

### 5.1 Stage 1 — Data Loading

**Requirements:**
- Support automated download using `mne.datasets.eegbci.load_data(subject, runs)`
- Support batch download for a configurable list of subjects (default: subjects 1–10 for dev, all 109 for full runs)
- Cache downloaded files locally; re-runs must not re-download existing files
- Load EDF files into MNE Raw objects with correct channel names and montage (`standard_1020`)
- Log subject, run ID, file path, and raw data shape for each loaded file

**Configuration Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `SUBJECTS` | `range(1, 11)` | Subject IDs to process |
| `RUNS` | `[3, 7]` | Run indices for left/right MI tasks |
| `DATA_PATH` | `./data/eegbci` | Local cache directory for EDF files |

---

### 5.2 Stage 2 — Signal Preprocessing

**Bandpass Filtering:**
- Apply a zero-phase FIR bandpass filter over 1–40 Hz using `mne.filter.filter_data`
- Filter must be applied before epoching to prevent filter artifacts at epoch boundaries
- Filter length: auto (MNE default), minimum phase: False (zero-phase)
- Rationale: 1 Hz lower bound removes DC drift; 40 Hz upper bound removes high-frequency EMG noise while preserving mu (8–12 Hz) and beta (13–30 Hz) bands

**Notch Filtering:**
- Apply a 60 Hz notch filter (US power line noise) using `mne.notch_filter`
- Also apply notch at 120 Hz (first harmonic) for robustness

**Re-referencing:**
- Apply common average reference (CAR) using `raw.set_eeg_reference('average')`
- CAR removes signals common to all electrodes, improving SNR for cortical sources

**Epoch Extraction:**
- Extract epochs time-locked to motor imagery event onset markers (T1, T2)
- Epoch window: **-0.5 s to +4.0 s** relative to event onset
- Baseline correction: mean subtraction over the -0.5 to 0.0 s pre-stimulus window
- Event IDs: `{'left': 1, 'right': 2}` mapped from annotations T1, T2

**Artifact Rejection:**
- Reject epochs with peak-to-peak amplitude exceeding **150 µV** (EEG) using the `reject` parameter in `mne.Epochs`
- Log number of rejected epochs per subject
- If more than 30% of epochs are rejected for a subject, flag that subject for review

---

### 5.3 Stage 3 — Feature Engineering

Two feature extraction pathways must be implemented and can be run independently or evaluated comparatively.

#### Pathway A: Power Spectral Density (PSD) Features
- Compute PSD using Welch's method via `mne.time_frequency.psd_array_welch`
  - FFT window: 2 seconds, 50% overlap, Hanning window
  - Frequency resolution: 0.5 Hz per bin
- Extract band power by averaging PSD within canonical frequency bands:

| Band | Range | Notes |
|---|---|---|
| Delta | 1–4 Hz | Include for baseline comparison |
| Theta | 4–8 Hz | |
| Mu (Alpha) | 8–12 Hz | **Primary motor imagery band** |
| Low Beta | 13–20 Hz | **Primary motor imagery band** |
| High Beta | 20–30 Hz | |
| Low Gamma | 30–40 Hz | |

- Channel selection: all 64 channels initially; optionally restrict to motor cortex ROI (C3, C4, Cz, FC3, FC4, CP3, CP4) for a 7-channel focused model
- Feature vector per epoch: `(n_channels × n_bands)` → flattened to 1D
- Apply `StandardScaler` normalization after feature extraction

#### Pathway B: Common Spatial Patterns (CSP)
- Apply CSP using `mne.decoding.CSP` or sklearn-compatible wrapper
- Number of CSP components: `n_components = 4` (configurable)
- CSP maximizes variance ratio between classes in the spatial domain
- Output: `(n_epochs × n_components)` feature matrix
- Combine with log-variance feature: `log(variance(CSP-filtered signal))`

#### Pathway C: Raw Epoch Flattening (for CNN only)
- Flatten 3D epoch array `(n_epochs × n_channels × n_timepoints)` for direct CNN input
- No manual feature extraction; CNN learns features from raw signal
- Normalize per channel across the dataset using z-score

---

### 5.4 Stage 4 — Model Training

#### Model A: Logistic Regression (Baseline)
- **Framework:** scikit-learn `LogisticRegression`
- **Solver:** lbfgs with L2 regularization
- **Hyperparameter:** C (inverse regularization strength) — grid search over `[0.001, 0.01, 0.1, 1.0, 10.0]`
- **Input:** PSD band power features (Pathway A) or CSP features (Pathway B)
- **Class weight:** `'balanced'` to handle potential class imbalance
- **Max iterations:** 1000

#### Model B: EEGNet CNN (Primary)
Architecture: EEGNet — a compact, BCI-optimized CNN (Lawhern et al., 2018). Implemented in PyTorch.

| Layer | Type | Output Shape | Notes |
|---|---|---|---|
| 1 | Temporal Conv2D | (F1, C, T/2) | F1=8, kernel=(1,64), padding=same |
| 2 | BatchNorm + ELU | same | Depth multiplier D=2 |
| 3 | Depthwise Conv2D | (F1×D, 1, T/2) | Spatial filter per temporal feature |
| 4 | AvgPool + Dropout | (F1×D, 1, T/8) | Pool=(1,4), dropout=0.5 |
| 5 | Separable Conv2D | (F2, 1, T/8) | F2=16, kernel=(1,16) |
| 6 | AvgPool + Dropout | (F2, 1, T/32) | Pool=(1,8), dropout=0.5 |
| 7 | Flatten + Dense | (2,) | Softmax output, 2 classes |

- **Optimizer:** Adam, lr=0.001, batch size=32, 300 epochs
- **Early stopping:** patience=30 on validation loss
- **Loss:** Cross-entropy
- **Train/val split:** 80/20 within training set
- **Data augmentation (optional):** Gaussian noise (σ=0.01) and temporal jitter (±50 ms)

#### Cross-Validation Strategy
- **Primary:** 10-fold stratified cross-validation on epoch-level data
- **Advanced:** Leave-One-Subject-Out (LOSO) for subject-independent evaluation
- Report mean ± std accuracy across folds for all models

---

### 5.5 Stage 5 — Evaluation & Reporting
- Compute and report: Accuracy, Precision, Recall, F1-score (per class and macro), AUC-ROC, Cohen's Kappa
- Generate confusion matrix heatmap (seaborn) for each model
- Generate ROC curve with AUC annotation for each model
- Generate per-subject accuracy bar chart for LOSO evaluation
- Export all metrics to a JSON summary file and a CSV results table
- Statistical comparison between models using paired t-test (p < 0.05 significance threshold)

### 5.6 Stage 6 — Visualization
- **Topographic maps:** Scalp distribution of ERD during MI using `mne.viz.plot_topomap`
- **PSD plots:** Compare left vs. right MI power spectra at C3 and C4 electrodes
- **Epoch butterfly plot:** Visual quality check of cleaned epochs
- **Feature importance:** Logistic regression coefficient weights mapped to electrode/band
- **Training curves:** Loss and accuracy over epochs for CNN
- All figures saved as high-resolution PNG (300 DPI) and PDF in `outputs/figures/`

---

## 6. Technical Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| Language | Python 3.10+ | Primary language |
| EEG I/O & Processing | MNE-Python >= 1.6 | Load EDF, filter, epoch, ICA, viz, CSP |
| ML Baseline | scikit-learn >= 1.4 | Logistic regression, cross-validation, metrics |
| Deep Learning | PyTorch >= 2.1 | EEGNet CNN implementation |
| Numerical | NumPy, SciPy | Array operations, signal processing utilities |
| Visualization | Matplotlib, Seaborn | Plots, topomaps, confusion matrices |
| Data I/O | Pandas | Results tables, CSV export |
| Notebooks | JupyterLab | Exploratory analysis and figure generation |
| Environment | conda / venv + requirements.txt | Reproducible environment |
| Version Control | Git + GitHub | Source control |
| Experiment Tracking (opt) | MLflow or W&B | Hyperparameter and metric logging |

---

## 7. Repository Structure

```
eeg-bci-decoder/
├── data/                      # Downloaded EDF files (gitignored)
│   └── eegbci/                # PhysioNet EEGBCI cache
├── notebooks/                 # Exploratory Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   └── 04_model_comparison.ipynb
├── src/                       # Core pipeline modules
│   ├── __init__.py
│   ├── config.py              # All configuration constants
│   ├── data_loader.py         # PhysioNet download + MNE loading
│   ├── preprocessing.py       # Filtering, epoching, rejection
│   ├── features.py            # PSD, CSP, raw extraction
│   ├── models/
│   │   ├── logistic.py        # LR + grid search
│   │   └── eegnet.py          # PyTorch EEGNet
│   ├── evaluate.py            # Metrics + cross-validation
│   └── visualize.py           # All plotting functions
├── outputs/
│   ├── figures/               # Saved plots
│   ├── models/                # Serialized trained models
│   └── results/               # JSON + CSV metrics
├── tests/                     # Unit tests
├── train.py                   # Main training entrypoint
├── evaluate.py                # Standalone evaluation script
├── requirements.txt
├── environment.yml            # Conda env spec
└── README.md
```

---

## 8. Implementation Phases & Timeline

| # | Phase | Deliverables | Duration |
|---|---|---|---|
| 1 | Environment & Data | conda env, data download for 5 subjects, EDF loading verified, raw signal plotted | 1–2 days |
| 2 | Preprocessing Pipeline | Bandpass/notch filter applied, epochs extracted, artifact rejection validated, butterfly plots | 2–3 days |
| 3 | Feature Extraction | PSD band power + CSP features, feature matrix shapes confirmed, class separation visible in PSD | 2–3 days |
| 4 | Baseline Model | Logistic regression trained, 10-fold CV reported, confusion matrix + ROC generated | 1–2 days |
| 5 | CNN Model | EEGNet implemented in PyTorch, training loop with early stopping, training curves saved | 3–5 days |
| 6 | Evaluation & Viz | All metrics computed, all figures generated, per-subject results table, final README | 2–3 days |
| 7 | Scale & Polish | Run on all 109 subjects, LOSO cross-validation, model comparison | 3–5 days |

**Total estimated effort: 14–23 focused development days.**

---

## 9. Key Algorithms & Signal Processing Notes

### 9.1 Event-Related Desynchronization (ERD)
Motor imagery suppresses mu (8–12 Hz) and beta (13–30 Hz) oscillations contralaterally — known as ERD. Left hand MI suppresses power over right motor cortex (C4); right hand MI suppresses power over left motor cortex (C3). The power asymmetry between C3 and C4 is the primary discriminative signal. Features should capture this lateralization.

### 9.2 Common Spatial Patterns (CSP)
CSP finds spatial filters **W** such that **W**ᵀ**Σ₁W** is diagonal and **W**ᵀ**Σ₂W = I**. This maximizes the ratio of signal variance between class 1 and class 2, making class differences maximally visible. The first and last n/2 components capture the most discriminative spatial information. CSP is the single most effective feature extraction method for MI-BCI and should be a primary pathway.

### 9.3 Welch PSD Estimation
Welch's method divides the signal into overlapping windows, applies a Hanning taper, computes the periodogram for each window, and averages. This reduces the variance of the spectral estimate at the cost of frequency resolution. For a 4-second epoch at 160 Hz (640 samples), a 2-second window (320 samples) with 50% overlap gives 3 averaged periodograms — yielding 0.5 Hz frequency resolution, sufficient to resolve individual EEG bands.

### 9.4 EEGNet Architecture Rationale
EEGNet uses depthwise convolutions to learn spatial filters (analogous to CSP) and separable convolutions to learn temporal dynamics. Its compact design (< 2K parameters for 64 channels) prevents overfitting on small EEG datasets, generalizes across subjects, and is suitable for deployment on resource-constrained hardware. It is the recommended CNN architecture for EEG classification tasks with fewer than 10,000 training examples.

---

## 10. Risks & Mitigations

| Risk | Severity | Description | Mitigation |
|---|---|---|---|
| High inter-subject variability | High | EEG varies significantly between subjects, making cross-subject models much weaker than within-subject | Train/evaluate within-subject first; use LOSO only as secondary benchmark |
| Data leakage across folds | High | If epochs from the same trial are split across train/test, accuracy is artificially inflated | Ensure CV splits at epoch level, not trial level; verify with manual inspection |
| Overfitting (CNN) | Medium | EEG datasets are small by DL standards; CNNs easily overfit | Use EEGNet (compact), early stopping, dropout=0.5, and data augmentation |
| Poor artifact rejection | Medium | Muscle or ocular artifacts can masquerade as signal | Combine amplitude thresholding with visual inspection; optionally apply ICA |
| PhysioNet download reliability | Low | Network issues during batch download of all 109 subjects | Implement retry logic; develop on subset (10 subjects) before full run |
| Inconsistent MNE API versions | Low | MNE-Python has breaking changes between versions | Pin MNE version in requirements.txt; test on specified version |

---

## 11. Future Extensions (v2+)

### Short-Term
- **Multi-class MI:** Extend to 4-class (left hand, right hand, both feet, tongue) using BCI Competition IV Dataset 2a
- **ICA artifact removal:** Add Independent Component Analysis for automatic ocular/muscle artifact removal
- **Riemannian geometry classifiers:** Implement MDM (Minimum Distance to Mean) using pyRiemann — state-of-the-art for MI-BCI
- **Hyperparameter optimization:** Add Optuna or Ray Tune for automated CNN architecture search

### Medium-Term
- **Real-time decoding:** Integrate with LSL (Lab Streaming Layer) for online classification from live EEG streams
- **Transfer learning:** Pre-train on all 109 subjects, fine-tune on new subject with minimal calibration data
- **Explainability:** Grad-CAM visualization on EEGNet to identify which time-frequency-spatial features drive predictions
- **Subject-adaptive calibration:** Implement Riemannian alignment to reduce session-to-session variability

### Long-Term
- **Hardware integration:** Deploy pipeline to OpenBCI Cyton or g.tec Unicorn for real-world testing
- **Robotic control:** Connect decoded motor intention to a robot arm or exoskeleton via ROS or direct serial
- **Closed-loop BCI:** Add neurofeedback loop providing real-time ERD feedback to improve signal quality
- **Edge deployment:** Optimize EEGNet for inference on Raspberry Pi or NVIDIA Jetson Nano

---

## 12. References

- Goldberger, A.L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23): e215–e220.
- Lawhern, V.J. et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces. *Journal of Neural Engineering*, 15(5).
- Pfurtscheller, G. & Lopes da Silva, F.H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11): 1842–1857.
- Blankertz, B. et al. (2008). The BCI Competition 2003: Progress and Perspectives in Detection and Discrimination of EEG Single Trials. *IEEE TNSRE*.
- Lotte, F. et al. (2018). A review of classification algorithms for EEG-based BCI: a 10-year update. *Journal of Neural Engineering*, 15(3).
- Roy, Y. et al. (2019). Deep learning-based electroencephalography analysis: a systematic review. *Journal of Neural Engineering*, 16(5).
- MNE-Python documentation: [mne.tools](https://mne.tools)
We need to define the tasks and features to build our project, progressing from a barebones setup to a minimal viable product (MVP), to a feature-rich polished version.
