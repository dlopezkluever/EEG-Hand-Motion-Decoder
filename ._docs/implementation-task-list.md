# EEG Hand Movement Decoder — Implementation Task List

Iterative development plan from barebones setup to full-featured BCI pipeline.
Each phase builds on the previous, delivering a functional product at every stage.

---

## Phase 0: Setup

**Goal:** A barebones project skeleton that runs — environment configured, data downloadable, and a single raw EEG signal loaded and printed to console. Nothing is usable yet, but the foundation compiles and executes.

---

### 0.1 Environment & Project Scaffold

- [x] Create conda environment with Python 3.10+ and install core dependencies: `mne`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `pytorch`, `jupyterlab`
- [x] Generate `requirements.txt` and `environment.yml` pinning all dependency versions
- [x] Create the full directory structure per PRD Section 7 (`src/`, `src/models/`, `data/`, `notebooks/`, `outputs/figures/`, `outputs/models/`, `outputs/results/`, `tests/`)
- [x] Create placeholder `__init__.py` files and empty module files (`config.py`, `data_loader.py`, `preprocessing.py`, `features.py`, `evaluate.py`, `visualize.py`, `models/logistic.py`, `models/eegnet.py`)
- [x] Add `.gitignore` ignoring `data/`, `outputs/models/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.pyc`, conda/venv dirs

### 0.2 Configuration Module

- [x] Implement `src/config.py` with all configurable constants from the PRD: `SUBJECTS` (default `range(1, 11)`), `RUNS` (`[3, 7]`), `DATA_PATH` (`./data/eegbci`), frequency band definitions, filter parameters, epoch window settings, artifact rejection threshold
- [x] Add model hyperparameter defaults: LR grid search values, EEGNet architecture params (F1, F2, D, dropout, lr, batch size, epochs, patience)
- [x] Add output path constants for figures, models, and results directories
- [x] Add a `get_config()` function that returns all settings as a dictionary for logging/reproducibility
- [x] Verify config imports cleanly from other modules with a simple test script

### 0.3 Data Download & Raw Loading

- [x] Implement `src/data_loader.py` — function `download_data(subjects, runs)` using `mne.datasets.eegbci.load_data()` with local caching to `DATA_PATH`
- [x] Implement function `load_raw(subject, runs)` that loads EDF files into MNE `Raw` objects, concatenates runs, and sets the `standard_1020` montage
- [x] Add logging: print subject ID, run ID, file path, raw data shape, sampling rate, and channel count for each loaded file
- [x] Add retry logic (3 attempts with backoff) for PhysioNet download failures
- [x] Write a `__main__` guard in `data_loader.py` that downloads and loads Subject 1 Runs [3, 7] and prints a summary — verify this runs end-to-end

### 0.4 Smoke Test & Validation

- [x] Run `data_loader.py` standalone — confirm EDF files download to `data/eegbci/`, Raw object loads with 64 channels at 160 Hz
- [x] Print raw data shape, channel names, and first 5 annotation events to console
- [x] Create `tests/test_data_loader.py` with a basic test: load Subject 1, assert channel count == 64, sampling rate == 160, annotations exist
- [x] Verify the full project imports without errors: `from src.config import *; from src.data_loader import *`

---

## Phase 1: MVP — Preprocessing + Baseline Model + Evaluation

**Goal:** A working end-to-end pipeline: raw EEG in → preprocessed epochs → PSD features → logistic regression classifier → accuracy score printed. This is the first version that delivers the project's primary value — a trained BCI decoder with a measurable accuracy above chance (50%).

---

### 1.1 Signal Preprocessing — Filtering

- [ ] Implement `src/preprocessing.py` — function `apply_filters(raw)` that applies: (1) bandpass FIR filter 1–40 Hz via `raw.filter()`, (2) notch filter at 60 Hz and 120 Hz via `raw.notch_filter()`
- [ ] Apply common average reference (CAR) via `raw.set_eeg_reference('average', projection=True)`
- [ ] Log filter parameters applied (low freq, high freq, notch freqs, filter type)
- [ ] Verify filtered signal: print before/after data shape, confirm no NaN values, confirm channel count unchanged
- [ ] Add unit test: load Subject 1, apply filters, assert data shape unchanged and values differ from raw

### 1.2 Signal Preprocessing — Epoch Extraction

- [ ] Implement function `extract_epochs(raw, tmin=-0.5, tmax=4.0, reject_threshold=150e-6)` that reads event annotations, maps T1→left (class 0), T2→right (class 1), and creates `mne.Epochs`
- [ ] Apply baseline correction over the pre-stimulus window (−0.5 to 0.0 s)
- [ ] Configure artifact rejection: reject epochs with peak-to-peak amplitude > 150 µV
- [ ] Log: total epochs extracted, epochs per class, number rejected, rejection percentage; flag subjects with >30% rejection rate
- [ ] Add unit test: extract epochs for Subject 1, assert epochs exist for both classes, assert epoch shape is (n_epochs, 64, n_timepoints)

### 1.3 Feature Engineering — PSD Band Power (Pathway A)

- [ ] Implement `src/features.py` — function `extract_psd_features(epochs)` using `mne.time_frequency.psd_array_welch` (2s window, 50% overlap, Hanning)
- [ ] Compute average band power for each of the 6 canonical bands (Delta 1–4, Theta 4–8, Mu 8–12, Low Beta 13–20, High Beta 20–30, Low Gamma 30–40 Hz) across all 64 channels
- [ ] Flatten feature vector per epoch to shape `(n_epochs, 64 × 6)` = `(n_epochs, 384)`, extract labels array `y`
- [ ] Apply `StandardScaler` normalization to the feature matrix
- [ ] Add unit test: extract features for Subject 1, assert feature matrix shape is `(n_epochs, 384)`, no NaN/Inf values

### 1.4 Baseline Model — Logistic Regression

- [ ] Implement `src/models/logistic.py` — function `train_logistic(X, y)` using `sklearn.LogisticRegression` with `solver='lbfgs'`, `class_weight='balanced'`, `max_iter=1000`
- [ ] Implement 10-fold stratified cross-validation using `sklearn.model_selection.StratifiedKFold`
- [ ] Report per-fold accuracy and mean ± std accuracy across folds
- [ ] Print classification to console: accuracy, F1-score (macro), AUC-ROC
- [ ] Add unit test: train on Subject 1 features, assert accuracy > 0.5 (above chance)

### 1.5 Basic Evaluation & Metrics

- [ ] Implement `src/evaluate.py` — function `compute_metrics(y_true, y_pred, y_prob)` returning accuracy, precision, recall, F1 (per-class and macro), AUC-ROC, Cohen's Kappa
- [ ] Implement function `save_results(metrics, model_name, subject_id)` that exports metrics to JSON in `outputs/results/`
- [ ] Generate and save a confusion matrix heatmap (seaborn) to `outputs/figures/`
- [ ] Print a formatted summary table of all metrics to console
- [ ] Add unit test: pass known y_true/y_pred arrays, assert metric values are correct

### 1.6 Main Training Entrypoint

- [ ] Implement `train.py` that orchestrates the full pipeline: load data → preprocess → extract features → train model → evaluate → save results
- [ ] Support configurable subject list via command-line argument or config
- [ ] Loop over all configured subjects, aggregate per-subject results into a summary CSV via Pandas
- [ ] Print a final summary: per-subject accuracy table and overall mean accuracy
- [ ] Verify full pipeline runs end-to-end for subjects 1–5 and produces accuracy above 50%

---

## Phase 2: CNN Model + CSP Features

**Goal:** Add the EEGNet CNN as the primary model and CSP as a second feature pathway, enabling model comparison. The pipeline now supports multiple feature extraction strategies and multiple classifiers, evaluated side-by-side.

---

### 2.1 Feature Engineering — Common Spatial Patterns (Pathway B)

- [ ] Implement function `extract_csp_features(epochs, n_components=4)` in `src/features.py` using `mne.decoding.CSP`
- [ ] Output feature matrix of shape `(n_epochs, n_components)` with log-variance transformation applied
- [ ] Integrate CSP pathway into `train.py` — train logistic regression on CSP features and compare to PSD features
- [ ] Log CSP spatial filter weights for interpretability
- [ ] Add unit test: extract CSP features for Subject 1, assert shape `(n_epochs, 4)`, no NaN values

### 2.2 EEGNet CNN — Architecture

- [ ] Implement `src/models/eegnet.py` — PyTorch `nn.Module` class `EEGNet` following the architecture from PRD Section 5.4: temporal Conv2D → BatchNorm + ELU → depthwise Conv2D → AvgPool + Dropout → separable Conv2D → AvgPool + Dropout → Flatten + Dense (softmax, 2 classes)
- [ ] Parameterize architecture: `F1=8`, `F2=16`, `D=2`, `dropout=0.5`, configurable `n_channels` and `n_timepoints`
- [ ] Implement `count_parameters()` utility to verify compact model size (< 2K params for 64 channels)
- [ ] Add unit test: instantiate EEGNet, pass a random tensor of correct shape through forward pass, assert output shape is `(batch, 2)`

### 2.3 EEGNet CNN — Training Loop

- [ ] Implement function `train_eegnet(X, y, config)` with: Adam optimizer (lr=0.001), cross-entropy loss, batch size=32, max 300 epochs
- [ ] Implement early stopping with patience=30 on validation loss (80/20 train/val split)
- [ ] Implement feature extraction Pathway C: raw epoch array `(n_epochs, n_channels, n_timepoints)` with per-channel z-score normalization
- [ ] Log training loss, validation loss, and validation accuracy per epoch; save training curves data
- [ ] Return trained model, training history, and best validation accuracy

### 2.4 EEGNet Cross-Validation

- [ ] Wrap EEGNet training in 10-fold stratified cross-validation (same fold splits as logistic regression for fair comparison)
- [ ] Report per-fold accuracy, mean ± std accuracy, F1, and AUC-ROC
- [ ] Save the best model weights (by validation loss) per fold to `outputs/models/`
- [ ] Generate predictions across all folds for confusion matrix and ROC curve generation
- [ ] Compare EEGNet results to logistic regression (PSD) and logistic regression (CSP) in a summary table

### 2.5 Model Comparison Framework

- [ ] Update `train.py` to run all three model-feature combinations: LR+PSD, LR+CSP, EEGNet+Raw
- [ ] Collect all results into a unified comparison CSV: model, feature type, subject, accuracy, F1, AUC-ROC
- [ ] Print a side-by-side comparison table to console after training completes
- [ ] Implement paired t-test (scipy) between model accuracies across folds for statistical significance (p < 0.05)
- [ ] Save the comparison summary to `outputs/results/model_comparison.json`

---

## Phase 3: Full Visualization & Evaluation Suite

**Goal:** Complete all visualization and evaluation deliverables from the PRD. The pipeline now produces publication-quality figures, detailed per-subject reports, and a comprehensive results package.

---

### 3.1 EEG Signal Visualizations

- [ ] Implement `src/visualize.py` — function `plot_topomap(epochs)` generating scalp topographic maps of ERD during left vs. right MI using `mne.viz.plot_topomap`
- [ ] Implement function `plot_psd_comparison(epochs)` comparing left vs. right MI power spectra specifically at C3 and C4 electrodes
- [ ] Implement function `plot_butterfly(epochs)` for epoch quality visual inspection
- [ ] Save all figures as PNG (300 DPI) and PDF to `outputs/figures/`
- [ ] Add a `generate_all_signal_figures(epochs, subject_id)` convenience function

### 3.2 Model Performance Visualizations

- [ ] Implement function `plot_confusion_matrix(y_true, y_pred, model_name)` using seaborn heatmap with annotations
- [ ] Implement function `plot_roc_curve(y_true, y_prob, model_name)` with AUC annotation
- [ ] Implement function `plot_training_curves(history)` showing loss and accuracy over epochs for EEGNet
- [ ] Implement function `plot_feature_importance(model, feature_names)` mapping logistic regression coefficients to electrode/band combinations
- [ ] Implement function `plot_subject_accuracy_bar(results_df)` showing per-subject accuracy comparison across models

### 3.3 Comprehensive Evaluation Report

- [ ] Update `evaluate.py` to compute all PRD metrics: Accuracy, Precision, Recall, F1 (per-class + macro), AUC-ROC, Cohen's Kappa — for every model-feature combination
- [ ] Export full metrics to both JSON and CSV in `outputs/results/`
- [ ] Generate per-subject accuracy breakdown tables
- [ ] Add aggregate statistics: mean, std, min, max, median accuracy across subjects
- [ ] Auto-generate a text-based evaluation summary saved to `outputs/results/evaluation_report.txt`

### 3.4 Exploratory Jupyter Notebooks

- [ ] Create `notebooks/01_data_exploration.ipynb`: load raw data for one subject, plot raw signals, inspect annotations and events, display channel montage
- [ ] Create `notebooks/02_preprocessing.ipynb`: demonstrate filtering effects (before/after PSD), show epoch extraction, visualize artifact rejection
- [ ] Create `notebooks/03_feature_extraction.ipynb`: visualize PSD band power distributions, plot CSP spatial patterns, show class separation in feature space
- [ ] Create `notebooks/04_model_comparison.ipynb`: load saved results, generate all comparison plots, reproduce key figures from the pipeline

---

## Phase 4: Scale, Harden & Polish

**Goal:** Scale to the full 109-subject dataset, add LOSO cross-validation, implement hyperparameter tuning, and harden the pipeline for robustness. The project is now a complete, benchmarkable BCI system ready for portfolio presentation.

---

### 4.1 Full Dataset Scaling (109 Subjects)

- [ ] Update `config.py` to support `SUBJECTS = range(1, 110)` and batch-process all 109 subjects
- [ ] Implement parallel/batched data loading with progress bars (tqdm) and robust error handling per subject
- [ ] Add subject-level result caching: skip subjects whose results already exist in `outputs/results/` to enable resumable runs
- [ ] Profile memory usage and implement epoch-level data generators if memory exceeds available RAM
- [ ] Run full pipeline on all 109 subjects; save complete results CSV

### 4.2 Leave-One-Subject-Out (LOSO) Cross-Validation

- [ ] Implement LOSO CV in `src/evaluate.py`: for each test subject, train on all other subjects' data, evaluate on held-out subject
- [ ] Aggregate LOSO results: per-subject accuracy, overall mean ± std
- [ ] Generate per-subject accuracy bar chart for LOSO evaluation
- [ ] Compare within-subject (10-fold) vs. cross-subject (LOSO) performance in a summary table
- [ ] Document expected accuracy drop for cross-subject generalization

### 4.3 Hyperparameter Tuning — Logistic Regression

- [ ] Implement grid search over regularization strength `C = [0.001, 0.01, 0.1, 1.0, 10.0]` using `sklearn.model_selection.GridSearchCV`
- [ ] Report best C per subject and overall best C
- [ ] Log grid search results (all C values and corresponding CV scores) to JSON
- [ ] Re-run evaluation with optimal hyperparameters and update results
- [ ] Compare tuned vs. default performance

### 4.4 Data Augmentation for EEGNet

- [ ] Implement Gaussian noise augmentation: add noise with σ=0.01 to training epochs
- [ ] Implement temporal jitter augmentation: randomly shift epoch window by ±50 ms
- [ ] Apply augmentation only to training set (never validation/test)
- [ ] Compare EEGNet performance with and without augmentation
- [ ] Save augmentation configuration and results to comparison table

### 4.5 Pipeline Hardening & Reproducibility

- [ ] Add random seed control: set seeds for NumPy, PyTorch, and scikit-learn globally via config for reproducibility
- [ ] Add comprehensive logging (Python `logging` module) to all pipeline stages with timestamps
- [ ] Implement command-line argument parsing in `train.py` (argparse): `--subjects`, `--models`, `--features`, `--cv-folds`, `--output-dir`
- [ ] Add pipeline-level error handling: catch and log per-subject failures without crashing the full run
- [ ] Verify reproducibility: run pipeline twice with same seed, assert identical results

---

## Phase 5: Advanced Features & Future-Readiness

**Goal:** Stretch enhancements that push toward published state-of-the-art and prepare the codebase for future extensions (real-time decoding, multi-class, transfer learning). These are optional improvements beyond the core deliverable.

---

### 5.1 ICA Artifact Removal

- [ ] Implement optional ICA-based artifact removal in `preprocessing.py` using `mne.preprocessing.ICA`
- [ ] Auto-detect and exclude EOG (eye blink) and EMG (muscle) components
- [ ] Compare pipeline accuracy with amplitude-rejection-only vs. ICA+amplitude rejection
- [ ] Make ICA configurable (on/off) via `config.py`
- [ ] Log ICA components excluded per subject

### 5.2 Motor Cortex ROI Channel Selection

- [ ] Implement optional channel subset selection: restrict to motor cortex ROI channels (C3, C4, Cz, FC3, FC4, CP3, CP4)
- [ ] Compare 7-channel ROI vs. full 64-channel performance for all models
- [ ] Analyze whether reduced channel count improves or degrades accuracy (less noise vs. less information)
- [ ] Save ROI comparison results alongside full-channel results
- [ ] Document findings on optimal channel selection for MI-BCI

### 5.3 Experiment Tracking Integration

- [ ] Integrate MLflow or Weights & Biases for experiment tracking
- [ ] Log all hyperparameters, metrics, and artifacts (figures, models) per run
- [ ] Enable comparison of runs across different configurations via the tracking UI
- [ ] Add experiment tags: model type, feature type, subject count, CV strategy
- [ ] Document setup instructions for the experiment tracker in README

### 5.4 Standalone Evaluation Script

- [ ] Implement `evaluate.py` (root-level) as a standalone script that loads saved models and re-evaluates on specified subjects
- [ ] Support loading both sklearn (pickle/joblib) and PyTorch (state dict) models
- [ ] Generate all evaluation figures and metrics without retraining
- [ ] Add `--model-path` and `--data-subjects` arguments for flexible evaluation
- [ ] Useful for evaluating on new subjects or regenerating figures

### 5.5 Documentation & Portfolio Readiness

- [ ] Write comprehensive `README.md`: project overview, setup instructions, usage, results summary with key figures, architecture diagram, references
- [ ] Add inline docstrings to all public functions in `src/` modules
- [ ] Create a results summary section in README with accuracy tables and sample figures
- [ ] Ensure all outputs are cleanly organized and figures are publication-quality
- [ ] Tag a v1.0 release on GitHub with the complete pipeline and results

---

## Phase Summary

| Phase | Name | Delivers | Key Outcome |
|-------|------|----------|-------------|
| 0 | Setup | Project skeleton, environment, data loading | Raw EEG loads and prints to console |
| 1 | MVP | Preprocessing + PSD features + Logistic Regression + metrics | First accuracy score above chance (~65–75%) |
| 2 | CNN + CSP | EEGNet model, CSP features, model comparison | Multiple models compared, best model identified |
| 3 | Visualization | All figures, notebooks, comprehensive reports | Publication-quality outputs |
| 4 | Scale & Harden | 109 subjects, LOSO, tuning, augmentation, CLI | Complete benchmarkable system |
| 5 | Advanced | ICA, ROI selection, experiment tracking, docs | Portfolio-ready project |
