# Pipeline Results Overview

Evaluation results from running the EEG BCI pipeline on 3 subjects (Subjects 1, 2, 3) from the PhysioNet EEGBCI dataset. Two test configurations were run: a baseline pipeline and a fully-featured pipeline with ICA artifact removal, hyperparameter tuning, and data augmentation.

---

## Table of Contents

- [Model Comparison Summary](#model-comparison-summary)
- [Per-Subject Accuracy Breakdown](#per-subject-accuracy-breakdown)
- [Within-Subject Cross-Validation (10-Fold)](#within-subject-cross-validation-10-fold)
- [Leave-One-Subject-Out (LOSO) Cross-Validation](#leave-one-subject-out-loso-cross-validation)
- [Hyperparameter Tuning Results](#hyperparameter-tuning-results)
- [Statistical Tests](#statistical-tests)
- [Standalone Model Evaluation](#standalone-model-evaluation)
- [Signal Visualizations](#signal-visualizations)
- [Model Visualizations](#model-visualizations)
- [Test Log Summary](#test-log-summary)

---

## Model Comparison Summary

Results from the comprehensive run: `python train.py --subjects 1 2 3 --no-cache --ica --tune --augment`

| Model | Mean Accuracy | Std | Mean F1 (Macro) | Mean AUC-ROC |
|-------|:---:|:---:|:---:|:---:|
| LR + PSD | 0.5556 | 0.0770 | 0.5542 | 0.6098 |
| **LR + CSP** | **0.8778** | **0.1262** | **0.8774** | **0.9274** |
| EEGNet + Raw | 0.7778 | 0.1262 | 0.7757 | 0.8577 |

Best model by mean accuracy: **LR + CSP (87.8%)**

![Subject Accuracy Comparison](outputs/figures/subject_accuracy_comparison.png)

---

## Per-Subject Accuracy Breakdown

| Subject | LR + PSD | LR + CSP | EEGNet | Best Model |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.6000 | **0.9667** | 0.8667 | LR + CSP |
| 2 | 0.6000 | **0.7333** | 0.6333 | LR + CSP |
| 3 | 0.4667 | **0.9333** | 0.8333 | LR + CSP |

CSP features consistently outperform PSD band-power features for logistic regression, which is expected -- CSP is specifically designed to maximize the variance ratio between two classes for motor imagery tasks.

---

## Within-Subject Cross-Validation (10-Fold)

### Subject 1 (ICA: 1 EOG component removed)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LR + PSD | 0.67 | 0.67 | 0.67 | 0.67 | 1.00 | 0.67 | 0.33 | 0.67 | 0.67 | 0.00 | 0.60 |
| LR + CSP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | **0.97** |
| EEGNet | 1.00 | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 0.67 | 1.00 | 1.00 | 0.87 |

### Subject 2 (ICA: 2 EOG components removed)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LR + PSD | 0.33 | 0.67 | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 | 0.33 | 1.00 | 0.67 | 0.60 |
| LR + CSP | 0.67 | 1.00 | 0.67 | 0.33 | 1.00 | 1.00 | 1.00 | 0.67 | 0.33 | 0.67 | **0.73** |
| EEGNet | 0.33 | 1.00 | 0.33 | 1.00 | 0.67 | 0.67 | 0.67 | 0.67 | 1.00 | 0.67 | 0.63 |

### Subject 3 (ICA: 1 EOG component removed)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LR + PSD | 0.33 | 0.33 | 0.33 | 1.00 | 0.33 | 0.33 | 0.67 | 0.33 | 0.67 | 0.33 | 0.47 |
| LR + CSP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.67 | 1.00 | 1.00 | **0.93** |
| EEGNet | 1.00 | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.67 | 0.67 | 0.83 |

High fold-to-fold variance (e.g., 0.00 to 1.00) is expected given the small number of epochs per subject (30 trials, 3 per fold).

---

## Leave-One-Subject-Out (LOSO) Cross-Validation

Run: `python train.py --loso --subjects 1 2 3 --loso-models LR_PSD`

| Test Subject | LR + PSD Accuracy |
|:---:|:---:|
| 1 | 0.7333 |
| 2 | 0.5667 |
| 3 | 0.5333 |
| **Mean** | **0.6111 +/- 0.0875** |

| Metric | Value |
|--------|:---:|
| Mean Accuracy | 0.6111 |
| Macro F1 | 0.6111 |
| AUC-ROC | 0.6136 |

Cross-subject generalization is lower than within-subject performance, which is consistent with the known challenge of inter-subject variability in EEG-based BCIs. With only 3 subjects and PSD features, this baseline is expected. CSP and EEGNet LOSO would likely perform better with more subjects.

---

## Hyperparameter Tuning Results

GridSearchCV over C = [0.001, 0.01, 0.1, 1.0, 10.0] for LR + PSD:

| Subject | Default Accuracy | Tuned Accuracy | Delta | Best C |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.6000 | 0.5333 | -0.0667 | 0.1 |
| 2 | 0.6000 | 0.6333 | +0.0333 | 1.0 |
| 3 | 0.4667 | 0.5000 | +0.0333 | 0.01 |
| **Mean** | **0.5556** | **0.5444** | **-0.0111** | -- |

Tuning did not improve overall performance for LR + PSD. This is expected with only 30 epochs per subject -- the inner CV loop of GridSearchCV has very few samples, making the selected C unstable across folds.

---

## Statistical Tests

Pairwise paired t-tests between model accuracies (n=3 subjects):

| Comparison | t-statistic | p-value | Significant (p < 0.05) |
|-----------|:---:|:---:|:---:|
| LR_PSD vs LR_CSP | -3.263 | 0.0825 | No |
| LR_PSD vs EEGNet_Raw | -2.250 | 0.1533 | No |
| LR_CSP vs EEGNet_Raw | inf | 0.0000 | Not reliable |

No comparisons reached statistical significance at p < 0.05. With only 3 subjects, statistical power is very low. The `t=inf` result for LR_CSP vs EEGNet_Raw arises because both models had identical standard deviations, making the denominator of the t-statistic zero. A larger subject pool (10+) is needed for meaningful statistical comparisons.

---

## Standalone Model Evaluation

Run: `python evaluate.py --model-path outputs/models/eegnet_fold01.pt --data-subjects 1`

Re-evaluated a saved EEGNet model (fold 1 checkpoint from the ICA+augment run) on Subject 1:

| Metric | Value |
|--------|:---:|
| Accuracy | 0.8667 |
| Precision (macro) | 0.8750 |
| Recall (macro) | 0.8616 |
| F1 (macro) | 0.8643 |
| AUC-ROC | 0.8795 |
| Cohen's Kappa | 0.7297 |

This confirms that saved models load correctly and produce consistent predictions.

![Confusion Matrix - EEGNet Eval](outputs/figures/confusion_EEGNet_Raw_eval_subject001.png)
![ROC Curve - EEGNet Eval](outputs/figures/roc_EEGNet_Raw_eval_subject001.png)

---

## Signal Visualizations

### Topographic Maps (Scalp Voltage Distribution)

| Subject 1 | Subject 2 | Subject 3 |
|:---------:|:---------:|:---------:|
| ![Topomap S1](outputs/figures/topomap_subject001.png) | ![Topomap S2](outputs/figures/topomap_subject002.png) | ![Topomap S3](outputs/figures/topomap_subject003.png) |

### PSD Comparison (Left vs Right Motor Imagery)

| Subject 1 | Subject 2 | Subject 3 |
|:---------:|:---------:|:---------:|
| ![PSD S1](outputs/figures/psd_comparison_subject001.png) | ![PSD S2](outputs/figures/psd_comparison_subject002.png) | ![PSD S3](outputs/figures/psd_comparison_subject003.png) |

### Butterfly Plots (ERP Waveforms)

| Subject 1 | Subject 2 | Subject 3 |
|:---------:|:---------:|:---------:|
| ![Butterfly S1](outputs/figures/butterfly_subject001.png) | ![Butterfly S2](outputs/figures/butterfly_subject002.png) | ![Butterfly S3](outputs/figures/butterfly_subject003.png) |

---

## Model Visualizations

### Confusion Matrices

**LR + PSD:**

| Subject 1 | Subject 2 | Subject 3 |
|:---------:|:---------:|:---------:|
| ![CM PSD S1](outputs/figures/confusion_LR_PSD_subject001.png) | ![CM PSD S2](outputs/figures/confusion_LR_PSD_subject002.png) | ![CM PSD S3](outputs/figures/confusion_LR_PSD_subject003.png) |

**LR + CSP:**

| Subject 1 | Subject 2 | Subject 3 |
|:---------:|:---------:|:---------:|
| ![CM CSP S1](outputs/figures/confusion_LR_CSP_subject001.png) | ![CM CSP S2](outputs/figures/confusion_LR_CSP_subject002.png) | ![CM CSP S3](outputs/figures/confusion_LR_CSP_subject003.png) |

**EEGNet:**

| Subject 1 | Subject 2 | Subject 3 |
|:---------:|:---------:|:---------:|
| ![CM EEGNet S1](outputs/figures/confusion_EEGNet_Raw_subject001.png) | ![CM EEGNet S2](outputs/figures/confusion_EEGNet_Raw_subject002.png) | ![CM EEGNet S3](outputs/figures/confusion_EEGNet_Raw_subject003.png) |

### ROC Curves

**Per-model ROC:**

| Subject 1 | Subject 2 | Subject 3 |
|:---------:|:---------:|:---------:|
| ![ROC PSD S1](outputs/figures/roc_LR_PSD_subject001.png) | ![ROC PSD S2](outputs/figures/roc_LR_PSD_subject002.png) | ![ROC PSD S3](outputs/figures/roc_LR_PSD_subject003.png) |
| ![ROC CSP S1](outputs/figures/roc_LR_CSP_subject001.png) | ![ROC CSP S2](outputs/figures/roc_LR_CSP_subject002.png) | ![ROC CSP S3](outputs/figures/roc_LR_CSP_subject003.png) |
| ![ROC EEGNet S1](outputs/figures/roc_EEGNet_Raw_subject001.png) | ![ROC EEGNet S2](outputs/figures/roc_EEGNet_Raw_subject002.png) | ![ROC EEGNet S3](outputs/figures/roc_EEGNet_Raw_subject003.png) |

**Combined ROC Comparison (All Models):**

| Subject 1 | Subject 2 | Subject 3 |
|:---------:|:---------:|:---------:|
| ![ROC Comp S1](outputs/figures/roc_comparison_subject001.png) | ![ROC Comp S2](outputs/figures/roc_comparison_subject002.png) | ![ROC Comp S3](outputs/figures/roc_comparison_subject003.png) |

### EEGNet Training Curves

| Subject 1 | Subject 2 | Subject 3 |
|:---------:|:---------:|:---------:|
| ![Train S1](outputs/figures/training_curves_EEGNet_subject001.png) | ![Train S2](outputs/figures/training_curves_EEGNet_subject002.png) | ![Train S3](outputs/figures/training_curves_EEGNet_subject003.png) |

### Feature Importance (LR + PSD)

| Subject 1 | Subject 2 | Subject 3 |
|:---------:|:---------:|:---------:|
| ![FI S1](outputs/figures/feature_importance_LR_PSD_subject001.png) | ![FI S2](outputs/figures/feature_importance_LR_PSD_subject002.png) | ![FI S3](outputs/figures/feature_importance_LR_PSD_subject003.png) |

---

## Test Log Summary

Four pipeline tests were run to validate all features end-to-end:

### Test 1: Smoke Test (Baseline, 1 Subject)

```
python train.py --subjects 1 --no-cache
```

- Duration: ~3.5 minutes
- Status: PASSED
- All 3 models trained and evaluated (LR+PSD, LR+CSP, EEGNet)
- All outputs generated: JSON results, CSV comparison, confusion matrices, ROC curves, training curves, topomaps, PSD plots, butterfly plots
- Best model: EEGNet (90.0%)

### Test 2: Full Features (ICA + Tuning + Augmentation, 3 Subjects)

```
python train.py --subjects 1 2 3 --no-cache --ica --tune --augment
```

- Duration: ~55 minutes (CPU-only, augmented EEGNet trains for 300 epochs per fold)
- Status: PASSED
- ICA correctly identified 1-2 EOG components per subject
- Hyperparameter tuning completed for all subjects
- Data augmentation tripled training set (27 -> 81 epochs per fold via Gaussian noise + temporal jitter)
- Best model: LR + CSP (87.8%)

### Test 3: LOSO Cross-Validation (3 Subjects)

```
python train.py --loso --subjects 1 2 3 --loso-models LR_PSD
```

- Duration: ~3 seconds
- Status: PASSED
- Cross-subject evaluation completed
- Results saved to `outputs/results/loso_results.json`

### Test 4: Standalone Evaluation

```
python evaluate.py --model-path outputs/models/eegnet_fold01.pt --data-subjects 1
```

- Duration: ~4 seconds
- Status: PASSED
- Loaded saved EEGNet checkpoint and evaluated on Subject 1
- Produced confusion matrix and ROC curve

All tests completed without errors. All output files and figures generated correctly.
