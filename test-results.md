Some Verification Tests:

  Quick smoke test (1 subject, ~2-3 min):
  python train.py --subjects 1 --no-cache

  Test with Phase 5 features (1 subject, ~3-5 min):
  python train.py --subjects 1 --no-cache --ica --roi

  Test everything together (3 subjects, ~10-15 min):
  python train.py --subjects 1 2 3 --no-cache --ica --tune --augment

  Test standalone evaluation:
  python evaluate.py --model-path outputs/models/eegnet_fold01.pt --data-subjects 1
  python evaluate.py --results-dir outputs/results/

  Test LOSO (needs 3+ subjects):

---------------------------------------------------------------------------------

Summary:
  Test Results Summary

  Test 1: Smoke test (--subjects 1 --no-cache)

  - LR+PSD: 63.3% acc - reasonable for single-subject baseline
  - LR+CSP: 86.7% acc - strong
  - EEGNet: 90.0% acc - best performer
  - All outputs saved correctly (JSON, CSV, figures)

  Test 2: Full features (--subjects 1 2 3 --ica --tune --augment)

  - ICA working correctly (1-2 components excluded per subject)
  - Tuning, augmentation, CSP, EEGNet all functional
  - LR+CSP was best at 87.8%, EEGNet at 77.8%
  - All 3 subjects completed successfully

  Test 3: LOSO (--loso --subjects 1 2 3 --loso-models LR_PSD)

  - Cross-subject accuracy: 61.1% - expected to be lower than within-subject
  - Completed cleanly

  Test 4: Standalone evaluation (evaluate.py)

  - Loaded saved EEGNet model and re-evaluated on Subject 1: 86.7% acc
  - Confusion matrix and ROC saved correctly



Quick smoke test (1 subject, ~2-3 min):
  python train.py --subjects 1 --no-cache

Results:


2026-03-10 00:08:09,533  INFO      Global random seeds set to 42
2026-03-10 00:08:09,533  INFO      Pipeline configuration: {'PROJECT_ROOT': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI'), 'DATA_PATH': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/data/eegbci'), 'OUTPUT_DIR': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/outputs'), 'FIGURES_DIR': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/outputs/figures'), 'MODELS_DIR': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/outputs/models'), 'RESULTS_DIR': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/outputs/results'), 'SUBJECTS': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109], 'CACHE_RESULTS': True, 'RUNS': [3, 7], 'SAMPLING_RATE': 160, 'N_CHANNELS': 64, 'BANDPASS_LOW': 1.0, 'BANDPASS_HIGH': 40.0, 'NOTCH_FREQS': [60.0, 120.0], 'EPOCH_TMIN': -0.5, 'EPOCH_TMAX': 4.0, 'BASELINE': (None, 0), 'REJECT_THRESHOLD': None, 'REJECT_WARN_RATIO': 0.3, 'USE_ICA': False, 'ICA_N_COMPONENTS': 20, 'ICA_METHOD': 'fastica', 'ICA_RANDOM_STATE': 42, 'ICA_EOG_THRESHOLD': 
3.0, 'ICA_MUSCLE_THRESHOLD': 1.0, 'EVENT_ID': {'left': 1, 'right': 2}, 'PSD_FMIN': 1.0, 'PSD_FMAX': 40.0, 'PSD_N_FFT': 320, 'PSD_N_OVERLAP': 160, 'PSD_WINDOW': 'hann', 'FREQ_BANDS': {'delta': (1, 4), 'theta': (4, 8), 'mu': (8, 12), 'low_beta': (13, 20), 'high_beta': (20, 30), 'low_gamma': (30, 40)}, 'USE_ROI_CHANNELS': False, 'MOTOR_ROI_CHANNELS': ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4'], 'CSP_N_COMPONENTS': 4, 'LR_SOLVER': 'lbfgs', 'LR_MAX_ITER': 1000, 'LR_CLASS_WEIGHT': 'balanced', 'LR_C_GRID': [0.001, 0.01, 0.1, 1.0, 10.0], 'EEGNET_F1': 8, 'EEGNET_F2': 16, 'EEGNET_D': 2, 'EEGNET_DROPOUT': 0.5, 'EEGNET_LR': 0.001, 'EEGNET_BATCH_SIZE': 32, 'EEGNET_MAX_EPOCHS': 300, 'EEGNET_PATIENCE': 30, 'CV_N_FOLDS': 10, 'RANDOM_SEED': 42, 'AUGMENT_GAUSSIAN_STD': 0.01, 'AUGMENT_TEMPORAL_JITTER_MS': 50, 'USE_MLFLOW': False, 
'MLFLOW_EXPERIMENT_NAME': 'EEG-BCI-Pipeline', 'MLFLOW_TRACKING_URI': 'file:./mlruns'}
2026-03-10 00:08:09,533  INFO      Processing 1 subjects: [1]
2026-03-10 00:08:09,533  INFO      Options: cache=False, tune=False, augment=False, loso=False, ica=False, roi=False  
Subjects:   0%|                                                                                | 0/1 [00:00<?, ?it/s]2026-03-10 00:08:09,533  INFO      ============================================================
2026-03-10 00:08:09,533  INFO      Processing Subject 001
2026-03-10 00:08:09,533  INFO      ============================================================
2026-03-10 00:08:09,697  INFO      Downloaded  subject=001  file=S001R03.edf
2026-03-10 00:08:09,697  INFO      Downloaded  subject=001  file=S001R07.edf
2026-03-10 00:08:09,697  INFO      Download complete: 1 subjects, 2 total files.
2026-03-10 00:08:10,612  INFO      Loaded  subject=001  runs=[3, 7]  channels=64  sfreq=160 Hz  n_times=40000  duration=250.0 s
2026-03-10 00:08:10,635  INFO      Applying filters: bandpass 1.0–40.0 Hz, notch [60.0, 120.0] Hz, FIR
2026-03-10 00:08:11,022  INFO      Filtering complete: shape before=(64, 40000)  after=(64, 40000)  channels=64
2026-03-10 00:08:11,029  INFO      Event mapping: {'left': 2, 'right': 3} (from raw annotations: {'T0': 1, 'T1': 2, 'T2': 3})
2026-03-10 00:08:11,102  INFO      Epochs: total=30  kept=30  rejected=0 (0.0%)  left=16  right=14
2026-03-10 00:08:11,103  INFO      --- Signal Visualizations ---
2026-03-10 00:08:12,724  INFO      Saved figure: topomap_subject001 (.png + .pdf)
2026-03-10 00:08:13,566  INFO      Saved figure: psd_comparison_subject001 (.png + .pdf)
2026-03-10 00:08:14,666  INFO      Saved figure: butterfly_subject001 (.png + .pdf)
2026-03-10 00:08:14,666  INFO      Generated 6 signal figures for Subject 001
2026-03-10 00:08:14,666  INFO      --- LR + PSD ---
2026-03-10 00:08:14,966  INFO      PSD features: X=(30, 384)  y=(30,)  bands=6  channels=64
2026-03-10 00:08:15,004  INFO        Fold  1/10 — accuracy: 0.6667
2026-03-10 00:08:15,009  INFO        Fold  2/10 — accuracy: 0.6667
2026-03-10 00:08:15,025  INFO        Fold  3/10 — accuracy: 1.0000
2026-03-10 00:08:15,039  INFO        Fold  4/10 — accuracy: 0.6667
2026-03-10 00:08:15,051  INFO        Fold  5/10 — accuracy: 0.6667
2026-03-10 00:08:15,062  INFO        Fold  6/10 — accuracy: 1.0000
2026-03-10 00:08:15,072  INFO        Fold  7/10 — accuracy: 0.3333
2026-03-10 00:08:15,084  INFO        Fold  8/10 — accuracy: 0.3333
2026-03-10 00:08:15,095  INFO        Fold  9/10 — accuracy: 0.3333
2026-03-10 00:08:15,108  INFO        Fold 10/10 — accuracy: 0.6667
2026-03-10 00:08:15,116  INFO      Logistic Regression — Accuracy: 0.6333 +/- 0.2333  F1: 0.6329  AUC: 0.6339

==================================================
Logistic Regression — Cross-Validation Results
==================================================
  Fold  1: 0.6667
  Fold  2: 0.6667
  Fold  3: 1.0000
  Fold  4: 0.6667
  Fold  5: 0.6667
  Fold  6: 1.0000
  Fold  7: 0.3333
  Fold  8: 0.3333
  Fold  9: 0.3333
  Fold 10: 0.6667
--------------------------------------------------
  Mean accuracy:  0.6333 +/- 0.2333
  Macro F1-score: 0.6329
  AUC-ROC:        0.6339
==================================================


=======================================================
  LR_PSD — Subject 1 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.6333
  precision_macro               0.6333
  precision_left                0.6667
  precision_right               0.6000
  recall_macro                  0.6339
  recall_left                   0.6250
  recall_right                  0.6429
  f1_macro                      0.6329
  f1_left                       0.6452
  f1_right                      0.6207
  auc_roc                       0.6339
  cohens_kappa                  0.2667
  cv_mean_accuracy              0.6333
  cv_std_accuracy               0.2333
=======================================================

2026-03-10 00:08:15,134  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\LR_PSD_subject001.json
2026-03-10 00:08:15,238  INFO      Confusion matrix saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\figures\confusion_LR_PSD_subject001.png
2026-03-10 00:08:15,427  INFO      Saved figure: roc_LR_PSD_subject001 (.png + .pdf)
2026-03-10 00:08:15,712  INFO      Saved figure: feature_importance_LR_PSD_subject001 (.png + .pdf)
2026-03-10 00:08:15,712  INFO      --- LR + CSP ---
Computing rank from data with rank=None
    Using tolerance 0.00043 (2.2e-16 eps * 64 dim * 3.1e+10  max singular value)
    Estimated rank (data): 63
    data: rank 63 computed from 64 data channels with 0 projectors
    Setting small data eigenvalues to zero (without PCA)
Reducing data rank from 64 -> 63
Estimating class=0 covariance using EMPIRICAL
Done.
Estimating class=1 covariance using EMPIRICAL
Done.
    Setting small data eigenvalues to zero (without PCA)
2026-03-10 00:08:16,130  INFO      CSP features: X=(30, 4)  y=(30,)  components=4
2026-03-10 00:08:16,131  INFO      CSP filter weight ranges: ['comp0: [-30677.7560, 38498.8045]', 'comp1: [-136674.9551, 152002.3233]', 'comp2: [-44081.9356, 51148.8804]', 'comp3: [-118809.0613, 105150.3812]']
2026-03-10 00:08:16,141  INFO        Fold  1/10 — accuracy: 1.0000
2026-03-10 00:08:16,145  INFO        Fold  2/10 — accuracy: 1.0000
2026-03-10 00:08:16,152  INFO        Fold  3/10 — accuracy: 0.6667
2026-03-10 00:08:16,156  INFO        Fold  4/10 — accuracy: 1.0000
2026-03-10 00:08:16,161  INFO        Fold  5/10 — accuracy: 1.0000
2026-03-10 00:08:16,166  INFO        Fold  6/10 — accuracy: 0.6667
2026-03-10 00:08:16,172  INFO        Fold  7/10 — accuracy: 0.6667
2026-03-10 00:08:16,176  INFO        Fold  8/10 — accuracy: 1.0000
2026-03-10 00:08:16,179  INFO        Fold  9/10 — accuracy: 0.6667
2026-03-10 00:08:16,183  INFO        Fold 10/10 — accuracy: 1.0000
2026-03-10 00:08:16,185  INFO      Logistic Regression — Accuracy: 0.8667 +/- 0.1633  F1: 0.8667  AUC: 0.9732

==================================================
Logistic Regression — Cross-Validation Results
==================================================
  Fold  1: 1.0000
  Fold  2: 1.0000
  Fold  3: 0.6667
  Fold  4: 1.0000
  Fold  5: 1.0000
  Fold  6: 0.6667
  Fold  7: 0.6667
  Fold  8: 1.0000
  Fold  9: 0.6667
  Fold 10: 1.0000
--------------------------------------------------
  Mean accuracy:  0.8667 +/- 0.1633
  Macro F1-score: 0.8667
  AUC-ROC:        0.9732
==================================================


=======================================================
  LR_CSP — Subject 1 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.8667
  precision_macro               0.8705
  precision_left                0.9286
  precision_right               0.8125
  recall_macro                  0.8705
  recall_left                   0.8125
  recall_right                  0.9286
  f1_macro                      0.8667
  f1_left                       0.8667
  f1_right                      0.8667
  auc_roc                       0.9732
  cohens_kappa                  0.7345
  cv_mean_accuracy              0.8667
  cv_std_accuracy               0.1633
=======================================================

2026-03-10 00:08:16,200  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\LR_CSP_subject001.json
2026-03-10 00:08:16,294  INFO      Confusion matrix saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\figures\confusion_LR_CSP_subject001.png
2026-03-10 00:08:16,500  INFO      Saved figure: roc_LR_CSP_subject001 (.png + .pdf)
2026-03-10 00:08:16,500  INFO      --- EEGNet + Raw ---
2026-03-10 00:08:16,530  INFO      Raw features: X=(30, 64, 721)  y=(30,)  (per-channel z-score normalized)
2026-03-10 00:08:16,544  INFO      EEGNet Fold 1/10 — train=27, test=3
2026-03-10 00:08:16,544  INFO      Training EEGNet on device: cpu
2026-03-10 00:08:16,926  INFO      EEGNet parameters: 2834
2026-03-10 00:08:19,741  INFO        Epoch   1/300 — train_loss: 0.6375  val_loss: 0.7018  val_acc: 0.5000
2026-03-10 00:08:22,249  INFO        Epoch  25/300 — train_loss: 0.1787  val_loss: 0.7610  val_acc: 0.5000
2026-03-10 00:08:24,756  INFO        Epoch  50/300 — train_loss: 0.0309  val_loss: 0.4986  val_acc: 0.8333
2026-03-10 00:08:27,193  INFO        Epoch  75/300 — train_loss: 0.0081  val_loss: 0.4342  val_acc: 0.8333
2026-03-10 00:08:29,510  INFO        Epoch 100/300 — train_loss: 0.0042  val_loss: 0.3555  val_acc: 0.8333
2026-03-10 00:08:32,379  INFO        Epoch 125/300 — train_loss: 0.0045  val_loss: 0.3659  val_acc: 0.6667
2026-03-10 00:08:34,834  INFO      Early stopping at epoch 139 (patience=30). Best val_loss: 0.3124
2026-03-10 00:08:35,081  INFO        Fold  1/10 — accuracy: 1.0000
2026-03-10 00:08:35,289  INFO      EEGNet Fold 2/10 — train=27, test=3
2026-03-10 00:08:35,289  INFO      Training EEGNet on device: cpu
2026-03-10 00:08:35,353  INFO      EEGNet parameters: 2834
2026-03-10 00:08:35,463  INFO        Epoch   1/300 — train_loss: 0.6282  val_loss: 0.6733  val_acc: 0.5000
2026-03-10 00:08:38,413  INFO        Epoch  25/300 — train_loss: 0.1426  val_loss: 0.3272  val_acc: 1.0000
2026-03-10 00:08:41,861  INFO        Epoch  50/300 — train_loss: 0.0233  val_loss: 0.0988  val_acc: 1.0000
2026-03-10 00:08:45,029  INFO        Epoch  75/300 — train_loss: 0.0055  val_loss: 0.1325  val_acc: 1.0000
2026-03-10 00:08:46,645  INFO      Early stopping at epoch 87 (patience=30). Best val_loss: 0.0897
2026-03-10 00:08:46,667  INFO        Fold  2/10 — accuracy: 0.6667
2026-03-10 00:08:46,678  INFO      EEGNet Fold 3/10 — train=27, test=3
2026-03-10 00:08:46,678  INFO      Training EEGNet on device: cpu
2026-03-10 00:08:46,691  INFO      EEGNet parameters: 2834
2026-03-10 00:08:46,839  INFO        Epoch   1/300 — train_loss: 0.7627  val_loss: 0.6625  val_acc: 0.8333
2026-03-10 00:08:49,708  INFO        Epoch  25/300 — train_loss: 0.2200  val_loss: 0.3206  val_acc: 0.8333
2026-03-10 00:08:52,626  INFO        Epoch  50/300 — train_loss: 0.0572  val_loss: 0.1561  val_acc: 1.0000
2026-03-10 00:08:57,090  INFO        Epoch  75/300 — train_loss: 0.0237  val_loss: 0.1256  val_acc: 1.0000
2026-03-10 00:09:00,712  INFO        Epoch 100/300 — train_loss: 0.0092  val_loss: 0.0988  val_acc: 1.0000
2026-03-10 00:09:04,447  INFO        Epoch 125/300 — train_loss: 0.0049  val_loss: 0.0853  val_acc: 1.0000
2026-03-10 00:09:07,600  INFO        Epoch 150/300 — train_loss: 0.0048  val_loss: 0.0714  val_acc: 1.0000
2026-03-10 00:09:10,800  INFO        Epoch 175/300 — train_loss: 0.0019  val_loss: 0.0716  val_acc: 1.0000
2026-03-10 00:09:14,075  INFO        Epoch 200/300 — train_loss: 0.0022  val_loss: 0.0667  val_acc: 1.0000
2026-03-10 00:09:16,942  INFO        Epoch 225/300 — train_loss: 0.0012  val_loss: 0.0603  val_acc: 1.0000
2026-03-10 00:09:20,060  INFO        Epoch 250/300 — train_loss: 0.0018  val_loss: 0.0521  val_acc: 1.0000
2026-03-10 00:09:23,021  INFO        Epoch 275/300 — train_loss: 0.0014  val_loss: 0.0586  val_acc: 1.0000
2026-03-10 00:09:23,283  INFO      Early stopping at epoch 277 (patience=30). Best val_loss: 0.0512
2026-03-10 00:09:23,298  INFO        Fold  3/10 — accuracy: 0.6667
2026-03-10 00:09:23,314  INFO      EEGNet Fold 4/10 — train=27, test=3
2026-03-10 00:09:23,314  INFO      Training EEGNet on device: cpu
2026-03-10 00:09:23,329  INFO      EEGNet parameters: 2834
2026-03-10 00:09:23,432  INFO        Epoch   1/300 — train_loss: 0.7232  val_loss: 0.7004  val_acc: 0.5000
2026-03-10 00:09:26,557  INFO        Epoch  25/300 — train_loss: 0.1913  val_loss: 0.4087  val_acc: 0.8333
2026-03-10 00:09:29,501  INFO        Epoch  50/300 — train_loss: 0.0241  val_loss: 0.6188  val_acc: 0.8333
2026-03-10 00:09:30,352  INFO      Early stopping at epoch 57 (patience=30). Best val_loss: 0.4051
2026-03-10 00:09:30,374  INFO        Fold  4/10 — accuracy: 1.0000
2026-03-10 00:09:30,384  INFO      EEGNet Fold 5/10 — train=27, test=3
2026-03-10 00:09:30,385  INFO      Training EEGNet on device: cpu
2026-03-10 00:09:30,393  INFO      EEGNet parameters: 2834
2026-03-10 00:09:30,510  INFO        Epoch   1/300 — train_loss: 0.7387  val_loss: 0.6986  val_acc: 0.5000
2026-03-10 00:09:34,075  INFO        Epoch  25/300 — train_loss: 0.2887  val_loss: 0.2976  val_acc: 1.0000
2026-03-10 00:09:37,706  INFO        Epoch  50/300 — train_loss: 0.0443  val_loss: 0.1607  val_acc: 1.0000
2026-03-10 00:09:40,196  INFO        Epoch  75/300 — train_loss: 0.0116  val_loss: 0.1274  val_acc: 1.0000
2026-03-10 00:09:42,975  INFO      Early stopping at epoch 98 (patience=30). Best val_loss: 0.1264
2026-03-10 00:09:42,997  INFO        Fold  5/10 — accuracy: 1.0000
2026-03-10 00:09:43,009  INFO      EEGNet Fold 6/10 — train=27, test=3
2026-03-10 00:09:43,009  INFO      Training EEGNet on device: cpu
2026-03-10 00:09:43,021  INFO      EEGNet parameters: 2834
2026-03-10 00:09:43,150  INFO        Epoch   1/300 — train_loss: 0.7774  val_loss: 0.6853  val_acc: 0.6667
2026-03-10 00:09:45,981  INFO        Epoch  25/300 — train_loss: 0.2213  val_loss: 0.3307  val_acc: 0.8333
2026-03-10 00:09:49,842  INFO        Epoch  50/300 — train_loss: 0.0395  val_loss: 0.1673  val_acc: 1.0000
2026-03-10 00:09:53,132  INFO        Epoch  75/300 — train_loss: 0.0149  val_loss: 0.1277  val_acc: 1.0000
2026-03-10 00:09:55,781  INFO        Epoch 100/300 — train_loss: 0.0069  val_loss: 0.1356  val_acc: 1.0000
2026-03-10 00:09:56,407  INFO      Early stopping at epoch 106 (patience=30). Best val_loss: 0.1269
2026-03-10 00:09:56,423  INFO        Fold  6/10 — accuracy: 1.0000
2026-03-10 00:09:56,430  INFO      EEGNet Fold 7/10 — train=27, test=3
2026-03-10 00:09:56,430  INFO      Training EEGNet on device: cpu
2026-03-10 00:09:56,441  INFO      EEGNet parameters: 2834
2026-03-10 00:09:56,545  INFO        Epoch   1/300 — train_loss: 0.7550  val_loss: 0.6724  val_acc: 0.8333
2026-03-10 00:09:59,413  INFO        Epoch  25/300 — train_loss: 0.1425  val_loss: 0.2790  val_acc: 1.0000

2026-03-10 00:10:02,350  INFO        Epoch  50/300 — train_loss: 0.0226  val_loss: 0.1830  val_acc: 1.0000
2026-03-10 00:10:05,165  INFO        Epoch  75/300 — train_loss: 0.0080  val_loss: 0.1718  val_acc: 1.0000
2026-03-10 00:10:08,303  INFO        Epoch 100/300 — train_loss: 0.0049  val_loss: 0.1612  val_acc: 1.0000
2026-03-10 00:10:11,270  INFO        Epoch 125/300 — train_loss: 0.0029  val_loss: 0.1537  val_acc: 1.0000
2026-03-10 00:10:14,571  INFO        Epoch 150/300 — train_loss: 0.0041  val_loss: 0.1625  val_acc: 1.0000
2026-03-10 00:10:17,100  INFO      Early stopping at epoch 168 (patience=30). Best val_loss: 0.1502
2026-03-10 00:10:17,125  INFO        Fold  7/10 — accuracy: 0.6667
2026-03-10 00:10:17,136  INFO      EEGNet Fold 8/10 — train=27, test=3
2026-03-10 00:10:17,137  INFO      Training EEGNet on device: cpu
2026-03-10 00:10:17,148  INFO      EEGNet parameters: 2834
2026-03-10 00:10:17,337  INFO        Epoch   1/300 — train_loss: 0.7471  val_loss: 0.6774  val_acc: 0.6667
2026-03-10 00:10:20,394  INFO        Epoch  25/300 — train_loss: 0.2278  val_loss: 0.6291  val_acc: 0.5000
2026-03-10 00:10:23,264  INFO        Epoch  50/300 — train_loss: 0.0569  val_loss: 0.5883  val_acc: 0.6667
2026-03-10 00:10:26,319  INFO        Epoch  75/300 — train_loss: 0.0122  val_loss: 0.5309  val_acc: 0.8333
2026-03-10 00:10:29,304  INFO        Epoch 100/300 — train_loss: 0.0064  val_loss: 0.4908  val_acc: 0.8333
2026-03-10 00:10:32,121  INFO        Epoch 125/300 — train_loss: 0.0025  val_loss: 0.3861  val_acc: 0.8333
2026-03-10 00:10:34,877  INFO        Epoch 150/300 — train_loss: 0.0019  val_loss: 0.3888  val_acc: 0.8333
2026-03-10 00:10:37,758  INFO        Epoch 175/300 — train_loss: 0.0029  val_loss: 0.4052  val_acc: 0.8333
2026-03-10 00:10:38,928  INFO      Early stopping at epoch 184 (patience=30). Best val_loss: 0.3564
2026-03-10 00:10:38,947  INFO        Fold  8/10 — accuracy: 1.0000
2026-03-10 00:10:38,958  INFO      EEGNet Fold 9/10 — train=27, test=3
2026-03-10 00:10:38,959  INFO      Training EEGNet on device: cpu
2026-03-10 00:10:38,972  INFO      EEGNet parameters: 2834
2026-03-10 00:10:39,106  INFO        Epoch   1/300 — train_loss: 0.7262  val_loss: 0.6883  val_acc: 0.6667
2026-03-10 00:10:41,788  INFO        Epoch  25/300 — train_loss: 0.2046  val_loss: 0.3989  val_acc: 1.0000
2026-03-10 00:10:44,481  INFO        Epoch  50/300 — train_loss: 0.0801  val_loss: 0.1174  val_acc: 1.0000
2026-03-10 00:10:47,298  INFO        Epoch  75/300 — train_loss: 0.0202  val_loss: 0.0455  val_acc: 1.0000
2026-03-10 00:10:50,319  INFO        Epoch 100/300 — train_loss: 0.0063  val_loss: 0.0240  val_acc: 1.0000
2026-03-10 00:10:53,159  INFO        Epoch 125/300 — train_loss: 0.0052  val_loss: 0.0185  val_acc: 1.0000
2026-03-10 00:10:55,853  INFO        Epoch 150/300 — train_loss: 0.0036  val_loss: 0.0142  val_acc: 1.0000
2026-03-10 00:10:58,605  INFO        Epoch 175/300 — train_loss: 0.0030  val_loss: 0.0175  val_acc: 1.0000
2026-03-10 00:11:00,083  INFO      Early stopping at epoch 187 (patience=30). Best val_loss: 0.0133
2026-03-10 00:11:00,098  INFO        Fold  9/10 — accuracy: 1.0000
2026-03-10 00:11:00,115  INFO      EEGNet Fold 10/10 — train=27, test=3
2026-03-10 00:11:00,115  INFO      Training EEGNet on device: cpu
2026-03-10 00:11:00,131  INFO      EEGNet parameters: 2834
2026-03-10 00:11:00,269  INFO        Epoch   1/300 — train_loss: 0.6607  val_loss: 0.6963  val_acc: 0.3333
2026-03-10 00:11:03,532  INFO        Epoch  25/300 — train_loss: 0.2174  val_loss: 0.3913  val_acc: 1.0000
2026-03-10 00:11:06,662  INFO        Epoch  50/300 — train_loss: 0.0218  val_loss: 0.1036  val_acc: 1.0000
2026-03-10 00:11:10,612  INFO        Epoch  75/300 — train_loss: 0.0084  val_loss: 0.0736  val_acc: 1.0000
2026-03-10 00:11:13,123  INFO        Epoch 100/300 — train_loss: 0.0054  val_loss: 0.0591  val_acc: 1.0000
2026-03-10 00:11:15,693  INFO        Epoch 125/300 — train_loss: 0.0027  val_loss: 0.0437  val_acc: 1.0000
2026-03-10 00:11:18,176  INFO        Epoch 150/300 — train_loss: 0.0041  val_loss: 0.0440  val_acc: 1.0000
2026-03-10 00:11:20,668  INFO        Epoch 175/300 — train_loss: 0.0026  val_loss: 0.0326  val_acc: 1.0000
2026-03-10 00:11:23,119  INFO        Epoch 200/300 — train_loss: 0.0022  val_loss: 0.0301  val_acc: 1.0000
2026-03-10 00:11:25,592  INFO        Epoch 225/300 — train_loss: 0.0011  val_loss: 0.0254  val_acc: 1.0000
2026-03-10 00:11:28,164  INFO        Epoch 250/300 — train_loss: 0.0007  val_loss: 0.0237  val_acc: 1.0000
2026-03-10 00:11:31,083  INFO        Epoch 275/300 — train_loss: 0.0013  val_loss: 0.0203  val_acc: 1.0000
2026-03-10 00:11:33,760  INFO        Epoch 300/300 — train_loss: 0.0007  val_loss: 0.0188  val_acc: 1.0000
2026-03-10 00:11:33,774  INFO        Fold 10/10 — accuracy: 1.0000
2026-03-10 00:11:33,837  INFO      EEGNet — Accuracy: 0.9000 +/- 0.1528  F1: 0.8990  AUC: 0.9688

==================================================
EEGNet — Cross-Validation Results
==================================================
  Fold  1: 1.0000
  Fold  2: 0.6667
  Fold  3: 0.6667
  Fold  4: 1.0000
  Fold  5: 1.0000
  Fold  6: 1.0000
  Fold  7: 0.6667
  Fold  8: 1.0000
  Fold  9: 1.0000
  Fold 10: 1.0000
--------------------------------------------------
  Mean accuracy:  0.9000 +/- 0.1528
  Macro F1-score: 0.8990
  AUC-ROC:        0.9688
==================================================


=======================================================
  EEGNet_Raw — Subject 1 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.9000
  precision_macro               0.9027
  precision_left                0.8824
  precision_right               0.9231
  recall_macro                  0.8973
  recall_left                   0.9375
  recall_right                  0.8571
  f1_macro                      0.8990
  f1_left                       0.9091
  f1_right                      0.8889
  auc_roc                       0.9688
  cohens_kappa                  0.7982
  cv_mean_accuracy              0.9000
  cv_std_accuracy               0.1528
=======================================================

2026-03-10 00:11:33,869  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\EEGNet_Raw_subject001.json
2026-03-10 00:11:34,182  INFO      Confusion matrix saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\figures\confusion_EEGNet_Raw_subject001.png
2026-03-10 00:11:35,917  INFO      Saved figure: roc_EEGNet_Raw_subject001 (.png + .pdf)
2026-03-10 00:11:36,391  INFO      Saved figure: training_curves_EEGNet_subject001 (.png + .pdf)
2026-03-10 00:11:36,654  INFO      Saved figure: roc_comparison_subject001 (.png + .pdf)
Subjects: 100%|███████████████████████████████████████████████████████████████████████| 1/1 [03:27<00:00, 207.14s/it]
2026-03-10 00:11:36,706  INFO      Comparison CSV saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\model_comparison.csv
2026-03-10 00:11:36,708  INFO      Comparison JSON saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\model_comparison.json
2026-03-10 00:11:37,298  INFO      Saved figure: subject_accuracy_comparison (.png + .pdf)
2026-03-10 00:11:37,301  INFO      Full metrics CSV saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\full_metrics.csv
2026-03-10 00:11:37,308  INFO      Evaluation report saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\evaluation_report.txt
2026-03-10 00:11:37,308  INFO      Aggregate stats saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\aggregate_stats.json
===========================================================================
  EEG BCI PIPELINE — COMPREHENSIVE EVALUATION REPORT
  Generated: 2026-03-10 00:11:37
===========================================================================

Per-Subject Accuracy Breakdown
===========================================================================
 Subject        LR+PSD        LR+CSP        EEGNet          Best
---------------------------------------------------------------------------
       1        0.6333        0.8667        0.9000        EEGNet


Aggregate Statistics Across Subjects
===========================================================================
  Model               Mean      Std      Min      Max   Median     N
  --------------- -------- -------- -------- -------- -------- -----
  LR_PSD            0.6333      nan   0.6333   0.6333   0.6333     1
  LR_CSP            0.8667      nan   0.8667   0.8667   0.8667     1
  EEGNet_Raw        0.9000      nan   0.9000   0.9000   0.9000     1

F1 (Macro) and AUC-ROC Summary
===========================================================================
  Model              Mean F1   Mean AUC
  --------------- ---------- ----------
  LR_PSD              0.6329     0.6339
  LR_CSP              0.8667     0.9732
  EEGNet_Raw          0.8990     0.9688

Pairwise Statistical Tests (paired t-test)
===========================================================================
  LR_PSD vs LR_CSP: t=nan, p=nan — not significant
  LR_PSD vs EEGNet_Raw: t=nan, p=nan — not significant
  LR_CSP vs EEGNet_Raw: t=nan, p=nan — not significant

Best model by mean accuracy: EEGNet_Raw (0.9000)

===========================================================================
  END OF REPORT
===========================================================================

================================================================================
  PIPELINE COMPLETE — Model Comparison
================================================================================

  Per-Subject Accuracy:
 subject  lr_psd_accuracy  lr_csp_accuracy  eegnet_accuracy
       1           0.6333           0.8667           0.9000

--------------------------------------------------------------------------------
  Model Summary:
  Model             Mean Acc    Std Acc    Mean F1   Mean AUC
  --------------- ---------- ---------- ---------- ----------
  LR + PSD            0.6333        nan     0.6329     0.6339
  LR + CSP            0.8667        nan     0.8667     0.9732
  EEGNet + Raw        0.9000        nan     0.8990     0.9688

  Pairwise t-tests (paired, p < 0.05):
  LR_PSD vs LR_CSP: t=nan, p=nan — significant: no
  LR_PSD vs EEGNet_Raw: t=nan, p=nan — significant: no
  LR_CSP vs EEGNet_Raw: t=nan, p=nan — significant: no

  Subjects completed: 1/1
================================================================================







==================================================================================================================================================
================================================================================
==================================================================================================================================================
================================================================================
==================================================================================================================================================

$  python train.py --subjects 1 2 3 --no-cache --ica --tune --augment
2026-03-10 00:12:45,318  INFO      Global random seeds set to 42
2026-03-10 00:12:45,319  INFO      Pipeline configuration: {'PROJECT_ROOT': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI'), 'DATA_PATH': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/data/eegbci'), 'OUTPUT_DIR': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/outputs'), 'FIGURES_DIR': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/outputs/figures'), 'MODELS_DIR': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/outputs/models'), 'RESULTS_DIR': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/outputs/results'), 'SUBJECTS': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109], 'CACHE_RESULTS': True, 'RUNS': [3, 7], 'SAMPLING_RATE': 160, 'N_CHANNELS': 64, 'BANDPASS_LOW': 1.0, 'BANDPASS_HIGH': 40.0, 'NOTCH_FREQS': [60.0, 120.0], 'EPOCH_TMIN': -0.5, 'EPOCH_TMAX': 4.0, 'BASELINE': (None, 0), 'REJECT_THRESHOLD': None, 'REJECT_WARN_RATIO': 0.3, 'USE_ICA': True, 'ICA_N_COMPONENTS': 20, 'ICA_METHOD': 'fastica', 'ICA_RANDOM_STATE': 42, 'ICA_EOG_THRESHOLD': 3.0, 'ICA_MUSCLE_THRESHOLD': 1.0, 'EVENT_ID': {'left': 1, 'right': 2}, 'PSD_FMIN': 1.0, 'PSD_FMAX': 40.0, 'PSD_N_FFT': 
320, 'PSD_N_OVERLAP': 160, 'PSD_WINDOW': 'hann', 'FREQ_BANDS': {'delta': (1, 4), 'theta': (4, 8), 'mu': (8, 12), 'low_beta': (13, 20), 'high_beta': (20, 30), 'low_gamma': (30, 40)}, 'USE_ROI_CHANNELS': False, 'MOTOR_ROI_CHANNELS': ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4'], 'CSP_N_COMPONENTS': 4, 'LR_SOLVER': 'lbfgs', 'LR_MAX_ITER': 1000, 'LR_CLASS_WEIGHT': 'balanced', 'LR_C_GRID': [0.001, 0.01, 0.1, 1.0, 10.0], 'EEGNET_F1': 8, 'EEGNET_F2': 16, 'EEGNET_D': 2, 'EEGNET_DROPOUT': 0.5, 'EEGNET_LR': 0.001, 'EEGNET_BATCH_SIZE': 32, 'EEGNET_MAX_EPOCHS': 300, 'EEGNET_PATIENCE': 30, 'CV_N_FOLDS': 10, 'RANDOM_SEED': 42, 'AUGMENT_GAUSSIAN_STD': 0.01, 'AUGMENT_TEMPORAL_JITTER_MS': 50, 'USE_MLFLOW': False, 'MLFLOW_EXPERIMENT_NAME': 'EEG-BCI-Pipeline', 'MLFLOW_TRACKING_URI': 'file:./mlruns'}
2026-03-10 00:12:45,319  INFO      Processing 3 subjects: [1, 2, 3]
2026-03-10 00:12:45,319  INFO      Options: cache=False, tune=True, augment=True, loso=False, ica=True, roi=False     
Subjects:   0%|                                                                                | 0/3 [00:00<?, ?it/s]2026-03-10 00:12:45,319  INFO      ============================================================
2026-03-10 00:12:45,319  INFO      Processing Subject 001
2026-03-10 00:12:45,319  INFO      ============================================================
2026-03-10 00:12:45,450  INFO      Downloaded  subject=001  file=S001R03.edf
2026-03-10 00:12:45,450  INFO      Downloaded  subject=001  file=S001R07.edf
2026-03-10 00:12:45,450  INFO      Download complete: 1 subjects, 2 total files.
2026-03-10 00:12:46,199  INFO      Loaded  subject=001  runs=[3, 7]  channels=64  sfreq=160 Hz  n_times=40000  duration=250.0 s
2026-03-10 00:12:46,211  INFO      Applying filters: bandpass 1.0–40.0 Hz, notch [60.0, 120.0] Hz, FIR
2026-03-10 00:12:46,433  INFO      Filtering complete: shape before=(64, 40000)  after=(64, 40000)  channels=64
2026-03-10 00:12:48,039  INFO      ICA fitted: method=fastica, n_components=20, n_iter=66
2026-03-10 00:12:48,552  INFO      ICA artifact removal: excluded 1 components [(0, 'EOG (via Fp1)')]
2026-03-10 00:12:48,552  INFO      ICA applied: excluded 1 components
2026-03-10 00:12:48,552  INFO      Event mapping: {'left': 2, 'right': 3} (from raw annotations: {'T0': 1, 'T1': 2, 'T2': 3})
2026-03-10 00:12:48,624  INFO      Epochs: total=30  kept=30  rejected=0 (0.0%)  left=16  right=14
2026-03-10 00:12:48,624  INFO      --- Signal Visualizations ---
2026-03-10 00:12:50,197  INFO      Saved figure: topomap_subject001 (.png + .pdf)
2026-03-10 00:12:50,933  INFO      Saved figure: psd_comparison_subject001 (.png + .pdf)
2026-03-10 00:12:51,951  INFO      Saved figure: butterfly_subject001 (.png + .pdf)
2026-03-10 00:12:51,951  INFO      Generated 6 signal figures for Subject 001
2026-03-10 00:12:51,951  INFO      --- LR + PSD ---
2026-03-10 00:12:52,340  INFO      PSD features: X=(30, 384)  y=(30,)  bands=6  channels=64
2026-03-10 00:12:52,369  INFO        Fold  1/10 — accuracy: 0.6667
2026-03-10 00:12:52,383  INFO        Fold  2/10 — accuracy: 0.6667
2026-03-10 00:12:52,398  INFO        Fold  3/10 — accuracy: 0.6667
2026-03-10 00:12:52,408  INFO        Fold  4/10 — accuracy: 0.6667
2026-03-10 00:12:52,420  INFO        Fold  5/10 — accuracy: 1.0000
2026-03-10 00:12:52,431  INFO        Fold  6/10 — accuracy: 0.6667
2026-03-10 00:12:52,441  INFO        Fold  7/10 — accuracy: 0.3333
2026-03-10 00:12:52,452  INFO        Fold  8/10 — accuracy: 0.6667
2026-03-10 00:12:52,459  INFO        Fold  9/10 — accuracy: 0.6667
2026-03-10 00:12:52,473  INFO        Fold 10/10 — accuracy: 0.0000
2026-03-10 00:12:52,479  INFO      Logistic Regression — Accuracy: 0.6000 +/- 0.2494  F1: 0.5982  AUC: 0.6205

==================================================
Logistic Regression — Cross-Validation Results
==================================================
  Fold  1: 0.6667
  Fold  2: 0.6667
  Fold  3: 0.6667
  Fold  4: 0.6667
  Fold  5: 1.0000
  Fold  6: 0.6667
  Fold  7: 0.3333
  Fold  8: 0.6667
  Fold  9: 0.6667
  Fold 10: 0.0000
--------------------------------------------------
  Mean accuracy:  0.6000 +/- 0.2494
  Macro F1-score: 0.5982
  AUC-ROC:        0.6205
==================================================


=======================================================
  LR_PSD — Subject 1 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.6000
  precision_macro               0.6111
  precision_left                0.6667
  precision_right               0.5556
  recall_macro                  0.6071
  recall_left                   0.5000
  recall_right                  0.7143
  f1_macro                      0.5982
  f1_left                       0.5714
  f1_right                      0.6250
  auc_roc                       0.6205
  cohens_kappa                  0.2105
  cv_mean_accuracy              0.6000
  cv_std_accuracy               0.2494
=======================================================

2026-03-10 00:12:52,498  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\LR_PSD_subject001.json
2026-03-10 00:12:52,631  INFO      Confusion matrix saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\figures\confusion_LR_PSD_subject001.png
2026-03-10 00:12:52,838  INFO      Saved figure: roc_LR_PSD_subject001 (.png + .pdf)
2026-03-10 00:12:53,137  INFO      Saved figure: feature_importance_LR_PSD_subject001 (.png + .pdf)
2026-03-10 00:12:53,138  INFO      --- LR + PSD (Tuned) ---
2026-03-10 00:13:05,413  INFO        Fold  1/10 — accuracy: 0.6667  best_C: 0.1000
2026-03-10 00:13:05,468  INFO        Fold  2/10 — accuracy: 0.6667  best_C: 0.0100
2026-03-10 00:13:05,515  INFO        Fold  3/10 — accuracy: 0.6667  best_C: 0.1000
2026-03-10 00:13:05,566  INFO        Fold  4/10 — accuracy: 0.6667  best_C: 10.0000
2026-03-10 00:13:05,608  INFO        Fold  5/10 — accuracy: 1.0000  best_C: 0.0100
2026-03-10 00:13:05,658  INFO        Fold  6/10 — accuracy: 0.6667  best_C: 1.0000
2026-03-10 00:13:05,712  INFO        Fold  7/10 — accuracy: 0.0000  best_C: 0.1000
2026-03-10 00:13:05,784  INFO        Fold  8/10 — accuracy: 0.3333  best_C: 0.0100
2026-03-10 00:13:05,834  INFO        Fold  9/10 — accuracy: 0.6667  best_C: 1.0000
2026-03-10 00:13:05,878  INFO        Fold 10/10 — accuracy: 0.0000  best_C: 10.0000
2026-03-10 00:13:05,878  INFO      Tuned LR — Accuracy: 0.5333 +/- 0.3055  F1: 0.5333  AUC: 0.5625  Best C: 0.1

=======================================================
Logistic Regression (Tuned) — Cross-Validation Results
=======================================================
  Fold  1: 0.6667  (C=0.1)
  Fold  2: 0.6667  (C=0.01)
  Fold  3: 0.6667  (C=0.1)
  Fold  4: 0.6667  (C=10.0)
  Fold  5: 1.0000  (C=0.01)
  Fold  6: 0.6667  (C=1.0)
  Fold  7: 0.0000  (C=0.1)
  Fold  8: 0.3333  (C=0.01)
  Fold  9: 0.6667  (C=1.0)
  Fold 10: 0.0000  (C=10.0)
-------------------------------------------------------
  Mean accuracy:  0.5333 +/- 0.3055
  Macro F1-score: 0.5333
  AUC-ROC:        0.5625
  Overall best C: 0.1
=======================================================


=======================================================
  LR_PSD_Tuned — Subject 1 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.5333
  precision_macro               0.5357
  precision_left                0.5714
  precision_right               0.5000
  recall_macro                  0.5357
  recall_left                   0.5000
  recall_right                  0.5714
  f1_macro                      0.5333
  f1_left                       0.5333
  f1_right                      0.5333
  auc_roc                       0.5625
  cohens_kappa                  0.0708
  cv_mean_accuracy              0.5333
  cv_std_accuracy               0.3055
  best_C                        0.1000
=======================================================

2026-03-10 00:13:05,905  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\LR_PSD_Tuned_subject001.json
2026-03-10 00:13:05,909  INFO        Tuned vs Default: 0.5333 vs 0.6000 (delta=-0.0667)
2026-03-10 00:13:05,909  INFO      --- LR + CSP ---
Computing rank from data with rank=None
    Using tolerance 0.00028 (2.2e-16 eps * 64 dim * 2e+10  max singular value)
    Estimated rank (data): 62
    data: rank 62 computed from 64 data channels with 0 projectors
    Setting small data eigenvalues to zero (without PCA)
Reducing data rank from 64 -> 62
Estimating class=0 covariance using EMPIRICAL
Done.
Estimating class=1 covariance using EMPIRICAL
Done.
    Setting small data eigenvalues to zero (without PCA)
2026-03-10 00:13:06,202  INFO      CSP features: X=(30, 4)  y=(30,)  components=4
2026-03-10 00:13:06,203  INFO      CSP filter weight ranges: ['comp0: [-50012.1216, 35404.4346]', 'comp1: [-188928.6751, 186755.9542]', 'comp2: [-87594.7221, 51439.0721]', 'comp3: [-37081.3941, 29411.8117]']
2026-03-10 00:13:06,212  INFO        Fold  1/10 — accuracy: 1.0000
2026-03-10 00:13:06,217  INFO        Fold  2/10 — accuracy: 1.0000
2026-03-10 00:13:06,222  INFO        Fold  3/10 — accuracy: 1.0000
2026-03-10 00:13:06,227  INFO        Fold  4/10 — accuracy: 1.0000
2026-03-10 00:13:06,234  INFO        Fold  5/10 — accuracy: 1.0000
2026-03-10 00:13:06,239  INFO        Fold  6/10 — accuracy: 1.0000
2026-03-10 00:13:06,243  INFO        Fold  7/10 — accuracy: 1.0000
2026-03-10 00:13:06,247  INFO        Fold  8/10 — accuracy: 0.6667
2026-03-10 00:13:06,253  INFO        Fold  9/10 — accuracy: 1.0000
2026-03-10 00:13:06,258  INFO        Fold 10/10 — accuracy: 1.0000
2026-03-10 00:13:06,262  INFO      Logistic Regression — Accuracy: 0.9667 +/- 0.1000  F1: 0.9666  AUC: 0.9955

==================================================
Logistic Regression — Cross-Validation Results
==================================================
  Fold  1: 1.0000
  Fold  2: 1.0000
  Fold  3: 1.0000
  Fold  4: 1.0000
  Fold  5: 1.0000
  Fold  6: 1.0000
  Fold  7: 1.0000
  Fold  8: 0.6667
  Fold  9: 1.0000
  Fold 10: 1.0000
--------------------------------------------------
  Mean accuracy:  0.9667 +/- 0.1000
  Macro F1-score: 0.9666
  AUC-ROC:        0.9955
==================================================


=======================================================
  LR_CSP — Subject 1 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.9667
  precision_macro               0.9667
  precision_left                1.0000
  precision_right               0.9333
  recall_macro                  0.9688
  recall_left                   0.9375
  recall_right                  1.0000
  f1_macro                      0.9666
  f1_left                       0.9677
  f1_right                      0.9655
  auc_roc                       0.9955
  cohens_kappa                  0.9333
  cv_mean_accuracy              0.9667
  cv_std_accuracy               0.1000
=======================================================

2026-03-10 00:13:06,280  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\LR_CSP_subject001.json
2026-03-10 00:13:06,460  INFO      Confusion matrix saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\figures\confusion_LR_CSP_subject001.png
2026-03-10 00:13:06,699  INFO      Saved figure: roc_LR_CSP_subject001 (.png + .pdf)
2026-03-10 00:13:06,699  INFO      --- EEGNet + Raw ---
2026-03-10 00:13:06,738  INFO      Raw features: X=(30, 64, 721)  y=(30,)  (per-channel z-score normalized)
2026-03-10 00:13:06,740  INFO      --- EEGNet + Raw (Augmented) ---
2026-03-10 00:13:06,773  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:13:06,786  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:13:06,802  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:13:06,807  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:13:06,807  INFO      EEGNet Fold 1/10 — train=81, test=3
2026-03-10 00:13:06,808  INFO      Training EEGNet on device: cpu
2026-03-10 00:13:07,096  INFO      EEGNet parameters: 2834
2026-03-10 00:13:11,679  INFO        Epoch   1/300 — train_loss: 0.7028  val_loss: 0.6755  val_acc: 0.7647
2026-03-10 00:13:19,484  INFO        Epoch  25/300 — train_loss: 0.0627  val_loss: 0.0248  val_acc: 1.0000
2026-03-10 00:13:30,486  INFO        Epoch  50/300 — train_loss: 0.0103  val_loss: 0.0027  val_acc: 1.0000
2026-03-10 00:13:39,028  INFO        Epoch  75/300 — train_loss: 0.0060  val_loss: 0.0012  val_acc: 1.0000
2026-03-10 00:14:53,472  INFO        Epoch 100/300 — train_loss: 0.0018  val_loss: 0.0006  val_acc: 1.0000
2026-03-10 00:15:07,913  INFO        Epoch 125/300 — train_loss: 0.0009  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:15:17,456  INFO        Epoch 150/300 — train_loss: 0.0013  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:15:25,396  INFO        Epoch 175/300 — train_loss: 0.0009  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:16:36,560  INFO        Epoch 200/300 — train_loss: 0.0005  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:18:01,579  INFO        Epoch 225/300 — train_loss: 0.0005  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:19:29,494  INFO        Epoch 250/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:20:51,710  INFO        Epoch 275/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:22:05,866  INFO        Epoch 300/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:22:06,280  INFO        Fold  1/10 — accuracy: 1.0000
2026-03-10 00:22:06,614  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:22:06,644  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:22:06,689  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:22:06,694  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:22:06,695  INFO      EEGNet Fold 2/10 — train=81, test=3
2026-03-10 00:22:06,697  INFO      Training EEGNet on device: cpu
2026-03-10 00:22:06,935  INFO      EEGNet parameters: 2834
2026-03-10 00:22:10,451  INFO        Epoch   1/300 — train_loss: 0.7089  val_loss: 0.6856  val_acc: 0.7059
2026-03-10 00:23:37,567  INFO        Epoch  25/300 — train_loss: 0.0469  val_loss: 0.0261  val_acc: 1.0000
2026-03-10 00:24:16,231  INFO        Epoch  50/300 — train_loss: 0.0091  val_loss: 0.0038  val_acc: 1.0000
2026-03-10 00:24:24,594  INFO        Epoch  75/300 — train_loss: 0.0057  val_loss: 0.0019  val_acc: 1.0000
2026-03-10 00:24:33,365  INFO        Epoch 100/300 — train_loss: 0.0020  val_loss: 0.0010  val_acc: 1.0000
2026-03-10 00:24:42,594  INFO        Epoch 125/300 — train_loss: 0.0018  val_loss: 0.0006  val_acc: 1.0000
2026-03-10 00:24:51,098  INFO        Epoch 150/300 — train_loss: 0.0008  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:24:59,445  INFO        Epoch 175/300 — train_loss: 0.0008  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:25:41,824  INFO        Epoch 200/300 — train_loss: 0.0005  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:26:57,410  INFO        Epoch 225/300 — train_loss: 0.0004  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:28:15,648  INFO        Epoch 250/300 — train_loss: 0.0006  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:28:24,674  INFO        Epoch 275/300 — train_loss: 0.0010  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:28:32,522  INFO        Epoch 300/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:28:32,675  INFO        Fold  2/10 — accuracy: 0.6667
2026-03-10 00:28:32,845  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:28:32,868  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:28:32,888  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:28:32,888  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:28:32,888  INFO      EEGNet Fold 3/10 — train=81, test=3
2026-03-10 00:28:32,888  INFO      Training EEGNet on device: cpu
2026-03-10 00:28:32,942  INFO      EEGNet parameters: 2834
2026-03-10 00:28:33,477  INFO        Epoch   1/300 — train_loss: 0.7277  val_loss: 0.6856  val_acc: 0.5294
2026-03-10 00:28:43,861  INFO        Epoch  25/300 — train_loss: 0.0584  val_loss: 0.0410  val_acc: 1.0000
2026-03-10 00:28:54,713  INFO        Epoch  50/300 — train_loss: 0.0095  val_loss: 0.0047  val_acc: 1.0000
2026-03-10 00:29:05,409  INFO        Epoch  75/300 — train_loss: 0.0029  val_loss: 0.0021  val_acc: 1.0000
2026-03-10 00:29:12,473  INFO        Epoch 100/300 — train_loss: 0.0025  val_loss: 0.0012  val_acc: 1.0000
2026-03-10 00:29:21,359  INFO        Epoch 125/300 — train_loss: 0.0012  val_loss: 0.0006  val_acc: 1.0000
2026-03-10 00:29:28,256  INFO        Epoch 150/300 — train_loss: 0.0008  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:29:35,319  INFO        Epoch 175/300 — train_loss: 0.0005  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:29:43,143  INFO        Epoch 200/300 — train_loss: 0.0005  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:29:49,773  INFO        Epoch 225/300 — train_loss: 0.0010  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:29:58,047  INFO        Epoch 250/300 — train_loss: 0.0005  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:30:05,331  INFO        Epoch 275/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:30:12,756  INFO        Epoch 300/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:30:12,875  INFO        Fold  3/10 — accuracy: 0.6667
2026-03-10 00:30:12,938  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:30:12,954  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:30:12,987  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:30:12,994  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:30:12,994  INFO      EEGNet Fold 4/10 — train=81, test=3
2026-03-10 00:30:12,995  INFO      Training EEGNet on device: cpu
2026-03-10 00:30:13,043  INFO      EEGNet parameters: 2834
2026-03-10 00:30:13,311  INFO        Epoch   1/300 — train_loss: 0.6533  val_loss: 0.6848  val_acc: 0.6471
2026-03-10 00:30:19,789  INFO        Epoch  25/300 — train_loss: 0.0432  val_loss: 0.0306  val_acc: 1.0000
2026-03-10 00:30:26,582  INFO        Epoch  50/300 — train_loss: 0.0083  val_loss: 0.0042  val_acc: 1.0000
2026-03-10 00:30:33,236  INFO        Epoch  75/300 — train_loss: 0.0036  val_loss: 0.0016  val_acc: 1.0000
2026-03-10 00:30:39,985  INFO        Epoch 100/300 — train_loss: 0.0038  val_loss: 0.0009  val_acc: 1.0000
2026-03-10 00:30:46,799  INFO        Epoch 125/300 — train_loss: 0.0025  val_loss: 0.0006  val_acc: 1.0000
2026-03-10 00:30:53,344  INFO        Epoch 150/300 — train_loss: 0.0013  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:30:59,941  INFO        Epoch 175/300 — train_loss: 0.0008  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:31:06,576  INFO        Epoch 200/300 — train_loss: 0.0006  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:31:14,158  INFO        Epoch 225/300 — train_loss: 0.0003  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:31:21,236  INFO        Epoch 250/300 — train_loss: 0.0016  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:31:27,873  INFO        Epoch 275/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:31:34,667  INFO        Epoch 300/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:31:34,692  INFO        Fold  4/10 — accuracy: 1.0000
2026-03-10 00:31:34,756  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:31:34,788  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:31:34,843  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:31:34,851  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:31:34,852  INFO      EEGNet Fold 5/10 — train=81, test=3
2026-03-10 00:31:34,852  INFO      Training EEGNet on device: cpu
2026-03-10 00:31:34,898  INFO      EEGNet parameters: 2834
2026-03-10 00:31:35,333  INFO        Epoch   1/300 — train_loss: 0.7052  val_loss: 0.6927  val_acc: 0.4706
2026-03-10 00:31:44,071  INFO        Epoch  25/300 — train_loss: 0.0481  val_loss: 0.0213  val_acc: 1.0000
2026-03-10 00:31:51,017  INFO        Epoch  50/300 — train_loss: 0.0053  val_loss: 0.0039  val_acc: 1.0000
2026-03-10 00:31:57,804  INFO        Epoch  75/300 — train_loss: 0.0029  val_loss: 0.0015  val_acc: 1.0000
2026-03-10 00:32:04,705  INFO        Epoch 100/300 — train_loss: 0.0019  val_loss: 0.0009  val_acc: 1.0000
2026-03-10 00:32:11,334  INFO        Epoch 125/300 — train_loss: 0.0031  val_loss: 0.0006  val_acc: 1.0000
2026-03-10 00:32:17,971  INFO        Epoch 150/300 — train_loss: 0.0022  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:32:24,501  INFO        Epoch 175/300 — train_loss: 0.0009  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:32:31,064  INFO        Epoch 200/300 — train_loss: 0.0007  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:32:37,728  INFO        Epoch 225/300 — train_loss: 0.0006  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:32:44,329  INFO        Epoch 250/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:32:51,110  INFO        Epoch 275/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:32:57,630  INFO        Epoch 300/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:32:57,638  INFO        Fold  5/10 — accuracy: 1.0000
2026-03-10 00:32:57,675  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:32:57,688  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:32:57,705  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:32:57,710  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:32:57,710  INFO      EEGNet Fold 6/10 — train=81, test=3
2026-03-10 00:32:57,711  INFO      Training EEGNet on device: cpu
2026-03-10 00:32:57,732  INFO      EEGNet parameters: 2834
2026-03-10 00:32:57,986  INFO        Epoch   1/300 — train_loss: 0.6742  val_loss: 0.6790  val_acc: 0.7059
2026-03-10 00:33:04,406  INFO        Epoch  25/300 — train_loss: 0.0729  val_loss: 0.0697  val_acc: 1.0000
2026-03-10 00:33:11,095  INFO        Epoch  50/300 — train_loss: 0.0141  val_loss: 0.0146  val_acc: 1.0000
2026-03-10 00:33:17,612  INFO        Epoch  75/300 — train_loss: 0.0051  val_loss: 0.0040  val_acc: 1.0000
2026-03-10 00:33:24,206  INFO        Epoch 100/300 — train_loss: 0.0027  val_loss: 0.0021  val_acc: 1.0000
2026-03-10 00:33:30,716  INFO        Epoch 125/300 — train_loss: 0.0028  val_loss: 0.0014  val_acc: 1.0000
2026-03-10 00:33:37,216  INFO        Epoch 150/300 — train_loss: 0.0011  val_loss: 0.0008  val_acc: 1.0000
2026-03-10 00:33:43,867  INFO        Epoch 175/300 — train_loss: 0.0014  val_loss: 0.0006  val_acc: 1.0000
2026-03-10 00:33:50,610  INFO        Epoch 200/300 — train_loss: 0.0007  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:33:57,191  INFO        Epoch 225/300 — train_loss: 0.0008  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:34:03,789  INFO        Epoch 250/300 — train_loss: 0.0004  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:34:10,735  INFO        Epoch 275/300 — train_loss: 0.0002  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:34:17,481  INFO        Epoch 300/300 — train_loss: 0.0004  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:34:17,498  INFO        Fold  6/10 — accuracy: 1.0000
2026-03-10 00:34:17,523  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:34:17,540  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:34:17,562  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:34:17,567  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:34:17,567  INFO      EEGNet Fold 7/10 — train=81, test=3
2026-03-10 00:34:17,568  INFO      Training EEGNet on device: cpu
2026-03-10 00:34:17,586  INFO      EEGNet parameters: 2834
2026-03-10 00:34:17,844  INFO        Epoch   1/300 — train_loss: 0.6497  val_loss: 0.6754  val_acc: 0.6471
2026-03-10 00:34:24,313  INFO        Epoch  25/300 — train_loss: 0.0381  val_loss: 0.0665  val_acc: 1.0000
2026-03-10 00:34:31,053  INFO        Epoch  50/300 — train_loss: 0.0120  val_loss: 0.0071  val_acc: 1.0000
2026-03-10 00:34:37,641  INFO        Epoch  75/300 — train_loss: 0.0031  val_loss: 0.0037  val_acc: 1.0000
2026-03-10 00:34:44,277  INFO        Epoch 100/300 — train_loss: 0.0038  val_loss: 0.0018  val_acc: 1.0000
2026-03-10 00:34:50,918  INFO        Epoch 125/300 — train_loss: 0.0017  val_loss: 0.0009  val_acc: 1.0000
2026-03-10 00:34:57,579  INFO        Epoch 150/300 — train_loss: 0.0014  val_loss: 0.0007  val_acc: 1.0000
2026-03-10 00:35:04,186  INFO        Epoch 175/300 — train_loss: 0.0014  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:35:10,769  INFO        Epoch 200/300 — train_loss: 0.0007  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:35:17,360  INFO        Epoch 225/300 — train_loss: 0.0014  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:35:24,199  INFO        Epoch 250/300 — train_loss: 0.0006  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:35:30,782  INFO        Epoch 275/300 — train_loss: 0.0002  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:35:37,455  INFO        Epoch 300/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:35:37,466  INFO        Fold  7/10 — accuracy: 0.6667
2026-03-10 00:35:37,500  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:35:37,518  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:35:37,539  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:35:37,544  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:35:37,544  INFO      EEGNet Fold 8/10 — train=81, test=3
2026-03-10 00:35:37,545  INFO      Training EEGNet on device: cpu
2026-03-10 00:35:37,565  INFO      EEGNet parameters: 2834
2026-03-10 00:35:37,825  INFO        Epoch   1/300 — train_loss: 0.7466  val_loss: 0.6835  val_acc: 0.7059
2026-03-10 00:35:44,275  INFO        Epoch  25/300 — train_loss: 0.0377  val_loss: 0.0297  val_acc: 1.0000
2026-03-10 00:35:50,969  INFO        Epoch  50/300 — train_loss: 0.0110  val_loss: 0.0057  val_acc: 1.0000
2026-03-10 00:35:57,421  INFO        Epoch  75/300 — train_loss: 0.0042  val_loss: 0.0024  val_acc: 1.0000
2026-03-10 00:36:03,955  INFO        Epoch 100/300 — train_loss: 0.0016  val_loss: 0.0014  val_acc: 1.0000
2026-03-10 00:36:10,450  INFO        Epoch 125/300 — train_loss: 0.0016  val_loss: 0.0009  val_acc: 1.0000
2026-03-10 00:36:16,970  INFO        Epoch 150/300 — train_loss: 0.0011  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:36:23,516  INFO        Epoch 175/300 — train_loss: 0.0010  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:36:30,028  INFO        Epoch 200/300 — train_loss: 0.0005  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:36:36,672  INFO        Epoch 225/300 — train_loss: 0.0006  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:36:43,218  INFO        Epoch 250/300 — train_loss: 0.0008  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:36:49,735  INFO        Epoch 275/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:36:56,296  INFO        Epoch 300/300 — train_loss: 0.0005  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:36:56,303  INFO        Fold  8/10 — accuracy: 0.6667
2026-03-10 00:36:56,348  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:36:56,360  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:36:56,387  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:36:56,389  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:36:56,389  INFO      EEGNet Fold 9/10 — train=81, test=3
2026-03-10 00:36:56,394  INFO      Training EEGNet on device: cpu
2026-03-10 00:36:56,418  INFO      EEGNet parameters: 2834
2026-03-10 00:36:56,688  INFO        Epoch   1/300 — train_loss: 0.6777  val_loss: 0.6826  val_acc: 0.7059
2026-03-10 00:37:04,064  INFO        Epoch  25/300 — train_loss: 0.0587  val_loss: 0.0713  val_acc: 1.0000
2026-03-10 00:37:10,878  INFO        Epoch  50/300 — train_loss: 0.0095  val_loss: 0.0078  val_acc: 1.0000
2026-03-10 00:37:17,470  INFO        Epoch  75/300 — train_loss: 0.0041  val_loss: 0.0029  val_acc: 1.0000
2026-03-10 00:37:24,476  INFO        Epoch 100/300 — train_loss: 0.0060  val_loss: 0.0019  val_acc: 1.0000
2026-03-10 00:37:31,987  INFO        Epoch 125/300 — train_loss: 0.0012  val_loss: 0.0011  val_acc: 1.0000
2026-03-10 00:37:38,639  INFO        Epoch 150/300 — train_loss: 0.0017  val_loss: 0.0006  val_acc: 1.0000
2026-03-10 00:37:45,251  INFO        Epoch 175/300 — train_loss: 0.0013  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:37:51,816  INFO        Epoch 200/300 — train_loss: 0.0008  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:37:58,361  INFO        Epoch 225/300 — train_loss: 0.0010  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:38:04,970  INFO        Epoch 250/300 — train_loss: 0.0003  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:38:11,461  INFO        Epoch 275/300 — train_loss: 0.0002  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:38:18,126  INFO        Epoch 300/300 — train_loss: 0.0002  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:38:18,147  INFO        Fold  9/10 — accuracy: 1.0000
2026-03-10 00:38:18,182  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:38:18,197  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:38:18,218  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:38:18,223  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:38:18,224  INFO      EEGNet Fold 10/10 — train=81, test=3
2026-03-10 00:38:18,224  INFO      Training EEGNet on device: cpu
2026-03-10 00:38:18,244  INFO      EEGNet parameters: 2834
2026-03-10 00:38:18,486  INFO        Epoch   1/300 — train_loss: 0.7221  val_loss: 0.6686  val_acc: 0.7059
2026-03-10 00:38:24,892  INFO        Epoch  25/300 — train_loss: 0.0845  val_loss: 0.0453  val_acc: 1.0000
2026-03-10 00:38:31,362  INFO        Epoch  50/300 — train_loss: 0.0107  val_loss: 0.0060  val_acc: 1.0000
2026-03-10 00:38:37,983  INFO        Epoch  75/300 — train_loss: 0.0058  val_loss: 0.0019  val_acc: 1.0000
2026-03-10 00:38:45,269  INFO        Epoch 100/300 — train_loss: 0.0050  val_loss: 0.0008  val_acc: 1.0000
2026-03-10 00:38:52,141  INFO        Epoch 125/300 — train_loss: 0.0011  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:39:00,064  INFO        Epoch 150/300 — train_loss: 0.0010  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:39:06,954  INFO        Epoch 175/300 — train_loss: 0.0013  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:39:13,578  INFO        Epoch 200/300 — train_loss: 0.0009  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:39:20,162  INFO        Epoch 225/300 — train_loss: 0.0010  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:39:26,749  INFO        Epoch 250/300 — train_loss: 0.0007  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:39:33,203  INFO        Epoch 275/300 — train_loss: 0.0005  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:39:39,747  INFO        Epoch 300/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:39:39,761  INFO        Fold 10/10 — accuracy: 1.0000
2026-03-10 00:39:39,899  INFO      EEGNet — Accuracy: 0.8667 +/- 0.1633  F1: 0.8643  AUC: 0.9420

==================================================
EEGNet — Cross-Validation Results
==================================================
  Fold  1: 1.0000
  Fold  2: 0.6667
  Fold  3: 0.6667
  Fold  4: 1.0000
  Fold  5: 1.0000
  Fold  6: 1.0000
  Fold  7: 0.6667
  Fold  8: 0.6667
  Fold  9: 1.0000
  Fold 10: 1.0000
--------------------------------------------------
  Mean accuracy:  0.8667 +/- 0.1633
  Macro F1-score: 0.8643
  AUC-ROC:        0.9420
==================================================


=======================================================
  EEGNet_Raw — Subject 1 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.8667
  precision_macro               0.8750
  precision_left                0.8333
  precision_right               0.9167
  recall_macro                  0.8616
  recall_left                   0.9375
  recall_right                  0.7857
  f1_macro                      0.8643
  f1_left                       0.8824
  f1_right                      0.8462
  auc_roc                       0.9420
  cohens_kappa                  0.7297
  cv_mean_accuracy              0.8667
  cv_std_accuracy               0.1633
=======================================================

2026-03-10 00:39:40,008  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\EEGNet_Raw_subject001.json
2026-03-10 00:39:40,606  INFO      Confusion matrix saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\figures\confusion_EEGNet_Raw_subject001.png
2026-03-10 00:39:40,930  INFO      Saved figure: roc_EEGNet_Raw_subject001 (.png + .pdf)
2026-03-10 00:39:43,565  INFO      Saved figure: training_curves_EEGNet_subject001 (.png + .pdf)
2026-03-10 00:39:43,792  INFO      Saved figure: roc_comparison_subject001 (.png + .pdf)
Subjects:  33%|███████████████████████▎                                              | 1/3 [26:58<53:56, 1618.48s/it]2026-03-10 00:39:43,803  INFO      ============================================================
2026-03-10 00:39:43,804  INFO      Processing Subject 002
2026-03-10 00:39:43,804  INFO      ============================================================
2026-03-10 00:39:43,910  INFO      Downloaded  subject=002  file=S002R03.edf
2026-03-10 00:39:43,911  INFO      Downloaded  subject=002  file=S002R07.edf
2026-03-10 00:39:43,911  INFO      Download complete: 1 subjects, 2 total files.
2026-03-10 00:39:44,095  INFO      Loaded  subject=002  runs=[3, 7]  channels=64  sfreq=160 Hz  n_times=39360  duration=246.0 s
2026-03-10 00:39:44,095  INFO      Applying filters: bandpass 1.0–40.0 Hz, notch [60.0, 120.0] Hz, FIR
2026-03-10 00:39:44,317  INFO      Filtering complete: shape before=(64, 39360)  after=(64, 39360)  channels=64
2026-03-10 00:39:44,887  INFO      ICA fitted: method=fastica, n_components=20, n_iter=31
2026-03-10 00:39:45,270  INFO      ICA artifact removal: excluded 2 components [(0, 'EOG (via Fp1)'), (1, 'EOG (via Fpz)')]
2026-03-10 00:39:45,270  INFO      ICA applied: excluded 2 components
2026-03-10 00:39:45,286  INFO      Event mapping: {'left': 2, 'right': 3} (from raw annotations: {'T0': 1, 'T1': 2, 'T2': 3})
2026-03-10 00:39:45,337  INFO      Epochs: total=30  kept=30  rejected=0 (0.0%)  left=15  right=15
2026-03-10 00:39:45,337  INFO      --- Signal Visualizations ---
2026-03-10 00:39:46,506  INFO      Saved figure: topomap_subject002 (.png + .pdf)
2026-03-10 00:39:47,251  INFO      Saved figure: psd_comparison_subject002 (.png + .pdf)
2026-03-10 00:39:48,869  INFO      Saved figure: butterfly_subject002 (.png + .pdf)
2026-03-10 00:39:48,871  INFO      Generated 6 signal figures for Subject 002
2026-03-10 00:39:48,871  INFO      --- LR + PSD ---
2026-03-10 00:39:49,206  INFO      PSD features: X=(30, 384)  y=(30,)  bands=6  channels=64
2026-03-10 00:39:49,244  INFO        Fold  1/10 — accuracy: 0.3333
2026-03-10 00:39:49,259  INFO        Fold  2/10 — accuracy: 0.6667
2026-03-10 00:39:49,282  INFO        Fold  3/10 — accuracy: 0.3333
2026-03-10 00:39:49,302  INFO        Fold  4/10 — accuracy: 0.6667
2026-03-10 00:39:49,320  INFO        Fold  5/10 — accuracy: 0.6667
2026-03-10 00:39:49,336  INFO        Fold  6/10 — accuracy: 0.6667
2026-03-10 00:39:49,353  INFO        Fold  7/10 — accuracy: 0.6667
2026-03-10 00:39:49,369  INFO        Fold  8/10 — accuracy: 0.3333
2026-03-10 00:39:49,388  INFO        Fold  9/10 — accuracy: 1.0000
2026-03-10 00:39:49,404  INFO        Fold 10/10 — accuracy: 0.6667
2026-03-10 00:39:49,409  INFO      Logistic Regression — Accuracy: 0.6000 +/- 0.2000  F1: 0.6000  AUC: 0.6711

==================================================
Logistic Regression — Cross-Validation Results
==================================================
  Fold  1: 0.3333
  Fold  2: 0.6667
  Fold  3: 0.3333
  Fold  4: 0.6667
  Fold  5: 0.6667
  Fold  6: 0.6667
  Fold  7: 0.6667
  Fold  8: 0.3333
  Fold  9: 1.0000
  Fold 10: 0.6667
--------------------------------------------------
  Mean accuracy:  0.6000 +/- 0.2000
  Macro F1-score: 0.6000
  AUC-ROC:        0.6711
==================================================


=======================================================
  LR_PSD — Subject 2 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.6000
  precision_macro               0.6000
  precision_left                0.6000
  precision_right               0.6000
  recall_macro                  0.6000
  recall_left                   0.6000
  recall_right                  0.6000
  f1_macro                      0.6000
  f1_left                       0.6000
  f1_right                      0.6000
  auc_roc                       0.6711
  cohens_kappa                  0.2000
  cv_mean_accuracy              0.6000
  cv_std_accuracy               0.2000
=======================================================

2026-03-10 00:39:49,435  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\LR_PSD_subject002.json
2026-03-10 00:39:49,643  INFO      Confusion matrix saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\figures\confusion_LR_PSD_subject002.png
2026-03-10 00:39:49,853  INFO      Saved figure: roc_LR_PSD_subject002 (.png + .pdf)
2026-03-10 00:39:50,191  INFO      Saved figure: feature_importance_LR_PSD_subject002 (.png + .pdf)
2026-03-10 00:39:50,191  INFO      --- LR + PSD (Tuned) ---
2026-03-10 00:40:08,135  INFO        Fold  1/10 — accuracy: 0.3333  best_C: 1.0000
2026-03-10 00:40:08,224  INFO        Fold  2/10 — accuracy: 0.6667  best_C: 10.0000
2026-03-10 00:40:08,337  INFO        Fold  3/10 — accuracy: 0.3333  best_C: 1.0000
2026-03-10 00:40:08,411  INFO        Fold  4/10 — accuracy: 0.6667  best_C: 1.0000
2026-03-10 00:40:08,458  INFO        Fold  5/10 — accuracy: 0.3333  best_C: 0.0100
2026-03-10 00:40:08,513  INFO        Fold  6/10 — accuracy: 0.6667  best_C: 1.0000
2026-03-10 00:40:08,568  INFO        Fold  7/10 — accuracy: 1.0000  best_C: 0.1000
2026-03-10 00:40:08,625  INFO        Fold  8/10 — accuracy: 0.3333  best_C: 1.0000
2026-03-10 00:40:08,686  INFO        Fold  9/10 — accuracy: 1.0000  best_C: 1.0000
2026-03-10 00:40:08,735  INFO        Fold 10/10 — accuracy: 1.0000  best_C: 0.0100
2026-03-10 00:40:08,913  INFO      Tuned LR — Accuracy: 0.6333 +/- 0.2769  F1: 0.6329  AUC: 0.6489  Best C: 1.0

=======================================================
Logistic Regression (Tuned) — Cross-Validation Results
=======================================================
  Fold  1: 0.3333  (C=1.0)
  Fold  2: 0.6667  (C=10.0)
  Fold  3: 0.3333  (C=1.0)
  Fold  4: 0.6667  (C=1.0)
  Fold  5: 0.3333  (C=0.01)
  Fold  6: 0.6667  (C=1.0)
  Fold  7: 1.0000  (C=0.1)
  Fold  8: 0.3333  (C=1.0)
  Fold  9: 1.0000  (C=1.0)
  Fold 10: 1.0000  (C=0.01)
-------------------------------------------------------
  Mean accuracy:  0.6333 +/- 0.2769
  Macro F1-score: 0.6329
  AUC-ROC:        0.6489
  Overall best C: 1.0
=======================================================


=======================================================
  LR_PSD_Tuned — Subject 2 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.6333
  precision_macro               0.6339
  precision_left                0.6250
  precision_right               0.6429
  recall_macro                  0.6333
  recall_left                   0.6667
  recall_right                  0.6000
  f1_macro                      0.6329
  f1_left                       0.6452
  f1_right                      0.6207
  auc_roc                       0.6489
  cohens_kappa                  0.2667
  cv_mean_accuracy              0.6333
  cv_std_accuracy               0.2769
  best_C                        1.0000
=======================================================

2026-03-10 00:40:08,991  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\LR_PSD_Tuned_subject002.json
2026-03-10 00:40:09,006  INFO        Tuned vs Default: 0.6333 vs 0.6000 (delta=+0.0333)
2026-03-10 00:40:09,007  INFO      --- LR + CSP ---
Computing rank from data with rank=None
    Using tolerance 0.00012 (2.2e-16 eps * 64 dim * 8.3e+09  max singular value)
    Estimated rank (data): 61
    data: rank 61 computed from 64 data channels with 0 projectors
    Setting small data eigenvalues to zero (without PCA)
Reducing data rank from 64 -> 61
Estimating class=0 covariance using EMPIRICAL
Done.
Estimating class=1 covariance using EMPIRICAL
Done.
    Setting small data eigenvalues to zero (without PCA)
2026-03-10 00:40:09,478  INFO      CSP features: X=(30, 4)  y=(30,)  components=4
2026-03-10 00:40:09,479  INFO      CSP filter weight ranges: ['comp0: [-33814.4018, 59589.8757]', 'comp1: [-93321.2293, 55153.0268]', 'comp2: [-37512.2269, 19001.2561]', 'comp3: [-32127.2034, 27592.9494]']
2026-03-10 00:40:09,488  INFO        Fold  1/10 — accuracy: 0.6667
2026-03-10 00:40:09,491  INFO        Fold  2/10 — accuracy: 1.0000
2026-03-10 00:40:09,495  INFO        Fold  3/10 — accuracy: 0.6667
2026-03-10 00:40:09,499  INFO        Fold  4/10 — accuracy: 0.3333
2026-03-10 00:40:09,505  INFO        Fold  5/10 — accuracy: 1.0000
2026-03-10 00:40:09,508  INFO        Fold  6/10 — accuracy: 1.0000
2026-03-10 00:40:09,517  INFO        Fold  7/10 — accuracy: 1.0000
2026-03-10 00:40:09,523  INFO        Fold  8/10 — accuracy: 0.6667
2026-03-10 00:40:09,531  INFO        Fold  9/10 — accuracy: 0.3333
2026-03-10 00:40:09,538  INFO        Fold 10/10 — accuracy: 0.6667
2026-03-10 00:40:09,541  INFO      Logistic Regression — Accuracy: 0.7333 +/- 0.2494  F1: 0.7321  AUC: 0.7911

==================================================
Logistic Regression — Cross-Validation Results
==================================================
  Fold  1: 0.6667
  Fold  2: 1.0000
  Fold  3: 0.6667
  Fold  4: 0.3333
  Fold  5: 1.0000
  Fold  6: 1.0000
  Fold  7: 1.0000
  Fold  8: 0.6667
  Fold  9: 0.3333
  Fold 10: 0.6667
--------------------------------------------------
  Mean accuracy:  0.7333 +/- 0.2494
  Macro F1-score: 0.7321
  AUC-ROC:        0.7911
==================================================


=======================================================
  LR_CSP — Subject 2 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.7333
  precision_macro               0.7376
  precision_left                0.7059
  precision_right               0.7692
  recall_macro                  0.7333
  recall_left                   0.8000
  recall_right                  0.6667
  f1_macro                      0.7321
  f1_left                       0.7500
  f1_right                      0.7143
  auc_roc                       0.7911
  cohens_kappa                  0.4667
  cv_mean_accuracy              0.7333
  cv_std_accuracy               0.2494
=======================================================

2026-03-10 00:40:09,564  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\LR_CSP_subject002.json
2026-03-10 00:40:09,907  INFO      Confusion matrix saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\figures\confusion_LR_CSP_subject002.png
2026-03-10 00:40:10,173  INFO      Saved figure: roc_LR_CSP_subject002 (.png + .pdf)
2026-03-10 00:40:10,173  INFO      --- EEGNet + Raw ---
2026-03-10 00:40:10,201  INFO      Raw features: X=(30, 64, 721)  y=(30,)  (per-channel z-score normalized)
2026-03-10 00:40:10,203  INFO      --- EEGNet + Raw (Augmented) ---
2026-03-10 00:40:10,255  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:40:10,266  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:40:10,282  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:40:10,282  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:40:10,282  INFO      EEGNet Fold 1/10 — train=81, test=3
2026-03-10 00:40:10,282  INFO      Training EEGNet on device: cpu
2026-03-10 00:40:10,558  INFO      EEGNet parameters: 2834
2026-03-10 00:40:12,613  INFO        Epoch   1/300 — train_loss: 0.6842  val_loss: 0.6818  val_acc: 0.6471
2026-03-10 00:40:19,791  INFO        Epoch  25/300 — train_loss: 0.0882  val_loss: 0.0975  val_acc: 1.0000
2026-03-10 00:40:26,766  INFO        Epoch  50/300 — train_loss: 0.0111  val_loss: 0.0102  val_acc: 1.0000
2026-03-10 00:40:33,688  INFO        Epoch  75/300 — train_loss: 0.0048  val_loss: 0.0035  val_acc: 1.0000
2026-03-10 00:40:40,501  INFO        Epoch 100/300 — train_loss: 0.0047  val_loss: 0.0019  val_acc: 1.0000
2026-03-10 00:40:47,204  INFO        Epoch 125/300 — train_loss: 0.0013  val_loss: 0.0011  val_acc: 1.0000
2026-03-10 00:40:53,975  INFO        Epoch 150/300 — train_loss: 0.0013  val_loss: 0.0008  val_acc: 1.0000
2026-03-10 00:41:00,876  INFO        Epoch 175/300 — train_loss: 0.0010  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:41:07,804  INFO        Epoch 200/300 — train_loss: 0.0010  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:41:14,626  INFO        Epoch 225/300 — train_loss: 0.0003  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:41:21,267  INFO        Epoch 250/300 — train_loss: 0.0002  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:41:28,353  INFO        Epoch 275/300 — train_loss: 0.0004  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:41:35,189  INFO        Epoch 300/300 — train_loss: 0.0006  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:41:35,262  INFO        Fold  1/10 — accuracy: 0.3333
2026-03-10 00:41:35,352  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:41:35,360  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:41:35,407  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:41:35,407  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:41:35,407  INFO      EEGNet Fold 2/10 — train=81, test=3
2026-03-10 00:41:35,407  INFO      Training EEGNet on device: cpu
2026-03-10 00:41:35,463  INFO      EEGNet parameters: 2834
2026-03-10 00:41:35,792  INFO        Epoch   1/300 — train_loss: 0.7099  val_loss: 0.6942  val_acc: 0.4706
2026-03-10 00:41:43,158  INFO        Epoch  25/300 — train_loss: 0.0785  val_loss: 0.1358  val_acc: 1.0000
2026-03-10 00:41:49,938  INFO        Epoch  50/300 — train_loss: 0.0112  val_loss: 0.0093  val_acc: 1.0000
2026-03-10 00:41:56,592  INFO        Epoch  75/300 — train_loss: 0.0034  val_loss: 0.0028  val_acc: 1.0000
2026-03-10 00:42:03,778  INFO        Epoch 100/300 — train_loss: 0.0022  val_loss: 0.0014  val_acc: 1.0000
2026-03-10 00:42:10,740  INFO        Epoch 125/300 — train_loss: 0.0007  val_loss: 0.0008  val_acc: 1.0000
2026-03-10 00:42:17,343  INFO        Epoch 150/300 — train_loss: 0.0009  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:42:23,998  INFO        Epoch 175/300 — train_loss: 0.0006  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:42:30,594  INFO        Epoch 200/300 — train_loss: 0.0011  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:42:38,654  INFO        Epoch 225/300 — train_loss: 0.0005  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:42:45,190  INFO        Epoch 250/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:42:51,736  INFO        Epoch 275/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:42:58,348  INFO        Epoch 300/300 — train_loss: 0.0001  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:42:58,387  INFO        Fold  2/10 — accuracy: 1.0000
2026-03-10 00:42:58,449  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:42:58,463  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:42:58,486  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:42:58,493  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:42:58,494  INFO      EEGNet Fold 3/10 — train=81, test=3
2026-03-10 00:42:58,494  INFO      Training EEGNet on device: cpu
2026-03-10 00:42:58,533  INFO      EEGNet parameters: 2834
2026-03-10 00:42:58,791  INFO        Epoch   1/300 — train_loss: 0.7172  val_loss: 0.6790  val_acc: 0.7647
2026-03-10 00:43:05,306  INFO        Epoch  25/300 — train_loss: 0.1795  val_loss: 0.2004  val_acc: 1.0000
2026-03-10 00:43:12,203  INFO        Epoch  50/300 — train_loss: 0.0190  val_loss: 0.0152  val_acc: 1.0000
2026-03-10 00:43:18,721  INFO        Epoch  75/300 — train_loss: 0.0058  val_loss: 0.0041  val_acc: 1.0000
2026-03-10 00:43:25,467  INFO        Epoch 100/300 — train_loss: 0.0030  val_loss: 0.0020  val_acc: 1.0000
2026-03-10 00:43:32,196  INFO        Epoch 125/300 — train_loss: 0.0013  val_loss: 0.0012  val_acc: 1.0000
2026-03-10 00:43:38,952  INFO        Epoch 150/300 — train_loss: 0.0014  val_loss: 0.0008  val_acc: 1.0000
2026-03-10 00:43:45,688  INFO        Epoch 175/300 — train_loss: 0.0008  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:43:52,548  INFO        Epoch 200/300 — train_loss: 0.0015  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:43:59,159  INFO        Epoch 225/300 — train_loss: 0.0011  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:44:05,888  INFO        Epoch 250/300 — train_loss: 0.0003  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:44:12,626  INFO        Epoch 275/300 — train_loss: 0.0005  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:44:19,233  INFO        Epoch 300/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:44:19,257  INFO        Fold  3/10 — accuracy: 0.3333
2026-03-10 00:44:19,293  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:44:19,312  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:44:19,330  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:44:19,338  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:44:19,338  INFO      EEGNet Fold 4/10 — train=81, test=3
2026-03-10 00:44:19,338  INFO      Training EEGNet on device: cpu
2026-03-10 00:44:19,357  INFO      EEGNet parameters: 2834
2026-03-10 00:44:19,611  INFO        Epoch   1/300 — train_loss: 0.7399  val_loss: 0.6933  val_acc: 0.5294
2026-03-10 00:44:26,097  INFO        Epoch  25/300 — train_loss: 0.0819  val_loss: 0.1444  val_acc: 1.0000
2026-03-10 00:44:32,720  INFO        Epoch  50/300 — train_loss: 0.0122  val_loss: 0.0116  val_acc: 1.0000
2026-03-10 00:44:39,382  INFO        Epoch  75/300 — train_loss: 0.0079  val_loss: 0.0035  val_acc: 1.0000
2026-03-10 00:44:46,093  INFO        Epoch 100/300 — train_loss: 0.0028  val_loss: 0.0018  val_acc: 1.0000
2026-03-10 00:44:52,689  INFO        Epoch 125/300 — train_loss: 0.0013  val_loss: 0.0011  val_acc: 1.0000
2026-03-10 00:44:59,564  INFO        Epoch 150/300 — train_loss: 0.0009  val_loss: 0.0007  val_acc: 1.0000
2026-03-10 00:45:07,096  INFO        Epoch 175/300 — train_loss: 0.0006  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:45:14,673  INFO        Epoch 200/300 — train_loss: 0.0015  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:45:21,865  INFO        Epoch 225/300 — train_loss: 0.0007  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:45:28,967  INFO        Epoch 250/300 — train_loss: 0.0003  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:45:36,073  INFO        Epoch 275/300 — train_loss: 0.0003  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:45:43,129  INFO        Epoch 300/300 — train_loss: 0.0004  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:45:43,146  INFO        Fold  4/10 — accuracy: 1.0000
2026-03-10 00:45:43,193  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:45:43,207  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:45:43,224  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:45:43,229  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:45:43,230  INFO      EEGNet Fold 5/10 — train=81, test=3
2026-03-10 00:45:43,230  INFO      Training EEGNet on device: cpu
2026-03-10 00:45:43,252  INFO      EEGNet parameters: 2834
2026-03-10 00:45:43,523  INFO        Epoch   1/300 — train_loss: 0.7479  val_loss: 0.6917  val_acc: 0.6471
2026-03-10 00:45:49,960  INFO        Epoch  25/300 — train_loss: 0.1332  val_loss: 0.1309  val_acc: 1.0000
2026-03-10 00:45:56,403  INFO        Epoch  50/300 — train_loss: 0.0149  val_loss: 0.0130  val_acc: 1.0000
2026-03-10 00:46:02,821  INFO        Epoch  75/300 — train_loss: 0.0039  val_loss: 0.0043  val_acc: 1.0000
2026-03-10 00:46:09,302  INFO        Epoch 100/300 — train_loss: 0.0013  val_loss: 0.0018  val_acc: 1.0000
2026-03-10 00:46:15,850  INFO        Epoch 125/300 — train_loss: 0.0014  val_loss: 0.0013  val_acc: 1.0000
2026-03-10 00:46:22,424  INFO        Epoch 150/300 — train_loss: 0.0008  val_loss: 0.0008  val_acc: 1.0000
2026-03-10 00:46:28,899  INFO        Epoch 175/300 — train_loss: 0.0006  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:46:35,382  INFO        Epoch 200/300 — train_loss: 0.0006  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:46:41,853  INFO        Epoch 225/300 — train_loss: 0.0003  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:46:48,344  INFO        Epoch 250/300 — train_loss: 0.0003  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:46:54,860  INFO        Epoch 275/300 — train_loss: 0.0004  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:47:01,545  INFO        Epoch 300/300 — train_loss: 0.0004  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:47:01,545  INFO        Fold  5/10 — accuracy: 0.6667
2026-03-10 00:47:01,592  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:47:01,607  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:47:01,626  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:47:01,631  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:47:01,631  INFO      EEGNet Fold 6/10 — train=81, test=3
2026-03-10 00:47:01,631  INFO      Training EEGNet on device: cpu
2026-03-10 00:47:01,654  INFO      EEGNet parameters: 2834
2026-03-10 00:47:01,923  INFO        Epoch   1/300 — train_loss: 0.7471  val_loss: 0.6926  val_acc: 0.5294
2026-03-10 00:47:08,422  INFO        Epoch  25/300 — train_loss: 0.0994  val_loss: 0.1675  val_acc: 1.0000
2026-03-10 00:47:15,021  INFO        Epoch  50/300 — train_loss: 0.0127  val_loss: 0.0129  val_acc: 1.0000
2026-03-10 00:47:21,532  INFO        Epoch  75/300 — train_loss: 0.0052  val_loss: 0.0036  val_acc: 1.0000
2026-03-10 00:47:29,185  INFO        Epoch 100/300 — train_loss: 0.0014  val_loss: 0.0017  val_acc: 1.0000
2026-03-10 00:47:35,976  INFO        Epoch 125/300 — train_loss: 0.0012  val_loss: 0.0011  val_acc: 1.0000
2026-03-10 00:47:42,423  INFO        Epoch 150/300 — train_loss: 0.0014  val_loss: 0.0007  val_acc: 1.0000
2026-03-10 00:47:48,881  INFO        Epoch 175/300 — train_loss: 0.0011  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:47:55,504  INFO        Epoch 200/300 — train_loss: 0.0004  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:48:02,032  INFO        Epoch 225/300 — train_loss: 0.0004  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:48:08,500  INFO        Epoch 250/300 — train_loss: 0.0008  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:48:14,969  INFO        Epoch 275/300 — train_loss: 0.0003  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:48:21,568  INFO        Epoch 300/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:48:21,568  INFO        Fold  6/10 — accuracy: 1.0000
2026-03-10 00:48:21,615  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:48:21,626  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:48:21,644  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:48:21,648  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:48:21,649  INFO      EEGNet Fold 7/10 — train=81, test=3
2026-03-10 00:48:21,649  INFO      Training EEGNet on device: cpu
2026-03-10 00:48:21,667  INFO      EEGNet parameters: 2834
2026-03-10 00:48:21,930  INFO        Epoch   1/300 — train_loss: 0.6971  val_loss: 0.6894  val_acc: 0.6471
2026-03-10 00:48:28,298  INFO        Epoch  25/300 — train_loss: 0.0972  val_loss: 0.1713  val_acc: 1.0000
2026-03-10 00:48:34,809  INFO        Epoch  50/300 — train_loss: 0.0103  val_loss: 0.0135  val_acc: 1.0000
2026-03-10 00:48:41,324  INFO        Epoch  75/300 — train_loss: 0.0037  val_loss: 0.0039  val_acc: 1.0000
2026-03-10 00:48:47,856  INFO        Epoch 100/300 — train_loss: 0.0024  val_loss: 0.0020  val_acc: 1.0000
2026-03-10 00:48:54,785  INFO        Epoch 125/300 — train_loss: 0.0026  val_loss: 0.0013  val_acc: 1.0000
2026-03-10 00:49:02,548  INFO        Epoch 150/300 — train_loss: 0.0011  val_loss: 0.0008  val_acc: 1.0000
2026-03-10 00:49:09,173  INFO        Epoch 175/300 — train_loss: 0.0011  val_loss: 0.0006  val_acc: 1.0000
2026-03-10 00:49:15,771  INFO        Epoch 200/300 — train_loss: 0.0005  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:49:22,377  INFO        Epoch 225/300 — train_loss: 0.0004  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:49:28,908  INFO        Epoch 250/300 — train_loss: 0.0004  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:49:35,534  INFO        Epoch 275/300 — train_loss: 0.0004  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:49:42,012  INFO        Epoch 300/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:49:42,028  INFO        Fold  7/10 — accuracy: 1.0000
2026-03-10 00:49:42,069  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:49:42,082  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:49:42,095  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:49:42,105  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:49:42,105  INFO      EEGNet Fold 8/10 — train=81, test=3
2026-03-10 00:49:42,105  INFO      Training EEGNet on device: cpu
2026-03-10 00:49:42,122  INFO      EEGNet parameters: 2834
2026-03-10 00:49:42,392  INFO        Epoch   1/300 — train_loss: 0.7588  val_loss: 0.6862  val_acc: 0.6471
2026-03-10 00:49:48,742  INFO        Epoch  25/300 — train_loss: 0.1672  val_loss: 0.1991  val_acc: 1.0000
2026-03-10 00:49:55,266  INFO        Epoch  50/300 — train_loss: 0.0134  val_loss: 0.0129  val_acc: 1.0000
2026-03-10 00:50:01,830  INFO        Epoch  75/300 — train_loss: 0.0063  val_loss: 0.0034  val_acc: 1.0000
2026-03-10 00:50:08,439  INFO        Epoch 100/300 — train_loss: 0.0027  val_loss: 0.0014  val_acc: 1.0000
2026-03-10 00:50:14,987  INFO        Epoch 125/300 — train_loss: 0.0019  val_loss: 0.0007  val_acc: 1.0000
2026-03-10 00:50:21,563  INFO        Epoch 150/300 — train_loss: 0.0012  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:50:28,126  INFO        Epoch 175/300 — train_loss: 0.0006  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:50:34,610  INFO        Epoch 200/300 — train_loss: 0.0007  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:50:41,310  INFO        Epoch 225/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:50:47,835  INFO        Epoch 250/300 — train_loss: 0.0005  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:50:54,513  INFO        Epoch 275/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:51:01,135  INFO        Epoch 300/300 — train_loss: 0.0004  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:51:01,155  INFO        Fold  8/10 — accuracy: 0.3333
2026-03-10 00:51:01,195  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:51:01,207  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:51:01,224  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:51:01,229  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:51:01,229  INFO      EEGNet Fold 9/10 — train=81, test=3
2026-03-10 00:51:01,229  INFO      Training EEGNet on device: cpu
2026-03-10 00:51:01,247  INFO      EEGNet parameters: 2834
2026-03-10 00:51:01,503  INFO        Epoch   1/300 — train_loss: 0.6645  val_loss: 0.6907  val_acc: 0.4706
2026-03-10 00:51:07,908  INFO        Epoch  25/300 — train_loss: 0.1309  val_loss: 0.1313  val_acc: 1.0000
2026-03-10 00:51:14,519  INFO        Epoch  50/300 — train_loss: 0.0116  val_loss: 0.0091  val_acc: 1.0000
2026-03-10 00:51:21,225  INFO        Epoch  75/300 — train_loss: 0.0050  val_loss: 0.0035  val_acc: 1.0000
2026-03-10 00:51:27,734  INFO        Epoch 100/300 — train_loss: 0.0037  val_loss: 0.0018  val_acc: 1.0000
2026-03-10 00:51:34,319  INFO        Epoch 125/300 — train_loss: 0.0016  val_loss: 0.0010  val_acc: 1.0000
2026-03-10 00:51:40,946  INFO        Epoch 150/300 — train_loss: 0.0009  val_loss: 0.0007  val_acc: 1.0000
2026-03-10 00:51:47,488  INFO        Epoch 175/300 — train_loss: 0.0007  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:51:54,126  INFO        Epoch 200/300 — train_loss: 0.0005  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:52:00,689  INFO        Epoch 225/300 — train_loss: 0.0004  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:52:07,408  INFO        Epoch 250/300 — train_loss: 0.0006  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:52:13,989  INFO        Epoch 275/300 — train_loss: 0.0003  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:52:20,536  INFO        Epoch 300/300 — train_loss: 0.0004  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:52:20,559  INFO        Fold  9/10 — accuracy: 0.0000
2026-03-10 00:52:20,590  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:52:20,609  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:52:20,631  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:52:20,636  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:52:20,637  INFO      EEGNet Fold 10/10 — train=81, test=3
2026-03-10 00:52:20,637  INFO      Training EEGNet on device: cpu
2026-03-10 00:52:20,657  INFO      EEGNet parameters: 2834
2026-03-10 00:52:21,017  INFO        Epoch   1/300 — train_loss: 0.7238  val_loss: 0.6903  val_acc: 0.7059
2026-03-10 00:52:27,357  INFO        Epoch  25/300 — train_loss: 0.1100  val_loss: 0.2009  val_acc: 0.9412
2026-03-10 00:52:33,950  INFO        Epoch  50/300 — train_loss: 0.0138  val_loss: 0.0198  val_acc: 1.0000
2026-03-10 00:52:40,749  INFO        Epoch  75/300 — train_loss: 0.0126  val_loss: 0.0049  val_acc: 1.0000
2026-03-10 00:52:47,332  INFO        Epoch 100/300 — train_loss: 0.0032  val_loss: 0.0021  val_acc: 1.0000
2026-03-10 00:52:53,925  INFO        Epoch 125/300 — train_loss: 0.0017  val_loss: 0.0013  val_acc: 1.0000
2026-03-10 00:53:00,597  INFO        Epoch 150/300 — train_loss: 0.0011  val_loss: 0.0008  val_acc: 1.0000
2026-03-10 00:53:07,201  INFO        Epoch 175/300 — train_loss: 0.0006  val_loss: 0.0006  val_acc: 1.0000
2026-03-10 00:53:13,781  INFO        Epoch 200/300 — train_loss: 0.0006  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:53:20,411  INFO        Epoch 225/300 — train_loss: 0.0010  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:53:29,119  INFO        Epoch 250/300 — train_loss: 0.0003  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:53:38,513  INFO        Epoch 275/300 — train_loss: 0.0002  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:53:47,053  INFO        Epoch 300/300 — train_loss: 0.0004  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:53:47,073  INFO        Fold 10/10 — accuracy: 0.6667
2026-03-10 00:53:47,157  INFO      EEGNet — Accuracy: 0.6333 +/- 0.3480  F1: 0.6296  AUC: 0.7067


==================================================
EEGNet — Cross-Validation Results
==================================================
  Fold  1: 0.3333
  Fold  2: 1.0000
  Fold  3: 0.3333
  Fold  4: 1.0000
  Fold  5: 0.6667
  Fold  6: 1.0000
  Fold  7: 1.0000
  Fold  8: 0.3333
  Fold  9: 0.0000
  Fold 10: 0.6667
--------------------------------------------------
  Mean accuracy:  0.6333 +/- 0.3480
  Macro F1-score: 0.6296
  AUC-ROC:        0.7067
==================================================


=======================================================
  EEGNet_Raw — Subject 2 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.6333
  precision_macro               0.6389
  precision_left                0.6667
  precision_right               0.6111
  recall_macro                  0.6333
  recall_left                   0.5333
  recall_right                  0.7333
  f1_macro                      0.6296
  f1_left                       0.5926
  f1_right                      0.6667
  auc_roc                       0.7067
  cohens_kappa                  0.2667
  cv_mean_accuracy              0.6333
  cv_std_accuracy               0.3480
=======================================================

2026-03-10 00:53:47,216  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\EEGNet_Raw_subject002.json
2026-03-10 00:53:47,675  INFO      Confusion matrix saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\figures\confusion_EEGNet_Raw_subject002.png
2026-03-10 00:53:47,997  INFO      Saved figure: roc_EEGNet_Raw_subject002 (.png + .pdf)
2026-03-10 00:53:48,571  INFO      Saved figure: training_curves_EEGNet_subject002 (.png + .pdf)
2026-03-10 00:53:48,766  INFO      Saved figure: roc_comparison_subject002 (.png + .pdf)
Subjects:  67%|██████████████████████████████████████████████▋                       | 2/3 [41:03<19:23, 1163.48s/it]2026-03-10 00:53:48,791  INFO      ============================================================
2026-03-10 00:53:48,791  INFO      Processing Subject 003
2026-03-10 00:53:48,791  INFO      ============================================================
2026-03-10 00:53:48,923  INFO      Downloaded  subject=003  file=S003R03.edf
2026-03-10 00:53:48,923  INFO      Downloaded  subject=003  file=S003R07.edf
2026-03-10 00:53:48,923  INFO      Download complete: 1 subjects, 2 total files.
2026-03-10 00:53:49,276  INFO      Loaded  subject=003  runs=[3, 7]  channels=64  sfreq=160 Hz  n_times=40000  duration=250.0 s
2026-03-10 00:53:49,292  INFO      Applying filters: bandpass 1.0–40.0 Hz, notch [60.0, 120.0] Hz, FIR
2026-03-10 00:53:49,675  INFO      Filtering complete: shape before=(64, 40000)  after=(64, 40000)  channels=64
2026-03-10 00:53:50,901  INFO      ICA fitted: method=fastica, n_components=20, n_iter=29
2026-03-10 00:53:51,378  INFO      ICA artifact removal: excluded 3 components [(0, 'EOG (via Fp1)'), (1, 'EOG (via Fp1)'), (2, 'EOG (via Fp2)')]
2026-03-10 00:53:51,380  INFO      ICA applied: excluded 3 components
2026-03-10 00:53:51,383  INFO      Event mapping: {'left': 2, 'right': 3} (from raw annotations: {'T0': 1, 'T1': 2, 'T2': 3})
2026-03-10 00:53:51,424  INFO      Epochs: total=30  kept=30  rejected=0 (0.0%)  left=15  right=15
2026-03-10 00:53:51,424  INFO      --- Signal Visualizations ---
2026-03-10 00:53:52,638  INFO      Saved figure: topomap_subject003 (.png + .pdf)
2026-03-10 00:53:53,402  INFO      Saved figure: psd_comparison_subject003 (.png + .pdf)
2026-03-10 00:53:56,553  INFO      Saved figure: butterfly_subject003 (.png + .pdf)
2026-03-10 00:53:56,553  INFO      Generated 6 signal figures for Subject 003
2026-03-10 00:53:56,553  INFO      --- LR + PSD ---
2026-03-10 00:53:56,855  INFO      PSD features: X=(30, 384)  y=(30,)  bands=6  channels=64
2026-03-10 00:53:56,920  INFO        Fold  1/10 — accuracy: 0.6667
2026-03-10 00:53:56,935  INFO        Fold  2/10 — accuracy: 0.6667
2026-03-10 00:53:56,948  INFO        Fold  3/10 — accuracy: 0.0000
2026-03-10 00:53:56,968  INFO        Fold  4/10 — accuracy: 0.6667
2026-03-10 00:53:56,984  INFO        Fold  5/10 — accuracy: 0.3333
2026-03-10 00:53:56,996  INFO        Fold  6/10 — accuracy: 0.3333
2026-03-10 00:53:57,009  INFO        Fold  7/10 — accuracy: 0.6667
2026-03-10 00:53:57,020  INFO        Fold  8/10 — accuracy: 0.6667
2026-03-10 00:53:57,031  INFO        Fold  9/10 — accuracy: 0.3333
2026-03-10 00:53:57,042  INFO        Fold 10/10 — accuracy: 0.3333
2026-03-10 00:53:57,047  INFO      Logistic Regression — Accuracy: 0.4667 +/- 0.2211  F1: 0.4643  AUC: 0.5378

==================================================
Logistic Regression — Cross-Validation Results
==================================================
  Fold  1: 0.6667
  Fold  2: 0.6667
  Fold  3: 0.0000
  Fold  4: 0.6667
  Fold  5: 0.3333
  Fold  6: 0.3333
  Fold  7: 0.6667
  Fold  8: 0.6667
  Fold  9: 0.3333
  Fold 10: 0.3333
--------------------------------------------------
  Mean accuracy:  0.4667 +/- 0.2211
  Macro F1-score: 0.4643
  AUC-ROC:        0.5378
==================================================


=======================================================
  LR_PSD — Subject 3 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.4667
  precision_macro               0.4661
  precision_left                0.4615
  precision_right               0.4706
  recall_macro                  0.4667
  recall_left                   0.4000
  recall_right                  0.5333
  f1_macro                      0.4643
  f1_left                       0.4286
  f1_right                      0.5000
  auc_roc                       0.5378
  cohens_kappa                 -0.0667
  cv_mean_accuracy              0.4667
  cv_std_accuracy               0.2211
=======================================================

2026-03-10 00:53:57,067  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\LR_PSD_subject003.json
2026-03-10 00:53:57,185  INFO      Confusion matrix saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\figures\confusion_LR_PSD_subject003.png
2026-03-10 00:53:57,406  INFO      Saved figure: roc_LR_PSD_subject003 (.png + .pdf)
2026-03-10 00:53:57,703  INFO      Saved figure: feature_importance_LR_PSD_subject003 (.png + .pdf)
2026-03-10 00:53:57,703  INFO      --- LR + PSD (Tuned) ---
2026-03-10 00:54:12,095  INFO        Fold  1/10 — accuracy: 0.6667  best_C: 1.0000
2026-03-10 00:54:12,165  INFO        Fold  2/10 — accuracy: 0.6667  best_C: 0.1000
2026-03-10 00:54:12,240  INFO        Fold  3/10 — accuracy: 0.0000  best_C: 0.0010
2026-03-10 00:54:12,293  INFO        Fold  4/10 — accuracy: 0.6667  best_C: 10.0000
2026-03-10 00:54:12,338  INFO        Fold  5/10 — accuracy: 0.3333  best_C: 10.0000
2026-03-10 00:54:12,404  INFO        Fold  6/10 — accuracy: 0.3333  best_C: 0.0010
2026-03-10 00:54:12,454  INFO        Fold  7/10 — accuracy: 0.3333  best_C: 0.0100
2026-03-10 00:54:12,507  INFO        Fold  8/10 — accuracy: 0.6667  best_C: 0.0100
2026-03-10 00:54:12,559  INFO        Fold  9/10 — accuracy: 0.3333  best_C: 0.1000
2026-03-10 00:54:12,604  INFO        Fold 10/10 — accuracy: 0.6667  best_C: 0.0010
2026-03-10 00:54:12,626  INFO      Tuned LR — Accuracy: 0.4667 +/- 0.2211  F1: 0.4667  AUC: 0.5111  Best C: 0.001

=======================================================
Logistic Regression (Tuned) — Cross-Validation Results
=======================================================
  Fold  1: 0.6667  (C=1.0)
  Fold  2: 0.6667  (C=0.1)
  Fold  3: 0.0000  (C=0.001)
  Fold  4: 0.6667  (C=10.0)
  Fold  5: 0.3333  (C=10.0)
  Fold  6: 0.3333  (C=0.001)
  Fold  7: 0.3333  (C=0.01)
  Fold  8: 0.6667  (C=0.01)
  Fold  9: 0.3333  (C=0.1)
  Fold 10: 0.6667  (C=0.001)
-------------------------------------------------------
  Mean accuracy:  0.4667 +/- 0.2211
  Macro F1-score: 0.4667
  AUC-ROC:        0.5111
  Overall best C: 0.001
=======================================================


=======================================================
  LR_PSD_Tuned — Subject 3 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.4667
  precision_macro               0.4667
  precision_left                0.4667
  precision_right               0.4667
  recall_macro                  0.4667
  recall_left                   0.4667
  recall_right                  0.4667
  f1_macro                      0.4667
  f1_left                       0.4667
  f1_right                      0.4667
  auc_roc                       0.5111
  cohens_kappa                 -0.0667
  cv_mean_accuracy              0.4667
  cv_std_accuracy               0.2211
  best_C                        0.0010
=======================================================

2026-03-10 00:54:12,664  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\LR_PSD_Tuned_subject003.json
2026-03-10 00:54:12,675  INFO        Tuned vs Default: 0.4667 vs 0.4667 (delta=+0.0000)
2026-03-10 00:54:12,675  INFO      --- LR + CSP ---
Computing rank from data with rank=None
    Using tolerance 0.00019 (2.2e-16 eps * 64 dim * 1.4e+10  max singular value)
    Estimated rank (data): 60
    data: rank 60 computed from 64 data channels with 0 projectors
    Setting small data eigenvalues to zero (without PCA)
Reducing data rank from 64 -> 60
Estimating class=0 covariance using EMPIRICAL
Done.
Estimating class=1 covariance using EMPIRICAL
Done.
    Setting small data eigenvalues to zero (without PCA)
2026-03-10 00:54:13,073  INFO      CSP features: X=(30, 4)  y=(30,)  components=4
2026-03-10 00:54:13,074  INFO      CSP filter weight ranges: ['comp0: [-65342.3154, 65823.9472]', 'comp1: [-37978.0240, 30596.7446]', 'comp2: [-87591.9822, 90793.8914]', 'comp3: [-21605.1241, 34889.6843]']
2026-03-10 00:54:13,082  INFO        Fold  1/10 — accuracy: 1.0000
2026-03-10 00:54:13,085  INFO        Fold  2/10 — accuracy: 1.0000
2026-03-10 00:54:13,089  INFO        Fold  3/10 — accuracy: 1.0000
2026-03-10 00:54:13,094  INFO        Fold  4/10 — accuracy: 0.6667
2026-03-10 00:54:13,099  INFO        Fold  5/10 — accuracy: 1.0000
2026-03-10 00:54:13,105  INFO        Fold  6/10 — accuracy: 1.0000
2026-03-10 00:54:13,111  INFO        Fold  7/10 — accuracy: 0.6667
2026-03-10 00:54:13,116  INFO        Fold  8/10 — accuracy: 1.0000
2026-03-10 00:54:13,120  INFO        Fold  9/10 — accuracy: 1.0000
2026-03-10 00:54:13,124  INFO        Fold 10/10 — accuracy: 1.0000
2026-03-10 00:54:13,126  INFO      Logistic Regression — Accuracy: 0.9333 +/- 0.1333  F1: 0.9333  AUC: 0.9956

==================================================
Logistic Regression — Cross-Validation Results
==================================================
  Fold  1: 1.0000
  Fold  2: 1.0000
  Fold  3: 1.0000
  Fold  4: 0.6667
  Fold  5: 1.0000
  Fold  6: 1.0000
  Fold  7: 0.6667
  Fold  8: 1.0000
  Fold  9: 1.0000
  Fold 10: 1.0000
--------------------------------------------------
  Mean accuracy:  0.9333 +/- 0.1333
  Macro F1-score: 0.9333
  AUC-ROC:        0.9956
==================================================


=======================================================
  LR_CSP — Subject 3 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.9333
  precision_macro               0.9333
  precision_left                0.9333
  precision_right               0.9333
  recall_macro                  0.9333
  recall_left                   0.9333
  recall_right                  0.9333
  f1_macro                      0.9333
  f1_left                       0.9333
  f1_right                      0.9333
  auc_roc                       0.9956
  cohens_kappa                  0.8667
  cv_mean_accuracy              0.9333
  cv_std_accuracy               0.1333
=======================================================

2026-03-10 00:54:13,145  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\LR_CSP_subject003.json
2026-03-10 00:54:13,668  INFO      Confusion matrix saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\figures\confusion_LR_CSP_subject003.png
2026-03-10 00:54:14,289  INFO      Saved figure: roc_LR_CSP_subject003 (.png + .pdf)
2026-03-10 00:54:14,289  INFO      --- EEGNet + Raw ---
2026-03-10 00:54:14,313  INFO      Raw features: X=(30, 64, 721)  y=(30,)  (per-channel z-score normalized)
2026-03-10 00:54:14,315  INFO      --- EEGNet + Raw (Augmented) ---
2026-03-10 00:54:14,353  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:54:14,376  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:54:14,407  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:54:14,414  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:54:14,414  INFO      EEGNet Fold 1/10 — train=81, test=3
2026-03-10 00:54:14,415  INFO      Training EEGNet on device: cpu
2026-03-10 00:54:14,644  INFO      EEGNet parameters: 2834
2026-03-10 00:54:15,599  INFO        Epoch   1/300 — train_loss: 0.7143  val_loss: 0.6793  val_acc: 0.7059
2026-03-10 00:54:22,670  INFO        Epoch  25/300 — train_loss: 0.0332  val_loss: 0.0184  val_acc: 1.0000
2026-03-10 00:54:29,469  INFO        Epoch  50/300 — train_loss: 0.0032  val_loss: 0.0022  val_acc: 1.0000
2026-03-10 00:54:36,188  INFO        Epoch  75/300 — train_loss: 0.0030  val_loss: 0.0010  val_acc: 1.0000
2026-03-10 00:54:42,953  INFO        Epoch 100/300 — train_loss: 0.0018  val_loss: 0.0006  val_acc: 1.0000
2026-03-10 00:54:50,763  INFO        Epoch 125/300 — train_loss: 0.0012  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:54:58,020  INFO        Epoch 150/300 — train_loss: 0.0007  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:55:04,829  INFO        Epoch 175/300 — train_loss: 0.0014  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:55:11,718  INFO        Epoch 200/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:55:18,392  INFO        Epoch 225/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:55:25,054  INFO        Epoch 250/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:55:31,692  INFO        Epoch 275/300 — train_loss: 0.0004  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:55:38,723  INFO        Epoch 300/300 — train_loss: 0.0001  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:55:38,824  INFO        Fold  1/10 — accuracy: 1.0000
2026-03-10 00:55:38,960  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:55:38,976  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:55:39,095  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:55:39,095  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:55:39,095  INFO      EEGNet Fold 2/10 — train=81, test=3
2026-03-10 00:55:39,095  INFO      Training EEGNet on device: cpu
2026-03-10 00:55:39,139  INFO      EEGNet parameters: 2834
2026-03-10 00:55:39,557  INFO        Epoch   1/300 — train_loss: 0.7419  val_loss: 0.6676  val_acc: 0.8235
2026-03-10 00:55:45,998  INFO        Epoch  25/300 — train_loss: 0.0279  val_loss: 0.0144  val_acc: 1.0000
2026-03-10 00:55:52,574  INFO        Epoch  50/300 — train_loss: 0.0037  val_loss: 0.0027  val_acc: 1.0000
2026-03-10 00:55:59,126  INFO        Epoch  75/300 — train_loss: 0.0018  val_loss: 0.0013  val_acc: 1.0000
2026-03-10 00:56:05,798  INFO        Epoch 100/300 — train_loss: 0.0011  val_loss: 0.0008  val_acc: 1.0000
2026-03-10 00:56:12,480  INFO        Epoch 125/300 — train_loss: 0.0010  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:56:19,016  INFO        Epoch 150/300 — train_loss: 0.0006  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:56:25,641  INFO        Epoch 175/300 — train_loss: 0.0005  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:56:32,220  INFO        Epoch 200/300 — train_loss: 0.0004  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:56:38,922  INFO        Epoch 225/300 — train_loss: 0.0004  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:56:45,666  INFO        Epoch 250/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:56:53,520  INFO        Epoch 275/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:57:01,447  INFO        Epoch 300/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:57:01,454  INFO        Fold  2/10 — accuracy: 0.6667
2026-03-10 00:57:01,515  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:57:01,532  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:57:01,554  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:57:01,558  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:57:01,558  INFO      EEGNet Fold 3/10 — train=81, test=3
2026-03-10 00:57:01,558  INFO      Training EEGNet on device: cpu
2026-03-10 00:57:01,585  INFO      EEGNet parameters: 2834
2026-03-10 00:57:01,861  INFO        Epoch   1/300 — train_loss: 0.7398  val_loss: 0.6795  val_acc: 0.8235
2026-03-10 00:57:08,707  INFO        Epoch  25/300 — train_loss: 0.0394  val_loss: 0.0203  val_acc: 1.0000
2026-03-10 00:57:15,747  INFO        Epoch  50/300 — train_loss: 0.0055  val_loss: 0.0032  val_acc: 1.0000
2026-03-10 00:57:22,502  INFO        Epoch  75/300 — train_loss: 0.0023  val_loss: 0.0014  val_acc: 1.0000
2026-03-10 00:57:30,359  INFO        Epoch 100/300 — train_loss: 0.0012  val_loss: 0.0008  val_acc: 1.0000
2026-03-10 00:57:37,221  INFO        Epoch 125/300 — train_loss: 0.0007  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:57:43,960  INFO        Epoch 150/300 — train_loss: 0.0008  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:57:50,580  INFO        Epoch 175/300 — train_loss: 0.0008  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:57:57,177  INFO        Epoch 200/300 — train_loss: 0.0007  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:58:03,782  INFO        Epoch 225/300 — train_loss: 0.0004  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:58:10,469  INFO        Epoch 250/300 — train_loss: 0.0006  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:58:17,062  INFO        Epoch 275/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:58:23,770  INFO        Epoch 300/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:58:23,785  INFO        Fold  3/10 — accuracy: 0.6667
2026-03-10 00:58:23,824  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:58:23,840  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:58:23,860  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:58:23,865  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:58:23,865  INFO      EEGNet Fold 4/10 — train=81, test=3
2026-03-10 00:58:23,865  INFO      Training EEGNet on device: cpu
2026-03-10 00:58:23,889  INFO      EEGNet parameters: 2834
2026-03-10 00:58:24,141  INFO        Epoch   1/300 — train_loss: 0.6735  val_loss: 0.6962  val_acc: 0.5294
2026-03-10 00:58:30,599  INFO        Epoch  25/300 — train_loss: 0.0540  val_loss: 0.0493  val_acc: 1.0000
2026-03-10 00:58:37,220  INFO        Epoch  50/300 — train_loss: 0.0047  val_loss: 0.0044  val_acc: 1.0000
2026-03-10 00:58:43,838  INFO        Epoch  75/300 — train_loss: 0.0029  val_loss: 0.0017  val_acc: 1.0000
2026-03-10 00:58:50,707  INFO        Epoch 100/300 — train_loss: 0.0015  val_loss: 0.0008  val_acc: 1.0000
2026-03-10 00:58:58,858  INFO        Epoch 125/300 — train_loss: 0.0023  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 00:59:06,311  INFO        Epoch 150/300 — train_loss: 0.0006  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 00:59:13,220  INFO        Epoch 175/300 — train_loss: 0.0005  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 00:59:21,126  INFO        Epoch 200/300 — train_loss: 0.0004  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 00:59:28,355  INFO        Epoch 225/300 — train_loss: 0.0005  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:59:35,513  INFO        Epoch 250/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:59:42,609  INFO        Epoch 275/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:59:49,173  INFO        Epoch 300/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 00:59:49,190  INFO        Fold  4/10 — accuracy: 1.0000
2026-03-10 00:59:49,229  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 00:59:49,242  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 00:59:49,269  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 00:59:49,274  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 00:59:49,274  INFO      EEGNet Fold 5/10 — train=81, test=3
2026-03-10 00:59:49,274  INFO      Training EEGNet on device: cpu
2026-03-10 00:59:49,296  INFO      EEGNet parameters: 2834
2026-03-10 00:59:49,560  INFO        Epoch   1/300 — train_loss: 0.6970  val_loss: 0.6802  val_acc: 0.7059
2026-03-10 00:59:55,920  INFO        Epoch  25/300 — train_loss: 0.0189  val_loss: 0.0150  val_acc: 1.0000
2026-03-10 01:00:02,440  INFO        Epoch  50/300 — train_loss: 0.0040  val_loss: 0.0022  val_acc: 1.0000
2026-03-10 01:00:08,950  INFO        Epoch  75/300 — train_loss: 0.0016  val_loss: 0.0010  val_acc: 1.0000
2026-03-10 01:00:15,550  INFO        Epoch 100/300 — train_loss: 0.0010  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 01:00:22,437  INFO        Epoch 125/300 — train_loss: 0.0014  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 01:00:29,000  INFO        Epoch 150/300 — train_loss: 0.0005  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 01:00:35,597  INFO        Epoch 175/300 — train_loss: 0.0004  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 01:00:42,154  INFO        Epoch 200/300 — train_loss: 0.0006  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:00:48,710  INFO        Epoch 225/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:00:55,529  INFO        Epoch 250/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:01:02,029  INFO        Epoch 275/300 — train_loss: 0.0001  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:01:08,474  INFO        Epoch 300/300 — train_loss: 0.0001  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:01:08,493  INFO        Fold  5/10 — accuracy: 0.6667
2026-03-10 01:01:08,526  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 01:01:08,540  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 01:01:08,557  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 01:01:08,562  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 01:01:08,562  INFO      EEGNet Fold 6/10 — train=81, test=3
2026-03-10 01:01:08,562  INFO      Training EEGNet on device: cpu
2026-03-10 01:01:08,584  INFO      EEGNet parameters: 2834
2026-03-10 01:01:08,836  INFO        Epoch   1/300 — train_loss: 0.7594  val_loss: 0.6712  val_acc: 0.7059
2026-03-10 01:01:15,078  INFO        Epoch  25/300 — train_loss: 0.0491  val_loss: 0.0370  val_acc: 1.0000
2026-03-10 01:01:21,564  INFO        Epoch  50/300 — train_loss: 0.0089  val_loss: 0.0048  val_acc: 1.0000
2026-03-10 01:01:28,096  INFO        Epoch  75/300 — train_loss: 0.0023  val_loss: 0.0022  val_acc: 1.0000
2026-03-10 01:01:34,875  INFO        Epoch 100/300 — train_loss: 0.0016  val_loss: 0.0010  val_acc: 1.0000
2026-03-10 01:01:43,966  INFO        Epoch 125/300 — train_loss: 0.0009  val_loss: 0.0005  val_acc: 1.0000
2026-03-10 01:01:50,544  INFO        Epoch 150/300 — train_loss: 0.0009  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 01:01:57,028  INFO        Epoch 175/300 — train_loss: 0.0009  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 01:02:03,798  INFO        Epoch 200/300 — train_loss: 0.0005  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 01:02:10,608  INFO        Epoch 225/300 — train_loss: 0.0003  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 01:02:17,220  INFO        Epoch 250/300 — train_loss: 0.0005  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:02:23,772  INFO        Epoch 275/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:02:30,263  INFO        Epoch 300/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:02:30,263  INFO        Fold  6/10 — accuracy: 0.6667
2026-03-10 01:02:30,314  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 01:02:30,331  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 01:02:30,352  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 01:02:30,357  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 01:02:30,357  INFO      EEGNet Fold 7/10 — train=81, test=3
2026-03-10 01:02:30,358  INFO      Training EEGNet on device: cpu
2026-03-10 01:02:30,380  INFO      EEGNet parameters: 2834
2026-03-10 01:02:30,720  INFO        Epoch   1/300 — train_loss: 0.6510  val_loss: 0.6899  val_acc: 0.5882
2026-03-10 01:02:37,031  INFO        Epoch  25/300 — train_loss: 0.0214  val_loss: 0.0109  val_acc: 1.0000
2026-03-10 01:02:43,594  INFO        Epoch  50/300 — train_loss: 0.0032  val_loss: 0.0016  val_acc: 1.0000
2026-03-10 01:02:50,127  INFO        Epoch  75/300 — train_loss: 0.0022  val_loss: 0.0007  val_acc: 1.0000
2026-03-10 01:02:56,704  INFO        Epoch 100/300 — train_loss: 0.0016  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 01:03:03,284  INFO        Epoch 125/300 — train_loss: 0.0004  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 01:03:10,080  INFO        Epoch 150/300 — train_loss: 0.0005  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 01:03:16,675  INFO        Epoch 175/300 — train_loss: 0.0004  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:03:23,267  INFO        Epoch 200/300 — train_loss: 0.0004  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:03:29,906  INFO        Epoch 225/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:03:36,475  INFO        Epoch 250/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:03:43,095  INFO        Epoch 275/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:03:49,781  INFO        Epoch 300/300 — train_loss: 0.0003  val_loss: 0.0000  val_acc: 1.0000
2026-03-10 01:03:49,794  INFO        Fold  7/10 — accuracy: 1.0000
2026-03-10 01:03:49,828  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 01:03:49,844  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 01:03:49,867  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 01:03:49,872  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 01:03:49,872  INFO      EEGNet Fold 8/10 — train=81, test=3
2026-03-10 01:03:49,872  INFO      Training EEGNet on device: cpu
2026-03-10 01:03:49,893  INFO      EEGNet parameters: 2834
2026-03-10 01:03:50,160  INFO        Epoch   1/300 — train_loss: 0.7297  val_loss: 0.6920  val_acc: 0.4118
2026-03-10 01:03:56,863  INFO        Epoch  25/300 — train_loss: 0.0738  val_loss: 0.1268  val_acc: 0.9412
2026-03-10 01:04:03,387  INFO        Epoch  50/300 — train_loss: 0.0086  val_loss: 0.0070  val_acc: 1.0000
2026-03-10 01:04:10,026  INFO        Epoch  75/300 — train_loss: 0.0050  val_loss: 0.0023  val_acc: 1.0000
2026-03-10 01:04:17,104  INFO        Epoch 100/300 — train_loss: 0.0022  val_loss: 0.0011  val_acc: 1.0000
2026-03-10 01:04:23,946  INFO        Epoch 125/300 — train_loss: 0.0012  val_loss: 0.0006  val_acc: 1.0000
2026-03-10 01:04:30,473  INFO        Epoch 150/300 — train_loss: 0.0008  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 01:04:37,019  INFO        Epoch 175/300 — train_loss: 0.0007  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 01:04:43,610  INFO        Epoch 200/300 — train_loss: 0.0006  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 01:04:50,251  INFO        Epoch 225/300 — train_loss: 0.0004  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:04:56,840  INFO        Epoch 250/300 — train_loss: 0.0005  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:05:03,575  INFO        Epoch 275/300 — train_loss: 0.0004  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:05:10,178  INFO        Epoch 300/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:05:10,203  INFO        Fold  8/10 — accuracy: 1.0000
2026-03-10 01:05:10,238  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 01:05:10,252  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 01:05:10,272  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 01:05:10,278  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 01:05:10,278  INFO      EEGNet Fold 9/10 — train=81, test=3
2026-03-10 01:05:10,278  INFO      Training EEGNet on device: cpu
2026-03-10 01:05:10,300  INFO      EEGNet parameters: 2834
2026-03-10 01:05:10,619  INFO        Epoch   1/300 — train_loss: 0.6774  val_loss: 0.6741  val_acc: 0.6471
2026-03-10 01:05:16,969  INFO        Epoch  25/300 — train_loss: 0.0405  val_loss: 0.0274  val_acc: 1.0000
2026-03-10 01:05:23,551  INFO        Epoch  50/300 — train_loss: 0.0070  val_loss: 0.0043  val_acc: 1.0000
2026-03-10 01:05:30,064  INFO        Epoch  75/300 — train_loss: 0.0034  val_loss: 0.0017  val_acc: 1.0000
2026-03-10 01:05:36,660  INFO        Epoch 100/300 — train_loss: 0.0024  val_loss: 0.0010  val_acc: 1.0000
2026-03-10 01:05:43,262  INFO        Epoch 125/300 — train_loss: 0.0013  val_loss: 0.0006  val_acc: 1.0000
2026-03-10 01:05:49,861  INFO        Epoch 150/300 — train_loss: 0.0006  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 01:05:56,562  INFO        Epoch 175/300 — train_loss: 0.0005  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 01:06:03,079  INFO        Epoch 200/300 — train_loss: 0.0005  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 01:06:09,658  INFO        Epoch 225/300 — train_loss: 0.0008  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 01:06:16,253  INFO        Epoch 250/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:06:22,767  INFO        Epoch 275/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:06:29,266  INFO        Epoch 300/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:06:29,284  INFO        Fold  9/10 — accuracy: 0.6667
2026-03-10 01:06:29,318  INFO      Gaussian-noise augmentation: 27 -> 54 epochs (std=0.0100)
2026-03-10 01:06:29,336  INFO      Temporal-jitter augmentation: 27 -> 54 epochs (max_shift=8 samples)
2026-03-10 01:06:29,356  INFO      Combined augmentation: 27 -> 81 epochs (x3)
2026-03-10 01:06:29,361  INFO        Augmented training data: 27 -> 81 epochs
2026-03-10 01:06:29,361  INFO      EEGNet Fold 10/10 — train=81, test=3
2026-03-10 01:06:29,361  INFO      Training EEGNet on device: cpu
2026-03-10 01:06:29,380  INFO      EEGNet parameters: 2834
2026-03-10 01:06:29,641  INFO        Epoch   1/300 — train_loss: 0.6811  val_loss: 0.6857  val_acc: 0.5294
2026-03-10 01:06:35,955  INFO        Epoch  25/300 — train_loss: 0.0190  val_loss: 0.0163  val_acc: 1.0000
2026-03-10 01:06:42,478  INFO        Epoch  50/300 — train_loss: 0.0048  val_loss: 0.0028  val_acc: 1.0000
2026-03-10 01:06:49,075  INFO        Epoch  75/300 — train_loss: 0.0022  val_loss: 0.0012  val_acc: 1.0000
2026-03-10 01:06:55,564  INFO        Epoch 100/300 — train_loss: 0.0009  val_loss: 0.0007  val_acc: 1.0000
2026-03-10 01:07:03,001  INFO        Epoch 125/300 — train_loss: 0.0016  val_loss: 0.0004  val_acc: 1.0000
2026-03-10 01:07:09,799  INFO        Epoch 150/300 — train_loss: 0.0005  val_loss: 0.0003  val_acc: 1.0000
2026-03-10 01:07:16,423  INFO        Epoch 175/300 — train_loss: 0.0005  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 01:07:23,066  INFO        Epoch 200/300 — train_loss: 0.0003  val_loss: 0.0002  val_acc: 1.0000
2026-03-10 01:07:30,632  INFO        Epoch 225/300 — train_loss: 0.0002  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:07:37,263  INFO        Epoch 250/300 — train_loss: 0.0007  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:07:43,751  INFO        Epoch 275/300 — train_loss: 0.0003  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:07:50,226  INFO        Epoch 300/300 — train_loss: 0.0001  val_loss: 0.0001  val_acc: 1.0000
2026-03-10 01:07:50,251  INFO        Fold 10/10 — accuracy: 1.0000
2026-03-10 01:07:50,331  INFO      EEGNet — Accuracy: 0.8333 +/- 0.1667  F1: 0.8331  AUC: 0.9244

==================================================
EEGNet — Cross-Validation Results
==================================================
  Fold  1: 1.0000
  Fold  2: 0.6667
  Fold  3: 0.6667
  Fold  4: 1.0000
  Fold  5: 0.6667
  Fold  6: 0.6667
  Fold  7: 1.0000
  Fold  8: 1.0000
  Fold  9: 0.6667
  Fold 10: 1.0000
--------------------------------------------------
  Mean accuracy:  0.8333 +/- 0.1667
  Macro F1-score: 0.8331
  AUC-ROC:        0.9244
==================================================


=======================================================
  EEGNet_Raw — Subject 3 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.8333
  precision_macro               0.8348
  precision_left                0.8571
  precision_right               0.8125
  recall_macro                  0.8333
  recall_left                   0.8000
  recall_right                  0.8667
  f1_macro                      0.8331
  f1_left                       0.8276
  f1_right                      0.8387
  auc_roc                       0.9244
  cohens_kappa                  0.6667
  cv_mean_accuracy              0.8333
  cv_std_accuracy               0.1667
=======================================================

2026-03-10 01:07:50,377  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\EEGNet_Raw_subject003.json
2026-03-10 01:07:50,737  INFO      Confusion matrix saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\figures\confusion_EEGNet_Raw_subject003.png
2026-03-10 01:07:51,018  INFO      Saved figure: roc_EEGNet_Raw_subject003 (.png + .pdf)
2026-03-10 01:07:51,491  INFO      Saved figure: training_curves_EEGNet_subject003 (.png + .pdf)
2026-03-10 01:07:51,717  INFO      Saved figure: roc_comparison_subject003 (.png + .pdf)
Subjects: 100%|██████████████████████████████████████████████████████████████████████| 3/3 [55:06<00:00, 1102.14s/it]
2026-03-10 01:07:51,845  INFO      Comparison CSV saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\model_comparison.csv
C:\Users\Daniel Lopez\AppData\Roaming\Python\Python312\site-packages\scipy\stats\_axis_nan_policy.py:423: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
  return hypotest_fun_in(*args, **kwds)
2026-03-10 01:07:51,942  INFO      Comparison JSON saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\model_comparison.json
2026-03-10 01:07:52,239  INFO      Saved figure: subject_accuracy_comparison (.png + .pdf)
2026-03-10 01:07:52,239  INFO      Full metrics CSV saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\full_metrics.csv
2026-03-10 01:07:52,273  INFO      Evaluation report saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\evaluation_report.txt
2026-03-10 01:07:52,274  INFO      Aggregate stats saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\aggregate_stats.json
===========================================================================
  EEG BCI PIPELINE — COMPREHENSIVE EVALUATION REPORT
  Generated: 2026-03-10 01:07:52
===========================================================================

Per-Subject Accuracy Breakdown
===========================================================================
 Subject        LR+PSD        LR+CSP        EEGNet          Best
---------------------------------------------------------------------------
       1        0.6000        0.9667        0.8667        LR+CSP
       2        0.6000        0.7333        0.6333        LR+CSP
       3        0.4667        0.9333        0.8333        LR+CSP


Aggregate Statistics Across Subjects
===========================================================================
  Model               Mean      Std      Min      Max   Median     N
  --------------- -------- -------- -------- -------- -------- -----
  LR_PSD            0.5556   0.0770   0.4667   0.6000   0.6000     3
  LR_CSP            0.8778   0.1262   0.7333   0.9667   0.9333     3
  EEGNet_Raw        0.7778   0.1262   0.6333   0.8667   0.8333     3

F1 (Macro) and AUC-ROC Summary
===========================================================================
  Model              Mean F1   Mean AUC
  --------------- ---------- ----------
  LR_PSD              0.5542     0.6098
  LR_CSP              0.8774     0.9274
  EEGNet_Raw          0.7757     0.8577

Pairwise Statistical Tests (paired t-test)
===========================================================================
  LR_PSD vs LR_CSP: t=-3.2628, p=0.0825 — not significant
  LR_PSD vs EEGNet_Raw: t=-2.2502, p=0.1533 — not significant
  LR_CSP vs EEGNet_Raw: t=inf, p=0.0000 — SIGNIFICANT

Best model by mean accuracy: LR_CSP (0.8778)

===========================================================================
  END OF REPORT
===========================================================================

================================================================================
  PIPELINE COMPLETE — Model Comparison
================================================================================

  Per-Subject Accuracy:
 subject  lr_psd_accuracy  lr_csp_accuracy  eegnet_accuracy
       1           0.6000           0.9667           0.8667
       2           0.6000           0.7333           0.6333
       3           0.4667           0.9333           0.8333

--------------------------------------------------------------------------------
  Model Summary:
  Model             Mean Acc    Std Acc    Mean F1   Mean AUC
  --------------- ---------- ---------- ---------- ----------
  LR + PSD            0.5556     0.0770     0.5542     0.6098
  LR + CSP            0.8778     0.1262     0.8774     0.9274
  EEGNet + Raw        0.7778     0.1262     0.7757     0.8577

  Pairwise t-tests (paired, p < 0.05):
  LR_PSD vs LR_CSP: t=-3.263, p=0.0825 — significant: no
  LR_PSD vs EEGNet_Raw: t=-2.250, p=0.1533 — significant: no
  LR_CSP vs EEGNet_Raw: t=inf, p=0.0000 — significant: YES *

--------------------------------------------------------------------------------
  Hyperparameter Tuning Results (LR + PSD):
  Metric                  Default      Tuned      Delta
  -------------------- ---------- ---------- ----------
  Mean Accuracy            0.5556     0.5444    -0.0111

  Subjects completed: 3/3
================================================================================


==================================================================================================================================================
================================================================================
==================================================================================================================================================
================================================================================
==================================================================================================================================================

Test LOSO (needs 3+ subjects):

$  python train.py --loso --subjects 1 2 3 --loso-models LR_PSD
2026-03-10 01:14:56,954  INFO      Global random seeds set to 42
2026-03-10 01:14:56,954  INFO      Pipeline configuration: {'PROJECT_ROOT': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI'), 'DATA_PATH': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/data/eegbci'), 'OUTPUT_DIR': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/outputs'), 'FIGURES_DIR': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/outputs/figures'), 'MODELS_DIR': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/outputs/models'), 'RESULTS_DIR': WindowsPath('C:/Users/Daniel Lopez/Desktop/Neet-a-thon/BCI/outputs/results'), 'SUBJECTS': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109], 'CACHE_RESULTS': True, 'RUNS': [3, 7], 'SAMPLING_RATE': 160, 'N_CHANNELS': 64, 'BANDPASS_LOW': 1.0, 'BANDPASS_HIGH': 40.0, 'NOTCH_FREQS': [60.0, 120.0], 'EPOCH_TMIN': -0.5, 'EPOCH_TMAX': 4.0, 'BASELINE': (None, 0), 'REJECT_THRESHOLD': None, 'REJECT_WARN_RATIO': 0.3, 'USE_ICA': False, 'ICA_N_COMPONENTS': 20, 'ICA_METHOD': 'fastica', 'ICA_RANDOM_STATE': 42, 'ICA_EOG_THRESHOLD': 
3.0, 'ICA_MUSCLE_THRESHOLD': 1.0, 'EVENT_ID': {'left': 1, 'right': 2}, 'PSD_FMIN': 1.0, 'PSD_FMAX': 40.0, 'PSD_N_FFT': 320, 'PSD_N_OVERLAP': 160, 'PSD_WINDOW': 'hann', 'FREQ_BANDS': {'delta': (1, 4), 'theta': (4, 8), 'mu': (8, 12), 'low_beta': (13, 20), 'high_beta': (20, 30), 'low_gamma': (30, 40)}, 'USE_ROI_CHANNELS': False, 'MOTOR_ROI_CHANNELS': ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4'], 'CSP_N_COMPONENTS': 4, 'LR_SOLVER': 'lbfgs', 'LR_MAX_ITER': 1000, 'LR_CLASS_WEIGHT': 'balanced', 'LR_C_GRID': [0.001, 0.01, 0.1, 1.0, 10.0], 'EEGNET_F1': 8, 'EEGNET_F2': 16, 'EEGNET_D': 2, 'EEGNET_DROPOUT': 0.5, 'EEGNET_LR': 0.001, 'EEGNET_BATCH_SIZE': 32, 'EEGNET_MAX_EPOCHS': 300, 'EEGNET_PATIENCE': 30, 'CV_N_FOLDS': 10, 'RANDOM_SEED': 42, 'AUGMENT_GAUSSIAN_STD': 0.01, 'AUGMENT_TEMPORAL_JITTER_MS': 50, 'USE_MLFLOW': False, 
'MLFLOW_EXPERIMENT_NAME': 'EEG-BCI-Pipeline', 'MLFLOW_TRACKING_URI': 'file:./mlruns'}
2026-03-10 01:14:56,969  INFO      Processing 3 subjects: [1, 2, 3]
2026-03-10 01:14:56,970  INFO      Options: cache=True, tune=False, augment=False, loso=True, ica=False, roi=False    
2026-03-10 01:14:56,970  INFO      LOSO mode: 3 subjects, models=['LR_PSD']
Loading subjects:   0%|                                                                        | 0/3 [00:00<?, ?it/s2 
026-03-10 01:14:57,090  INFO      Downloaded  subject=001  file=S001R03.edf
2026-03-10 01:14:57,090  INFO      Downloaded  subject=001  file=S001R07.edf
2026-03-10 01:14:57,090  INFO      Download complete: 1 subjects, 2 total files.
2026-03-10 01:14:57,798  INFO      Loaded  subject=001  runs=[3, 7]  channels=64  sfreq=160 Hz  n_times=40000  duration=250.0 s
2026-03-10 01:14:57,798  INFO      Applying filters: bandpass 1.0–40.0 Hz, notch [60.0, 120.0] Hz, FIR
2026-03-10 01:14:58,006  INFO      Filtering complete: shape before=(64, 40000)  after=(64, 40000)  channels=64
2026-03-10 01:14:58,006  INFO      Event mapping: {'left': 2, 'right': 3} (from raw annotations: {'T0': 1, 'T1': 2, 'T2': 3})
2026-03-10 01:14:58,081  INFO      Epochs: total=30  kept=30  rejected=0 (0.0%)  left=16  right=14
2026-03-10 01:14:58,328  INFO      PSD features: X=(30, 384)  y=(30,)  bands=6  channels=64
Loading subjects:  33%|█████████████████████▎                                          | 1/3 [00:01<00:02,  1.36s/it]2026-03-10 01:14:58,422  INFO      Downloaded  subject=002  file=S002R03.edf
2026-03-10 01:14:58,422  INFO      Downloaded  subject=002  file=S002R07.edf
2026-03-10 01:14:58,422  INFO      Download complete: 1 subjects, 2 total files.
2026-03-10 01:14:58,688  INFO      Loaded  subject=002  runs=[3, 7]  channels=64  sfreq=160 Hz  n_times=39360  duration=246.0 s
2026-03-10 01:14:58,704  INFO      Applying filters: bandpass 1.0–40.0 Hz, notch [60.0, 120.0] Hz, FIR
2026-03-10 01:14:58,878  INFO      Filtering complete: shape before=(64, 39360)  after=(64, 39360)  channels=64
2026-03-10 01:14:58,894  INFO      Event mapping: {'left': 2, 'right': 3} (from raw annotations: {'T0': 1, 'T1': 2, 'T2': 3})
2026-03-10 01:14:58,952  INFO      Epochs: total=30  kept=30  rejected=0 (0.0%)  left=15  right=15
2026-03-10 01:14:59,219  INFO      PSD features: X=(30, 384)  y=(30,)  bands=6  channels=64
Loading subjects:  67%|██████████████████████████████████████████▋                     | 2/3 [00:02<00:01,  1.08s/it]2026-03-10 01:14:59,331  INFO      Downloaded  subject=003  file=S003R03.edf
2026-03-10 01:14:59,331  INFO      Downloaded  subject=003  file=S003R07.edf
2026-03-10 01:14:59,331  INFO      Download complete: 1 subjects, 2 total files.
2026-03-10 01:14:59,517  INFO      Loaded  subject=003  runs=[3, 7]  channels=64  sfreq=160 Hz  n_times=40000  duration=250.0 s
2026-03-10 01:14:59,532  INFO      Applying filters: bandpass 1.0–40.0 Hz, notch [60.0, 120.0] Hz, FIR
2026-03-10 01:14:59,728  INFO      Filtering complete: shape before=(64, 40000)  after=(64, 40000)  channels=64
2026-03-10 01:14:59,728  INFO      Event mapping: {'left': 2, 'right': 3} (from raw annotations: {'T0': 1, 'T1': 2, 'T2': 3})
2026-03-10 01:14:59,782  INFO      Epochs: total=30  kept=30  rejected=0 (0.0%)  left=15  right=15
2026-03-10 01:14:59,996  INFO      PSD features: X=(30, 384)  y=(30,)  bands=6  channels=64
Loading subjects: 100%|████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.01s/it]
2026-03-10 01:15:00,044  INFO      LOSO — test subject 001: accuracy=0.7333
2026-03-10 01:15:00,064  INFO      LOSO — test subject 002: accuracy=0.5667
2026-03-10 01:15:00,083  INFO      LOSO — test subject 003: accuracy=0.5333
2026-03-10 01:15:00,087  INFO      LOSO LR_PSD — Accuracy: 0.6111 +/- 0.0875  F1: 0.6111  AUC: 0.6136
2026-03-10 01:15:00,592  INFO      Saved figure: subject_accuracy_comparison (.png + .pdf)
2026-03-10 01:15:00,608  INFO      LOSO results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\loso_results.json

======================================================================
  LOSO Cross-Validation Results
======================================================================

  LR_PSD:
    Mean accuracy: 0.6111 +/- 0.0875
    Macro F1:      0.6111
    AUC-ROC:       0.6136
======================================================================


Daniel Lopez@DanielLopez MINGW64 ~/Desktop/Neet-a-thon/BCI (main)
$  python evaluate.py --model-path outputs/models/eegnet_fold01.pt --data-subjects 1
2026-03-10 01:21:12,391  INFO      Evaluating model: outputs\models\eegnet_fold01.pt (type=EEGNet_Raw)
2026-03-10 01:21:12,539  INFO      Downloaded  subject=001  file=S001R03.edf
2026-03-10 01:21:12,540  INFO      Downloaded  subject=001  file=S001R07.edf
2026-03-10 01:21:12,540  INFO      Download complete: 1 subjects, 2 total files.
2026-03-10 01:21:12,939  INFO      Loaded  subject=001  runs=[3, 7]  channels=64  sfreq=160 Hz  n_times=40000  duration=250.0 s
2026-03-10 01:21:12,954  INFO      Applying filters: bandpass 1.0–40.0 Hz, notch [60.0, 120.0] Hz, FIR
2026-03-10 01:21:13,157  INFO      Filtering complete: shape before=(64, 40000)  after=(64, 40000)  channels=64
2026-03-10 01:21:13,157  INFO      Event mapping: {'left': 2, 'right': 3} (from raw annotations: {'T0': 1, 'T1': 2, 'T2': 3})
2026-03-10 01:21:13,219  INFO      Epochs: total=30  kept=30  rejected=0 (0.0%)  left=16  right=14
2026-03-10 01:21:13,248  INFO      Raw features: X=(30, 64, 721)  y=(30,)  (per-channel z-score normalized)
2026-03-10 01:21:15,585  INFO      Evaluating on Subject 001
2026-03-10 01:21:15,686  INFO      Downloaded  subject=001  file=S001R03.edf
2026-03-10 01:21:15,686  INFO      Downloaded  subject=001  file=S001R07.edf
2026-03-10 01:21:15,686  INFO      Download complete: 1 subjects, 2 total files.
2026-03-10 01:21:15,819  INFO      Loaded  subject=001  runs=[3, 7]  channels=64  sfreq=160 Hz  n_times=40000  duration=250.0 s
2026-03-10 01:21:15,835  INFO      Applying filters: bandpass 1.0–40.0 Hz, notch [60.0, 120.0] Hz, FIR
2026-03-10 01:21:16,001  INFO      Filtering complete: shape before=(64, 40000)  after=(64, 40000)  channels=64
2026-03-10 01:21:16,001  INFO      Event mapping: {'left': 2, 'right': 3} (from raw annotations: {'T0': 1, 'T1': 2, 'T2': 3})
2026-03-10 01:21:16,060  INFO      Epochs: total=30  kept=30  rejected=0 (0.0%)  left=16  right=14
2026-03-10 01:21:16,084  INFO      Raw features: X=(30, 64, 721)  y=(30,)  (per-channel z-score normalized)

=======================================================
  EEGNet_Raw — Subject 1 — Evaluation Metrics
=======================================================
  Metric                         Value
  ------------------------- ----------
  accuracy                      0.8667
  precision_macro               0.8750
  precision_left                0.8333
  precision_right               0.9167
  recall_macro                  0.8616
  recall_left                   0.9375
  recall_right                  0.7857
  f1_macro                      0.8643
  f1_left                       0.8824
  f1_right                      0.8462
  auc_roc                       0.8795
  cohens_kappa                  0.7297
=======================================================

2026-03-10 01:21:16,173  INFO      Results saved to C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI\outputs\results\EEGNet_Raw_eval_subject001.json
2026-03-10 01:21:16,771  INFO      Saved figure: confusion_EEGNet_Raw_eval_subject001 (.png + .pdf)
2026-03-10 01:21:17,000  INFO      Saved figure: roc_EEGNet_Raw_eval_subject001 (.png + .pdf)

============================================================
  Standalone Evaluation — EEGNet_Raw
  Model: outputs\models\eegnet_fold01.pt
============================================================
  Subjects evaluated: 1
  Mean accuracy:      0.8667 +/- 0.0000
  Mean F1 (macro):    0.8643
  Mean AUC-ROC:       0.8795
============================================================

