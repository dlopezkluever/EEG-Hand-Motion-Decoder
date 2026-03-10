The Core Neuroscience Concepts

  EEG (Electroencephalography)

  A cap with 64 electrodes sits on someone's scalp. Each electrode picks up tiny      
  voltage fluctuations (microvolts) caused by large groups of neurons firing together 
  underneath. It's noisy, indirect, but cheap and fast (160 readings per second in    
  your dataset).

  Motor Imagery (MI)

  When you imagine moving your left hand (without actually doing it), your brain      
  produces a measurable signature over the motor cortex — the strip of brain that     
  controls movement. This is what the system tries to detect.

  The Key Signal: Event-Related Desynchronization (ERD)

  This is the actual phenomenon your classifier exploits. Here's how it works:        

  - Your brain constantly produces rhythmic oscillations in specific frequency bands: 
    - Mu rhythm (8–12 Hz) — the "idle" rhythm of the motor cortex
    - Beta rhythm (13–30 Hz) — also present over motor areas
  - When you imagine moving your right hand, the mu/beta power drops (desynchronizes) 
  over the left motor cortex (electrode C3) — and vice versa
  - This happens contralaterally (opposite side) because the left brain controls the  
  right body and vice versa

  So the whole game is: measure the power difference between C3 and C4 in the mu/beta 
  bands, and you can tell which hand someone is imagining moving.

  ---
  How Your Pipeline Works (Stage by Stage)

  Stage 1 — Data Loading

  You're using a free, public dataset from PhysioNet: 109 people, each with EEG       
  recordings while they imagined moving their left or right hand. The data comes as   
  .edf files (a standard EEG format). MNE-Python downloads and loads these
  automatically.

  Stage 2 — Preprocessing

  Raw EEG is extremely noisy. This stage cleans it up:

  1. Bandpass filter (1–40 Hz): Keeps only the frequencies you care about (mu and beta   live in 8–30 Hz). Removes slow drift below 1 Hz and high-frequency muscle noise    
  above 40 Hz
  2. Notch filter (60 Hz): Removes electrical interference from power lines
  3. Common Average Reference: Subtracts the average of all electrodes from each      
  electrode — removes noise common to all sensors
  4. Epoch extraction: Chops the continuous recording into 4.5-second segments (called   epochs), each one time-locked to when the person was told "imagine left" or        
  "imagine right"
  5. Artifact rejection: Throws away any epoch where the voltage spiked too high      
  (probably a muscle twitch or eye blink, not brain activity)

  Stage 3 — Feature Extraction

  Now you need to turn each epoch (a 64-channel × ~720-sample time series) into a     
  compact set of numbers a classifier can use. Three approaches:

  - Pathway A (PSD band power): For each channel, compute how much power is in each   
  frequency band (delta, theta, mu, beta, gamma). This gives you a 384-dimensional    
  feature vector per epoch. This directly captures the ERD phenomenon.
  - Pathway B (CSP — Common Spatial Patterns): A mathematical trick that finds the    
  optimal linear combinations of channels that maximize the difference between left   
  and right classes. It's basically asking: "what weighted mix of electrodes best     
  separates left from right?" This is the most effective classical method for motor   
  imagery.
  - Pathway C (Raw signal): Feed the raw epoch directly into a neural network and let 
  it learn its own features.

  Stage 4 — Classification

  Two models:

  - Logistic Regression (baseline): A simple linear classifier. Takes the PSD or CSP
  features and draws a decision boundary. It's the "can we do better than a coin      
  flip?" test.
  - EEGNet (primary): A small convolutional neural network specifically designed for  
  EEG. Its layers mirror what the classical pipeline does — temporal convolutions     
  learn frequency filters, depthwise convolutions learn spatial filters (like CSP),   
  and separable convolutions combine them. It's only ~2,000 parameters, which prevents   overfitting on small EEG datasets.

  Stage 5 — Evaluation

  Measure how well the models work using cross-validation (train on 9 folds, test on  
  1, repeat 10 times). Key metrics: accuracy, F1-score, AUC-ROC. The target is 75%    
  accuracy (chance is 50%).

  Stage 6 — Visualization

  Topographic scalp maps showing the ERD pattern, power spectrum comparisons at C3 vs 
  C4, confusion matrices, ROC curves, training loss curves.

  ---
  Why it's Achievable

  - The dataset is free and well-studied — you know what accuracy to expect
  - MNE-Python handles all the EEG-specific heavy lifting (file I/O, filtering,       
  epoching, CSP, visualization)
  - The classification problem is binary (left vs right) — the simplest possible BCI  
  task
  - Your PRD has a clean phased approach: Phase 0 just loads data, Phase 1 gets you a 
  working decoder, later phases add complexity

  Why it's Non-Trivial

  - EEG signals are very noisy (microvolts measured through skull and scalp)
  - There's huge variability between people — a model trained on one person may not   
  work on another
  - The discriminative signal (ERD) is subtle — it's a small power modulation in a    
  specific frequency band at specific electrodes

----


  ---

    
● Phase 0 complete. Here's what was built:

  ┌─────────────┬─────────────────────────────────────────────────────────────────┐   
  │    Task     │                             Status                              │   
  ├─────────────┼─────────────────────────────────────────────────────────────────┤   
  │ 0.1         │ Directory structure, requirements.txt, environment.yml,         │   
  │ Scaffold    │ .gitignore, all placeholder modules                             │   
  ├─────────────┼─────────────────────────────────────────────────────────────────┤   
  │ 0.2 Config  │ src/config.py — 43 config keys covering dataset, filtering,     │   
  │             │ epochs, PSD bands, CSP, LR, EEGNet, CV, seeds                   │   
  ├─────────────┼─────────────────────────────────────────────────────────────────┤   
  │ 0.3 Data    │ src/data_loader.py — download_data() with retry logic,          │   
  │ Loader      │ load_raw() with montage setup, standalone smoke test            │   
  ├─────────────┼─────────────────────────────────────────────────────────────────┤   
  │ 0.4 Tests   │ 6/6 tests passing — config imports, data download, 64 channels, │   
  │             │  160 Hz, annotations verified                                   │   
  └─────────────┴─────────────────────────────────────────────────────────────────┘   

  Smoke test confirmed: Subject 1 loaded with 64 channels, 160 Hz, 250s duration, 62  
  annotations (T0/T1/T2 events).


  
   Phase 1: MVP -- Complete

  Modules implemented:

  ┌────────────────────────┬───────────────────────────────────────────────────────┐  
  │          File          │                     What it does                      │  
  ├────────────────────────┼───────────────────────────────────────────────────────┤  
  │                        │ apply_filters() (bandpass 1-40 Hz FIR + notch + CAR)  │  
  │ src/preprocessing.py   │ and extract_epochs() (event mapping, baseline         │  
  │                        │ correction, artifact rejection with logging)          │  
  ├────────────────────────┼───────────────────────────────────────────────────────┤  
  │                        │ extract_psd_features() -- Welch PSD across 6          │  
  │ src/features.py        │ frequency bands x 64 channels = 384 features,         │  
  │                        │ StandardScaler normalized                             │  
  ├────────────────────────┼───────────────────────────────────────────────────────┤  
  │ src/models/logistic.py │ train_logistic() -- 10-fold stratified CV with        │  
  │                        │ per-fold accuracy, F1, AUC-ROC reporting              │  
  ├────────────────────────┼───────────────────────────────────────────────────────┤  
  │                        │ compute_metrics() (12 metrics including Cohen's       │  
  │ src/evaluate.py        │ Kappa), save_results() (JSON),                        │  
  │                        │ plot_confusion_matrix() (PNG heatmap),                │  
  │                        │ print_metrics_table()                                 │  
  ├────────────────────────┼───────────────────────────────────────────────────────┤  
  │ train.py               │ Full pipeline orchestrator with --subjects CLI arg,   │  
  │                        │ per-subject error handling, summary CSV output        │  
  └────────────────────────┴───────────────────────────────────────────────────────┘  

  Tests added (26 total, all passing):

  - tests/test_preprocessing.py -- 7 tests (filtering shape/values/NaN, epoch
  existence/classes/shape)
  - tests/test_features.py -- 5 tests (shape 384, labels, NaN/Inf checks)
  - tests/test_logistic.py -- 4 tests (above chance, fold count, predictions, AUC     
  range)
  - tests/test_evaluate.py -- 4 tests (perfect/random predictions, key presence,      
  types)

  Results (subjects 1-5):

  - Overall mean accuracy: 67.3% (above 50% chance threshold)
  - Per-subject JSON metrics saved to outputs/results/
  - Confusion matrix PNGs saved to outputs/figures/
  - Summary CSV at outputs/results/summary_logistic_regression.csv

  Notable adaptation:

  - Artifact rejection threshold set to None (disabled) because raw PhysioNet EEGBCI  
  data routinely exceeds 150 µV peak-to-peak without ICA cleanup. The rejection       
  infrastructure is fully in place and will be activated in Phase 5 after ICA is      
  added.

   Phase 1 Walkthrough

  Phase 1 builds the first working end-to-end BCI decoder. Raw EEG signals go in, a   
  classification accuracy comes out. Here's the data flow:

  Raw EEG → Filter → Epoch → Extract Features → Train Model → Evaluate

  Let me walk through each stage.

  ---
  1. Signal Preprocessing — Filtering (src/preprocessing.py)

  apply_filters(raw) cleans the raw EEG signal in three steps:

  1. Bandpass filter (1–40 Hz) — Removes frequencies outside the range we care about. 
  Below 1 Hz is slow drift (electrode movement, breathing). Above 40 Hz is mostly     
  muscle noise. The motor imagery signals we want (mu rhythm at 8–12 Hz, beta at 13–30 
  Hz) sit comfortably inside this range.
  2. Notch filter (60 Hz) — Removes power line interference. The US electrical grid   
  runs at 60 Hz and contaminates EEG recordings. (120 Hz harmonic was skipped because 
  our sampling rate of 160 Hz means the Nyquist limit is 80 Hz — you literally can't  
  represent 120 Hz in this data.)
  3. Common Average Reference (CAR) — EEG measures voltage differences, so you need a 
  reference point. CAR subtracts the average of all 64 channels from each channel.    
  This removes noise that's common across the whole scalp, making localized brain     
  activity (like motor cortex activation) stand out more.

  The function works on a copy so the original raw data is preserved.

  ---
  2. Epoch Extraction (src/preprocessing.py)

  extract_epochs(raw) chops the continuous signal into trial-sized chunks:

  - The PhysioNet data has annotations marking when each motor imagery task started:  
  T1 = "imagine left hand", T2 = "imagine right hand"
  - For each event, we cut a window from -0.5s before to 4.0s after the cue — giving  
  us 4.5 seconds per trial
  - Baseline correction subtracts the mean voltage during the pre-stimulus period     
  (-0.5 to 0s) from the whole epoch. This zeroes out any DC offset so each trial      
  starts from a common baseline
  - Artifact rejection is architecturally supported (reject epochs where any channel  
  exceeds a peak-to-peak threshold), but currently disabled because the raw data is   
  too noisy without ICA cleaning — that comes in Phase 5

  The result: ~30 labeled epochs per subject (roughly 15 left, 15 right).

  ---
  3. PSD Feature Extraction (src/features.py)

  extract_psd_features(epochs) converts raw time-domain signals into frequency-domain 
  features the classifier can use:

  1. Welch's method computes the Power Spectral Density (PSD) — how much power exists 
  at each frequency. Uses a 2-second window with 50% overlap and Hanning tapering to  
  reduce spectral leakage.
  2. Band power averaging — Rather than using every frequency bin, we average power   
  into 6 neuroscience-meaningful bands:
    - Delta (1–4 Hz) — deep sleep, not very useful here but included for completeness 
    - Theta (4–8 Hz) — drowsiness, memory
    - Mu (8–12 Hz) — key band — suppressed over motor cortex during motor imagery     
  (Event-Related Desynchronization)
    - Low Beta (13–20 Hz) — also modulated by motor imagery
    - High Beta (20–30 Hz) — motor planning
    - Low Gamma (30–40 Hz) — higher-level processing
  3. Flatten — 64 channels × 6 bands = 384 features per epoch
  4. StandardScaler normalizes each feature to zero mean and unit variance, preventing
  features with larger magnitudes from dominating the classifier

  The output: a matrix of shape (n_epochs, 384) and a label array y where 0=left,     
  1=right.

  ---
  4. Logistic Regression (src/models/logistic.py)

  train_logistic(X, y) trains and evaluates the baseline classifier:

  - Logistic Regression with class_weight='balanced' (adjusts for any class imbalance) 
  and L-BFGS solver
  - 10-fold Stratified Cross-Validation — splits data into 10 parts, trains on 9,     
  tests on 1, rotates. "Stratified" means each fold preserves the left/right class    
  ratio. With only 30 epochs, each fold has just 3 test samples, which explains the   
  high variance (accuracies jump between 0.33, 0.67, and 1.0)
  - Reports per-fold accuracy, mean ± std, macro F1, and AUC-ROC

  ---
  5. Evaluation (src/evaluate.py)

  compute_metrics() calculates 12 metrics from the cross-validated predictions:       
  - Accuracy, Precision, Recall, F1 (each per-class and macro-averaged)
  - AUC-ROC (how well the model separates classes across all thresholds)
  - Cohen's Kappa (accuracy adjusted for chance agreement)

  save_results() exports to JSON. plot_confusion_matrix() generates a seaborn heatmap 
  PNG showing true vs predicted labels.

  ---
  6. Pipeline Orchestrator (train.py)

  main() ties everything together:
  - Loops over subjects, runs the full pipeline for each
  - Catches per-subject errors without crashing the whole run
  - Aggregates all results into a summary CSV via Pandas
  - Prints a final comparison table
  - Supports --subjects 1 2 3 CLI argument

  ---
  Results

  Across subjects 1–5, the model achieved 67.3% mean accuracy — above the 50% chance  
  baseline. The high variance per fold (std ~0.23–0.29) is expected with only 30      
  epochs per subject and 3 samples per fold. Later phases will improve this with      
  better features (CSP), better models (EEGNet), and more data (109 subjects).        
