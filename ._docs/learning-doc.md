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

----

  The Big Picture

  Your project is a Brain-Computer Interface (BCI) — software that reads brain signals   (EEG) and figures out whether a person is imagining moving their left hand or their   right hand. That's it. Left or right — a binary classification problem.

  Phase 1 built the first working version: read brain data → clean it up → extract    
  numerical features → train a simple classifier. Phase 2 added two more approaches so   we can compare them and find what works best.

  ---
  What's EEG Data, Actually?

  Imagine 64 tiny sensors (electrodes) placed all over someone's scalp, like a cap.   
  Each sensor records tiny voltage fluctuations caused by brain activity — about 160  
  readings per second (160 Hz). So for one 4.5-second trial of someone imagining a    
  hand movement, each sensor captures ~721 numbers. That's your raw data: a matrix of 
  shape (64 channels, 721 timepoints).

  The data comes from PhysioNet — a public dataset where 109 people did this exact    
  experiment.

  ---
  The Three Approaches We Now Have

  Think of it like three different strategies to solve the same problem:

  Strategy 1: LR + PSD (built in Phase 1)

  "What frequencies is the brain producing?"

  - PSD = Power Spectral Density. This breaks a brain signal into its frequency       
  components — like how a prism splits white light into a rainbow. Brain signals have 
  meaningful frequency bands:
    - Mu rhythm (8–12 Hz): This is the key one for motor imagery. When you imagine    
  moving your LEFT hand, the mu rhythm decreases on the RIGHT side of your brain (and 
  vice versa). This is called Event-Related Desynchronization (ERD).
    - Beta (13–30 Hz): Also changes during motor imagery.
    - We compute power in 6 bands total across all 64 channels = 384 features per     
  trial.
  - LR = Logistic Regression. A simple, well-understood classifier. It draws a line   
  (hyperplane) in the 384-dimensional feature space that best separates "left" from   
  "right" trials.

  This got ~63% accuracy on Subject 1. Above chance (50%), but not great.

  ---
  Strategy 2: LR + CSP (new in Phase 2)

  "What spatial patterns on the scalp distinguish left from right?"

  This is the clever one. CSP = Common Spatial Patterns.

  Here's the intuition: when you imagine moving your left hand, certain electrodes    
  (especially C4, on the right side of the scalp) show more activity, while others    
  show less. CSP is an algorithm that learns which combinations of electrodes are most   different between the two classes.

  It finds spatial filters — weighted combinations of all 64 channels — that maximize 
  the variance for one class while minimizing it for the other. Think of it like: "if 
  I mix these channels together in just the right proportions, I get a signal that's  
  very active during left-hand imagination and very quiet during right-hand
  imagination."

  We use 4 CSP components (2 favoring each class), giving us just 4 features per      
  trial. Despite having way fewer features than PSD (4 vs 384), it often works better 
  because those 4 features are specifically designed to capture the left-vs-right     
  difference.

  This got ~87% accuracy on Subject 1 — a big improvement.

  The relevant code in src/features.py:

  def extract_csp_features(epochs, n_components=4):
      csp = CSP(n_components=4, log=True)  # log=True applies log-variance
      X = csp.fit_transform(data, y)       # learns filters AND transforms
      # X shape: (n_epochs, 4) — just 4 numbers per trial!

  ---
  Strategy 3: EEGNet + Raw (new in Phase 2)

  "Let a neural network figure it out from the raw signals."

  Instead of manually designing features (PSD bands, CSP filters), we feed the raw EEG   data directly into a Convolutional Neural Network (CNN) and let it learn what      
  patterns matter.

  EEGNet is a published architecture specifically designed for EEG. It's tiny compared   to image CNNs — only 2,834 parameters (a typical image CNN has millions). Here's   
  what each layer does:

  Raw EEG: (64 channels × 721 timepoints)
      ↓
  [Temporal Conv] — learns frequency filters (like our PSD bands,
                    but the network discovers them automatically)
      ↓
  [Depthwise Conv] — learns spatial filters (like CSP, but learned
                     jointly with everything else)
      ↓
  [Separable Conv] — combines the above into higher-level patterns
      ↓
  [Dense Layer] — outputs: "left" or "right"

  The architecture in src/models/eegnet.py follows this flow. Key design choices:     

  - ELU activation (not ReLU) — works better for EEG
  - Average pooling (not max pooling) — preserves timing information
  - Dropout = 0.5 — prevents overfitting on small datasets
  - Early stopping — monitors validation loss and stops training when it stops        
  improving (patience=30 epochs)

  This got ~83% accuracy on Subject 1.

  The raw data just needs per-channel z-score normalization (subtract mean, divide by 
  standard deviation for each channel) — no hand-crafted features needed:

  def extract_raw_features(epochs):
      mean = data.mean(axis=(0, 2), keepdims=True)  # per-channel mean
      std = data.std(axis=(0, 2), keepdims=True)     # per-channel std
      X = (data - mean) / std                         # normalize

  ---
  Cross-Validation: How We Measure Accuracy

  We can't just train on all the data and test on the same data — that would be       
  cheating (memorization). Instead we use 10-fold stratified cross-validation:        

  1. Split the ~30 trials into 10 groups
  2. Train on 9 groups, test on the 1 held-out group
  3. Repeat 10 times, each group gets a turn as the test set
  4. Average the 10 accuracy scores

  "Stratified" means each fold keeps the same ratio of left/right trials. All three   
  models use the exact same fold splits (same random seed) so the comparison is fair. 

  ---
  The Comparison Framework (train.py)

  The updated pipeline now runs all three strategies for each subject and produces:   

  1. Per-subject accuracy table — how each model did on each person
  2. Summary statistics — mean accuracy, F1-score, AUC-ROC across subjects
  3. Paired t-tests — statistical test asking "is the difference between models real, 
  or just random chance?" (needs multiple subjects to work)
  4. Saved outputs — CSV, JSON, confusion matrix plots for every combination

  ---
  Why Three Approaches?

  ┌────────────┬─────────────────────────────────┬───────────────────────────────┐    
  │  Approach  │              Pros               │             Cons              │    
  ├────────────┼─────────────────────────────────┼───────────────────────────────┤    
  │ LR + PSD   │ Simple, interpretable, fast     │ Ignores spatial patterns      │    
  ├────────────┼─────────────────────────────────┼───────────────────────────────┤    
  │ LR + CSP   │ Captures spatial brain patterns │ Needs careful tuning of       │    
  │            │  directly                       │ frequency bands               │    
  ├────────────┼─────────────────────────────────┼───────────────────────────────┤    
  │ EEGNet +   │ Learns everything automatically │ Needs more data, harder to    │    
  │ Raw        │                                 │ interpret                     │    
  └────────────┴─────────────────────────────────┴───────────────────────────────┘    

  In BCI research, CSP + LR is the classic "gold standard" baseline, while EEGNet     
  represents the modern deep learning approach. Having both lets you benchmark        
  properly.

  ---


Here's how to verify each part of Phase 3:

  3.1 & 3.2 — Visualizations (quick smoke test)

  Run this in your terminal from the project root:

  python -c "
  import sys; sys.path.insert(0, '.')
  from src.data_loader import download_data, load_raw
  from src.preprocessing import apply_filters, extract_epochs
  from src.visualize import *

  # Load one subject
  download_data(subjects=[1])
  raw = load_raw(subject=1)
  filtered = apply_filters(raw)
  epochs = extract_epochs(filtered)

  # Test all signal visualizations
  generate_all_signal_figures(epochs, subject_id=1)

  # Test model performance visualizations
  import numpy as np
  y_true = np.array([0,0,1,1,0,1,0,1])
  y_pred = np.array([0,1,1,1,0,0,0,1])
  y_prob = np.array([0.2,0.6,0.8,0.9,0.3,0.4,0.1,0.7])
  plot_confusion_matrix(y_true, y_pred, 'test_model', 1)
  plot_roc_curve(y_true, y_prob, 'test_model', 1)
  plot_training_curves({'train_loss':[1,.8,.5], 'val_loss':[1.1,.9,.6], 'val_accuracy':[.5,.6,.7]})        

  print('Done — check outputs/figures/')
  "

  Then open outputs/figures/ and confirm you see:
  - topomap_subject001.png + .pdf
  - psd_comparison_subject001.png + .pdf
  - butterfly_subject001.png + .pdf
  - confusion_test_model_subject001.png + .pdf
  - roc_test_model_subject001.png + .pdf
  - training_curves_EEGNet_subject000.png + .pdf

  3.3 — Evaluation Report

  python -c "
  import sys; sys.path.insert(0, '.')
  import pandas as pd
  from src.evaluate import generate_evaluation_report, compute_aggregate_stats

  # Fake results for 3 subjects
  df = pd.DataFrame([
      {'subject':1, 'lr_psd_accuracy':0.72, 'lr_csp_accuracy':0.68, 'eegnet_accuracy':0.65,
  'lr_psd_f1':0.71, 'lr_csp_f1':0.67, 'eegnet_f1':0.64, 'lr_psd_auc':0.78, 'lr_csp_auc':0.74,
  'eegnet_auc':0.70},
      {'subject':2, 'lr_psd_accuracy':0.80, 'lr_csp_accuracy':0.75, 'eegnet_accuracy':0.77,
  'lr_psd_f1':0.79, 'lr_csp_f1':0.74, 'eegnet_f1':0.76, 'lr_psd_auc':0.85, 'lr_csp_auc':0.80,
  'eegnet_auc':0.82},
      {'subject':3, 'lr_psd_accuracy':0.60, 'lr_csp_accuracy':0.55, 'eegnet_accuracy':0.58,
  'lr_psd_f1':0.59, 'lr_csp_f1':0.54, 'eegnet_f1':0.57, 'lr_psd_auc':0.65, 'lr_csp_auc':0.60,
  'eegnet_auc':0.62},
  ])

  generate_evaluation_report(df)
  print('Check outputs/results/evaluation_report.txt')
  "

  Verify outputs/results/evaluation_report.txt contains:
  - Per-subject accuracy breakdown table
  - Aggregate stats (mean, std, min, max, median)
  - F1/AUC summary
  - Best model identified

  3.4 — Notebooks

  Just open each in JupyterLab and confirm cells are present:
  jupyter lab notebooks/

  Full end-to-end (takes a few minutes)

  python train.py --subjects 1 2

  This runs the complete pipeline with all new visualizations and generates the evaluation report. Check   
  outputs/figures/ and outputs/results/ for all outputs.

  ---
  Quick checklist:

  ┌────────────────────────────────────┬─────────────────────────────────────────────────┐
  │               Check                │                      Where                      │
  ├────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ Topomaps, PSD, butterfly PNGs+PDFs │ outputs/figures/                                │
  ├────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ ROC curves per model               │ outputs/figures/roc_*.png                       │
  ├────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ Training curves                    │ outputs/figures/training_curves_*.png           │
  ├────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ Feature importance bar chart       │ outputs/figures/feature_importance_*.png        │
  ├────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ Subject accuracy comparison        │ outputs/figures/subject_accuracy_comparison.png │
  ├────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ Evaluation report text             │ outputs/results/evaluation_report.txt           │
  ├────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ Aggregate stats JSON               │ outputs/results/aggregate_stats.json            │
  ├────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ Full metrics CSV                   │ outputs/results/full_metrics.csv                │
  ├────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ 4 notebooks exist                  │ notebooks/01_*.ipynb through 04_*.ipynb         │
  └────────────────────────────────────┴─────────────────────────────────────────────────┘

  Want me to finish populating the notebook cells first, or would you prefer to run a verification now?    



  ----


  Here's a summary of everything implemented for Phase 4: Scale, Harden & Polish:

  4.1 Full Dataset Scaling (109 Subjects)

  - src/config.py: SUBJECTS updated from range(1, 11) to range(1, 110), added CACHE_RESULTS = True
  - train.py: tqdm progress bars for subject loop, result caching (skip subjects with existing JSON results   in outputs/results/), _load_cached_result() to resume from cached data

  4.2 Leave-One-Subject-Out (LOSO) Cross-Validation

  - src/evaluate.py: Added run_loso_cv() — trains on all-but-one subjects, tests on held-out subject,      
  supports LR_PSD, LR_CSP, and EEGNet_Raw
  - src/evaluate.py: Added compare_within_vs_cross_subject() — generates comparison table with accuracy    
  drop
  - train.py: Added run_loso_mode() and --loso / --loso-models CLI flags

  4.3 Hyperparameter Tuning — Logistic Regression

  - src/models/logistic.py: Added train_logistic_tuned() with nested CV (outer 10-fold, inner 5-fold       
  GridSearchCV over C = [0.001, 0.01, 0.1, 1.0, 10.0]), reports best C per fold and overall
  - train.py: Integrated via --tune flag, saves grid search details to JSON, prints tuned vs default       
  comparison

  4.4 Data Augmentation for EEGNet

  - src/augmentation.py (new): augment_gaussian_noise() (σ=0.01), augment_temporal_jitter() (±8 samples /  
  ±50ms), apply_augmentation() (combines both, 3x training data)
  - src/models/eegnet.py: train_eegnet_cv() now accepts optional augment_fn parameter, applies to training 
  folds only
  - train.py: Integrated via --augment flag

  4.5 Pipeline Hardening & Reproducibility

  - train.py: set_global_seeds() sets numpy, torch, Python random, CUDA, cuDNN seeds
  - Extended CLI: --subjects, --no-cache, --tune, --augment, --loso, --loso-models, --cv-folds,
  --output-dir, --seed
  - Per-subject error handling already existed, now with cache-aware progress tracking

  ● Start small to verify, then scale up:

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
  python train.py --loso --subjects 1 2 3 --loso-models LR_PSD

  I'd recommend starting with the first command. Want me to run it?