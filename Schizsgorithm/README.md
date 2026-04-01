# Schizsgorithm: EEG-Based Schizophrenia Classification

Schizsgorithm is a machine learning pipeline for classifying schizophrenia from EEG recordings. The project uses MNE for EEG preprocessing, extracts statistical features from fixed-length signal windows, and evaluates classification performance with subject-aware cross-validation to reduce leakage across individuals.

---

## Overview

EEG is a noisy, high-variance biosignal with substantial variability across subjects and recording sessions. This project explores whether schizophrenia-related patterns can be identified from EEG recordings using a reproducible preprocessing and feature-based classification pipeline.

The core idea is:

1. Load raw EEG recordings from EDF files  
2. Preprocess the signal using MNE  
3. Split continuous recordings into fixed-length epochs  
4. Extract statistical features from each epoch  
5. Train and evaluate a classifier using group-aware cross-validation  

---

## Pipeline

### 1. Data Loading
EEG recordings are loaded from EDF files using `mne.io.read_raw_edf`.

- Files beginning with `h` are treated as healthy controls
- Files beginning with `s` are treated as schizophrenia cases

### 2. Preprocessing
Each EEG recording is preprocessed with:

- EEG re-referencing
- Band-pass filtering from **0.5 Hz to 60 Hz**
- Segmentation into **5-second fixed-length epochs** with **1-second overlap**

This converts each subject recording into multiple fixed-length EEG windows.

### 3. Feature Extraction
For each epoch, the pipeline computes statistical features channel-wise, including:

- Mean
- Standard deviation
- Peak-to-peak amplitude
- Variance
- Minimum
- Maximum
- Argmin
- Argmax
- RMS
- Absolute difference signal
- Skewness
- Kurtosis

These features are concatenated into a single feature vector for classification.

### 4. Modeling
The current implementation uses:

- `StandardScaler`
- `LogisticRegression`
- `GridSearchCV` for tuning the regularization parameter `C`

### 5. Evaluation
The model is evaluated using **GroupKFold**, where each subject is treated as a separate group.

This is important because it prevents epochs from the same subject from appearing in both training and validation folds, reducing subject leakage and giving a more realistic estimate of generalization.

---

## Why Group-Aware Validation Matters

A major risk in EEG classification is overly optimistic performance caused by leakage across windows from the same subject. This project explicitly addresses that by grouping all epochs from a single subject together during cross-validation.

That means the classifier is evaluated more like a real-world system:
- train on some subjects
- test on unseen subjects

---

## Current Method

### Preprocessing
- Re-reference EEG
- Filter from 0.5 to 60 Hz
- Epoch into 5-second windows with 1-second overlap

### Features
Handcrafted statistical descriptors computed per channel and concatenated into one feature vector.

### Classifier
- Logistic Regression
- Hyperparameter tuning over:
  - `C = [0.1, 0.6, 2, 3, 4, 7, 34]`

### Cross-Validation
- `GroupKFold(n_splits=5)`

---

## Repository Structure

```text
.
├── datafiles/              # EDF files
├── main.py                 # training / evaluation pipeline
└── README.md
