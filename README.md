# Dysarthric Speech Severity Classification

> Classifying dysarthric speech as **Mild / Moderate / Severe** using acoustic features (MFCCs, ZCR, Spectral Centroid) and comparing SVM vs Random Forest classifiers.

---

## Overview

Dysarthria is a motor speech disorder caused by neurological conditions — cerebral palsy, ALS, Parkinson's disease — that weakens the muscles used for speech. Patients are clinically categorised by severity based on intelligibility and articulation control.

This project builds a **complete speech processing and classification pipeline** that:

- Generates synthetic speech samples with controllable dysarthric characteristics (jitter, shimmer, noise)
- Visualises audio as waveforms and Mel-spectrograms
- Extracts acoustic features: **13 MFCCs**, **Zero Crossing Rate**, and **Spectral Centroid**
- Trains and compares **SVM (RBF kernel)** and **Random Forest** classifiers
- Evaluates with 5-fold stratified cross-validation and confusion matrices
- Connects the pipeline to real dysarthric speech research (TORGO database, clinical literature)

---

## Pipeline

```
Raw Audio (.wav)
      │
      ▼  librosa
Feature Extraction
  ├── MFCCs (13 coefficients) — vocal tract spectral envelope
  ├── Zero Crossing Rate      — voicing / breathiness
  └── Spectral Centroid       — pitch brightness proxy
      │
      ▼  scikit-learn
Classification
  ├── SVM (RBF kernel, C=10, gamma='scale')
  └── Random Forest (100 trees)
      │
      ▼
Evaluation
  ├── 5-Fold Stratified Cross-Validation
  ├── Hold-out Test Accuracy (30% split)
  └── Confusion Matrix + Classification Report
```

---

## Severity Classes

| Class | Noise Level | Pitch Jitter | Shimmer | Clinical Interpretation |
|-------|------------|--------------|---------|------------------------|
| **Mild** | Low | Low | Low | Near-normal speech with minor imprecision |
| **Moderate** | Medium | Medium | Medium | Noticeable articulation errors |
| **Severe** | High | High | High | Highly unintelligible, effortful speech |

> **Jitter** = cycle-to-cycle variation in fundamental frequency (F0 instability)  
> **Shimmer** = cycle-to-cycle variation in amplitude  
> Both are standard clinical measures used in dysarthric speech pathology.

---

## Results

| Model | CV Accuracy (5-fold) | Notes |
|-------|---------------------|-------|
| SVM (RBF kernel) | reported in notebook | Sensitive to C/gamma |
| **Random Forest** | **higher, lower variance** | Robust to feature scale differences |

**Random Forest outperforms SVM** due to: non-linear decision boundaries in MFCC space, scale insensitivity, and variance reduction through ensemble averaging.

Both models struggle most at the **mild/moderate boundary** — clinically expected, as even trained speech-language pathologists find this boundary ambiguous. The **severe class is cleanly separated** by both models.

---

## Outputs Generated

| File | Description |
|------|-------------|
| `audio_samples/` | 45 synthetic `.wav` files (15 per class) |
| `waveforms.png` | Amplitude vs time for all 3 severity classes |
| `spectrograms.png` | Mel-spectrograms showing harmonic structure differences |
| `feature_distributions.png` | MFCC-1, ZCR, and Spectral Centroid distributions by class |
| `confusion_matrices.png` | SVM and Random Forest confusion matrices |
| `feature_importance.png` | Random Forest Gini feature importances |
| `cv_comparison.png` | Per-fold cross-validation accuracy comparison |

---

## Repository Structure

```
dysarthric-speech-classifier/
├── dysarthric_speech_classifier.ipynb   # Full pipeline — run this
├── requirements.txt                      # Python dependencies
├── .gitignore
└── README.md
```

> **Note:** `audio_samples/` and output `.png` files are generated when you run the notebook — they are not committed to the repo.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/dysarthric-speech-classifier.git
cd dysarthric-speech-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the notebook
```bash
jupyter notebook dysarthric_speech_classifier.ipynb
```

Run all cells in order. The notebook will:
- Generate 45 synthetic audio samples in `audio_samples/`
- Extract features and train both classifiers
- Save all output plots as `.png` files

---

## Using Real Data (TORGO Database)

The notebook uses **synthetic audio** that mimics dysarthric acoustic characteristics. To use real dysarthric speech:

1. Request access to the **TORGO database**: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
2. Replace the `audio_samples/` generation block (Section 1) with your TORGO `.wav` file paths
3. Ensure file naming follows `{severity_label}_{id}.wav` or update the label extraction logic in Section 3

---

## Connection to Research Literature

This pipeline directly mirrors the baseline approach in:

- **Rudzicz et al. (2012)** — TORGO database paper; MFCC-based classification of dysarthric speech
- **Chandrashekar & Hegde (2017)** — ZCR and spectral features for dysarthric severity
- Standard clinical NLP progression: **handcrafted features + SVM/RF → CNN on spectrograms → wav2vec 2.0 / HuBERT fine-tuning**

This notebook covers the first two stages — providing a concrete foundation for understanding what deep learning models subsequently improve upon.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `librosa` | Audio loading, MFCC, ZCR, spectral feature extraction |
| `numpy` | Numerical operations |
| `pandas` | Feature matrix management |
| `matplotlib` | Waveform, spectrogram, and result plots |
| `seaborn` | Confusion matrix heatmaps |
| `scikit-learn` | SVM, Random Forest, cross-validation, metrics |
| `scipy` | WAV file I/O |

---

## Author

**[Your Friend's Name]**  
[Degree and Institution]  
[Contact / GitHub / LinkedIn]
