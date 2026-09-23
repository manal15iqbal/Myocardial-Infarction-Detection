# Myocardial Infarction Detection using ECG Feature Engineering & Ensemble Learning

An interpretable machine learning system that detects Myocardial Infarction (MI) from ECG recordings using manually engineered clinical features combined with ensemble learning — achieving **92.04% accuracy** and a **0.9713 ROC-AUC**, without relying on black-box deep learning.

---

## Overview

Early detection of Myocardial Infarction (heart attack) from ECG signals is critical for reducing mortality, but manual interpretation is time-consuming and variable across clinicians. Most automated approaches rely on complex deep learning models that are computationally expensive and hard to interpret.

This project takes a different approach: **clinically-grounded feature engineering + classical ML ensembling**, prioritizing interpretability and efficiency while matching or exceeding typical deep learning performance on this task.

---

## Pipeline Architecture

```
ECG Input → Signal Preprocessing → Filtering → R-Peak Detection
    → Feature Extraction (HR, QRS duration, ST-segment features, ...)
    → Feature Dataset → [Random Forest | SVM | XGBoost]
    → Weighted Ensemble Voting → Prediction (MI / Normal)
    → Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
```

---

## Dataset

- **Source:** [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) — a large, publicly available clinical ECG dataset
- ~6,000 records initially selected from the **MI** and **NORM (Normal)** classes
- **5,682 valid records** retained after cleaning and validation
- Leads used: **I, II, V1, V2** (4 of the standard 12 leads)
- Sampling rate: 100 Hz

---

## Feature Engineering

Rather than feeding raw signals into a deep network, ~80 clinically interpretable features are extracted per record across the selected leads, including:

| Category | Features |
|---|---|
| **ST-segment** | ST elevation (mean/max), ST depression (mean/max), ST slope |
| **T-wave** | T-wave mean/std amplitude, negative T-wave count |
| **QRS complex** | QRS duration (mean/max) |
| **R-wave** | R-wave amplitude (mean/std) |
| **Heart rate / RR interval** | Heart rate, RR variability, RR coefficient of variation |
| **Beat morphology** | Beat-to-beat amplitude mean & variability |
| **Signal statistics** | Signal mean, standard deviation, skewness |

Feature extraction uses **NeuroKit2** for ECG processing (R-peak detection, wave delineation via the DWT method) plus custom logic for ST-segment and beat-level statistics.

The top predictive features (via SHAP analysis) include RR variability, QRS duration, ST slope, R-wave amplitude, and T-wave characteristics — consistent with established clinical MI indicators.

---

## Methodology

1. **Signal preprocessing & cleaning** — invalid/incomplete/noisy records removed
2. **Feature extraction** — ~80 features across 4 leads via NeuroKit2 + custom logic
3. **Data cleaning** — inf/NaN handling, median imputation, outlier column removal
4. **Train/test split** — 95:5 stratified split (done *before* balancing to avoid leakage)
5. **Class balancing** — SMOTE applied only to the training set
6. **Feature scaling** — StandardScaler
7. **Feature selection** — `SelectKBest` with mutual information (top 80 features)
8. **Model training:**
   - **Random Forest** (1000 trees, class-balanced)
   - **SVM (RBF kernel)** — hyperparameters tuned via `GridSearchCV`
   - **XGBoost** (1000 estimators, tuned regularization)
9. **Weighted ensemble** — soft-voting on predicted probabilities:
   - RF: 0.30, SVM: 0.20, XGBoost: 0.50
   - Decision threshold optimized on validation data (0.49)
10. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix
11. **Interpretability** — SHAP analysis to explain feature contributions

---

## Results

| Model | Accuracy |
|---|---|
| Random Forest | 91.00% |
| SVM | 87.20% |
| XGBoost | 91.00% |
| **Weighted Ensemble** | **92.04%** |

**Final ensemble metrics:**
- Precision: 92.08%
- Recall: 92.04%
- F1-Score: 92.03%
- ROC-AUC: **0.9713**

**Confusion Matrix:**

|  | Predicted MI | Predicted Normal |
|---|---|---|
| **Actual MI** | 125 (TN) | 14 (FP) |
| **Actual Normal** | 9 (FN) | 141 (TP) |

The low false-negative count is particularly relevant clinically, since missed MI diagnoses carry serious risk.

---

## Repository Structure

```
.
├── Final_MI_Detection_Improved_92.ipynb   # Main notebook: full signal-based pipeline
├── final_ensemble_model.pkl               # Trained weighted ensemble (RF + SVM + XGBoost)
├── scaler.pkl                             # Fitted StandardScaler
├── selector.pkl                           # Fitted SelectKBest feature selector
├── label_encoder.pkl                      # Label encoder (MI / NORM)               
├── features_mi_only.pkl                   # Extracted feature matrix (X, y)
└── README.md
```

## Tech Stack

- **Language:** Python
- **Signal processing:** NeuroKit2, PyWavelets, WFDB
- **ML:** scikit-learn (Random Forest, SVM, SelectKBest, GridSearchCV), XGBoost, imbalanced-learn (SMOTE)
- **Interpretability:** SHAP
- **Visualization:** Matplotlib
- **Data:** PTB-XL (PhysioNet)
- **Serialization:** Pickle

---

## Getting Started

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install neurokit2 wfdb scikit-learn xgboost shap pandas numpy matplotlib seaborn PyWavelets imbalanced-learn
```

Then open `Final_MI_Detection_Improved_92.ipynb` in Jupyter/Colab and run cells sequentially. The notebook downloads the PTB-XL dataset directly from PhysioNet, so an internet connection is required on first run.

### Using the saved model

```python
import pickle

with open("final_ensemble_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)

# X_new: extracted feature dataframe for a new record
X_scaled = scaler.transform(X_new)
X_selected = selector.transform(X_scaled)
prediction = model.predict(X_selected)        # 0 = Normal, 1 = MI
probability = model.predict_proba(X_selected)  # MI probability
```

## Key Findings

- Manually engineered, clinically meaningful ECG features can rival deep learning approaches for MI detection while remaining fully interpretable.
- Ensemble learning (RF + SVM + XGBoost) consistently outperformed any single classifier.
- SHAP analysis confirmed that the model's decisions align with known clinical MI markers (ST-segment changes, QRS widening, T-wave inversion, RR variability).
  
## Limitations & Future Work

- Trained on a binary MI vs. Normal task; does not distinguish MI subtypes or other cardiac abnormalities.
- Small test set for the signal-based model (95:5 split → ~289 test records); results should be validated on a larger held-out set.
- Feature extraction depends on reliable R-peak/wave detection, which can degrade on noisy signals.
- Potential extensions: multi-class classification, deployment as a web app, external dataset validation, real-time ECG stream support.

