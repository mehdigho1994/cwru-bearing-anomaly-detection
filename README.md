# Bearing Anomaly Detection on CWRU Vibration Data

Unsupervised anomaly detection for rolling-element bearings using the
[Case Western Reserve University (CWRU) Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
public vibration dataset.

The pipeline extracts time- and frequency-domain features from drive-end
accelerometer windows and trains an Isolation Forest on **normal** operation
only. Fault conditions (inner race, ball, outer race) are evaluated as anomalies.

## Method

1. Load CWRU `.mat` files (`97`, `105`, `118`, `130`) from `data/raw/`
2. Segment the drive-end (DE) channel into overlapping windows
3. Compute features: RMS, peak, crest factor, kurtosis, skewness, peak-to-peak,
   spectral centroid, and high-band energy ratio
4. Standardize features and fit Isolation Forest on normal windows
5. Score held-out normal and fault windows; report classification metrics

If official downloads are unavailable, the loader can synthesize CWRU-compatible
signals for pipeline testing. Reported results in `results/` were produced from
the real CWRU files listed below.

## Dataset files used

Place these files in `data/raw/` (not redistributed in this repository):

| File | Condition |
|------|-----------|
| `97.mat` | Normal baseline |
| `105.mat` | Inner-race fault (0.007") |
| `118.mat` | Ball fault (0.007") |
| `130.mat` | Outer-race fault (0.007") |

Source: CWRU Bearing Data Center, 12 kHz drive-end recordings.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python run_demo.py
```

## Results (CWRU subset)

| Metric | Value |
|--------|-------|
| Accuracy | 0.987 |
| Precision | 0.983 |
| Recall | 1.000 |
| F1 | 0.992 |
| ROC-AUC | 1.000 |

Confusion matrix (rows = true label, columns = predicted; order: normal, anomaly):

```
[[57,  3],
 [ 0, 177]]
```

Figures:

- `results/anomaly_score_hist.png` — anomaly score distributions
- `results/rms_vs_kurtosis.png` — RMS vs kurtosis feature space
- `results/metrics.json` — full numeric summary
- `results/classification_report.txt` — sklearn classification report

## Repository layout

```
run_demo.py              End-to-end entry point
src/download_data.py     Data loading / optional synthesis
src/features.py          Windowing and feature extraction
src/train_eval.py        Training, evaluation, and plots
data/raw/                Local CWRU .mat files (gitignored)
models/                  Saved model artifacts (gitignored)
results/                 Metrics and figures
```

## Requirements

See `requirements.txt` (NumPy, SciPy, scikit-learn, Matplotlib, joblib, requests).

## Acknowledgement

Bearing vibration data are provided by the Case Western Reserve University
Bearing Data Center. This repository redistributes only code and derived
evaluation outputs, not the original `.mat` recordings.
