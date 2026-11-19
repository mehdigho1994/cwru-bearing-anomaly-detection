# Bearing Anomaly Detection (CWRU-style)

Small, runnable proof that you can take public bearing vibration data, extract simple features, and detect faults with a classic unsupervised model.

## What this does

1. Loads a **CWRU** drive-end subset (`97/105/118/130.mat`) when download works  
2. Otherwise synthesizes **CWRU-compatible** vibration (SKF 6205 fault orders) so the pipeline always runs  
3. Windows the DE channel and extracts RMS, kurtosis, crest factor, spectral features  
4. Fits **Isolation Forest** on normal windows only  
5. Reports F1 / ROC-AUC on held-out normal + fault windows  
6. Writes plots, metrics, and an **email one-liner** you can paste into outreach

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
python run_demo.py
```

Outputs:

| Path | Content |
|------|---------|
| `results/metrics.json` | Accuracy, F1, ROC-AUC, confusion matrix |
| `results/anomaly_score_hist.png` | Score separation |
| `results/rms_vs_kurtosis.png` | Feature scatter |
| `models/isolation_forest.joblib` | Trained model + scaler |
| `EMAIL_BLURB.md` | Ready-to-paste outreach sentence |

## Why this is a good email proof

Anyone can rewrite a cold email. Fewer people can point to:

> trained model + public protocol + numbers + repo link

That moves the conversation from “I have an idea” to “here is a working demo.”

## Data notes

- Official CWRU Bearing Data Center: [engineering.case.edu/bearingdatacenter](https://engineering.case.edu/bearingdatacenter)
- `.mat` files are **not redistributed** in this repo (license / hosting policy)
- `data/raw/SOURCE.txt` records whether the run used real CWRU or synthetic fallback

## Project layout

```
run_demo.py          # one command
src/download_data.py # CWRU download or synthetic
src/features.py      # windowing + features
src/train_eval.py    # Isolation Forest + plots
results/             # metrics + figures
EMAIL_BLURB.md       # outreach sentence
```

## Next upgrades (optional)

- Swap Isolation Forest for One-Class SVM or autoencoder
- Add Paderborn University bearing dataset
- Stream windows for near-real-time scoring
