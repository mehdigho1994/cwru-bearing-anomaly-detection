"""Train Isolation Forest on normal bearings and evaluate anomaly detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from download_data import ensure_data
from features import FEATURE_NAMES, build_dataset

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"


def run(seed: int = 42, contamination: float = 0.05) -> dict:
    source = ensure_data()
    X, y, labels = build_dataset()

    # Unsupervised setup: fit only on normal training windows
    normal_idx = np.where(y == 0)[0]
    anomaly_idx = np.where(y == 1)[0]

    normal_train, normal_test = train_test_split(
        normal_idx, test_size=0.3, random_state=seed
    )
    # Hold out part of anomalies for a balanced-ish test set
    _, anomaly_test = train_test_split(anomaly_idx, test_size=0.5, random_state=seed)

    train_idx = normal_train
    test_idx = np.concatenate([normal_test, anomaly_test])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    y_test = y[test_idx]

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train)

    # decision_function: higher = more normal; convert to anomaly score
    raw = -model.decision_function(X_test)
    pred = (model.predict(X_test) == -1).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    metrics = {
        "data_source": source,
        "n_train_normal": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_test_normal": int((y_test == 0).sum()),
        "n_test_anomaly": int((y_test == 1).sum()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_test, raw)),
        "average_precision": float(average_precision_score(y_test, raw)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "feature_names": FEATURE_NAMES,
        "model": "IsolationForest",
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "scaler": scaler}, MODELS / "isolation_forest.joblib")
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULTS / "classification_report.txt").write_text(
        classification_report(y_test, pred, target_names=["normal", "anomaly"]),
        encoding="utf-8",
    )

    _plot_score_hist(raw, y_test, RESULTS / "anomaly_score_hist.png")
    _plot_feature_scatter(X[test_idx], y_test, RESULTS / "rms_vs_kurtosis.png")

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics -> {RESULTS / 'metrics.json'}")
    print(f"Saved model   -> {MODELS / 'isolation_forest.joblib'}")
    return metrics


def _plot_score_hist(scores: np.ndarray, y_true: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(scores[y_true == 0], bins=30, alpha=0.7, label="normal", color="#2a6f97")
    ax.hist(scores[y_true == 1], bins=30, alpha=0.7, label="anomaly", color="#c1121f")
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_ylabel("Count")
    ax.set_title("Isolation Forest scores on held-out windows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_feature_scatter(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    # rms=0, kurtosis=3
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(X[y == 0, 0], X[y == 0, 3], s=18, alpha=0.65, label="normal", c="#2a6f97")
    ax.scatter(X[y == 1, 0], X[y == 1, 3], s=18, alpha=0.65, label="anomaly", c="#c1121f")
    ax.set_xlabel("RMS")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Feature space: RMS vs Kurtosis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contamination", type=float, default=0.05)
    args = parser.parse_args()
    run(seed=args.seed, contamination=args.contamination)


if __name__ == "__main__":
    main()
