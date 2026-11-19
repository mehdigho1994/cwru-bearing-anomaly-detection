"""Window vibration signals and extract simple time/frequency features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew

from download_data import FILES, RAW_DIR, load_de_signal

FEATURE_NAMES = [
    "rms",
    "peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "peak_to_peak",
    "spectral_centroid",
    "band_energy_ratio",
]


def window_signal(x: np.ndarray, window: int = 2048, step: int = 1024) -> np.ndarray:
    if len(x) < window:
        raise ValueError("Signal shorter than window")
    starts = range(0, len(x) - window + 1, step)
    return np.stack([x[s : s + window] for s in starts])


def extract_features(windows: np.ndarray, fs: int = 12000) -> np.ndarray:
    feats = []
    for w in windows:
        rms = float(np.sqrt(np.mean(w**2)))
        peak = float(np.max(np.abs(w)))
        crest = peak / (rms + 1e-12)
        kurt = float(kurtosis(w, fisher=False))
        sk = float(skew(w))
        p2p = float(np.max(w) - np.min(w))

        spectrum = np.abs(np.fft.rfft(w * np.hanning(len(w))))
        freqs = np.fft.rfftfreq(len(w), d=1.0 / fs)
        power = spectrum**2
        centroid = float(np.sum(freqs * power) / (np.sum(power) + 1e-12))
        # high-band share (rough fault impulse energy proxy)
        high = power[freqs >= 1000].sum()
        band_ratio = float(high / (power.sum() + 1e-12))

        feats.append([rms, peak, crest, kurt, sk, p2p, centroid, band_ratio])
    return np.asarray(feats, dtype=np.float64)


def build_dataset(
    window: int = 2048,
    step: int = 1024,
    max_windows_per_file: int | None = 200,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    labels: list[str] = []

    for filename, meta in FILES.items():
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run download_data.py first")
        signal = load_de_signal(path)
        windows = window_signal(signal, window=window, step=step)
        if max_windows_per_file is not None:
            windows = windows[:max_windows_per_file]
        feats = extract_features(windows)
        label = meta["label"]
        is_anomaly = 0 if label == "normal" else 1
        X_list.append(feats)
        y_list.append(np.full(len(feats), is_anomaly, dtype=np.int32))
        labels.extend([label] * len(feats))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, labels


if __name__ == "__main__":
    from download_data import ensure_data

    ensure_data()
    X, y, labels = build_dataset()
    print(f"X={X.shape}, anomalies={y.sum()}, normals={(y == 0).sum()}")
    print("label counts:", {k: labels.count(k) for k in sorted(set(labels))})
