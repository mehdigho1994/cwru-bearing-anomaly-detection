"""Download a small CWRU subset, or synthesize CWRU-compatible signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import requests
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# Drive-end 12 kHz, 0 HP load — classic CWRU starter subset
FILES = {
    "97.mat": {
        "label": "normal",
        "urls": [
            "https://engineering.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/97.mat",
            "http://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/97.mat",
        ],
    },
    "105.mat": {
        "label": "inner_race",
        "urls": [
            "https://engineering.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/105.mat",
            "http://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/105.mat",
        ],
    },
    "118.mat": {
        "label": "ball",
        "urls": [
            "https://engineering.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/118.mat",
            "http://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/118.mat",
        ],
    },
    "130.mat": {
        "label": "outer_race",
        "urls": [
            "https://engineering.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/130.mat",
            "http://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/130.mat",
        ],
    },
}

FS = 12000  # Hz
RPM = 1797
SECONDS = 10


def _download(url: str, dest: Path, timeout: int = 30) -> bool:
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        if len(response.content) < 1000:
            return False
        dest.write_bytes(response.content)
        return True
    except requests.RequestException:
        return False


def _fault_impulses(n: int, fault_hz: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / FS
    period = 1.0 / fault_hz
    signal = np.zeros(n)
    centers = np.arange(0, SECONDS, period)
    for c in centers:
        idx = int(c * FS)
        width = int(0.002 * FS)
        if idx + width >= n:
            break
        envelope = amplitude * np.exp(-np.linspace(0, 8, width))
        carrier = np.sin(2 * np.pi * 3000 * np.arange(width) / FS)
        signal[idx : idx + width] += envelope * carrier * (1 + 0.1 * rng.standard_normal(width))
    # add shaft harmonics
    shaft_hz = RPM / 60.0
    signal += 0.15 * np.sin(2 * np.pi * shaft_hz * t)
    signal += 0.05 * np.sin(2 * np.pi * 2 * shaft_hz * t)
    signal += 0.08 * rng.standard_normal(n)
    return signal.astype(np.float64)


def synthesize_cwru_compatible(dest_dir: Path) -> str:
    """Generate vibration .mat files shaped like CWRU DE channel arrays."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = FS * SECONDS
    shaft_hz = RPM / 60.0

    # Approximate SKF 6205-2RS geometry fault frequencies (orders × shaft)
    specs = {
        "97.mat": ("X097_DE_time", None, 0.08),  # normal: noise only + shaft
        "105.mat": ("X105_DE_time", 5.4152 * shaft_hz, 1.2),  # BPFI
        "118.mat": ("X118_DE_time", 4.7135 * shaft_hz, 0.9),  # BSF
        "130.mat": ("X130_DE_time", 3.5848 * shaft_hz, 1.4),  # BPFO
    }

    for filename, (key, fault_hz, amp) in specs.items():
        t = np.arange(n) / FS
        if fault_hz is None:
            signal = 0.12 * np.sin(2 * np.pi * shaft_hz * t)
            signal += 0.04 * np.sin(2 * np.pi * 2 * shaft_hz * t)
            signal += 0.07 * rng.standard_normal(n)
        else:
            signal = _fault_impulses(n, fault_hz, amp, rng)
        savemat(dest_dir / filename, {key: signal.reshape(-1, 1)})

    meta = dest_dir / "SOURCE.txt"
    meta.write_text(
        "synthetic_cwru_compatible\n"
        "Reason: official CWRU .mat download unavailable from this environment.\n"
        "Signals mimic DE-channel vibration with SKF 6205 fault orders.\n",
        encoding="utf-8",
    )
    return "synthetic_cwru_compatible"


def _read_source() -> str | None:
    meta = RAW_DIR / "SOURCE.txt"
    if not meta.exists():
        return None
    return meta.read_text(encoding="utf-8").strip().splitlines()[0]


def _all_mats_present() -> bool:
    return all((RAW_DIR / name).exists() and (RAW_DIR / name).stat().st_size > 1000 for name in FILES)


def ensure_data(force_synthetic: bool = False) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if force_synthetic:
        return synthesize_cwru_compatible(RAW_DIR)

    existing = _read_source()
    if existing and _all_mats_present():
        return existing

    for filename, meta in FILES.items():
        dest = RAW_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        ok = False
        for url in meta["urls"]:
            print(f"Downloading {filename} from {url} ...")
            if _download(url, dest):
                print(f"  OK -> {dest}")
                ok = True
                break
            print("  failed")
        if not ok:
            print("CWRU download failed; falling back to synthetic signals.")
            return synthesize_cwru_compatible(RAW_DIR)

    (RAW_DIR / "SOURCE.txt").write_text("cwru\n", encoding="utf-8")
    return "cwru"


def load_de_signal(mat_path: Path) -> np.ndarray:
    mat = loadmat(mat_path)
    # Prefer drive-end channel keys like X097_DE_time
    candidates = [k for k in mat if k.endswith("_DE_time") or "_DE_" in k]
    if not candidates:
        candidates = [k for k in mat if not k.startswith("__") and hasattr(mat[k], "shape")]
    if not candidates:
        raise ValueError(f"No usable array in {mat_path}")
    key = sorted(candidates)[0]
    arr = np.asarray(mat[key]).squeeze().astype(np.float64)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CWRU or synthetic bearing data")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    args = parser.parse_args()
    source = ensure_data(force_synthetic=args.synthetic)
    print(f"Data ready in {RAW_DIR} (source={source})")


if __name__ == "__main__":
    main()
