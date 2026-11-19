"""Run data preparation, training, and evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from download_data import ensure_data  # noqa: E402
from train_eval import run  # noqa: E402


def main() -> None:
    print("=== 1) Prepare data ===")
    source = ensure_data()
    print(f"Data source: {source}")

    print("\n=== 2) Train and evaluate ===")
    metrics = run()

    print("\n=== Summary ===")
    print(f"F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
    print(f"Metrics: {ROOT / 'results' / 'metrics.json'}")


if __name__ == "__main__":
    main()
