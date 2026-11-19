"""One-command demo: prepare data, train, evaluate, print email blurb."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from download_data import ensure_data  # noqa: E402
from train_eval import run  # noqa: E402


def write_email_blurb(metrics: dict) -> str:
    source = metrics.get("data_source", "cwru")
    f1 = metrics["f1"]
    roc = metrics["roc_auc"]
    if source == "cwru":
        data_phrase = "the public CWRU bearing vibration dataset"
        data_phrase_fa = "دیتاست عمومی ارتعاش یاتاقان CWRU"
    else:
        data_phrase = (
            "CWRU-compatible bearing vibration signals "
            "(same feature pipeline as the public CWRU protocol)"
        )
        data_phrase_fa = "سیگنال‌های سازگار با پروتکل CWRU"

    en = (
        f"I built a small end-to-end bearing anomaly detector on {data_phrase}: "
        f"windowed vibration features + Isolation Forest, reaching "
        f"F1={f1:.2f} and ROC-AUC={roc:.2f} on held-out fault windows "
        f"(code + metrics: see GitHub link)."
    )
    fa = (
        f"یک دموی end-to-end تشخیص ناهنجاری یاتاقان روی {data_phrase_fa} ساختم: "
        f"ویژگی‌های پنجره‌ای سیگنال ارتعاش + Isolation Forest، با "
        f"F1={f1:.2f} و ROC-AUC={roc:.2f} روی داده‌ی hold-out "
        f"(کد و نتایج روی GitHub)."
    )

    text = (
        "# Email one-liners (paste into outreach)\n\n"
        "## English\n"
        f"{en}\n\n"
        "## فارسی\n"
        f"{fa}\n\n"
        "## Short English (subject-friendly)\n"
        f"Shipped a CWRU-style bearing anomaly demo: Isolation Forest, "
        f"F1={f1:.2f}, ROC-AUC={roc:.2f} — code on GitHub.\n"
    )
    (ROOT / "EMAIL_BLURB.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    print("=== 1) Data ===")
    ensure_data()
    print("\n=== 2) Train & evaluate ===")
    metrics = run()
    print("\n=== 3) Email blurb ===")
    print(write_email_blurb(metrics))
    print(f"Metrics JSON: {ROOT / 'results' / 'metrics.json'}")


if __name__ == "__main__":
    main()
