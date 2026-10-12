"""Generate report figures from real experiment JSONs (no fabrication).

Reads:
  experiments/EXP-010-ENSEMBLE/production_ensemble_results.json

Writes PNGs to docs/evidence/ for embedding into the project report:
  fig_leaderboard.png    — test accuracy + balanced accuracy per model
  fig_calibration.png    — ECE before/after temperature scaling (champion ensemble)

Run: python scripts/generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "EXP-010-ENSEMBLE" / "production_ensemble_results.json"
OUT = ROOT / "docs" / "evidence"

CYAN = "#0e7490"
AMBER = "#b45309"
SLATE = "#334155"


def leaderboard_figure(rows: list[dict], out: Path) -> None:
    rows = sorted(rows, key=lambda r: r.get("test_acc") or 0, reverse=True)
    names = [r["name"] for r in rows]
    acc = [r.get("test_acc") or 0 for r in rows]
    bacc = [r.get("test_bacc") or 0 for r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = range(len(names))
    ax.barh([i + 0.2 for i in y], acc, height=0.38, color=CYAN, label="Test accuracy")
    ax.barh([i - 0.2 for i in y], bacc, height=0.38, color=AMBER, label="Balanced accuracy")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent (RAF-DB official test split, n=3068)")
    ax.set_title("Production leaderboard — real experiment metrics", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    for i, (a, b) in enumerate(zip(acc, bacc)):
        ax.text(a + 0.6, i + 0.2, f"{a:.2f}", va="center", fontsize=7, color=SLATE)
        ax.text(b + 0.6, i - 0.2, f"{b:.2f}", va="center", fontsize=7, color=SLATE)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def calibration_figure(cal: dict, out: Path) -> None:
    raw = cal.get("raw_ece", 0.0)
    scaled = cal.get("cal_ece", 0.0)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bars = ax.bar(
        ["Raw", f"T-scaled\n(T={cal.get('temperature', 0):.3f})"],
        [raw, scaled],
        color=[SLATE, CYAN],
        width=0.55,
    )
    for bar, v in zip(bars, [raw, scaled]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.4f}",
                ha="center", fontsize=8)
    ax.set_ylabel("Expected calibration error")
    ax.set_title(
        f"Champion ensemble calibration ({cal.get('target', '')})", fontsize=9
    )
    ax.set_ylim(0, max(raw, scaled) * 1.35 + 0.01)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS.read_text())
    leaderboard_figure(data["leaderboard"], OUT / "fig_leaderboard.png")
    calibration_figure(data["calibration"], OUT / "fig_calibration.png")
    print(f"Figures written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
