"""Figures for the report.

Two plots, both answering the project's actual question:
  1. grouped F1 bars, A vs B, subsets ordered by OOD severity;
  2. the delta with its 95% CI against severity — the trend RetrievalPRM's case rested on.

Matplotlib defaults only, no seaborn, no custom colour cycle: the point is the numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: these run on a cluster with no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _plottable(result: dict) -> list[dict]:
    """Only subsets with a defined F1 can be drawn; the rest have nothing to plot."""
    return [r for r in result["per_subset"] if r.get("well_defined", True)]


def plot_f1_comparison(result: dict, out_path: str | Path) -> Path:
    rows = _plottable(result)
    if not rows:
        raise ValueError(
            "No subset has a defined F1, so there is nothing to plot. This happens when "
            "every subset contains only error cases or only clean cases — typical of a "
            "small --limit pilot. Run on more data."
        )
    labels = [r["subset"] for r in rows]
    a_values = [100 * r["a"]["f1"] for r in rows]
    b_values = [100 * r["b"]["f1"] for r in rows]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width / 2, a_values, width, label="A — baseline")
    ax.bar(x + width / 2, b_values, width, label="B — + retrieval")

    for xi, (a, b) in enumerate(zip(a_values, b_values)):
        ax.text(xi - width / 2, a + 0.6, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(xi + width / 2, b + 0.6, f"{b:.1f}", ha="center", fontsize=8)

    ax.set_xticks(x, labels)
    ax.set_ylabel("ProcessBench F1")
    ax.set_title("Step-level error detection, by subset (ordered by OOD severity)")
    ax.set_ylim(0, max(a_values + b_values + [1]) * 1.18)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_ood_trend(result: dict, out_path: str | Path) -> Path:
    rows = _plottable(result)
    if not rows:
        raise ValueError(
            "No subset has a defined delta, so the trend cannot be plotted. See "
            "plot_f1_comparison for why."
        )
    labels = [r["subset"] for r in rows]
    deltas = [100 * r["delta_f1"] for r in rows]
    lows = [100 * r["delta_ci95"][0] for r in rows]
    highs = [100 * r["delta_ci95"][1] for r in rows]

    x = np.arange(len(labels))
    lower_error = [d - lo for d, lo in zip(deltas, lows)]
    upper_error = [hi - d for d, hi in zip(deltas, highs)]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.errorbar(x, deltas, yerr=[lower_error, upper_error], fmt="o-", capsize=4)
    ax.axhline(0.0, linestyle="--", linewidth=1, color="grey")

    trend = result.get("ood_trend", {})
    if trend.get("testable"):
        fit = np.poly1d([100 * trend["slope_per_ood_rank"], 100 * trend["intercept"]])
        ax.plot(x, fit(x), linestyle=":", label=f"slope {100 * trend['slope_per_ood_rank']:+.2f}/rank")
        ax.legend()

    ax.set_xticks(x, labels)
    ax.set_ylabel("Δ F1  (B − A)")
    ax.set_xlabel("increasing out-of-distribution severity →")
    ax.set_title("Does the retrieval gain grow with difficulty?")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
