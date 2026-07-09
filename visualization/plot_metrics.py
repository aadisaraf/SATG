"""Plotting utilities for training metrics and per-class analysis.

Functions
---------
plot_coverage_over_time
    Line chart of trust coverage ratio over training iterations.
plot_coverage_by_class
    Horizontal bar chart of per-class trust coverage, sorted ascending.
"""

from typing import Dict

import matplotlib  # type: ignore[import-untyped]

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np


def plot_coverage_over_time(csv_path: str, save_path: str) -> None:
    """Plot trust coverage ratio over training iterations.

    Loads ``metrics.csv`` (written by the trainer) which must contain at least
    the columns ``iteration`` and ``trust_coverage_ratio``.

    Args:
        csv_path: Path to the ``metrics.csv`` file.
        save_path: Path where the PNG plot will be saved.
    """
    # Use numpy to load CSV (avoids pandas dependency)
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")

    iterations = data["iteration"]
    coverage = data["trust_coverage_ratio"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, coverage, linewidth=1.5)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Trust Coverage Ratio")
    ax.set_title("Trust Coverage Ratio Over Training")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_coverage_by_class(per_class_dict: Dict[str, float], save_path: str) -> None:
    """Horizontal bar chart of per-class trust coverage, sorted ascending.

    Args:
        per_class_dict: Mapping of class name to its trust coverage ratio
            (e.g. ``{"road": 0.95, "sidewalk": 0.82, ...}``).
        save_path: Path where the PNG plot will be saved.
    """
    sorted_items = sorted(per_class_dict.items(), key=lambda x: x[1])  # ascending
    classes = [item[0] for item in sorted_items]
    coverages = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(classes, coverages, height=0.7)
    ax.set_xlabel("Trust Coverage Ratio")
    ax.set_title("Per-Class Trust Coverage (Sorted Ascending)")
    ax.set_xlim(0.0, 1.05)
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels at the end of each bar
    for bar, val in zip(bars, coverages):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.3f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
