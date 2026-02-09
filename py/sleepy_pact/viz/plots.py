"""Plotting functions for training metrics."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_pass_rate(
    metrics: list[dict],
    output_path: Path | None = None,
    title: str = "Pass Rate by Cycle",
) -> None:
    """Plot pass rate over training cycles.

    Args:
        metrics: List of cycle metrics with 'cycle' and 'pass_rate'.
        output_path: Path to save the plot.
        title: Plot title.
    """
    cycles = [m["cycle"] for m in metrics]
    pass_rates = [m["pass_rate"] * 100 for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(cycles, pass_rates, "b-o", linewidth=2, markersize=8)
    plt.fill_between(cycles, pass_rates, alpha=0.3)

    plt.xlabel("Training Cycle", fontsize=12)
    plt.ylabel("Pass Rate (%)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)

    # Add value labels
    for i, (c, p) in enumerate(zip(cycles, pass_rates)):
        plt.annotate(
            f"{p:.1f}%",
            (c, p),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_steps_to_green(
    metrics: list[dict],
    output_path: Path | None = None,
    title: str = "Median Steps to Green by Cycle",
) -> None:
    """Plot median steps to success over training cycles.

    Args:
        metrics: List of cycle metrics with 'cycle' and 'median_steps_to_green'.
        output_path: Path to save the plot.
        title: Plot title.
    """
    cycles = [m["cycle"] for m in metrics]
    steps = [m["median_steps_to_green"] for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(cycles, steps, "g-o", linewidth=2, markersize=8)
    plt.fill_between(cycles, steps, alpha=0.3, color="green")

    plt.xlabel("Training Cycle", fontsize=12)
    plt.ylabel("Median Steps to Green", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(steps) * 1.2 if steps else 5)

    # Add value labels
    for c, s in zip(cycles, steps):
        plt.annotate(
            f"{s:.1f}",
            (c, s),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_repeat_error_rate(
    metrics: list[dict],
    output_path: Path | None = None,
    title: str = "Repeat Error Rate by Cycle",
) -> None:
    """Plot repeat error rate over training cycles.

    Args:
        metrics: List of cycle metrics with 'cycle' and 'repeat_error_rate'.
        output_path: Path to save the plot.
        title: Plot title.
    """
    cycles = [m["cycle"] for m in metrics]
    rates = [m.get("repeat_error_rate", 0) * 100 for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(cycles, rates, "r-o", linewidth=2, markersize=8)
    plt.fill_between(cycles, rates, alpha=0.3, color="red")

    # Target line
    plt.axhline(y=20, color="orange", linestyle="--", label="Target (<20%)")

    plt.xlabel("Training Cycle", fontsize=12)
    plt.ylabel("Repeat Error Rate (%)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 105)

    # Add value labels
    for c, r in zip(cycles, rates):
        plt.annotate(
            f"{r:.1f}%",
            (c, r),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_error_distribution(
    error_counts: dict[str, int],
    output_path: Path | None = None,
    title: str = "Error Distribution",
    top_n: int = 10,
) -> None:
    """Plot distribution of error signatures.

    Args:
        error_counts: Dictionary mapping error signature to count.
        output_path: Path to save the plot.
        title: Plot title.
        top_n: Number of top errors to show.
    """
    # Sort by count
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_errors = sorted_errors[:top_n]

    if not sorted_errors:
        return

    labels = [e[0][:30] + "..." if len(e[0]) > 30 else e[0] for e in sorted_errors]
    counts = [e[1] for e in sorted_errors]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(labels)), counts, color="steelblue")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Count", fontsize=12)
    plt.title(title, fontsize=14)
    plt.gca().invert_yaxis()

    # Add count labels
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
        )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_training_loss(
    losses: list[float],
    output_path: Path | None = None,
    title: str = "Training Loss",
) -> None:
    """Plot training loss curve.

    Args:
        losses: List of loss values per step.
        output_path: Path to save the plot.
        title: Plot title.
    """
    steps = list(range(1, len(losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, "b-", linewidth=1, alpha=0.7)

    # Smoothed line
    if len(losses) > 10:
        window = min(len(losses) // 10, 50)
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        smooth_steps = steps[window - 1 :]
        plt.plot(smooth_steps, smoothed, "r-", linewidth=2, label="Smoothed")

    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    if len(losses) > 10:
        plt.legend()

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_comparison(
    before: dict[str, float],
    after: dict[str, float],
    output_path: Path | None = None,
    title: str = "Before vs After Training",
) -> None:
    """Plot before/after comparison.

    Args:
        before: Metrics before training.
        after: Metrics after training.
        output_path: Path to save the plot.
        title: Plot title.
    """
    metrics = ["pass_rate", "median_steps", "repeat_error"]
    labels = ["Pass Rate (%)", "Median Steps", "Repeat Error (%)"]

    before_vals = [
        before.get("pass_rate", 0) * 100,
        before.get("median_steps_to_green", 0),
        before.get("repeat_error_rate", 0) * 100,
    ]
    after_vals = [
        after.get("pass_rate", 0) * 100,
        after.get("median_steps_to_green", 0),
        after.get("repeat_error_rate", 0) * 100,
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, before_vals, width, label="Before", color="gray")
    bars2 = ax.bar(x + width / 2, after_vals, width, label="After", color="steelblue")

    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def generate_all_plots(
    metrics_path: Path,
    output_dir: Path,
) -> None:
    """Generate all standard plots from metrics file.

    Args:
        metrics_path: Path to metrics JSONL file.
        output_dir: Directory to save plots.
    """
    # Load metrics
    metrics = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))

    if not metrics:
        print("No metrics to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_pass_rate(metrics, output_dir / "pass_rate_by_cycle.png")
    plot_steps_to_green(metrics, output_dir / "steps_to_green_by_cycle.png")
    plot_repeat_error_rate(metrics, output_dir / "repeat_error_rate_by_cycle.png")

    # Error distribution from latest cycle
    if "error_signatures" in metrics[-1]:
        plot_error_distribution(
            metrics[-1]["error_signatures"],
            output_dir / "error_distribution.png",
        )

    # Before/after comparison
    if len(metrics) >= 2:
        plot_comparison(
            metrics[0],
            metrics[-1],
            output_dir / "before_after_comparison.png",
        )

    print(f"Plots saved to {output_dir}")
