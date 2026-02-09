#!/usr/bin/env python3
"""
Generate plots comparing model performance across training cycles.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --output docs/results.png
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: uv pip install matplotlib")


def load_eval_results(data_dir: Path) -> list[dict]:
    """Load all evaluation results from the data directory."""
    results = []
    eval_dir = data_dir / "eval_runs"

    if not eval_dir.exists():
        print(f"No eval_runs directory found at {eval_dir}")
        return results

    for result_file in sorted(eval_dir.glob("*.json")):
        try:
            with open(result_file) as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")

    return results


def load_episodes_summary(data_dir: Path) -> dict:
    """Load episode statistics from episodes directory."""
    episodes_dir = data_dir / "episodes"
    stats = {"total": 0, "passed": 0, "failed": 0, "by_family": {}}

    if not episodes_dir.exists():
        return stats

    for jsonl_file in episodes_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    ep = json.loads(line)
                    stats["total"] += 1
                    if ep.get("passed"):
                        stats["passed"] += 1
                    else:
                        stats["failed"] += 1

                    family = ep.get("error_family", "unknown")
                    if family not in stats["by_family"]:
                        stats["by_family"][family] = {"passed": 0, "failed": 0}
                    if ep.get("passed"):
                        stats["by_family"][family]["passed"] += 1
                    else:
                        stats["by_family"][family]["failed"] += 1

    return stats


def generate_text_report(results: list[dict], episodes: dict, output_path: Path = None):
    """Generate a text-based results report."""
    lines = []
    lines.append("=" * 60)
    lines.append("SLEEPY-CODER EVALUATION RESULTS")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 60)
    lines.append("")

    if not results:
        lines.append("No evaluation results found.")
        lines.append("")
        lines.append("Run evaluations with:")
        lines.append("  ./rust/target/release/sleepy-coder eval --cycle 0")
        lines.append("  ./rust/target/release/sleepy-coder eval --cycle 1 --model sleepy-coder-v2")
    else:
        # Sort by cycle
        results = sorted(results, key=lambda x: x.get("cycle", 0))

        lines.append("CYCLE COMPARISON")
        lines.append("-" * 60)
        lines.append(f"{'Cycle':<8} {'Model':<30} {'Pass Rate':<12} {'Passed':<8} {'Total':<8}")
        lines.append("-" * 60)

        for r in results:
            cycle = r.get("cycle", 0)
            model = r.get("model", "unknown")[:28]
            passed = r.get("passed", 0)
            total = r.get("total", 0)
            rate = (passed / total * 100) if total > 0 else 0
            lines.append(f"{cycle:<8} {model:<30} {rate:>6.1f}%     {passed:<8} {total:<8}")

        lines.append("-" * 60)
        lines.append("")

        # Calculate improvement
        if len(results) >= 2:
            baseline = results[0]
            latest = results[-1]
            baseline_rate = baseline.get("passed", 0) / max(baseline.get("total", 1), 1) * 100
            latest_rate = latest.get("passed", 0) / max(latest.get("total", 1), 1) * 100
            improvement = latest_rate - baseline_rate

            lines.append(f"IMPROVEMENT: {improvement:+.1f}% ({baseline_rate:.1f}% -> {latest_rate:.1f}%)")
            lines.append("")

    # Episode statistics
    if episodes["total"] > 0:
        lines.append("")
        lines.append("TRAINING EPISODES")
        lines.append("-" * 60)
        lines.append(f"Total episodes: {episodes['total']}")
        lines.append(f"  Passed: {episodes['passed']}")
        lines.append(f"  Failed: {episodes['failed']} (used for training)")
        lines.append("")

        if episodes["by_family"]:
            lines.append("BY ERROR FAMILY:")
            for family, counts in sorted(episodes["by_family"].items()):
                total = counts["passed"] + counts["failed"]
                rate = counts["passed"] / total * 100 if total > 0 else 0
                lines.append(f"  {family}: {counts['passed']}/{total} ({rate:.0f}%)")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")

    return report


def generate_plots(results: list[dict], output_path: Path):
    """Generate matplotlib plots comparing cycles."""
    if not HAS_MATPLOTLIB:
        print("Cannot generate plots: matplotlib not installed")
        return

    if not results:
        print("No results to plot")
        return

    # Sort by cycle
    results = sorted(results, key=lambda x: x.get("cycle", 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sleepy-Coder: Continual Learning Results", fontsize=14, fontweight="bold")

    # Plot 1: Pass rate by cycle
    ax1 = axes[0]
    cycles = [r.get("cycle", 0) for r in results]
    pass_rates = [(r.get("passed", 0) / max(r.get("total", 1), 1) * 100) for r in results]
    models = [r.get("model", "unknown")[:20] for r in results]

    colors = ["#4C72B0" if i == 0 else "#55A868" for i in range(len(cycles))]
    bars = ax1.bar(range(len(cycles)), pass_rates, color=colors, edgecolor="black", linewidth=1.2)

    ax1.set_xlabel("Training Cycle", fontsize=11)
    ax1.set_ylabel("Pass Rate (%)", fontsize=11)
    ax1.set_title("Pass Rate by Cycle", fontsize=12)
    ax1.set_xticks(range(len(cycles)))
    ax1.set_xticklabels([f"Cycle {c}\n({m})" for c, m in zip(cycles, models)], fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=pass_rates[0] if pass_rates else 0, color="red", linestyle="--", alpha=0.5, label="Baseline")

    # Add value labels on bars
    for bar, rate in zip(bars, pass_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Plot 2: Passed vs Failed counts
    ax2 = axes[1]
    passed = [r.get("passed", 0) for r in results]
    failed = [r.get("total", 0) - r.get("passed", 0) for r in results]

    x = range(len(cycles))
    width = 0.35
    bars1 = ax2.bar([i - width/2 for i in x], passed, width, label="Passed", color="#55A868", edgecolor="black")
    bars2 = ax2.bar([i + width/2 for i in x], failed, width, label="Failed", color="#C44E52", edgecolor="black")

    ax2.set_xlabel("Training Cycle", fontsize=11)
    ax2.set_ylabel("Number of Tasks", fontsize=11)
    ax2.set_title("Task Outcomes by Cycle", fontsize=12)
    ax2.set_xticks(range(len(cycles)))
    ax2.set_xticklabels([f"Cycle {c}" for c in cycles], fontsize=10)
    ax2.legend()

    # Add value labels
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate sleepy-coder results plots")
    parser.add_argument("--data-dir", "-d", default="data", help="Data directory")
    parser.add_argument("--output", "-o", default="docs/results.png", help="Output plot path")
    parser.add_argument("--report", "-r", default="docs/results.txt", help="Output report path")
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / args.data_dir

    print(f"Loading results from: {data_dir}")

    # Load data
    results = load_eval_results(data_dir)
    episodes = load_episodes_summary(data_dir)

    # Generate text report
    report_path = project_root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_text_report(results, episodes, report_path)

    # Generate plots
    if HAS_MATPLOTLIB:
        output_path = project_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generate_plots(results, output_path)


if __name__ == "__main__":
    main()
