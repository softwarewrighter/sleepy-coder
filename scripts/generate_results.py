#!/usr/bin/env python3
"""
Generate results plots and documentation from evaluation metrics.

Usage:
    python scripts/generate_results.py
"""

import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_metrics(metrics_file: Path) -> list[dict]:
    """Load metrics from JSONL file."""
    results = []
    if metrics_file.exists():
        with open(metrics_file) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results


def generate_comparison_plot(results: list[dict], output_path: Path):
    """Generate comparison bar chart."""
    if not results:
        print("No results to plot")
        return

    # Sort by cycle
    results = sorted(results, key=lambda x: x.get("cycle", 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Sleepy-Coder: Continual Learning Evaluation", fontsize=16, fontweight="bold")

    # Plot 1: Pass rate comparison
    ax1 = axes[0]
    cycles = [r.get("cycle", 0) for r in results]
    pass_rates = [r.get("pass_rate", 0) * 100 for r in results]
    models = [r.get("model", "unknown").split(":")[0][:15] for r in results]

    # Color based on improvement
    colors = []
    baseline = pass_rates[0] if pass_rates else 0
    for i, rate in enumerate(pass_rates):
        if i == 0:
            colors.append("#4C72B0")  # Blue for baseline
        elif rate >= baseline:
            colors.append("#55A868")  # Green for improvement
        else:
            colors.append("#C44E52")  # Red for regression

    bars = ax1.bar(range(len(cycles)), pass_rates, color=colors, edgecolor="black", linewidth=1.5)

    ax1.set_xlabel("Training Cycle", fontsize=12)
    ax1.set_ylabel("Pass Rate (%)", fontsize=12)
    ax1.set_title("Pass Rate by Training Cycle", fontsize=13)
    ax1.set_xticks(range(len(cycles)))
    ax1.set_xticklabels([f"Cycle {c}\n({m})" for c, m in zip(cycles, models)], fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=baseline, color="gray", linestyle="--", alpha=0.7, linewidth=2, label="Baseline")
    ax1.legend(loc="upper right")

    # Add grid
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars, pass_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Plot 2: Passed vs Failed stacked bar
    ax2 = axes[1]
    passed = [r.get("passed", 0) for r in results]
    failed = [r.get("failed", 0) for r in results]

    x = range(len(cycles))
    bars1 = ax2.bar(x, passed, label="Passed", color="#55A868", edgecolor="black", linewidth=1.5)
    bars2 = ax2.bar(x, failed, bottom=passed, label="Failed", color="#C44E52", edgecolor="black", linewidth=1.5)

    ax2.set_xlabel("Training Cycle", fontsize=12)
    ax2.set_ylabel("Number of Tasks", fontsize=12)
    ax2.set_title("Task Outcomes by Cycle", fontsize=13)
    ax2.set_xticks(range(len(cycles)))
    ax2.set_xticklabels([f"Cycle {c}" for c in cycles], fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    # Add labels
    for i, (p, f) in enumerate(zip(passed, failed)):
        ax2.text(i, p/2, str(p), ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax2.text(i, p + f/2, str(f), ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Plot saved to: {output_path}")
    plt.close()


def generate_results_doc(results: list[dict], output_path: Path):
    """Generate markdown results documentation."""
    results = sorted(results, key=lambda x: x.get("cycle", 0))

    lines = []
    lines.append("# Sleepy-Coder Evaluation Results")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Cycle | Model | Pass Rate | Passed | Failed | Median Steps |")
    lines.append("|-------|-------|-----------|--------|--------|--------------|")

    for r in results:
        cycle = r.get("cycle", 0)
        model = r.get("model", "unknown")
        rate = r.get("pass_rate", 0) * 100
        passed = r.get("passed", 0)
        failed = r.get("failed", 0)
        steps = r.get("median_steps_to_green", 0)
        lines.append(f"| {cycle} | {model} | {rate:.1f}% | {passed} | {failed} | {steps:.1f} |")

    lines.append("")

    # Calculate change
    if len(results) >= 2:
        baseline_rate = results[0].get("pass_rate", 0) * 100
        latest_rate = results[-1].get("pass_rate", 0) * 100
        change = latest_rate - baseline_rate

        lines.append("## Analysis")
        lines.append("")
        if change > 0:
            lines.append(f"**Improvement: +{change:.1f}%** ({baseline_rate:.1f}% → {latest_rate:.1f}%)")
        elif change < 0:
            lines.append(f"**Regression: {change:.1f}%** ({baseline_rate:.1f}% → {latest_rate:.1f}%)")
            lines.append("")
            lines.append("### Possible Causes of Regression:")
            lines.append("")
            lines.append("1. **Insufficient training data**: Only 23 failed episodes used for training")
            lines.append("2. **Overfitting to training distribution**: Model may have overfit to specific error patterns")
            lines.append("3. **Catastrophic forgetting**: Fine-tuning may have degraded general capabilities")
            lines.append("4. **Training hyperparameters**: Learning rate or steps may need tuning")
            lines.append("5. **Data quality**: Training on failed examples may include poor patterns")
            lines.append("")
            lines.append("### Recommended Next Steps:")
            lines.append("")
            lines.append("1. Generate more training data by running more evaluation cycles")
            lines.append("2. Use curriculum learning - start with easier examples")
            lines.append("3. Adjust LoRA rank (try r=8 instead of r=16)")
            lines.append("4. Lower learning rate (try 1e-4 instead of 2e-4)")
            lines.append("5. Add regularization or use LoRA dropout")
            lines.append("6. Filter training data to only include high-quality corrections")
        else:
            lines.append(f"**No change**: {baseline_rate:.1f}%")

    lines.append("")
    lines.append("## Results Plot")
    lines.append("")
    lines.append("![Evaluation Results](results.png)")
    lines.append("")

    lines.append("## Training Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Base Model | Qwen2.5-Coder-1.5B-Instruct |")
    lines.append("| LoRA Rank | 16 |")
    lines.append("| LoRA Alpha | 32 |")
    lines.append("| Training Steps | 500 |")
    lines.append("| Learning Rate | 2e-4 |")
    lines.append("| Batch Size | 4 |")
    lines.append("| Quantization | 4-bit NF4 (QLoRA) |")
    lines.append("")

    lines.append("## Raw Data")
    lines.append("")
    lines.append("```json")
    for r in results:
        lines.append(json.dumps(r))
    lines.append("```")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Documentation saved to: {output_path}")


def main():
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    metrics_file = project_root / "rust" / "data" / "episodes" / "metrics.jsonl"
    print(f"Loading metrics from: {metrics_file}")

    results = load_metrics(metrics_file)
    print(f"Found {len(results)} evaluation runs")

    if not results:
        print("No results found. Run evaluations first:")
        print("  ./rust/target/release/sleepy-coder eval --cycle 0")
        return

    # Generate outputs
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)

    generate_comparison_plot(results, docs_dir / "results.png")
    generate_results_doc(results, docs_dir / "results.md")

    print("\nDone! View results at:")
    print(f"  - {docs_dir / 'results.md'}")
    print(f"  - {docs_dir / 'results.png'}")


if __name__ == "__main__":
    main()
