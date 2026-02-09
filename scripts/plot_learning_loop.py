#!/usr/bin/env python3
"""
Sleepy Coder Learning Loop Visualization

Shows the full cycle: usage → errors → training → evaluation → repeat
with quality graphed in parallel to demonstrate measurable improvement.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
import numpy as np

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def load_metrics(metrics_path: Path) -> list[dict]:
    """Load metrics from JSONL file."""
    metrics = []
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
    return metrics


def load_training_metrics(runs_dir: Path) -> list[dict]:
    """Load training metrics from run directories."""
    training_runs = []
    for adapter_dir in sorted(runs_dir.glob("**/metrics.json")):
        try:
            with open(adapter_dir) as f:
                data = json.load(f)
                data['path'] = str(adapter_dir.parent)
                training_runs.append(data)
        except Exception:
            pass
    return training_runs


def create_learning_loop_plot(metrics: list[dict], training_runs: list[dict], output_dir: Path):
    """Create a comprehensive learning loop visualization."""
    fig = plt.figure(figsize=(16, 12))

    # Create a 2x2 grid of subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1.5], hspace=0.3, wspace=0.25)

    # ---- Plot 1: Quality Over Training Cycles (main focus) ----
    ax1 = fig.add_subplot(gs[0, :])

    if metrics:
        cycles = [m['cycle'] for m in metrics]
        pass_rates = [m['pass_rate'] * 100 for m in metrics]

        # Plot pass rate with markers
        ax1.plot(cycles, pass_rates, 'o-', linewidth=2.5, markersize=10,
                 color='#2196F3', label='Pass Rate (%)')

        # Add baseline reference line
        if pass_rates:
            baseline = pass_rates[0]
            ax1.axhline(y=baseline, color='#4CAF50', linestyle='--', linewidth=2,
                       label=f'Baseline ({baseline:.1f}%)')

        # Color regions based on improvement
        for i in range(1, len(cycles)):
            color = '#E8F5E9' if pass_rates[i] >= pass_rates[0] else '#FFEBEE'
            ax1.axvspan(cycles[i-1], cycles[i], alpha=0.3, color=color)

        # Add annotations
        for i, (cycle, rate) in enumerate(zip(cycles, pass_rates)):
            offset = 5 if i % 2 == 0 else -15
            ax1.annotate(f'{rate:.1f}%', (cycle, rate),
                        textcoords="offset points", xytext=(0, offset),
                        ha='center', fontsize=11, fontweight='bold')

        ax1.set_xlabel('Training Cycle')
        ax1.set_ylabel('Pass Rate (%)')
        ax1.set_title('Quality Over Training Iterations\n(Green = at/above baseline, Red = regression)',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.set_ylim(max(0, min(pass_rates) - 10), min(100, max(pass_rates) + 10))
        ax1.set_xticks(cycles)

    # ---- Plot 2: Task Outcomes by Cycle ----
    ax2 = fig.add_subplot(gs[1, 0])

    if metrics:
        cycles = [m['cycle'] for m in metrics]
        passed = [m['passed'] for m in metrics]
        failed = [m['failed'] for m in metrics]

        x = np.arange(len(cycles))
        width = 0.35

        bars1 = ax2.bar(x - width/2, passed, width, label='Passed', color='#4CAF50')
        bars2 = ax2.bar(x + width/2, failed, width, label='Failed', color='#F44336')

        ax2.set_xlabel('Cycle')
        ax2.set_ylabel('Number of Tasks')
        ax2.set_title('Task Outcomes by Cycle')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Cycle {c}' for c in cycles])
        ax2.legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    # ---- Plot 3: Training Loss Over Runs ----
    ax3 = fig.add_subplot(gs[1, 1])

    if training_runs:
        runs = list(range(len(training_runs)))
        losses = [r.get('train_loss', 0) for r in training_runs]

        ax3.plot(runs, losses, 'o-', linewidth=2, markersize=8, color='#FF9800')
        ax3.fill_between(runs, losses, alpha=0.3, color='#FF9800')

        ax3.set_xlabel('Training Run')
        ax3.set_ylabel('Training Loss')
        ax3.set_title('Training Loss Over Time')
        ax3.set_xticks(runs)
        ax3.set_xticklabels([f'Run {i+1}' for i in runs])

        for i, (run, loss) in enumerate(zip(runs, losses)):
            ax3.annotate(f'{loss:.3f}', (run, loss),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=10)

    # ---- Plot 4: The Learning Loop Diagram ----
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 3)
    ax4.axis('off')
    ax4.set_title('The Sleepy Coder Learning Loop', fontsize=14, fontweight='bold', pad=20)

    # Draw the loop boxes
    boxes = [
        (1, 1.5, 'DAY\n(Run Tasks)', '#E3F2FD'),
        (3, 1.5, 'CAPTURE\n(Errors + Fixes)', '#FFF3E0'),
        (5, 1.5, 'SLEEP\n(Train LoRA)', '#E8F5E9'),
        (7, 1.5, 'EVAL\n(Gate Check)', '#FCE4EC'),
        (9, 1.5, 'DEPLOY\n(If Pass)', '#F3E5F5'),
    ]

    for x, y, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x-0.7, y-0.5), 1.4, 1,
                                        boxstyle="round,pad=0.05,rounding_size=0.2",
                                        facecolor=color, edgecolor='gray', linewidth=2)
        ax4.add_patch(rect)
        ax4.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw arrows between boxes
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.7
        x2 = boxes[i+1][0] - 0.7
        ax4.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Draw feedback loop arrow
    ax4.annotate('', xy=(1, 0.8), xytext=(9, 0.8),
                arrowprops=dict(arrowstyle='->', color='#2196F3', lw=2,
                               connectionstyle='arc3,rad=0.3'))
    ax4.text(5, 0.3, 'Continuous Improvement Loop', ha='center',
             fontsize=11, fontstyle='italic', color='#2196F3')

    # Add timestamp
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
             ha='right', fontsize=9, color='gray')

    # Save the plot
    output_path = output_dir / 'learning_loop.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    # Also save individual smaller versions
    plt.savefig(output_dir / 'learning_loop.svg', format='svg', bbox_inches='tight')

    plt.close()
    return output_path


def create_simple_progress_plot(metrics: list[dict], output_dir: Path):
    """Create a simple, focused progress plot for demos."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if not metrics:
        ax.text(0.5, 0.5, 'No metrics data yet', ha='center', va='center', fontsize=14)
        plt.savefig(output_dir / 'progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    cycles = [m['cycle'] for m in metrics]
    pass_rates = [m['pass_rate'] * 100 for m in metrics]

    # Create gradient fill based on improvement
    colors = ['#4CAF50' if r >= pass_rates[0] else '#F44336' for r in pass_rates]

    # Plot with color-coded markers
    for i, (cycle, rate, color) in enumerate(zip(cycles, pass_rates, colors)):
        ax.scatter(cycle, rate, s=200, c=color, zorder=3, edgecolor='white', linewidth=2)

    # Connect with lines
    ax.plot(cycles, pass_rates, 'k-', linewidth=2, alpha=0.5, zorder=2)

    # Baseline
    baseline = pass_rates[0]
    ax.axhline(y=baseline, color='#2196F3', linestyle='--', linewidth=2,
               label=f'Baseline: {baseline:.1f}%', alpha=0.7)

    # Fill region above/below baseline
    ax.fill_between(cycles, pass_rates, baseline,
                    where=[r >= baseline for r in pass_rates],
                    alpha=0.2, color='#4CAF50', interpolate=True)
    ax.fill_between(cycles, pass_rates, baseline,
                    where=[r < baseline for r in pass_rates],
                    alpha=0.2, color='#F44336', interpolate=True)

    # Labels
    for cycle, rate in zip(cycles, pass_rates):
        ax.annotate(f'{rate:.1f}%', (cycle, rate),
                   textcoords="offset points", xytext=(0, 15),
                   ha='center', fontsize=12, fontweight='bold')

    ax.set_xlabel('Training Cycle', fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('Sleepy Coder: Quality Over Training Iterations',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(max(0, min(pass_rates) - 15), min(100, max(pass_rates) + 15))
    ax.set_xticks(cycles)

    # Add legend for colors
    green_patch = mpatches.Patch(color='#4CAF50', label='At/above baseline')
    red_patch = mpatches.Patch(color='#F44336', label='Below baseline')
    ax.legend(handles=[green_patch, red_patch], loc='lower right')

    plt.savefig(output_dir / 'progress.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_dir / 'progress.png'}")


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    metrics_path = project_root / 'rust' / 'data' / 'episodes' / 'metrics.jsonl'
    runs_dir = project_root / 'runs' / 'adapters'
    output_dir = project_root / 'docs' / 'viz'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics...")
    metrics = load_metrics(metrics_path)
    print(f"  Found {len(metrics)} evaluation cycles")

    print("Loading training runs...")
    training_runs = load_training_metrics(runs_dir)
    print(f"  Found {len(training_runs)} training runs")

    print("\nGenerating visualizations...")
    create_learning_loop_plot(metrics, training_runs, output_dir)
    create_simple_progress_plot(metrics, output_dir)

    # Print summary
    print("\n=== Summary ===")
    for m in metrics:
        status = "PASS" if m['pass_rate'] >= metrics[0]['pass_rate'] else "REGRESSED"
        print(f"  Cycle {m['cycle']}: {m['pass_rate']*100:.1f}% [{status}]")

    if metrics:
        baseline = metrics[0]['pass_rate']
        latest = metrics[-1]['pass_rate']
        delta = (latest - baseline) * 100
        print(f"\n  Delta from baseline: {delta:+.1f}%")
        if delta >= 0:
            print("  Status: Learning is working!")
        else:
            print("  Status: Still recovering from regression")


if __name__ == '__main__':
    main()
