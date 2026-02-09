#!/usr/bin/env python3
"""
Update the dashboard HTML with real metrics data.

Usage:
    python scripts/update_dashboard.py
"""

import json
import re
from datetime import datetime
from pathlib import Path


def load_metrics(metrics_file: Path) -> list[dict]:
    """Load metrics from JSONL file."""
    results = []
    if metrics_file.exists():
        with open(metrics_file) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return sorted(results, key=lambda x: x.get("cycle", 0))


def generate_dashboard_data(metrics: list[dict]) -> str:
    """Generate JavaScript data object from metrics."""
    if not metrics:
        return """
        const metricsData = {
            cycles: [],
            passRates: [],
            passed: [],
            failed: [],
            models: []
        };
        """

    cycles = [m.get("cycle", 0) for m in metrics]
    pass_rates = [round(m.get("pass_rate", 0) * 100, 1) for m in metrics]
    passed = [m.get("passed", 0) for m in metrics]
    failed = [m.get("failed", 0) for m in metrics]
    models = [m.get("model", "unknown").split(":")[0][:20] for m in metrics]

    return f"""
        const metricsData = {{
            cycles: {json.dumps(cycles)},
            passRates: {json.dumps(pass_rates)},
            passed: {json.dumps(passed)},
            failed: {json.dumps(failed)},
            models: {json.dumps(models)}
        }};
    """


def generate_timeline_html(metrics: list[dict]) -> str:
    """Generate timeline items HTML from metrics."""
    if not metrics:
        return ""

    html_items = []
    baseline_rate = metrics[0].get("pass_rate", 0) * 100 if metrics else 0

    for m in metrics:
        cycle = m.get("cycle", 0)
        model = m.get("model", "unknown")
        pass_rate = m.get("pass_rate", 0) * 100
        passed = m.get("passed", 0)
        failed = m.get("failed", 0)
        total = passed + failed
        run_id = m.get("run_id", "")

        # Determine status
        if cycle == 0:
            status = "success"
            title = f"Cycle {cycle}: Baseline Evaluation"
            change_html = ""
        else:
            change = pass_rate - baseline_rate
            if change >= 0:
                status = "success"
                change_html = f' | <span style="color: var(--accent-green);">Improvement: +{change:.1f}%</span>'
            else:
                status = "failure"
                change_html = f' | <span style="color: var(--accent-red);">Regression: {change:.1f}%</span>'
            title = f"Cycle {cycle}: Training Run"

        # Parse timestamp from run_id if available
        time_str = "Unknown"
        if run_id:
            try:
                # Format: eval_cycle0_20260209_035102
                parts = run_id.split("_")
                if len(parts) >= 4:
                    date_str = parts[-2]
                    time_part = parts[-1]
                    dt = datetime.strptime(f"{date_str}_{time_part}", "%Y%m%d_%H%M%S")
                    time_str = dt.strftime("%b %d, %Y %H:%M")
            except Exception:
                pass

        html_items.append(f"""
            <div class="timeline-item">
                <div class="timeline-dot {status}"></div>
                <div class="timeline-content">
                    <div class="timeline-header">
                        <span class="timeline-title-text">{title}</span>
                        <span class="timeline-time">{time_str}</span>
                    </div>
                    <div class="timeline-details">
                        Model: {model} | Pass Rate: {pass_rate:.1f}% ({passed}/{total}){change_html}
                    </div>
                </div>
            </div>
        """)

    return "\n".join(html_items)


def update_dashboard(dashboard_path: Path, metrics: list[dict]):
    """Update the dashboard HTML with new data."""
    with open(dashboard_path) as f:
        html = f.read()

    # Update the metricsData JavaScript object
    new_data = generate_dashboard_data(metrics)
    html = re.sub(
        r'const metricsData = \{[^}]+cycles:[^}]+\};',
        new_data.strip(),
        html,
        flags=re.DOTALL
    )

    # Update baseline rate display
    if metrics:
        baseline = metrics[0].get("pass_rate", 0) * 100
        html = re.sub(
            r'<div class="metric-value neutral" id="baseline-rate">[^<]+</div>',
            f'<div class="metric-value neutral" id="baseline-rate">{baseline:.1f}%</div>',
            html
        )

    with open(dashboard_path, "w") as f:
        f.write(html)

    print(f"Dashboard updated: {dashboard_path}")


def main():
    project_root = Path(__file__).parent.parent
    metrics_file = project_root / "rust" / "data" / "episodes" / "metrics.jsonl"
    dashboard_path = project_root / "docs" / "dashboard.html"

    print(f"Loading metrics from: {metrics_file}")
    metrics = load_metrics(metrics_file)
    print(f"Found {len(metrics)} evaluation runs")

    if not dashboard_path.exists():
        print(f"Dashboard not found: {dashboard_path}")
        return

    update_dashboard(dashboard_path, metrics)

    # Print summary
    if metrics:
        baseline = metrics[0].get("pass_rate", 0) * 100
        latest = metrics[-1].get("pass_rate", 0) * 100
        change = latest - baseline
        print(f"\nCurrent status:")
        print(f"  Baseline: {baseline:.1f}%")
        print(f"  Latest:   {latest:.1f}%")
        print(f"  Change:   {change:+.1f}%")


if __name__ == "__main__":
    main()
