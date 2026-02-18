#!/usr/bin/env python3
"""Generate SVG visualizations from forgetting analysis data for docs/viz/."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def load_data(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def strategy_comparison_chart(data: dict, out_path: Path):
    """Bar chart comparing baseline, routed, and averaged strategies."""
    strategies = ["Baseline", "Routed", "Averaged"]
    pass_rates = [
        data["baseline"]["pass_rate"] * 100,
        data["routed"]["pass_rate"] * 100,
        data["averaged"]["pass_rate"] * 100,
    ]
    colors = ["#6c757d", "#28a745", "#dc3545"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strategies, pass_rates, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

    for bar, rate in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_ylabel("Pass Rate (%)", fontsize=12)
    ax.set_title("Share Algorithm: Strategy Comparison (30 Frozen Koans)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=12)

    # Add annotations
    ax.annotate("Zero regressions", xy=(1, pass_rates[1]),
                xytext=(1.4, pass_rates[1] + 8),
                arrowprops=dict(arrowstyle="->", color="#28a745"),
                fontsize=10, color="#28a745", fontweight="bold")
    ax.annotate("rh_008 regressed", xy=(2, pass_rates[2]),
                xytext=(2.3, pass_rates[2] + 8),
                arrowprops=dict(arrowstyle="->", color="#dc3545"),
                fontsize=10, color="#dc3545")

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def per_family_chart(data: dict, out_path: Path):
    """Grouped bar chart showing per-family pass rates by strategy."""
    families = ["borrow_checker", "result_handling", "trait_bounds"]
    family_labels = ["Borrow Checker", "Result Handling", "Trait Bounds"]
    strategies = ["baseline", "routed", "averaged"]
    strategy_labels = ["Baseline", "Routed", "Averaged"]
    colors = ["#6c757d", "#28a745", "#dc3545"]

    x = np.arange(len(families))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (strat, label, color) in enumerate(zip(strategies, strategy_labels, colors)):
        rates = []
        for fam in families:
            fam_data = data[strat]["per_family"][fam]
            rates.append(fam_data["passed"] / fam_data["total"] * 100)
        bars = ax.bar(x + i * width, rates, width, label=label, color=color, edgecolor="white", linewidth=1)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Pass Rate (%)", fontsize=12)
    ax.set_title("Per-Family Pass Rates by Strategy", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(family_labels, fontsize=11)
    ax.set_ylim(0, 90)
    ax.legend(fontsize=11, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def forgetting_heatmap(data: dict, out_path: Path):
    """Visual heatmap showing per-koan forgetting under each coefficient."""
    koans = sorted(data["baseline"]["per_koan"].keys())
    coef_names = [
        "mut_borrow_conflict_v4",
        "double_mut_borrow_v4",
        "return_local_ref_v4",
        "missing_clone_v4",
        "missing_hash_v4",
        "missing_ord_v4",
        "option_ok_or_v4",
        "result_map_err_v4",
    ]
    coef_short = ["mut_bc", "dbl_mt", "ret_lr", "mis_cl", "mis_hs", "mis_or", "opt_ok", "res_me"]

    baseline = data["baseline"]["per_koan"]

    # Build matrix: 0=stayed fail, 1=stayed pass, 2=+GAIN, -1=-LOST
    n_koans = len(koans)
    n_coefs = len(coef_names)
    matrix = np.zeros((n_koans, n_coefs + 3))  # +3 for baseline, routed, averaged

    col_labels = ["BL"] + coef_short + ["ROUTED", "AVGD"]

    for i, koan in enumerate(koans):
        bl = baseline[koan]
        # Baseline column
        matrix[i, 0] = 1 if bl else 0

        # Per-coefficient columns
        for j, coef in enumerate(coef_names):
            coef_result = data["per_coefficient"][coef]["per_koan"][koan]
            if bl and coef_result:
                matrix[i, j + 1] = 1  # stayed pass
            elif bl and not coef_result:
                matrix[i, j + 1] = -1  # LOST
            elif not bl and coef_result:
                matrix[i, j + 1] = 2  # GAIN
            else:
                matrix[i, j + 1] = 0  # stayed fail

        # Routed column
        routed_result = data["routed"]["per_koan"][koan]
        if bl and routed_result:
            matrix[i, n_coefs + 1] = 1
        elif bl and not routed_result:
            matrix[i, n_coefs + 1] = -1
        elif not bl and routed_result:
            matrix[i, n_coefs + 1] = 2
        else:
            matrix[i, n_coefs + 1] = 0

        # Averaged column
        avg_result = data["averaged"]["per_koan"][koan]
        if bl and avg_result:
            matrix[i, n_coefs + 2] = 1
        elif bl and not avg_result:
            matrix[i, n_coefs + 2] = -1
        elif not bl and avg_result:
            matrix[i, n_coefs + 2] = 2
        else:
            matrix[i, n_coefs + 2] = 0

    # Custom colormap: -1=red, 0=dark gray, 1=green, 2=bright blue
    cmap = mcolors.ListedColormap(["#e74c3c", "#3a3a3a", "#27ae60", "#3498db"])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Labels
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(koans)))
    ax.set_yticklabels(koans, fontsize=9, fontfamily="monospace")

    # Add text annotations
    labels_map = {-1: "LOST", 0: ".", 1: "P", 2: "GAIN"}
    text_colors = {-1: "white", 0: "#888", 1: "white", 2: "white"}
    for i in range(n_koans):
        for j in range(len(col_labels)):
            val = int(matrix[i, j])
            label = labels_map[val]
            color = text_colors[val]
            fontsize = 7 if label in ("LOST", "GAIN") else 8
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=fontsize, color=color, fontweight="bold" if val in (-1, 2) else "normal")

    # Add family separators
    ax.axhline(y=9.5, color="white", linewidth=2)
    ax.axhline(y=19.5, color="white", linewidth=2)

    # Add column separators for routed/averaged
    ax.axvline(x=n_coefs + 0.5, color="white", linewidth=2)

    # Family labels on the right
    ax.text(len(col_labels) + 0.3, 4.5, "BC", fontsize=11, fontweight="bold", color="#3498db", va="center")
    ax.text(len(col_labels) + 0.3, 14.5, "RH", fontsize=11, fontweight="bold", color="#e67e22", va="center")
    ax.text(len(col_labels) + 0.3, 24.5, "TB", fontsize=11, fontweight="bold", color="#9b59b6", va="center")

    ax.set_title("Forgetting Heatmap: Per-Coefficient Impact on 30 Frozen Koans",
                 fontsize=14, fontweight="bold", pad=15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", label="P = Pass (unchanged)"),
        Patch(facecolor="#3a3a3a", label=".  = Fail (unchanged)"),
        Patch(facecolor="#3498db", label="+GAIN = Was fail, now pass"),
        Patch(facecolor="#e74c3c", label="-LOST = Was pass, now fail"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=4, fontsize=10, frameon=False)

    # Add totals row
    totals_y = n_koans + 0.8
    for j in range(len(col_labels)):
        col_pass = sum(1 for i in range(n_koans) if matrix[i, j] in (1, 2))
        ax.text(j, totals_y, f"{col_pass}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="#333")
    ax.text(-1.5, totals_y, "Total:", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_ylim(n_koans + 1.2, -0.5)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def coefficient_impact_chart(data: dict, out_path: Path):
    """Show net delta (gains - regressions) per coefficient."""
    coef_names = [
        "mut_borrow_conflict_v4",
        "double_mut_borrow_v4",
        "return_local_ref_v4",
        "missing_clone_v4",
        "missing_hash_v4",
        "missing_ord_v4",
        "option_ok_or_v4",
        "result_map_err_v4",
    ]
    coef_short = ["mut_bc", "dbl_mt", "ret_lr", "mis_cl", "mis_hs", "mis_or", "opt_ok", "res_me"]

    gains = []
    regressions = []
    for coef in coef_names:
        cd = data["per_coefficient"][coef]
        gains.append(len(cd["gains"]))
        regressions.append(-len(cd["regressions"]))

    x = np.arange(len(coef_short))
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x, gains, color="#28a745", label="Gains", width=0.6)
    ax.bar(x, regressions, color="#dc3545", label="Regressions", width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(coef_short, fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("Number of Koans", fontsize=12)
    ax.set_title("Per-Coefficient Impact: Gains vs Regressions", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-2.5, 2.5)

    # Annotate
    ax.text(4, -2.2, "missing_hash also regresses tb_005",
            fontsize=9, fontstyle="italic", color="#dc3545")
    ax.text(1, 1.7, "5 of 8 coefficients improve rh_002",
            fontsize=9, fontstyle="italic", color="#28a745")

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def experiment_timeline_chart(out_path: Path):
    """Timeline of all experiments showing pass rates."""
    experiments = [
        ("C0 Baseline", 76.7, "#6c757d"),
        ("C1 Naive LoRA", 60.0, "#dc3545"),
        ("C9-10 Minimal", 73.3, "#ffc107"),
        ("Share-6", 73.3, "#ffc107"),
        ("Share-51", 70.0, "#dc3545"),
        ("Share Full", 73.3, "#ffc107"),
        ("Prompt Eng.", 83.3, "#28a745"),
        ("Exp1a Analytical", 43.3, "#dc3545"),
        ("Exp1b v4 Routed", 50.0, "#28a745"),
    ]

    names = [e[0] for e in experiments]
    rates = [e[1] for e in experiments]
    colors = [e[2] for e in experiments]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), rates, color=colors, width=0.6, edgecolor="white", linewidth=1)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Pass Rate (%)", fontsize=12)
    ax.set_title("Sleepy-Coder Experiment Timeline", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.axhline(y=76.7, color="#6c757d", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(len(names) - 0.5, 78, "Baseline (76.7%)", fontsize=9, color="#6c757d", ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Note about different eval conditions
    ax.annotate("* Exp1a/1b: plain prompt,\n  bf16 (not Ollama Q4)",
                xy=(8, 50), xytext=(6.5, 25),
                arrowprops=dict(arrowstyle="->", color="#666"),
                fontsize=9, color="#666",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="#ccc"))

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    project_root = Path(__file__).parent.parent
    json_path = project_root / "runs" / "experiments" / "forgetting" / "forgetting_analysis.json"

    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        sys.exit(1)

    data = load_data(json_path)
    viz_dir = project_root / "docs" / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")
    strategy_comparison_chart(data, viz_dir / "strategy_comparison.svg")
    per_family_chart(data, viz_dir / "per_family_breakdown.svg")
    forgetting_heatmap(data, viz_dir / "forgetting_heatmap.svg")
    coefficient_impact_chart(data, viz_dir / "coefficient_impact.svg")
    experiment_timeline_chart(viz_dir / "experiment_timeline.svg")

    # Also save the raw JSON to docs for GitHub visibility
    results_json_path = project_root / "docs" / "viz" / "forgetting_analysis.json"
    import shutil
    shutil.copy2(json_path, results_json_path)
    print(f"  Copied: {results_json_path}")

    print("\nDone! Generated 5 SVG charts + 1 JSON in docs/viz/")


if __name__ == "__main__":
    main()
