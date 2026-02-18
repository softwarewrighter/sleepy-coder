#!/usr/bin/env python3
"""
Measure and visualize catastrophic forgetting across Share coefficient sets.

For each trained coefficient set, evaluates ALL 30 frozen koans to show:
1. Which koans improve (gain) vs regress (forget) vs stay the same
2. Per-family breakdown
3. Net delta (gains - regressions) per coefficient
4. ASCII heatmap of per-koan results

Usage:
    python scripts/measure_forgetting.py --share runs/share_phase2

Output: Table + heatmap + JSON results
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from share_inference import ShareInferenceEngine, detect_pattern
from eval_koans import (
    KoanEvaluator,
    NAMED_COEFFICIENTS,
    PATTERN_TO_COEF,
    load_frozen_koans,
)

os.environ["HF_HUB_OFFLINE"] = "1"


def measure_forgetting(share_dir: Path, output_dir: Path):
    """Run baseline + each coefficient set on all 30 koans."""
    output_dir.mkdir(parents=True, exist_ok=True)

    koans = load_frozen_koans()
    print(f"Loaded {len(koans)} frozen eval koans")

    print("Loading Share inference engine...")
    engine = ShareInferenceEngine(share_dir)
    evaluator = KoanEvaluator(engine)

    # Step 1: Baseline
    print("\n" + "=" * 60)
    print("BASELINE (no coefficients)")
    print("=" * 60)
    baseline_results = evaluator.run_eval(koans, strategy="baseline")
    baseline_pass = {r.task_id: r.passed for r in baseline_results.koan_results}
    print(f"\nBaseline: {baseline_results.passed}/{baseline_results.total} = "
          f"{baseline_results.pass_rate:.1%}")

    # Step 2: Each coefficient set individually
    all_coef_results = {}
    for coef_id in NAMED_COEFFICIENTS:
        print(f"\n{'─' * 60}")
        print(f"Coefficient: {coef_id}")
        print(f"{'─' * 60}")

        # Apply this coefficient to ALL koans (not routing, to measure forgetting)
        engine.restore_weights()
        try:
            engine.apply_coefficients(coef_id)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        results = evaluator.run_eval(koans, strategy="baseline")
        all_coef_results[coef_id] = results

        coef_pass = {r.task_id: r.passed for r in results.koan_results}

        # Compute deltas
        gains = [tid for tid in baseline_pass if not baseline_pass[tid] and coef_pass.get(tid, False)]
        regressions = [tid for tid in baseline_pass if baseline_pass[tid] and not coef_pass.get(tid, False)]

        print(f"\n  Pass: {results.passed}/{results.total} = {results.pass_rate:.1%}")
        print(f"  Gains (+): {len(gains)} {gains}")
        print(f"  Regressions (-): {len(regressions)} {regressions}")
        print(f"  Net delta: {len(gains) - len(regressions):+d}")

    # Step 3: Routed (use detect_pattern to select coefficient per koan)
    print(f"\n{'=' * 60}")
    print("ROUTED (pattern-matched coefficients)")
    print("=" * 60)
    routed_results = evaluator.run_eval(koans, strategy="routed")
    routed_pass = {r.task_id: r.passed for r in routed_results.koan_results}
    routed_patterns = {r.task_id: r.pattern_used for r in routed_results.koan_results}
    gains = [tid for tid in baseline_pass if not baseline_pass[tid] and routed_pass.get(tid, False)]
    regressions = [tid for tid in baseline_pass if baseline_pass[tid] and not routed_pass.get(tid, False)]
    print(f"\n  Pass: {routed_results.passed}/{routed_results.total} = {routed_results.pass_rate:.1%}")
    print(f"  Gains (+): {len(gains)} {gains}")
    print(f"  Regressions (-): {len(regressions)} {regressions}")
    print(f"  Net delta: {len(gains) - len(regressions):+d}")

    # Step 4: Averaged
    print(f"\n{'=' * 60}")
    print("AVERAGED (all coefficients averaged)")
    print("=" * 60)
    averaged_results = evaluator.run_eval(koans, strategy="averaged")
    avg_pass = {r.task_id: r.passed for r in averaged_results.koan_results}
    gains = [tid for tid in baseline_pass if not baseline_pass[tid] and avg_pass.get(tid, False)]
    regressions = [tid for tid in baseline_pass if baseline_pass[tid] and not avg_pass.get(tid, False)]
    print(f"\n  Pass: {averaged_results.passed}/{averaged_results.total} = {averaged_results.pass_rate:.1%}")
    print(f"  Gains (+): {len(gains)} {gains}")
    print(f"  Regressions (-): {len(regressions)} {regressions}")
    print(f"  Net delta: {len(gains) - len(regressions):+d}")

    engine.restore_weights()

    # =============================================
    # Print ASCII heatmap
    # =============================================
    print("\n" + "=" * 80)
    print("FORGETTING HEATMAP")
    print("=" * 80)
    koan_ids = sorted(baseline_pass.keys())

    # Header: coefficient names (abbreviated)
    coef_names = list(all_coef_results.keys()) + ["routed", "averaged"]
    abbrev = [c[:8] for c in coef_names]
    header = f"{'Koan':<8} {'BL':>3} " + " ".join(f"{a:>8}" for a in abbrev)
    print(header)
    print("-" * len(header))

    for tid in koan_ids:
        bl = "P" if baseline_pass[tid] else "."
        cells = []
        for coef_id in list(all_coef_results.keys()):
            coef_pass_map = {r.task_id: r.passed for r in all_coef_results[coef_id].koan_results}
            was_pass = baseline_pass[tid]
            now_pass = coef_pass_map.get(tid, False)
            if was_pass and now_pass:
                cells.append("   P    ")  # stayed passing
            elif was_pass and not now_pass:
                cells.append("  -LOST ")  # REGRESSION
            elif not was_pass and now_pass:
                cells.append("  +GAIN ")  # IMPROVEMENT
            else:
                cells.append("   .    ")  # stayed failing

        # Routed
        was_pass = baseline_pass[tid]
        now_pass = routed_pass.get(tid, False)
        if was_pass and now_pass:
            cells.append("   P    ")
        elif was_pass and not now_pass:
            cells.append("  -LOST ")
        elif not was_pass and now_pass:
            cells.append("  +GAIN ")
        else:
            cells.append("   .    ")

        # Averaged
        now_pass = avg_pass.get(tid, False)
        if was_pass and now_pass:
            cells.append("   P    ")
        elif was_pass and not now_pass:
            cells.append("  -LOST ")
        elif not was_pass and now_pass:
            cells.append("  +GAIN ")
        else:
            cells.append("   .    ")

        print(f"{tid:<8} {bl:>3} " + " ".join(cells))

    # Summary row
    print("-" * len(header))
    bl_total = sum(1 for v in baseline_pass.values() if v)
    totals = [f"  {bl_total:>2}/30  "]
    for coef_id in list(all_coef_results.keys()):
        passed = all_coef_results[coef_id].passed
        totals.append(f"  {passed:>2}/30  ")
    totals.append(f"  {routed_results.passed:>2}/30  ")
    totals.append(f"  {averaged_results.passed:>2}/30  ")
    print(f"{'Total':<8} {'':>3} " + " ".join(totals))

    # =============================================
    # Save JSON results
    # =============================================
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "share_dir": str(share_dir),
        "baseline": {
            "pass_rate": baseline_results.pass_rate,
            "passed": baseline_results.passed,
            "total": baseline_results.total,
            "per_family": baseline_results.per_family,
            "per_koan": {r.task_id: r.passed for r in baseline_results.koan_results},
        },
        "routed": {
            "pass_rate": routed_results.pass_rate,
            "passed": routed_results.passed,
            "total": routed_results.total,
            "per_family": routed_results.per_family,
            "per_koan": {r.task_id: r.passed for r in routed_results.koan_results},
            "routing_map": {r.task_id: r.pattern_used for r in routed_results.koan_results},
        },
        "averaged": {
            "pass_rate": averaged_results.pass_rate,
            "passed": averaged_results.passed,
            "total": averaged_results.total,
            "per_family": averaged_results.per_family,
            "per_koan": {r.task_id: r.passed for r in averaged_results.koan_results},
        },
        "per_coefficient": {},
    }

    for coef_id, results in all_coef_results.items():
        coef_pass_map = {r.task_id: r.passed for r in results.koan_results}
        gains = [tid for tid in baseline_pass if not baseline_pass[tid] and coef_pass_map.get(tid, False)]
        regressions = [tid for tid in baseline_pass if baseline_pass[tid] and not coef_pass_map.get(tid, False)]
        output_data["per_coefficient"][coef_id] = {
            "pass_rate": results.pass_rate,
            "passed": results.passed,
            "total": results.total,
            "per_family": results.per_family,
            "per_koan": coef_pass_map,
            "gains": gains,
            "regressions": regressions,
            "net_delta": len(gains) - len(regressions),
        }

    with open(output_dir / "forgetting_analysis.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_dir / 'forgetting_analysis.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Measure forgetting across Share coefficient sets")
    parser.add_argument("--share", default="runs/share_phase2", help="Share directory with basis + coefficients")
    parser.add_argument("--output", default="runs/experiments/forgetting", help="Output directory")
    args = parser.parse_args()

    measure_forgetting(Path(args.share), Path(args.output))
