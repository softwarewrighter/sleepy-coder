#!/usr/bin/env python3
"""
Experiment Orchestrator for Share Paper Experiments.

Experiment 1: Routing vs Averaging vs Baseline
Experiment 2: Coefficient-Only Training (if Exp 1 shows value)
Experiment 3: Sequential Learning Curve (if Exp 2 shows value)

Usage:
    python scripts/run_experiments.py experiment1
    python scripts/run_experiments.py experiment2 --pattern bc_003
    python scripts/run_experiments.py experiment3
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from share_inference import ShareInferenceEngine
from eval_koans import KoanEvaluator, EvalResults, load_frozen_koans, NAMED_COEFFICIENTS


SHARE_DIR = Path("runs/share_phase2")
RESULTS_DIR = Path("runs/experiments")


def save_results(results: EvalResults, output_dir: Path, name: str):
    """Save evaluation results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "strategy": results.strategy,
        "total": results.total,
        "passed": results.passed,
        "pass_rate": results.pass_rate,
        "per_family": results.per_family,
        "timestamp": datetime.now().isoformat(),
        "koans": [
            {
                "task_id": r.task_id,
                "family": r.family,
                "passed": r.passed,
                "pattern_used": r.pattern_used,
                "strategy": r.strategy,
                "elapsed_s": r.elapsed_s,
                "error_message": r.error_message[:500] if r.error_message else "",
            }
            for r in results.koan_results
        ],
    }
    output_path = output_dir / f"{name}.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {output_path}")


def experiment1():
    """Experiment 1: Routing vs Averaging vs Baseline.

    Three conditions evaluated on 30 frozen koans:
    - Baseline: Pure base model, no LoRA
    - Averaged: Average all 9 named coefficient sets
    - Routed: Detect error type, select matching coefficient
    """
    print("=" * 60)
    print("EXPERIMENT 1: Routing vs Averaging vs Baseline")
    print("=" * 60)

    output_dir = RESULTS_DIR / "routing_vs_averaging"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load koans
    koans = load_frozen_koans()
    print(f"\nLoaded {len(koans)} frozen eval koans")

    # Load model (one-time cost)
    print("\nLoading Share inference engine...")
    engine = ShareInferenceEngine(SHARE_DIR)
    evaluator = KoanEvaluator(engine)

    all_results = {}

    # Run all three conditions
    for strategy in ["baseline", "averaged", "routed"]:
        print(f"\n{'─' * 40}")
        print(f"Running: {strategy}")
        print(f"{'─' * 40}")

        results = evaluator.run_eval(koans, strategy=strategy)
        all_results[strategy] = results
        save_results(results, output_dir, strategy)

        print(f"\n{results.summary()}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<12} {'Pass Rate':>10} {'Passed':>8} {'Total':>7}")
    print("-" * 40)
    for strategy, results in all_results.items():
        print(f"{strategy:<12} {results.pass_rate:>9.1%} {results.passed:>8} {results.total:>7}")

    # Per-family comparison
    families = sorted(set().union(*(r.per_family.keys() for r in all_results.values())))
    print(f"\n{'Family':<20}", end="")
    for strategy in all_results:
        print(f" {strategy:>10}", end="")
    print()
    print("-" * (20 + 11 * len(all_results)))
    for fam in families:
        print(f"{fam:<20}", end="")
        for strategy, results in all_results.items():
            counts = results.per_family.get(fam, {"total": 0, "passed": 0})
            t = counts["total"]
            p = counts["passed"]
            rate = p / t if t > 0 else 0
            print(f" {rate:>9.0%}", end="")
        print()

    # Save combined results
    combined = {
        "experiment": "routing_vs_averaging",
        "timestamp": datetime.now().isoformat(),
        "comparison": {
            strategy: {
                "pass_rate": r.pass_rate,
                "passed": r.passed,
                "total": r.total,
                "per_family": r.per_family,
            }
            for strategy, r in all_results.items()
        },
        # Per-koan diff: which koans differ between strategies
        "routing_vs_baseline_diff": [
            {
                "task_id": routed.task_id,
                "family": routed.family,
                "baseline": base.passed,
                "routed": routed.passed,
                "pattern_used": routed.pattern_used,
            }
            for base, routed in zip(
                all_results["baseline"].koan_results,
                all_results["routed"].koan_results,
            )
            if base.passed != routed.passed
        ],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results saved to {output_dir / 'results.json'}")


def experiment2(pattern: str = "bc_003"):
    """Experiment 2: Coefficient-Only Training.

    Train a new coefficient set for a failing pattern, then re-evaluate
    all 30 koans to verify zero regressions.
    """
    print("=" * 60)
    print(f"EXPERIMENT 2: Coefficient-Only Training ({pattern})")
    print("=" * 60)
    print("\nNOT YET IMPLEMENTED")
    print("Requires: Experiment 1 shows routing > averaging")
    print("Will use CoefficientTrainer from share_complete.py")


def experiment3():
    """Experiment 3: Sequential Learning Curve.

    Train coefficients one at a time and track cumulative performance.
    """
    print("=" * 60)
    print("EXPERIMENT 3: Sequential Learning Curve")
    print("=" * 60)
    print("\nNOT YET IMPLEMENTED")
    print("Requires: Experiment 2 shows coefficient training works")
    print("Will train: bc_003 -> bc_005 -> bc_010 -> tb_002 -> ...")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/run_experiments.py experiment1")
        print("  python scripts/run_experiments.py experiment2 [--pattern PATTERN]")
        print("  python scripts/run_experiments.py experiment3")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "experiment1":
        experiment1()
    elif cmd == "experiment2":
        pattern = "bc_003"
        if "--pattern" in sys.argv:
            idx = sys.argv.index("--pattern")
            if idx + 1 < len(sys.argv):
                pattern = sys.argv[idx + 1]
        experiment2(pattern)
    elif cmd == "experiment3":
        experiment3()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
