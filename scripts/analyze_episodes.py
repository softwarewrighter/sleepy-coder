#!/usr/bin/env python3
"""
Analyze episodes across cycles to find improvements and regressions.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_episodes(cycle_file: Path) -> dict[str, dict]:
    """Load episodes keyed by task_id."""
    episodes = {}
    with open(cycle_file) as f:
        for line in f:
            if line.strip():
                ep = json.loads(line)
                episodes[ep["task_id"]] = ep
    return episodes


def load_koans(koans_dir: Path) -> dict[str, dict]:
    """Load koan tasks."""
    koans = {}
    for koan_file in koans_dir.glob("*.json"):
        with open(koan_file) as f:
            koan = json.load(f)
            koans[koan["id"]] = koan
    return koans


def main():
    episodes_dir = Path("rust/data/episodes")
    koans_dir = Path("data/koans")

    # Load all cycles
    cycles = {}
    for cycle_file in sorted(episodes_dir.glob("cycle_*.jsonl")):
        cycle_num = int(cycle_file.stem.split("_")[1])
        cycles[cycle_num] = load_episodes(cycle_file)

    print(f"Loaded {len(cycles)} cycles")

    # Load koans for context
    koans = load_koans(koans_dir) if koans_dir.exists() else {}
    print(f"Loaded {len(koans)} koans")

    # Get all task IDs
    all_tasks = set()
    for cycle_eps in cycles.values():
        all_tasks.update(cycle_eps.keys())

    # Build pass/fail matrix
    print("\n" + "=" * 60)
    print("PASS/FAIL MATRIX BY TASK")
    print("=" * 60)
    print(f"{'Task':<12} | " + " | ".join(f"C{c}" for c in sorted(cycles.keys())))
    print("-" * 60)

    # Track changes
    improved = []  # Failed in C0, passed later
    regressed = []  # Passed in C0, failed later
    consistent_pass = []
    consistent_fail = []

    for task_id in sorted(all_tasks):
        results = []
        for cycle_num in sorted(cycles.keys()):
            if task_id in cycles[cycle_num]:
                passed = cycles[cycle_num][task_id]["passed"]
                results.append("✓" if passed else "✗")
            else:
                results.append("-")

        # Analyze trajectory
        c0_passed = cycles[0].get(task_id, {}).get("passed", None)
        latest_passed = cycles[max(cycles.keys())].get(task_id, {}).get("passed", None)

        if c0_passed == False and latest_passed == True:
            improved.append(task_id)
        elif c0_passed == True and latest_passed == False:
            regressed.append(task_id)
        elif c0_passed == True and latest_passed == True:
            consistent_pass.append(task_id)
        elif c0_passed == False and latest_passed == False:
            consistent_fail.append(task_id)

        print(f"{task_id:<12} | " + "  | ".join(results))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Improved (failed→passed):     {len(improved)}")
    print(f"Regressed (passed→failed):    {len(regressed)}")
    print(f"Consistent pass:              {len(consistent_pass)}")
    print(f"Consistent fail:              {len(consistent_fail)}")

    # Show specific examples
    if improved:
        print("\n" + "=" * 60)
        print("IMPROVED TASKS (now working that weren't before)")
        print("=" * 60)
        for task_id in improved[:5]:
            if task_id in koans:
                koan = koans[task_id]
                print(f"\n### {task_id}: {koan.get('name', 'N/A')}")
                print(f"Error family: {koan.get('error_family', 'N/A')}")
                print(f"Buggy code:")
                print("```rust")
                print(koan.get("buggy_code", "N/A")[:500])
                print("```")

    if regressed:
        print("\n" + "=" * 60)
        print("REGRESSED TASKS (were working, now broken)")
        print("=" * 60)
        for task_id in regressed[:5]:
            if task_id in koans:
                koan = koans[task_id]
                print(f"\n### {task_id}: {koan.get('name', 'N/A')}")
                print(f"Error family: {koan.get('error_family', 'N/A')}")

    if consistent_fail:
        print("\n" + "=" * 60)
        print("CONSISTENT FAILURES (never worked)")
        print("=" * 60)
        for task_id in consistent_fail[:5]:
            if task_id in koans:
                koan = koans[task_id]
                print(f"\n### {task_id}: {koan.get('name', 'N/A')}")
                print(f"Error family: {koan.get('error_family', 'N/A')}")


if __name__ == "__main__":
    main()
