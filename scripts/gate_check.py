#!/usr/bin/env python3
"""
Gate Check for Sleepy Coder Learning Loop

Validates that a new adapter doesn't regress before deployment.
This is the KEY safety mechanism for continual learning.

Usage:
    python gate_check.py --baseline rust/data/episodes/cycle_0.jsonl \
                         --candidate rust/data/episodes/cycle_N.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass


@dataclass
class GateResult:
    passed: bool
    overall_pass_rate: float
    baseline_pass_rate: float
    family_results: dict
    task_regressions: list
    task_improvements: list
    warnings: list
    blockers: list


def load_episodes(path: Path) -> dict:
    """Load episodes keyed by task_id."""
    episodes = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                ep = json.loads(line)
                episodes[ep["task_id"]] = ep
    return episodes


def get_family(task_id: str) -> str:
    """Get error family from task_id prefix."""
    if task_id.startswith('bc_'):
        return 'BorrowChecker'
    elif task_id.startswith('rh_'):
        return 'ResultHandling'
    elif task_id.startswith('tb_'):
        return 'TraitBounds'
    return 'Other'


def compute_pass_rate(episodes: dict) -> float:
    """Compute overall pass rate."""
    if not episodes:
        return 0.0
    passed = sum(1 for ep in episodes.values() if ep.get('passed', False))
    return passed / len(episodes) * 100


def compute_family_rates(episodes: dict) -> dict:
    """Compute pass rate by error family."""
    families = {}
    for task_id, ep in episodes.items():
        family = get_family(task_id)
        if family not in families:
            families[family] = {'passed': 0, 'total': 0}
        families[family]['total'] += 1
        if ep.get('passed', False):
            families[family]['passed'] += 1

    return {
        fam: data['passed'] / data['total'] * 100 if data['total'] > 0 else 0
        for fam, data in families.items()
    }


def run_gate_check(
    baseline_path: Path,
    candidate_path: Path,
    min_pass_rate: float = None,
    max_family_drop: float = 10.0,
    critical_tasks: list = None,
) -> GateResult:
    """
    Run gate check comparing candidate to baseline.

    Args:
        baseline_path: Path to baseline episodes
        candidate_path: Path to candidate episodes
        min_pass_rate: Minimum pass rate (default: baseline rate)
        max_family_drop: Maximum allowed drop per family (%)
        critical_tasks: Tasks that must not regress

    Returns:
        GateResult with pass/fail and details
    """
    baseline = load_episodes(baseline_path)
    candidate = load_episodes(candidate_path)

    baseline_rate = compute_pass_rate(baseline)
    candidate_rate = compute_pass_rate(candidate)

    if min_pass_rate is None:
        min_pass_rate = baseline_rate

    baseline_families = compute_family_rates(baseline)
    candidate_families = compute_family_rates(candidate)

    # Find regressions and improvements
    regressions = []
    improvements = []
    for task_id in baseline:
        base_passed = baseline[task_id].get('passed', False)
        cand_passed = candidate.get(task_id, {}).get('passed', False)

        if base_passed and not cand_passed:
            regressions.append(task_id)
        elif not base_passed and cand_passed:
            improvements.append(task_id)

    # Build warnings and blockers
    warnings = []
    blockers = []

    # Check overall pass rate
    if candidate_rate < min_pass_rate:
        blockers.append(
            f"Pass rate {candidate_rate:.1f}% below minimum {min_pass_rate:.1f}%"
        )

    # Check family drops
    family_results = {}
    for family in set(baseline_families) | set(candidate_families):
        base = baseline_families.get(family, 0)
        cand = candidate_families.get(family, 0)
        drop = base - cand
        family_results[family] = {'baseline': base, 'candidate': cand, 'drop': drop}

        if drop > max_family_drop:
            blockers.append(
                f"{family} dropped {drop:.1f}% ({base:.1f}% → {cand:.1f}%)"
            )
        elif drop > 0:
            warnings.append(
                f"{family} dropped {drop:.1f}% ({base:.1f}% → {cand:.1f}%)"
            )

    # Check critical tasks
    if critical_tasks:
        for task_id in critical_tasks:
            if task_id in regressions:
                blockers.append(f"Critical task {task_id} regressed")

    # Add regression warnings
    if regressions:
        warnings.append(f"{len(regressions)} task(s) regressed: {regressions[:5]}")

    # Determine pass/fail
    passed = len(blockers) == 0

    return GateResult(
        passed=passed,
        overall_pass_rate=candidate_rate,
        baseline_pass_rate=baseline_rate,
        family_results=family_results,
        task_regressions=regressions,
        task_improvements=improvements,
        warnings=warnings,
        blockers=blockers,
    )


def print_result(result: GateResult):
    """Print gate check result."""
    status = "PASSED" if result.passed else "BLOCKED"
    color = "\033[92m" if result.passed else "\033[91m"
    reset = "\033[0m"

    print()
    print("=" * 60)
    print(f"GATE CHECK: {color}{status}{reset}")
    print("=" * 60)
    print()
    print(f"Overall pass rate: {result.overall_pass_rate:.1f}% (baseline: {result.baseline_pass_rate:.1f}%)")
    print()

    print("Family Results:")
    for family, data in result.family_results.items():
        indicator = "↓" if data['drop'] > 0 else "↑" if data['drop'] < 0 else "="
        print(f"  {family}: {data['candidate']:.1f}% (was {data['baseline']:.1f}%) {indicator}")
    print()

    if result.task_improvements:
        print(f"Improvements ({len(result.task_improvements)}):")
        for task in result.task_improvements[:5]:
            print(f"  + {task}")
        print()

    if result.task_regressions:
        print(f"Regressions ({len(result.task_regressions)}):")
        for task in result.task_regressions[:5]:
            print(f"  - {task}")
        print()

    if result.blockers:
        print(f"{color}BLOCKERS:{reset}")
        for blocker in result.blockers:
            print(f"  ! {blocker}")
        print()

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  ? {warning}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Gate check for adapter deployment")
    parser.add_argument("--baseline", "-b", required=True, help="Baseline episodes file")
    parser.add_argument("--candidate", "-c", required=True, help="Candidate episodes file")
    parser.add_argument("--min-rate", type=float, help="Minimum pass rate (default: baseline)")
    parser.add_argument("--max-drop", type=float, default=10.0, help="Max family drop %")
    parser.add_argument("--critical", nargs="+", help="Critical tasks that must not regress")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = run_gate_check(
        baseline_path=Path(args.baseline),
        candidate_path=Path(args.candidate),
        min_pass_rate=args.min_rate,
        max_family_drop=args.max_drop,
        critical_tasks=args.critical,
    )

    if args.json:
        print(json.dumps({
            'passed': result.passed,
            'overall_pass_rate': result.overall_pass_rate,
            'baseline_pass_rate': result.baseline_pass_rate,
            'blockers': result.blockers,
            'warnings': result.warnings,
            'improvements': result.task_improvements,
            'regressions': result.task_regressions,
        }, indent=2))
    else:
        print_result(result)

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
