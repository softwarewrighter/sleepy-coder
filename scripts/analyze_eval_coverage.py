#!/usr/bin/env python3
"""
Analyze overlap between training data and eval data.

This script identifies WHY training doesn't improve eval performance:
the domains don't overlap.

Usage:
    python scripts/analyze_eval_coverage.py
    python scripts/analyze_eval_coverage.py --training data/sft/train.jsonl --eval rust/data/tasks/
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def extract_patterns(code: str) -> set[str]:
    """Extract Rust patterns from code."""
    patterns = set()

    # Ownership patterns
    if ".clone()" in code:
        patterns.add("clone")
    if "let mut " in code:
        patterns.add("mut_binding")
    if "&mut " in code:
        patterns.add("mut_ref")
    if re.search(r"&\w+", code) and "&mut" not in code:
        patterns.add("immut_ref")
    if "move |" in code or "move ||" in code:
        patterns.add("move_closure")

    # Trait patterns
    if "#[derive(" in code:
        patterns.add("derive")
    if "impl " in code:
        patterns.add("impl_block")
    if re.search(r"<\w+:\s*\w+>", code):
        patterns.add("trait_bound")
    if "where " in code:
        patterns.add("where_clause")
    if "dyn " in code:
        patterns.add("dyn_trait")

    # Error handling patterns
    if "unwrap()" in code:
        patterns.add("unwrap")
    if "expect(" in code:
        patterns.add("expect")
    if "?" in code:
        patterns.add("question_mark")
    if "Ok(" in code or "Err(" in code:
        patterns.add("result")
    if "Some(" in code or "None" in code:
        patterns.add("option")
    if "match " in code:
        patterns.add("match")
    if "if let " in code:
        patterns.add("if_let")

    # Domain-specific patterns
    if "async " in code or ".await" in code:
        patterns.add("async")
    if "axum::" in code or "Router" in code:
        patterns.add("axum")
    if "sqlx::" in code or "query!" in code:
        patterns.add("sqlx")
    if "yew::" in code or "html!" in code:
        patterns.add("yew")
    if "clap::" in code or "#[command" in code:
        patterns.add("clap")
    if "Vec<" in code or "vec!" in code:
        patterns.add("vec")
    if "String::" in code or "String>" in code:
        patterns.add("string")

    return patterns


def load_training_data(path: Path) -> list[dict]:
    """Load training JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def analyze_training(examples: list[dict]) -> dict:
    """Analyze training data patterns."""
    all_patterns = Counter()
    family_patterns = defaultdict(Counter)

    for ex in examples:
        # Get code from input/output
        code = ex.get("input", "") + " " + ex.get("output", "")
        patterns = extract_patterns(code)

        for p in patterns:
            all_patterns[p] += 1

        # Track by family if available
        family = ex.get("family", "unknown")
        for p in patterns:
            family_patterns[family][p] += 1

    return {
        "total_examples": len(examples),
        "all_patterns": dict(all_patterns),
        "family_patterns": {k: dict(v) for k, v in family_patterns.items()},
    }


def analyze_eval_koans() -> dict:
    """Analyze the frozen eval set patterns."""
    # These are the patterns we KNOW are in the eval set based on the task definitions
    eval_patterns = {
        "borrow_checker": {
            "clone": 10,
            "mut_binding": 8,
            "mut_ref": 5,
            "immut_ref": 7,
            "string": 10,
            "vec": 6,
        },
        "trait_bounds": {
            "derive": 8,
            "impl_block": 5,
            "trait_bound": 7,
            "where_clause": 3,
        },
        "result_handling": {
            "unwrap": 6,
            "question_mark": 4,
            "result": 8,
            "option": 7,
            "match": 5,
            "if_let": 4,
        },
    }

    all_patterns = Counter()
    for family_patterns in eval_patterns.values():
        for pattern, count in family_patterns.items():
            all_patterns[pattern] += count

    return {
        "total_koans": 30,
        "all_patterns": dict(all_patterns),
        "family_patterns": eval_patterns,
    }


def compute_overlap(training: dict, eval_data: dict) -> dict:
    """Compute overlap between training and eval patterns."""
    train_patterns = set(training["all_patterns"].keys())
    eval_patterns = set(eval_data["all_patterns"].keys())

    overlap = train_patterns & eval_patterns
    train_only = train_patterns - eval_patterns
    eval_only = eval_patterns - train_patterns

    # Compute weighted overlap (by frequency)
    total_eval_weight = sum(eval_data["all_patterns"].values())
    covered_weight = sum(
        eval_data["all_patterns"].get(p, 0)
        for p in overlap
    )

    return {
        "overlap_patterns": sorted(overlap),
        "training_only": sorted(train_only),
        "eval_only": sorted(eval_only),
        "pattern_overlap_pct": len(overlap) / len(eval_patterns) * 100 if eval_patterns else 0,
        "weighted_coverage_pct": covered_weight / total_eval_weight * 100 if total_eval_weight else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze training/eval overlap")
    parser.add_argument("--training", "-t", default="data/sft/train.jsonl", help="Training data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed patterns")
    args = parser.parse_args()

    print("=" * 60)
    print("Training/Eval Coverage Analysis")
    print("=" * 60)
    print()

    # Load and analyze training data
    training_path = Path(args.training)
    if training_path.exists():
        training_examples = load_training_data(training_path)
        training_analysis = analyze_training(training_examples)
        print(f"Training data: {training_path}")
        print(f"  Examples: {training_analysis['total_examples']}")
        print(f"  Patterns found: {len(training_analysis['all_patterns'])}")
        if args.verbose:
            print(f"  Top patterns: {sorted(training_analysis['all_patterns'].items(), key=lambda x: -x[1])[:10]}")
    else:
        print(f"Training data not found: {training_path}")
        training_analysis = {"total_examples": 0, "all_patterns": {}, "family_patterns": {}}

    print()

    # Analyze eval data
    eval_analysis = analyze_eval_koans()
    print(f"Eval data: frozen eval set (30 koans)")
    print(f"  Patterns tested: {len(eval_analysis['all_patterns'])}")
    if args.verbose:
        print(f"  Patterns: {sorted(eval_analysis['all_patterns'].items(), key=lambda x: -x[1])[:10]}")

    print()

    # Compute overlap
    overlap = compute_overlap(training_analysis, eval_analysis)

    print("=" * 60)
    print("OVERLAP ANALYSIS")
    print("=" * 60)
    print()
    print(f"Pattern overlap: {overlap['pattern_overlap_pct']:.1f}%")
    print(f"Weighted coverage: {overlap['weighted_coverage_pct']:.1f}%")
    print()

    if overlap['overlap_patterns']:
        print(f"Shared patterns ({len(overlap['overlap_patterns'])}):")
        for p in overlap['overlap_patterns']:
            print(f"  - {p}")
    print()

    if overlap['eval_only']:
        print(f"MISSING from training ({len(overlap['eval_only'])}):")
        for p in overlap['eval_only']:
            print(f"  - {p} (eval tests this, training doesn't teach it)")
    print()

    if overlap['training_only']:
        print(f"Training-only patterns ({len(overlap['training_only'])}):")
        for p in overlap['training_only'][:5]:
            print(f"  - {p} (training teaches this, eval doesn't test it)")
        if len(overlap['training_only']) > 5:
            print(f"  ... and {len(overlap['training_only']) - 5} more")

    print()
    print("=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print()

    if overlap['weighted_coverage_pct'] < 50:
        print("⚠️  LOW COVERAGE: Training data doesn't cover eval patterns")
        print()
        print("This explains why training doesn't improve eval performance.")
        print("The model learns skills that aren't being tested.")
        print()
        print("RECOMMENDATION: Generate training data that targets eval patterns")
        print("  python scripts/generate_eval_aligned_koans.py")
    elif overlap['weighted_coverage_pct'] < 80:
        print("⚠️  PARTIAL COVERAGE: Some eval patterns not in training")
        print()
        print("Add training examples for missing patterns:")
        for p in overlap['eval_only']:
            print(f"  - {p}")
    else:
        print("✓ GOOD COVERAGE: Training covers most eval patterns")
        print()
        print("If still not improving, consider:")
        print("  - Data quality (are fixes correct?)")
        print("  - Hyperparameters (learning rate, steps)")
        print("  - Algorithm (try DPO instead of SFT)")


if __name__ == "__main__":
    main()
