#!/usr/bin/env python3
"""
Train 50 pattern-specific LoRA adapters for Share consolidation.

This script:
1. Splits eval-aligned data by pattern (51 unique patterns)
2. Trains one adapter per pattern (minimal steps, small r)
3. Records adapter paths for Share consolidation

Usage:
    python scripts/train_50_adapters.py --steps 30 --batch-size 4
    python scripts/train_50_adapters.py --patterns-only  # just split data, no training
    python scripts/train_50_adapters.py --consolidate   # run Share after training
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sft"
ADAPTERS_DIR = PROJECT_ROOT / "runs" / "adapters" / "pattern_adapters"


def load_eval_aligned_data() -> list[dict]:
    """Load the eval-aligned training data."""
    data_file = DATA_DIR / "eval_aligned.jsonl"
    if not data_file.exists():
        print(f"ERROR: {data_file} not found. Run generate_eval_aligned_koans.py first.")
        sys.exit(1)

    examples = []
    with open(data_file) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def split_by_pattern(examples: list[dict]) -> dict[str, list[dict]]:
    """Split examples by pattern."""
    by_pattern = defaultdict(list)
    for ex in examples:
        pattern = ex.get("pattern", "unknown")
        by_pattern[pattern].append(ex)
    return dict(by_pattern)


def save_pattern_data(pattern: str, examples: list[dict]) -> Path:
    """Save pattern-specific training data."""
    patterns_dir = DATA_DIR / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)

    output_file = patterns_dir / f"{pattern}.jsonl"
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return output_file


def train_single_adapter(pattern: str, data_file: Path, steps: int, lr: float, lora_r: int, skip_existing: bool = False) -> Path | None:
    """Train a single adapter for a pattern."""
    # Check if already trained
    pattern_dir = ADAPTERS_DIR / pattern
    if skip_existing and pattern_dir.exists():
        existing = list(pattern_dir.glob("*/adapter"))
        if existing:
            return existing[-1]  # Return latest

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ADAPTERS_DIR / pattern / timestamp

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "cuda" / "scripts" / "train.py"),
        "--data", str(data_file),
        "--steps", str(steps),
        "--output", str(output_dir),
        "--lr", str(lr),
        "--lora-r", str(lora_r),
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT / "cuda",
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ERROR training {pattern}: {result.stderr[:200]}")
        return None

    adapter_path = output_dir / "adapter"
    if adapter_path.exists():
        return adapter_path
    return None


def train_adapters_sequential(
    patterns_data: dict[str, list[dict]],
    steps: int,
    lr: float,
    lora_r: int,
    min_examples: int = 3,
    skip_existing: bool = False,
) -> list[Path]:
    """Train adapters sequentially."""
    adapter_paths = []

    # Filter patterns with enough examples
    valid_patterns = {p: d for p, d in patterns_data.items() if len(d) >= min_examples}
    print(f"\nTraining {len(valid_patterns)} adapters (min {min_examples} examples)...")
    print(f"Settings: steps={steps}, lr={lr}, lora_r={lora_r}, skip_existing={skip_existing}")
    print()

    for i, (pattern, examples) in enumerate(sorted(valid_patterns.items()), 1):
        print(f"[{i}/{len(valid_patterns)}] {pattern} ({len(examples)} examples)...", end=" ", flush=True)

        data_file = save_pattern_data(pattern, examples)
        adapter_path = train_single_adapter(pattern, data_file, steps, lr, lora_r, skip_existing)

        if adapter_path:
            adapter_paths.append(adapter_path)
            if skip_existing and (ADAPTERS_DIR / pattern).exists():
                print("CACHED")
            else:
                print("OK")
        else:
            print("FAILED")

    return adapter_paths


def train_adapters_parallel(
    patterns_data: dict[str, list[dict]],
    steps: int,
    lr: float,
    lora_r: int,
    min_examples: int = 3,
    max_workers: int = 2,
    skip_existing: bool = False,
) -> list[Path]:
    """Train adapters in parallel (2 at a time to share GPU)."""
    adapter_paths = []

    # Filter patterns with enough examples
    valid_patterns = {p: d for p, d in patterns_data.items() if len(d) >= min_examples}
    print(f"\nTraining {len(valid_patterns)} adapters in PARALLEL (max_workers={max_workers})...")
    print(f"Settings: steps={steps}, lr={lr}, lora_r={lora_r}, skip_existing={skip_existing}")
    print()

    # Save all data files first
    pattern_files = {}
    for pattern, examples in valid_patterns.items():
        pattern_files[pattern] = save_pattern_data(pattern, examples)

    # Train in parallel batches
    patterns_list = list(sorted(pattern_files.items()))
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for pattern, data_file in patterns_list:
            future = executor.submit(
                train_single_adapter, pattern, data_file, steps, lr, lora_r, skip_existing
            )
            futures[future] = pattern

        for future in as_completed(futures):
            pattern = futures[future]
            completed += 1
            try:
                adapter_path = future.result()
                if adapter_path:
                    adapter_paths.append(adapter_path)
                    print(f"[{completed}/{len(patterns_list)}] {pattern}: OK")
                else:
                    failed += 1
                    print(f"[{completed}/{len(patterns_list)}] {pattern}: FAILED")
            except Exception as e:
                failed += 1
                print(f"[{completed}/{len(patterns_list)}] {pattern}: ERROR - {e}")

    print(f"\nCompleted: {len(adapter_paths)}, Failed: {failed}")
    return adapter_paths


def run_share_consolidation(adapter_paths: list[Path], output_dir: Path, k: int = 10, p: int = 2):
    """Run Share consolidation on all adapters."""
    print(f"\n=== Running Share Consolidation ===")
    print(f"Adapters: {len(adapter_paths)}")
    print(f"Output: {output_dir}")
    print(f"k={k}, p={p}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "share_proper.py"),
        "consolidate",
        "-a", *[str(p) for p in adapter_paths],
        "-o", str(output_dir),
        "-k", str(k),
        "-p", str(p),
    ]

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Train 50 pattern-specific adapters")
    parser.add_argument("--steps", "-s", type=int, default=30, help="Training steps per adapter")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--min-examples", type=int, default=3, help="Min examples per pattern")
    parser.add_argument("--patterns-only", action="store_true", help="Only split data, don't train")
    parser.add_argument("--consolidate", action="store_true", help="Run Share after training")
    parser.add_argument("--k", type=int, default=10, help="Share k parameter")
    parser.add_argument("--p", type=int, default=2, help="Share p parameter")
    parser.add_argument("--parallel", "-j", type=int, default=1, help="Number of parallel training jobs")
    parser.add_argument("--skip-existing", action="store_true", help="Skip patterns already trained")
    args = parser.parse_args()

    # Load and split data
    print("=== Loading eval-aligned data ===")
    examples = load_eval_aligned_data()
    print(f"Loaded {len(examples)} examples")

    patterns_data = split_by_pattern(examples)
    print(f"Split into {len(patterns_data)} patterns")

    # Show pattern distribution
    print("\nPattern distribution:")
    for pattern, exs in sorted(patterns_data.items(), key=lambda x: -len(x[1])):
        marker = "✓" if len(exs) >= args.min_examples else "✗"
        print(f"  {marker} {pattern}: {len(exs)}")

    valid_count = sum(1 for exs in patterns_data.values() if len(exs) >= args.min_examples)
    print(f"\nValid patterns (>={args.min_examples} examples): {valid_count}/{len(patterns_data)}")

    if args.patterns_only:
        # Just save the pattern files
        print("\n=== Saving pattern data files ===")
        for pattern, exs in patterns_data.items():
            if len(exs) >= args.min_examples:
                path = save_pattern_data(pattern, exs)
                print(f"  {pattern}: {path}")
        print("Done. Run without --patterns-only to train.")
        return

    # Train adapters
    if args.parallel > 1:
        adapter_paths = train_adapters_parallel(
            patterns_data,
            steps=args.steps,
            lr=args.lr,
            lora_r=args.lora_r,
            min_examples=args.min_examples,
            max_workers=args.parallel,
            skip_existing=args.skip_existing,
        )
    else:
        adapter_paths = train_adapters_sequential(
            patterns_data,
            steps=args.steps,
            lr=args.lr,
            lora_r=args.lora_r,
            min_examples=args.min_examples,
            skip_existing=args.skip_existing,
        )

    print(f"\n=== Training Complete ===")
    print(f"Trained: {len(adapter_paths)} adapters")

    # Save adapter manifest
    manifest_file = ADAPTERS_DIR / "manifest.json"
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "adapters": [str(p) for p in adapter_paths],
        "settings": {
            "steps": args.steps,
            "lr": args.lr,
            "lora_r": args.lora_r,
        }
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_file}")

    # Consolidate if requested
    if args.consolidate and len(adapter_paths) >= 2:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        share_output = PROJECT_ROOT / "runs" / "share" / f"pattern_share_{timestamp}"
        success = run_share_consolidation(adapter_paths, share_output, args.k, args.p)
        if success:
            print(f"\nShare output: {share_output}")
        else:
            print("\nShare consolidation failed!")
    elif args.consolidate:
        print("\nSkipping Share: need at least 2 adapters")
    else:
        print(f"\nTo consolidate: python scripts/share_proper.py consolidate -a {' '.join(str(p) for p in adapter_paths[:5])} ...")


if __name__ == "__main__":
    main()
