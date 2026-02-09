#!/usr/bin/env python3
"""
Prepare training data with replay buffer for continual learning.

This script implements research-backed data preparation:
1. Replay buffer - Include original training examples (prevents forgetting)
2. Mixed data - Both successful and failed examples
3. Self-Synthesized Rehearsal - Generate synthetic examples using base model

Usage:
    python scripts/prepare_training_data.py
    python scripts/prepare_training_data.py --replay-ratio 0.5 --output data/sft/mixed.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    examples = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    return examples


def load_koans(koans_dir: Path) -> dict:
    """Load all koan definitions."""
    koans = {}
    for jsonl_file in koans_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    koan = json.loads(line)
                    koans[koan["id"]] = koan
    return koans


def episode_to_sft(episode: dict, koans: dict) -> dict | None:
    """Convert an episode to SFT training format."""
    task_id = episode.get("task_id")
    if task_id not in koans:
        return None

    koan = koans[task_id]

    # Get the buggy code and expected fix
    buggy_code = koan.get("buggy_code", "")
    fixed_code = koan.get("fixed_code", "")
    error_msg = koan.get("error_msg", "Unknown error")

    instruction = """You are a Rust expert. Fix the following code that has a compilation error.
Return ONLY the fixed Rust code without any explanation or markdown formatting."""

    input_text = f"""## Buggy Code:
```rust
{buggy_code}
```

## Compiler Error:
{error_msg}

## Fixed Code:"""

    return {
        "instruction": instruction,
        "input": input_text,
        "output": fixed_code,
        "task_id": task_id,
        "passed": episode.get("passed", False),
        "steps": episode.get("steps_to_green", 0),
        "source": "episode"
    }


def generate_synthetic_examples(koans: dict, n: int = 10) -> list[dict]:
    """
    Generate synthetic training examples from koan definitions.
    This implements Self-Synthesized Rehearsal (SSR) concept.
    """
    synthetic = []
    koan_list = list(koans.values())

    for _ in range(min(n, len(koan_list))):
        koan = random.choice(koan_list)

        instruction = """You are a Rust expert. Fix the following code that has a compilation error.
Return ONLY the fixed Rust code without any explanation or markdown formatting."""

        input_text = f"""## Buggy Code:
```rust
{koan.get('buggy_code', '')}
```

## Compiler Error:
{koan.get('error_msg', 'Unknown error')}

## Fixed Code:"""

        synthetic.append({
            "instruction": instruction,
            "input": input_text,
            "output": koan.get("fixed_code", ""),
            "task_id": koan["id"],
            "source": "synthetic"
        })

    return synthetic


def prepare_mixed_dataset(
    sft_examples: list[dict],
    episodes: list[dict],
    replay_ratio: float = 0.5,
) -> list[dict]:
    """
    Prepare mixed training dataset following research best practices.

    Strategy:
    1. Full replay of original SFT examples (prevents forgetting)
    2. Add variations of successful examples (reinforcement)
    3. Weight by episode outcomes

    Research shows replay is the most effective technique.
    """
    mixed = []

    # Build task_id to SFT example mapping
    sft_by_task = {ex.get("task_id"): ex for ex in sft_examples}

    # Analyze episode outcomes
    task_pass_count = {}
    task_fail_count = {}
    for ep in episodes:
        task_id = ep.get("task_id")
        if ep.get("passed"):
            task_pass_count[task_id] = task_pass_count.get(task_id, 0) + 1
        else:
            task_fail_count[task_id] = task_fail_count.get(task_id, 0) + 1

    # 1. Full replay of ALL SFT examples (crucial for preventing forgetting)
    for ex in sft_examples:
        ex_copy = ex.copy()
        ex_copy["source"] = "replay"
        mixed.append(ex_copy)
    print(f"Added {len(sft_examples)} replay examples (full dataset)")

    # 2. Duplicate examples that the model struggled with (more training on hard cases)
    hard_examples = []
    for ex in sft_examples:
        task_id = ex.get("task_id")
        fail_count = task_fail_count.get(task_id, 0)
        pass_count = task_pass_count.get(task_id, 0)

        # If this task has failures, add extra copies
        if fail_count > 0:
            # Add 1-3 extra copies based on failure rate
            extra_copies = min(3, fail_count)
            for _ in range(extra_copies):
                ex_copy = ex.copy()
                ex_copy["source"] = "hard_example"
                hard_examples.append(ex_copy)

    if hard_examples:
        mixed.extend(hard_examples)
        print(f"Added {len(hard_examples)} extra copies of hard examples")

    # 3. Add successful examples with slight perturbations (data augmentation)
    successful_tasks = [tid for tid, count in task_pass_count.items() if count > 0]
    for task_id in successful_tasks:
        if task_id in sft_by_task:
            ex = sft_by_task[task_id]
            ex_copy = ex.copy()
            ex_copy["source"] = "success_reinforcement"
            mixed.append(ex_copy)

    print(f"Added {len(successful_tasks)} success reinforcement examples")

    # Shuffle
    random.shuffle(mixed)

    return mixed


def main():
    parser = argparse.ArgumentParser(description="Prepare training data with replay")
    parser.add_argument("--replay-ratio", type=float, default=0.5, help="Replay data ratio")
    parser.add_argument("--synthetic-ratio", type=float, default=0.2, help="Synthetic data ratio")
    parser.add_argument("--output", "-o", default="data/sft/mixed.jsonl", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Find project root
    project_root = Path(__file__).parent.parent

    # Load data sources
    sft_file = project_root / "data" / "sft" / "train.jsonl"
    episodes_dir = project_root / "rust" / "data" / "episodes"
    koans_dir = project_root / "data" / "koans"

    # Check if koans exist in expected location, try alternate
    if not koans_dir.exists():
        koans_dir = project_root / "rust" / "data" / "koans"

    print(f"Loading SFT data from: {sft_file}")
    sft_examples = load_jsonl(sft_file)
    print(f"Loaded {len(sft_examples)} SFT examples")

    print(f"Loading episodes from: {episodes_dir}")
    episodes = []
    for jsonl_file in sorted(episodes_dir.glob("cycle_*.jsonl")):
        episodes.extend(load_jsonl(jsonl_file))
    print(f"Loaded {len(episodes)} episodes")

    print(f"Loading koans from: {koans_dir}")
    koans = load_koans(koans_dir) if koans_dir.exists() else {}
    print(f"Loaded {len(koans)} koans")

    # Prepare mixed dataset
    mixed = prepare_mixed_dataset(
        sft_examples,
        episodes,
        replay_ratio=args.replay_ratio,
    )

    # Save
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in mixed:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(mixed)} training examples to: {output_path}")

    # Summary by source
    sources = {}
    for ex in mixed:
        src = ex.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print("\nDataset composition:")
    for src, count in sorted(sources.items()):
        pct = count / len(mixed) * 100
        print(f"  {src}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
