#!/usr/bin/env python3
"""
Train separate LoRA adapters for each task family.

This implements the proper Share workflow:
1. Create distinct training sets per family
2. Train one adapter per family
3. Consolidate using Share
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sft"
ADAPTERS_DIR = PROJECT_ROOT / "runs" / "adapters"

# Task families based on existing koans
TASK_FAMILIES = {
    "borrow_checker": {
        "prefix": "bc_",
        "description": "Borrow checker errors (move, borrow, mutability)",
        "instruction": "You are a Rust borrow checker expert. Fix move semantics, borrowing conflicts, and mutability errors.",
    },
    "lifetimes": {
        "prefix": "lt_",
        "description": "Lifetime annotation errors",
        "instruction": "You are a Rust lifetime annotation expert. Fix missing or incorrect lifetime parameters.",
    },
    "trait_bounds": {
        "prefix": "tb_",
        "description": "Trait bound errors (Debug, Clone, Send, etc.)",
        "instruction": "You are a Rust trait system expert. Fix missing trait implementations and bounds.",
    },
    "result_handling": {
        "prefix": "rh_",
        "description": "Result/Option handling errors",
        "instruction": "You are a Rust error handling expert. Fix Result/Option usage, ? operator, and error propagation.",
    },
    "type_mismatch": {
        "prefix": "tm_",
        "description": "Type mismatch and conversion errors",
        "instruction": "You are a Rust type system expert. Fix type mismatches, conversions, and inference issues.",
    },
}


def load_koans() -> list[dict]:
    """Load koans from the existing train.jsonl."""
    koans = []
    if (DATA_DIR / "train.jsonl").exists():
        with open(DATA_DIR / "train.jsonl") as f:
            for line in f:
                if line.strip():
                    koans.append(json.loads(line))
    return koans


def generate_family_data(family_name: str, koans: list[dict]) -> list[dict]:
    """Generate training data for a specific family."""
    family = TASK_FAMILIES[family_name]
    prefix = family["prefix"]
    instruction = family["instruction"]

    # Filter koans for this family
    family_koans = [k for k in koans if k.get("task_id", "").startswith(prefix)]

    # Create training examples
    examples = []
    for koan in family_koans:
        # Use family-specific instruction
        example = {
            "instruction": instruction + "\nReturn ONLY the fixed Rust code without any explanation.",
            "input": koan.get("input", ""),
            "output": koan.get("output", ""),
            "task_id": koan.get("task_id", ""),
            "family": family_name,
        }
        examples.append(example)

    return examples


def expand_family_data(examples: list[dict], target_count: int = 20) -> list[dict]:
    """Expand family data by creating variations."""
    if len(examples) >= target_count:
        return examples

    expanded = examples.copy()

    # Duplicate with slight variations to reach target
    while len(expanded) < target_count:
        for ex in examples:
            if len(expanded) >= target_count:
                break
            # Create a variation (slightly different instruction phrasing)
            variation = ex.copy()
            variation["instruction"] = variation["instruction"].replace(
                "Return ONLY",
                "Please return only"
            )
            expanded.append(variation)

    return expanded[:target_count]


def save_family_data(family_name: str, examples: list[dict]):
    """Save family training data."""
    output_file = DATA_DIR / f"family_{family_name}.jsonl"
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples to {output_file}")
    return output_file


def train_adapter(family_name: str, data_file: Path, steps: int = 50) -> Path:
    """Train a LoRA adapter for a family."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ADAPTERS_DIR / f"family_{family_name}" / timestamp

    print(f"\n=== Training adapter for {family_name} ===")
    print(f"Data: {data_file}")
    print(f"Steps: {steps}")
    print(f"Output: {output_dir}")

    # Run training
    cmd = [
        "python", str(PROJECT_ROOT / "cuda" / "scripts" / "train.py"),
        "--data", str(data_file),
        "--steps", str(steps),
        "--output", str(output_dir),
        "--lr", "5e-5",  # Lower LR for minimal training
    ]

    result = subprocess.run(cmd, cwd=PROJECT_ROOT / "cuda")
    if result.returncode != 0:
        print(f"ERROR: Training failed for {family_name}")
        return None

    return output_dir / "adapter"


def main():
    parser = argparse.ArgumentParser(description="Train adapters per task family")
    parser.add_argument("--families", "-f", nargs="+",
                        default=list(TASK_FAMILIES.keys()),
                        help="Families to train")
    parser.add_argument("--steps", "-s", type=int, default=50,
                        help="Training steps per family")
    parser.add_argument("--generate-only", "-g", action="store_true",
                        help="Only generate data, don't train")
    parser.add_argument("--target-examples", "-t", type=int, default=20,
                        help="Target examples per family")
    args = parser.parse_args()

    # Load existing koans
    koans = load_koans()
    print(f"Loaded {len(koans)} koans from train.jsonl")

    # Generate and optionally train for each family
    adapter_paths = []

    for family_name in args.families:
        if family_name not in TASK_FAMILIES:
            print(f"Unknown family: {family_name}")
            continue

        print(f"\n{'='*50}")
        print(f"Family: {family_name}")
        print(f"{'='*50}")

        # Generate training data
        examples = generate_family_data(family_name, koans)
        print(f"Found {len(examples)} koans for {family_name}")

        if len(examples) == 0:
            # Generate synthetic examples for this family
            print(f"No koans found - generating synthetic examples")
            examples = generate_synthetic_examples(family_name)

        # Expand to target count
        examples = expand_family_data(examples, args.target_examples)
        print(f"Expanded to {len(examples)} examples")

        # Save
        data_file = save_family_data(family_name, examples)

        if not args.generate_only:
            # Train adapter
            adapter_path = train_adapter(family_name, data_file, args.steps)
            if adapter_path:
                adapter_paths.append(adapter_path)

    if adapter_paths:
        print(f"\n{'='*50}")
        print("Trained adapters:")
        for p in adapter_paths:
            print(f"  {p}")
        print(f"\nNext: Consolidate with Share:")
        print(f"  python scripts/share_proper.py consolidate -a {' '.join(str(p) for p in adapter_paths)} -o runs/share_families")


def generate_synthetic_examples(family_name: str) -> list[dict]:
    """Generate synthetic examples for a family."""
    family = TASK_FAMILIES[family_name]
    instruction = family["instruction"]

    examples = []

    if family_name == "borrow_checker":
        patterns = [
            ('fn main() { let s = String::from("hello"); let t = s; println!("{}", s); }',
             'fn main() { let s = String::from("hello"); let t = s.clone(); println!("{}", s); }'),
            ('fn main() { let mut v = vec![1, 2, 3]; let first = &v[0]; v.push(4); println!("{}", first); }',
             'fn main() { let mut v = vec![1, 2, 3]; v.push(4); let first = &v[0]; println!("{}", first); }'),
            ('fn get_str() -> &str { let s = String::from("hello"); &s }',
             'fn get_str() -> String { String::from("hello") }'),
        ]
    elif family_name == "lifetimes":
        patterns = [
            ("fn longest(x: &str, y: &str) -> &str { if x.len() > y.len() { x } else { y } }",
             "fn longest<'a>(x: &'a str, y: &'a str) -> &'a str { if x.len() > y.len() { x } else { y } }"),
            ("struct Ref { data: &i32 }",
             "struct Ref<'a> { data: &'a i32 }"),
        ]
    elif family_name == "trait_bounds":
        patterns = [
            ("struct Point { x: i32, y: i32 } fn main() { let p = Point { x: 1, y: 2 }; println!(\"{:?}\", p); }",
             "#[derive(Debug)] struct Point { x: i32, y: i32 } fn main() { let p = Point { x: 1, y: 2 }; println!(\"{:?}\", p); }"),
            ("fn clone_it<T>(x: T) -> T { x.clone() }",
             "fn clone_it<T: Clone>(x: T) -> T { x.clone() }"),
        ]
    elif family_name == "result_handling":
        patterns = [
            ("fn parse(s: &str) -> i32 { s.parse()? }",
             "fn parse(s: &str) -> Result<i32, std::num::ParseIntError> { Ok(s.parse()?) }"),
            ("fn get(opt: Option<i32>) -> i32 { opt.unwrap() }",
             "fn get(opt: Option<i32>) -> i32 { opt.unwrap_or(0) }"),
        ]
    elif family_name == "type_mismatch":
        patterns = [
            ('fn greet(name: String) { } fn main() { greet("Alice"); }',
             'fn greet(name: &str) { } fn main() { greet("Alice"); }'),
            ("fn sum() -> i32 { vec![1, 2, 3].iter().sum() }",
             "fn sum() -> i32 { vec![1, 2, 3].iter().sum::<i32>() }"),
        ]
    else:
        patterns = []

    for i, (buggy, fixed) in enumerate(patterns):
        examples.append({
            "instruction": instruction + "\nReturn ONLY the fixed Rust code.",
            "input": f"## Buggy Code:\n```rust\n{buggy}\n```\n\n## Compiler Error:\nUnknown error\n\n## Fixed Code:",
            "output": fixed,
            "task_id": f"{family['prefix']}syn_{i:03d}",
            "family": family_name,
        })

    return examples


if __name__ == "__main__":
    main()
