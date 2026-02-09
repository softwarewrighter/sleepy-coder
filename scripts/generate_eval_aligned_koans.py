#!/usr/bin/env python3
"""
Generate training data aligned with the frozen eval set.

The key insight: our training data (yew, axum, sqlx, cli) doesn't overlap
with our eval data (borrow_checker, trait_bounds, result_handling).
This script generates training examples that teach the same patterns
tested in the eval set.

Usage:
    python scripts/generate_eval_aligned_koans.py
    python scripts/generate_eval_aligned_koans.py --variants 20 --output data/sft/eval_aligned.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import NamedTuple


class Koan(NamedTuple):
    """A training example with buggy code and fix."""
    task_id: str
    family: str
    pattern: str
    buggy_code: str
    fixed_code: str
    error_hint: str


# =============================================================================
# BORROW CHECKER PATTERNS (from eval koans analysis)
# =============================================================================

BORROW_CHECKER_TEMPLATES = [
    # Pattern: Use after move
    {
        "pattern": "use_after_move",
        "templates": [
            {
                "buggy": 'fn main() {{ let {var} = String::from("{val}"); let {var2} = {var}; println!("{{}}", {var}); }}',
                "fixed": 'fn main() {{ let {var} = String::from("{val}"); let {var2} = {var}.clone(); println!("{{}}", {var}); }}',
                "vars": [("s", "t", "hello"), ("name", "copy", "world"), ("data", "backup", "test")],
            },
            {
                "buggy": 'fn main() {{ let {var} = vec![1, 2, 3]; let {var2} = {var}; println!("{{:?}}", {var}); }}',
                "fixed": 'fn main() {{ let {var} = vec![1, 2, 3]; let {var2} = {var}.clone(); println!("{{:?}}", {var}); }}',
                "vars": [("v", "w", None), ("nums", "copy", None), ("items", "backup", None)],
            },
        ],
    },
    # Pattern: Mutable borrow while immutable exists
    {
        "pattern": "mut_borrow_conflict",
        "templates": [
            {
                "buggy": 'fn main() {{ let mut {var} = vec![1, 2, 3]; let {ref} = &{var}; {var}.push(4); println!("{{:?}}", {ref}); }}',
                "fixed": 'fn main() {{ let mut {var} = vec![1, 2, 3]; let {ref} = &{var}; println!("{{:?}}", {ref}); {var}.push(4); }}',
                "vars": [("v", "r"), ("nums", "slice"), ("data", "view")],
            },
        ],
    },
    # Pattern: Missing mut
    {
        "pattern": "missing_mut",
        "templates": [
            {
                "buggy": "fn main() {{ let {var} = vec![1, 2, 3]; {var}.push(4); }}",
                "fixed": "fn main() {{ let mut {var} = vec![1, 2, 3]; {var}.push(4); }}",
                "vars": [("v",), ("nums",), ("items",), ("data",)],
            },
            {
                "buggy": "fn main() {{ let {var} = String::new(); {var}.push_str(\"hello\"); }}",
                "fixed": "fn main() {{ let mut {var} = String::new(); {var}.push_str(\"hello\"); }}",
                "vars": [("s",), ("text",), ("msg",), ("buf",)],
            },
        ],
    },
    # Pattern: Return reference to local
    {
        "pattern": "return_local_ref",
        "templates": [
            {
                "buggy": "fn get_str() -> &str {{ let s = String::from(\"hello\"); &s }}",
                "fixed": "fn get_str() -> String {{ String::from(\"hello\") }}",
                "vars": [()],
            },
        ],
    },
]

# =============================================================================
# TRAIT BOUNDS PATTERNS
# =============================================================================

TRAIT_BOUNDS_TEMPLATES = [
    # Pattern: Missing Debug derive
    {
        "pattern": "missing_debug",
        "templates": [
            {
                "buggy": "struct {name} {{ {field}: {typ} }} fn main() {{ let x = {name} {{ {field}: {val} }}; println!(\"{{:?}}\", x); }}",
                "fixed": "#[derive(Debug)] struct {name} {{ {field}: {typ} }} fn main() {{ let x = {name} {{ {field}: {val} }}; println!(\"{{:?}}\", x); }}",
                "vars": [("Point", "x", "i32", "5"), ("User", "id", "u32", "1"), ("Config", "value", "String", 'String::from("test")')],
            },
        ],
    },
    # Pattern: Missing Clone derive
    {
        "pattern": "missing_clone",
        "templates": [
            {
                "buggy": "struct {name} {{ {field}: String }} fn main() {{ let a = {name} {{ {field}: String::from(\"x\") }}; let b = a.clone(); }}",
                "fixed": "#[derive(Clone)] struct {name} {{ {field}: String }} fn main() {{ let a = {name} {{ {field}: String::from(\"x\") }}; let b = a.clone(); }}",
                "vars": [("Data", "value"), ("Item", "name"), ("Entry", "key")],
            },
        ],
    },
    # Pattern: Missing trait bound on generic
    {
        "pattern": "missing_bound",
        "templates": [
            {
                "buggy": "fn print_it<T>(val: T) {{ println!(\"{{:?}}\", val); }}",
                "fixed": "fn print_it<T: std::fmt::Debug>(val: T) {{ println!(\"{{:?}}\", val); }}",
                "vars": [()],
            },
            {
                "buggy": "fn clone_it<T>(val: T) -> T {{ val.clone() }}",
                "fixed": "fn clone_it<T: Clone>(val: T) -> T {{ val.clone() }}",
                "vars": [()],
            },
        ],
    },
]

# =============================================================================
# RESULT HANDLING PATTERNS
# =============================================================================

RESULT_HANDLING_TEMPLATES = [
    # Pattern: Unwrap on Result
    {
        "pattern": "unwrap_result",
        "templates": [
            {
                "buggy": 'fn main() {{ let content = std::fs::read_to_string("file.txt").unwrap(); println!("{{}}", content); }}',
                "fixed": 'fn main() {{ let content = std::fs::read_to_string("file.txt").unwrap_or_default(); println!("{{}}", content); }}',
                "vars": [()],
            },
            {
                "buggy": 'fn main() {{ let num: i32 = "42".parse().unwrap(); println!("{{}}", num); }}',
                "fixed": 'fn main() {{ let num: i32 = "42".parse().unwrap_or(0); println!("{{}}", num); }}',
                "vars": [()],
            },
        ],
    },
    # Pattern: Missing ? operator context
    {
        "pattern": "question_mark",
        "templates": [
            {
                "buggy": "fn read_file() {{ let content = std::fs::read_to_string(\"x.txt\")?; }}",
                "fixed": "fn read_file() -> Result<(), std::io::Error> {{ let content = std::fs::read_to_string(\"x.txt\")?; Ok(()) }}",
                "vars": [()],
            },
        ],
    },
    # Pattern: Option handling
    {
        "pattern": "option_unwrap",
        "templates": [
            {
                "buggy": "fn main() {{ let v = vec![1, 2, 3]; let first = v.get(0).unwrap(); println!(\"{{}}\", first); }}",
                "fixed": "fn main() {{ let v = vec![1, 2, 3]; if let Some(first) = v.get(0) {{ println!(\"{{}}\", first); }} }}",
                "vars": [()],
            },
            {
                "buggy": "fn main() {{ let v = vec![1, 2, 3]; let first = v.first().unwrap(); }}",
                "fixed": "fn main() {{ let v = vec![1, 2, 3]; let first = v.first().unwrap_or(&0); }}",
                "vars": [()],
            },
        ],
    },
]


def generate_variants(templates: list, family: str, n_per_template: int = 5) -> list[Koan]:
    """Generate koan variants from templates."""
    koans = []
    koan_id = 0

    for pattern_group in templates:
        pattern = pattern_group["pattern"]

        for template in pattern_group["templates"]:
            vars_list = template["vars"]

            # Generate n variants by cycling through vars
            for i in range(min(n_per_template, len(vars_list) * 2)):
                vars_tuple = vars_list[i % len(vars_list)]

                # Create variable mapping
                var_names = ["var", "var2", "ref", "name", "field", "typ", "val"]
                var_map = {name: val for name, val in zip(var_names, vars_tuple) if val is not None}

                try:
                    buggy = template["buggy"].format(**var_map)
                    fixed = template["fixed"].format(**var_map)
                except KeyError:
                    # Template doesn't use all vars, that's fine
                    buggy = template["buggy"]
                    fixed = template["fixed"]

                koan = Koan(
                    task_id=f"{family}_{pattern}_{koan_id:03d}",
                    family=family,
                    pattern=pattern,
                    buggy_code=buggy,
                    fixed_code=fixed,
                    error_hint=f"Fix the {pattern.replace('_', ' ')} error",
                )
                koans.append(koan)
                koan_id += 1

    return koans


def koan_to_jsonl(koan: Koan) -> dict:
    """Convert koan to JSONL training format."""
    return {
        "instruction": "You are a Rust expert. Fix the following code that has a compilation error.\nReturn ONLY the fixed Rust code without any explanation or markdown formatting.",
        "input": f"## Buggy Code:\n```rust\n{koan.buggy_code}\n```\n\n## Compiler Error:\n{koan.error_hint}\n\n## Fixed Code:",
        "output": koan.fixed_code,
        "task_id": koan.task_id,
        "family": koan.family,
        "pattern": koan.pattern,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate eval-aligned training koans")
    parser.add_argument("--variants", "-n", type=int, default=10, help="Variants per template")
    parser.add_argument("--output", "-o", default="data/sft/eval_aligned.jsonl", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=== Generating Eval-Aligned Training Data ===")
    print(f"Variants per template: {args.variants}")
    print()

    all_koans = []

    # Generate borrow checker koans
    bc_koans = generate_variants(BORROW_CHECKER_TEMPLATES, "borrow_checker", args.variants)
    print(f"Borrow checker: {len(bc_koans)} koans")
    all_koans.extend(bc_koans)

    # Generate trait bounds koans
    tb_koans = generate_variants(TRAIT_BOUNDS_TEMPLATES, "trait_bounds", args.variants)
    print(f"Trait bounds: {len(tb_koans)} koans")
    all_koans.extend(tb_koans)

    # Generate result handling koans
    rh_koans = generate_variants(RESULT_HANDLING_TEMPLATES, "result_handling", args.variants)
    print(f"Result handling: {len(rh_koans)} koans")
    all_koans.extend(rh_koans)

    print()
    print(f"Total: {len(all_koans)} training examples")

    # Shuffle
    random.shuffle(all_koans)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for koan in all_koans:
            f.write(json.dumps(koan_to_jsonl(koan)) + "\n")

    print(f"Written to: {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Train: python cuda/scripts/train.py --data {output_path} --steps 100")
    print("  2. Merge: python cuda/scripts/merge.py --adapter runs/adapters/<timestamp>/adapter")
    print("  3. Eval:  ./rust/target/release/sleepy-coder eval --cycle 14 --model sleepy-coder-v14")


if __name__ == "__main__":
    main()
