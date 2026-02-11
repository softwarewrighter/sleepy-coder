#!/usr/bin/env python3
"""
LLM Distillation: Generate training data using Claude.

This script uses a large LLM (Claude) to generate diverse training examples
for Rust error patterns that the small model fails on. This is a form of
knowledge distillation - transferring Claude's Rust knowledge to Qwen.
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

try:
    import anthropic
except ImportError:
    print("Please install anthropic: uv pip install anthropic")
    sys.exit(1)

# The 8 persistent failure patterns with detailed descriptions
FAILURE_PATTERNS = {
    "mut_borrow_conflict": {
        "family": "borrow_checker",
        "description": "Mutable borrow while immutable borrow exists",
        "error_code": "E0502",
        "example_buggy": 'fn main() { let mut v = vec![1, 2, 3]; let first = &v[0]; v.push(4); println!("{}", first); }',
        "example_fixed": 'fn main() { let mut v = vec![1, 2, 3]; let first = v[0]; v.push(4); println!("{}", first); }',
        "fix_strategy": "Copy the value instead of borrowing, or restructure to not have overlapping borrows",
    },
    "double_mut_borrow": {
        "family": "borrow_checker",
        "description": "Two mutable borrows of the same value at once",
        "error_code": "E0499",
        "example_buggy": 'fn main() { let mut s = String::from("hello"); let r1 = &mut s; let r2 = &mut s; println!("{} {}", r1, r2); }',
        "example_fixed": 'fn main() { let mut s = String::from("hello"); let r1 = &mut s; println!("{}", r1); let r2 = &mut s; println!("{}", r2); }',
        "fix_strategy": "Use borrows sequentially, not simultaneously",
    },
    "return_local_ref": {
        "family": "borrow_checker",
        "description": "Returning a reference to a local variable",
        "error_code": "E0515",
        "example_buggy": 'fn get_str() -> &str { let s = String::from("hello"); &s }',
        "example_fixed": 'fn get_str() -> String { String::from("hello") }',
        "fix_strategy": "Return owned value instead of reference, or use 'static lifetime for string literals",
    },
    "option_ok_or": {
        "family": "result_handling",
        "description": "Converting Option to Result using ok_or",
        "error_code": "E0308",
        "example_buggy": 'fn parse_first(v: &[&str]) -> Result<i32, &str> { v.first().map(|s| s.parse().unwrap()) }',
        "example_fixed": 'fn parse_first(v: &[&str]) -> Result<i32, &str> { v.first().ok_or("empty")?.parse().map_err(|_| "parse error") }',
        "fix_strategy": "Use ok_or or ok_or_else to convert Option to Result",
    },
    "result_map_err": {
        "family": "result_handling",
        "description": "Using ? in main requires Result return type",
        "error_code": "E0277",
        "example_buggy": 'fn main() { let n: i32 = "42".parse()?; println!("{}", n); }',
        "example_fixed": 'fn main() -> Result<(), Box<dyn std::error::Error>> { let n: i32 = "42".parse()?; println!("{}", n); Ok(()) }',
        "fix_strategy": "Add Result return type to main, or use match/unwrap instead of ?",
    },
    "missing_clone": {
        "family": "trait_bounds",
        "description": "Calling .clone() on type that doesn't implement Clone",
        "error_code": "E0599",
        "example_buggy": 'struct Data { value: i32 } fn main() { let d = Data { value: 42 }; let d2 = d.clone(); }',
        "example_fixed": '#[derive(Clone)] struct Data { value: i32 } fn main() { let d = Data { value: 42 }; let d2 = d.clone(); }',
        "fix_strategy": "Add #[derive(Clone)] to the struct",
    },
    "missing_hash": {
        "family": "trait_bounds",
        "description": "Using type as HashMap key without Hash trait",
        "error_code": "E0277",
        "example_buggy": 'use std::collections::HashMap; struct Key { id: i32 } fn main() { let mut map = HashMap::new(); map.insert(Key { id: 1 }, "value"); }',
        "example_fixed": 'use std::collections::HashMap; #[derive(Hash, PartialEq, Eq)] struct Key { id: i32 } fn main() { let mut map = HashMap::new(); map.insert(Key { id: 1 }, "value"); }',
        "fix_strategy": "Add #[derive(Hash, PartialEq, Eq)] to the struct",
    },
    "missing_ord": {
        "family": "trait_bounds",
        "description": "Calling .sort() on Vec of type without Ord trait",
        "error_code": "E0277",
        "example_buggy": 'struct Score { value: i32 } fn main() { let mut scores = vec![Score { value: 3 }, Score { value: 1 }]; scores.sort(); }',
        "example_fixed": '#[derive(PartialEq, Eq, PartialOrd, Ord)] struct Score { value: i32 } fn main() { let mut scores = vec![Score { value: 3 }, Score { value: 1 }]; scores.sort(); }',
        "fix_strategy": "Add #[derive(PartialEq, Eq, PartialOrd, Ord)] to the struct",
    },
}


def generate_prompt(pattern_name: str, pattern_info: dict, index: int) -> str:
    """Generate a prompt for Claude to create a training example."""
    return f"""Generate a unique Rust code example that demonstrates the "{pattern_name}" error pattern.

## Pattern Description
- **Error**: {pattern_info['description']}
- **Error Code**: {pattern_info['error_code']}
- **Fix Strategy**: {pattern_info['fix_strategy']}

## Reference Example
Buggy: `{pattern_info['example_buggy']}`
Fixed: `{pattern_info['example_fixed']}`

## Requirements
1. Create a DIFFERENT example from the reference (variation #{index + 1})
2. Use realistic variable names and context (not just x, y, z)
3. The buggy code must trigger error {pattern_info['error_code']}
4. The fixed code must compile successfully
5. Keep it concise (under 10 lines ideally)
6. Make it practical - something a developer might actually write

## Output Format (JSON only, no markdown)
{{
    "buggy_code": "the Rust code with the error",
    "error_message": "the key part of the rustc error message",
    "fixed_code": "the corrected Rust code"
}}

Respond with ONLY the JSON object, no explanation or markdown."""


def generate_example(client: anthropic.Anthropic, pattern_name: str, pattern_info: dict, index: int) -> dict | None:
    """Generate a single training example using Claude."""
    prompt = generate_prompt(pattern_name, pattern_info, index)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract text from response
        text = response.content[0].text.strip()

        # Try to parse as JSON
        # Handle potential markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)

        # Validate required fields
        if not all(k in data for k in ["buggy_code", "error_message", "fixed_code"]):
            print(f"  Missing required fields in response")
            return None

        return data

    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def format_training_example(pattern_name: str, pattern_info: dict, example: dict, index: int) -> dict:
    """Format an example for training."""
    instruction = """You are a Rust expert. Fix the following code that has a compilation error.
Return ONLY the fixed Rust code without any explanation or markdown formatting."""

    input_text = f"""## Buggy Code:
```rust
{example['buggy_code']}
```

## Compiler Error:
{example['error_message']}

## Fixed Code:"""

    return {
        "instruction": instruction,
        "input": input_text,
        "output": example['fixed_code'],
        "task_id": f"{pattern_info['family']}_{pattern_name}_{index:03d}",
        "family": pattern_info['family'],
        "pattern": pattern_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate training data using Claude")
    parser.add_argument("--count", "-n", type=int, default=25, help="Examples per pattern")
    parser.add_argument("--output", "-o", type=str, default="data/sft/distilled", help="Output directory")
    parser.add_argument("--pattern", "-p", type=str, help="Only generate for specific pattern")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without calling API")
    args = parser.parse_args()

    # Check for API key
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize client
    client = None if args.dry_run else anthropic.Anthropic()

    # Select patterns to generate
    patterns = FAILURE_PATTERNS
    if args.pattern:
        if args.pattern not in patterns:
            print(f"Unknown pattern: {args.pattern}")
            print(f"Available: {list(patterns.keys())}")
            sys.exit(1)
        patterns = {args.pattern: patterns[args.pattern]}

    print(f"Generating {args.count} examples for {len(patterns)} patterns")
    print(f"Output directory: {output_dir}")
    print()

    total_generated = 0

    for pattern_name, pattern_info in patterns.items():
        print(f"=== {pattern_name} ===")

        output_file = output_dir / f"{pattern_name}.jsonl"
        examples = []

        for i in range(args.count):
            print(f"  Generating example {i + 1}/{args.count}...", end=" ")

            if args.dry_run:
                prompt = generate_prompt(pattern_name, pattern_info, i)
                print("(dry run)")
                if i == 0:
                    print(f"  Sample prompt:\n{prompt[:500]}...")
                continue

            example = generate_example(client, pattern_name, pattern_info, i)

            if example:
                formatted = format_training_example(pattern_name, pattern_info, example, i)
                examples.append(formatted)
                print("OK")
                total_generated += 1
            else:
                print("FAILED")

            # Rate limiting
            time.sleep(0.5)

        if not args.dry_run and examples:
            with open(output_file, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            print(f"  Saved {len(examples)} examples to {output_file}")

        print()

    if not args.dry_run:
        print(f"Total generated: {total_generated}")

        # Also create a combined file
        combined_file = output_dir / "all_distilled.jsonl"
        all_examples = []
        for pattern_name in patterns:
            pattern_file = output_dir / f"{pattern_name}.jsonl"
            if pattern_file.exists():
                with open(pattern_file) as f:
                    for line in f:
                        all_examples.append(json.loads(line))

        with open(combined_file, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + "\n")

        print(f"Combined file: {combined_file} ({len(all_examples)} examples)")


if __name__ == "__main__":
    main()
