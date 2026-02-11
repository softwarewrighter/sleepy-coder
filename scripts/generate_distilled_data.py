#!/usr/bin/env python3
"""
Generate diverse training data for the 8 failure patterns.

This script creates variations of the failure patterns using templates.
Each variation has different variable names, types, and contexts.
"""

import json
import random
from pathlib import Path
from typing import Iterator

# Variation components
STRUCT_NAMES = [
    "User", "Player", "Item", "Order", "Message", "Record", "Entry", "Node",
    "Config", "Settings", "State", "Event", "Task", "Job", "Request", "Response",
    "Account", "Product", "Customer", "Document", "Session", "Token", "Cache",
    "Buffer", "Queue", "Stack", "Counter", "Timer", "Logger", "Handler",
]

FIELD_NAMES = [
    "id", "name", "value", "count", "data", "info", "status", "level",
    "score", "price", "amount", "total", "index", "size", "length", "capacity",
    "timestamp", "version", "priority", "weight", "height", "width", "depth",
]

COLLECTION_NAMES = [
    "items", "values", "entries", "records", "elements", "nodes", "data",
    "results", "users", "orders", "messages", "events", "tasks", "logs",
]

STRING_VARS = [
    "name", "text", "content", "message", "label", "title", "description",
    "input", "output", "prefix", "suffix", "key", "path", "url", "query",
]

INT_FIELDS = ["id", "count", "value", "score", "level", "index", "size", "priority"]


def generate_mut_borrow_conflict() -> Iterator[dict]:
    """Generate variations of mutable borrow while immutable borrow exists."""

    # Pattern 1: Vec with index access
    for coll in COLLECTION_NAMES[:10]:
        for field in ["first", "last", "item", "element", "value"]:
            buggy = f'fn main() {{ let mut {coll} = vec![1, 2, 3, 4, 5]; let {field} = &{coll}[0]; {coll}.push(6); println!("{{}}", {field}); }}'
            fixed = f'fn main() {{ let mut {coll} = vec![1, 2, 3, 4, 5]; let {field} = {coll}[0]; {coll}.push(6); println!("{{}}", {field}); }}'
            yield {
                "buggy_code": buggy,
                "error_message": f"cannot borrow `{coll}` as mutable because it is also borrowed as immutable",
                "fixed_code": fixed,
            }

    # Pattern 2: Vec with len() check
    for coll in COLLECTION_NAMES[:8]:
        buggy = f'fn main() {{ let mut {coll} = vec![1, 2, 3]; let len = &{coll}.len(); {coll}.clear(); println!("{{}}", len); }}'
        fixed = f'fn main() {{ let mut {coll} = vec![1, 2, 3]; let len = {coll}.len(); {coll}.clear(); println!("{{}}", len); }}'
        yield {
            "buggy_code": buggy,
            "error_message": f"cannot borrow `{coll}` as mutable because it is also borrowed as immutable",
            "fixed_code": fixed,
        }

    # Pattern 3: HashMap with get and insert
    for i, key_name in enumerate(["key", "id", "name", "index"][:4]):
        buggy = f'use std::collections::HashMap; fn main() {{ let mut map = HashMap::new(); map.insert("{key_name}", 1); let val = map.get("{key_name}"); map.insert("new", 2); println!("{{:?}}", val); }}'
        fixed = f'use std::collections::HashMap; fn main() {{ let mut map = HashMap::new(); map.insert("{key_name}", 1); let val = map.get("{key_name}").copied(); map.insert("new", 2); println!("{{:?}}", val); }}'
        yield {
            "buggy_code": buggy,
            "error_message": "cannot borrow `map` as mutable because it is also borrowed as immutable",
            "fixed_code": fixed,
        }


def generate_double_mut_borrow() -> Iterator[dict]:
    """Generate variations of double mutable borrow."""

    # Pattern 1: String with two &mut
    for var in STRING_VARS[:10]:
        buggy = f'fn main() {{ let mut {var} = String::from("hello"); let r1 = &mut {var}; let r2 = &mut {var}; r1.push_str(" "); r2.push_str("world"); }}'
        fixed = f'fn main() {{ let mut {var} = String::from("hello"); {{ let r1 = &mut {var}; r1.push_str(" "); }} let r2 = &mut {var}; r2.push_str("world"); }}'
        yield {
            "buggy_code": buggy,
            "error_message": f"cannot borrow `{var}` as mutable more than once at a time",
            "fixed_code": fixed,
        }

    # Pattern 2: Vec with two mutable borrows
    for coll in COLLECTION_NAMES[:8]:
        buggy = f'fn main() {{ let mut {coll} = vec![1, 2, 3]; let a = &mut {coll}; let b = &mut {coll}; a.push(4); b.push(5); }}'
        fixed = f'fn main() {{ let mut {coll} = vec![1, 2, 3]; {coll}.push(4); {coll}.push(5); }}'
        yield {
            "buggy_code": buggy,
            "error_message": f"cannot borrow `{coll}` as mutable more than once at a time",
            "fixed_code": fixed,
        }

    # Pattern 3: Struct field double borrow
    for struct in STRUCT_NAMES[:6]:
        buggy = f'struct {struct} {{ data: Vec<i32> }} fn main() {{ let mut s = {struct} {{ data: vec![1] }}; let r1 = &mut s.data; let r2 = &mut s.data; r1.push(2); r2.push(3); }}'
        fixed = f'struct {struct} {{ data: Vec<i32> }} fn main() {{ let mut s = {struct} {{ data: vec![1] }}; s.data.push(2); s.data.push(3); }}'
        yield {
            "buggy_code": buggy,
            "error_message": "cannot borrow `s.data` as mutable more than once at a time",
            "fixed_code": fixed,
        }


def generate_return_local_ref() -> Iterator[dict]:
    """Generate variations of returning reference to local variable."""

    # Pattern 1: Return &str from String
    for func in ["get_name", "create_label", "make_title", "build_message", "format_text"]:
        buggy = f'fn {func}() -> &str {{ let s = String::from("hello"); &s }}'
        fixed = f'fn {func}() -> String {{ String::from("hello") }}'
        yield {
            "buggy_code": buggy,
            "error_message": "cannot return reference to local variable `s`",
            "fixed_code": fixed,
        }

    # Pattern 2: Return &str from format!
    for func in ["get_greeting", "make_message", "create_response", "build_output"]:
        for name in ["user", "item", "value", "data"]:
            buggy = f'fn {func}({name}: &str) -> &str {{ let result = format!("Hello, {{}}", {name}); &result }}'
            fixed = f'fn {func}({name}: &str) -> String {{ format!("Hello, {{}}", {name}) }}'
            yield {
                "buggy_code": buggy,
                "error_message": "cannot return reference to local variable `result`",
                "fixed_code": fixed,
            }

    # Pattern 3: Return reference to local vec element
    for func in ["first_item", "get_element", "find_value"]:
        buggy = f'fn {func}() -> &i32 {{ let v = vec![1, 2, 3]; &v[0] }}'
        fixed = f'fn {func}() -> i32 {{ let v = vec![1, 2, 3]; v[0] }}'
        yield {
            "buggy_code": buggy,
            "error_message": "cannot return reference to local variable `v`",
            "fixed_code": fixed,
        }


def generate_option_ok_or() -> Iterator[dict]:
    """Generate variations of Option to Result conversion."""

    # Pattern 1: first() without ok_or
    for coll in COLLECTION_NAMES[:8]:
        buggy = f'fn get_first({coll}: &[i32]) -> Result<i32, &str> {{ {coll}.first().copied() }}'
        fixed = f'fn get_first({coll}: &[i32]) -> Result<i32, &str> {{ {coll}.first().copied().ok_or("empty") }}'
        yield {
            "buggy_code": buggy,
            "error_message": "expected `Result<i32, &str>`, found `Option<i32>`",
            "fixed_code": fixed,
        }

    # Pattern 2: HashMap get without ok_or
    for key in ["id", "name", "key", "index"]:
        buggy = f'use std::collections::HashMap; fn lookup(map: &HashMap<String, i32>, {key}: &str) -> Result<i32, &str> {{ map.get({key}).copied() }}'
        fixed = f'use std::collections::HashMap; fn lookup(map: &HashMap<String, i32>, {key}: &str) -> Result<i32, &str> {{ map.get({key}).copied().ok_or("not found") }}'
        yield {
            "buggy_code": buggy,
            "error_message": "expected `Result<i32, &str>`, found `Option<i32>`",
            "fixed_code": fixed,
        }

    # Pattern 3: find() without ok_or
    for coll in COLLECTION_NAMES[:6]:
        buggy = f'fn find_even({coll}: &[i32]) -> Result<i32, &str> {{ {coll}.iter().find(|x| *x % 2 == 0).copied() }}'
        fixed = f'fn find_even({coll}: &[i32]) -> Result<i32, &str> {{ {coll}.iter().find(|x| *x % 2 == 0).copied().ok_or("no even number") }}'
        yield {
            "buggy_code": buggy,
            "error_message": "expected `Result<i32, &str>`, found `Option<i32>`",
            "fixed_code": fixed,
        }


def generate_result_map_err() -> Iterator[dict]:
    """Generate variations of ? in main without Result return type."""

    # Pattern 1: parse with ?
    for var in ["num", "value", "count", "size", "amount"]:
        buggy = f'fn main() {{ let {var}: i32 = "42".parse()?; println!("{{}}", {var}); }}'
        fixed = f'fn main() -> Result<(), Box<dyn std::error::Error>> {{ let {var}: i32 = "42".parse()?; println!("{{}}", {var}); Ok(()) }}'
        yield {
            "buggy_code": buggy,
            "error_message": "the `?` operator can only be used in a function that returns `Result`",
            "fixed_code": fixed,
        }

    # Pattern 2: file read with ?
    for name in ["content", "data", "text", "input"]:
        buggy = f'fn main() {{ let {name} = std::fs::read_to_string("file.txt")?; println!("{{}}", {name}); }}'
        fixed = f'fn main() -> Result<(), Box<dyn std::error::Error>> {{ let {name} = std::fs::read_to_string("file.txt")?; println!("{{}}", {name}); Ok(()) }}'
        yield {
            "buggy_code": buggy,
            "error_message": "the `?` operator can only be used in a function that returns `Result`",
            "fixed_code": fixed,
        }

    # Pattern 3: env var with ?
    for var in ["path", "home", "config", "setting"]:
        buggy = f'fn main() {{ let {var} = std::env::var("HOME")?; println!("{{}}", {var}); }}'
        fixed = f'fn main() -> Result<(), Box<dyn std::error::Error>> {{ let {var} = std::env::var("HOME")?; println!("{{}}", {var}); Ok(()) }}'
        yield {
            "buggy_code": buggy,
            "error_message": "the `?` operator can only be used in a function that returns `Result`",
            "fixed_code": fixed,
        }


def generate_missing_clone() -> Iterator[dict]:
    """Generate variations of missing Clone trait."""

    for struct in STRUCT_NAMES[:20]:
        for field in INT_FIELDS[:4]:
            buggy = f'struct {struct} {{ {field}: i32 }} fn main() {{ let a = {struct} {{ {field}: 42 }}; let b = a.clone(); println!("{{:?}}", b.{field}); }}'
            fixed = f'#[derive(Clone)] struct {struct} {{ {field}: i32 }} fn main() {{ let a = {struct} {{ {field}: 42 }}; let b = a.clone(); println!("{{:?}}", b.{field}); }}'
            yield {
                "buggy_code": buggy,
                "error_message": f"no method named `clone` found for struct `{struct}`",
                "fixed_code": fixed,
            }


def generate_missing_hash() -> Iterator[dict]:
    """Generate variations of missing Hash trait for HashMap key."""

    for struct in STRUCT_NAMES[:15]:
        for field in INT_FIELDS[:4]:
            buggy = f'use std::collections::HashMap; struct {struct} {{ {field}: i32 }} fn main() {{ let mut map = HashMap::new(); map.insert({struct} {{ {field}: 1 }}, "value"); }}'
            fixed = f'use std::collections::HashMap; #[derive(Hash, PartialEq, Eq)] struct {struct} {{ {field}: i32 }} fn main() {{ let mut map = HashMap::new(); map.insert({struct} {{ {field}: 1 }}, "value"); }}'
            yield {
                "buggy_code": buggy,
                "error_message": f"the trait `Hash` is not implemented for `{struct}`",
                "fixed_code": fixed,
            }


def generate_missing_ord() -> Iterator[dict]:
    """Generate variations of missing Ord trait for sorting."""

    for struct in STRUCT_NAMES[:15]:
        for field in INT_FIELDS[:4]:
            buggy = f'struct {struct} {{ {field}: i32 }} fn main() {{ let mut items = vec![{struct} {{ {field}: 3 }}, {struct} {{ {field}: 1 }}]; items.sort(); }}'
            fixed = f'#[derive(PartialEq, Eq, PartialOrd, Ord)] struct {struct} {{ {field}: i32 }} fn main() {{ let mut items = vec![{struct} {{ {field}: 3 }}, {struct} {{ {field}: 1 }}]; items.sort(); }}'
            yield {
                "buggy_code": buggy,
                "error_message": f"the trait `Ord` is not implemented for `{struct}`",
                "fixed_code": fixed,
            }


GENERATORS = {
    "mut_borrow_conflict": ("borrow_checker", generate_mut_borrow_conflict),
    "double_mut_borrow": ("borrow_checker", generate_double_mut_borrow),
    "return_local_ref": ("borrow_checker", generate_return_local_ref),
    "option_ok_or": ("result_handling", generate_option_ok_or),
    "result_map_err": ("result_handling", generate_result_map_err),
    "missing_clone": ("trait_bounds", generate_missing_clone),
    "missing_hash": ("trait_bounds", generate_missing_hash),
    "missing_ord": ("trait_bounds", generate_missing_ord),
}


def format_training_example(pattern_name: str, family: str, example: dict, index: int) -> dict:
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
        "task_id": f"{family}_{pattern_name}_{index:03d}",
        "family": family,
        "pattern": pattern_name,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate distilled training data")
    parser.add_argument("--count", "-n", type=int, default=25, help="Examples per pattern")
    parser.add_argument("--output", "-o", type=str, default="data/sft/distilled", help="Output directory")
    parser.add_argument("--pattern", "-p", type=str, help="Only generate for specific pattern")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = GENERATORS
    if args.pattern:
        if args.pattern not in patterns:
            print(f"Unknown pattern: {args.pattern}")
            print(f"Available: {list(patterns.keys())}")
            return
        patterns = {args.pattern: patterns[args.pattern]}

    print(f"Generating {args.count} examples for {len(patterns)} patterns")
    print(f"Output directory: {output_dir}")
    print()

    all_examples = []

    for pattern_name, (family, generator) in patterns.items():
        print(f"=== {pattern_name} ===")

        output_file = output_dir / f"{pattern_name}.jsonl"
        examples = []

        for i, example in enumerate(generator()):
            if i >= args.count:
                break
            formatted = format_training_example(pattern_name, family, example, i)
            examples.append(formatted)

        with open(output_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        print(f"  Generated {len(examples)} examples -> {output_file}")
        all_examples.extend(examples)

    # Combined file
    combined_file = output_dir / "all_distilled.jsonl"
    with open(combined_file, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print()
    print(f"Total: {len(all_examples)} examples")
    print(f"Combined: {combined_file}")


if __name__ == "__main__":
    main()
