#!/usr/bin/env python3
"""
Generate training data with replay buffer to prevent forgetting.

Strategy:
1. Include 44 targeted examples for 7 failure patterns
2. Add 20 replay examples from patterns the model PASSES (to prevent forgetting)
3. Total: ~64 examples with 70% novel (failures) and 30% replay (passing patterns)
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TARGETED_FILE = PROJECT_ROOT / "data" / "sft" / "targeted_failures.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "sft" / "targeted_with_replay.jsonl"

# Replay examples from patterns the baseline PASSES (to prevent forgetting)
REPLAY_EXAMPLES = [
    # bc_001: use_after_move_clone - PASSES
    {
        "pattern": "use_after_move_clone",
        "buggy": 'fn main() { let s = String::from("hello"); let t = s; println!("{}", s); }',
        "fixed": 'fn main() { let s = String::from("hello"); let t = s.clone(); println!("{}", s); }',
        "error": "borrow of moved value: `s`",
    },
    {
        "pattern": "use_after_move_clone",
        "buggy": 'fn main() { let v = vec![1, 2, 3]; let w = v; println!("{:?}", v); }',
        "fixed": 'fn main() { let v = vec![1, 2, 3]; let w = v.clone(); println!("{:?}", v); }',
        "error": "borrow of moved value: `v`",
    },
    # bc_004: missing_mut - PASSES
    {
        "pattern": "missing_mut",
        "buggy": "fn main() { let x = 5; x = 10; }",
        "fixed": "fn main() { let mut x = 5; x = 10; }",
        "error": "cannot assign twice to immutable variable",
    },
    {
        "pattern": "missing_mut",
        "buggy": "fn main() { let s = String::new(); s.push_str(\"hi\"); }",
        "fixed": "fn main() { let mut s = String::new(); s.push_str(\"hi\"); }",
        "error": "cannot borrow as mutable",
    },
    # tb_001: missing_debug - PASSES
    {
        "pattern": "missing_debug",
        "buggy": 'struct Point { x: i32, y: i32 } fn main() { let p = Point { x: 1, y: 2 }; println!("{:?}", p); }',
        "fixed": '#[derive(Debug)] struct Point { x: i32, y: i32 } fn main() { let p = Point { x: 1, y: 2 }; println!("{:?}", p); }',
        "error": "`Point` doesn't implement `Debug`",
    },
    {
        "pattern": "missing_debug",
        "buggy": 'struct User { name: String } fn main() { let u = User { name: "Alice".into() }; println!("{:?}", u); }',
        "fixed": '#[derive(Debug)] struct User { name: String } fn main() { let u = User { name: "Alice".into() }; println!("{:?}", u); }',
        "error": "`User` doesn't implement `Debug`",
    },
    # rh_002: option_unwrap_or - PASSES
    {
        "pattern": "option_unwrap_or",
        "buggy": "fn get(opt: Option<i32>) -> i32 { opt.unwrap() }",
        "fixed": "fn get(opt: Option<i32>) -> i32 { opt.unwrap_or(0) }",
        "error": "panics if None",
    },
    {
        "pattern": "option_unwrap_or",
        "buggy": 'fn name(opt: Option<String>) -> String { opt.unwrap() }',
        "fixed": 'fn name(opt: Option<String>) -> String { opt.unwrap_or_default() }',
        "error": "panics if None",
    },
    # rh_003: option_to_result_map_err - PASSES
    {
        "pattern": "option_to_result_map_err",
        "buggy": "fn convert(opt: Option<i32>) -> Result<i32, String> { Ok(opt?) }",
        "fixed": 'fn convert(opt: Option<i32>) -> Result<i32, String> { opt.ok_or("missing".to_string()) }',
        "error": "the `?` operator can only be used on `Result`s",
    },
    # tb_003: missing_copy - PASSES
    {
        "pattern": "missing_copy",
        "buggy": "struct Num { v: i32 } fn double(n: Num) -> i32 { n.v * 2 } fn main() { let n = Num { v: 5 }; println!(\"{} {}\", double(n), double(n)); }",
        "fixed": "#[derive(Clone, Copy)] struct Num { v: i32 } fn double(n: Num) -> i32 { n.v * 2 } fn main() { let n = Num { v: 5 }; println!(\"{} {}\", double(n), double(n)); }",
        "error": "use of moved value",
    },
    # tb_004: missing_generic_bound - PASSES
    {
        "pattern": "missing_generic_bound",
        "buggy": 'fn print_debug<T>(val: T) { println!("{:?}", val); }',
        "fixed": 'fn print_debug<T: std::fmt::Debug>(val: T) { println!("{:?}", val); }',
        "error": "doesn't implement `Debug`",
    },
    {
        "pattern": "missing_generic_bound",
        "buggy": "fn clone_it<T>(x: T) -> T { x.clone() }",
        "fixed": "fn clone_it<T: Clone>(x: T) -> T { x.clone() }",
        "error": "no method named `clone`",
    },
    # tb_005: missing_default - PASSES
    {
        "pattern": "missing_default",
        "buggy": "struct Config { timeout: u32 } fn main() { let c: Config = Default::default(); }",
        "fixed": "#[derive(Default)] struct Config { timeout: u32 } fn main() { let c: Config = Default::default(); }",
        "error": "doesn't implement `Default`",
    },
    # tb_006: missing_partial_eq - PASSES
    {
        "pattern": "missing_partial_eq",
        "buggy": 'struct Point { x: i32, y: i32 } fn main() { let p1 = Point { x: 1, y: 2 }; let p2 = Point { x: 1, y: 2 }; println!("{}", p1 == p2); }',
        "fixed": '#[derive(PartialEq)] struct Point { x: i32, y: i32 } fn main() { let p1 = Point { x: 1, y: 2 }; let p2 = Point { x: 1, y: 2 }; println!("{}", p1 == p2); }',
        "error": "binary operation `==` cannot be applied",
    },
    # rh_006: if_let_option - PASSES
    {
        "pattern": "if_let_option",
        "buggy": "fn check(opt: Option<i32>) { if opt.is_some() { let v = opt.unwrap(); println!(\"{}\", v); } }",
        "fixed": "fn check(opt: Option<i32>) { if let Some(v) = opt { println!(\"{}\", v); } }",
        "error": "use if let instead of unwrap",
    },
    # rh_007: match_result - PASSES
    {
        "pattern": "match_result",
        "buggy": 'fn handle(r: Result<i32, String>) { if r.is_ok() { println!("{}", r.unwrap()); } else { println!("err"); } }',
        "fixed": 'fn handle(r: Result<i32, String>) { match r { Ok(v) => println!("{}", v), Err(_) => println!("err") } }',
        "error": "use match instead of unwrap",
    },
    # rh_010: propagate_error - PASSES
    {
        "pattern": "propagate_error",
        "buggy": "fn parse(s: &str) -> Result<i32, std::num::ParseIntError> { let n = s.parse().unwrap(); Ok(n) }",
        "fixed": "fn parse(s: &str) -> Result<i32, std::num::ParseIntError> { let n = s.parse()?; Ok(n) }",
        "error": "use ? to propagate errors",
    },
]


def format_example(pattern: str, buggy: str, fixed: str, error: str, is_replay: bool = False) -> dict:
    """Format a single training example."""
    return {
        "instruction": "You are a Rust compiler error fixer. Fix the buggy code based on the error message. Return ONLY the fixed Rust code without explanation.",
        "input": f"## Buggy Code:\n```rust\n{buggy}\n```\n\n## Compiler Error:\n{error}\n\n## Fixed Code:",
        "output": fixed,
        "pattern": pattern,
        "family": "replay" if is_replay else "targeted_failure",
    }


def main():
    # Load targeted examples
    with open(TARGETED_FILE) as f:
        targeted = [json.loads(line) for line in f]

    print(f"Loaded {len(targeted)} targeted examples")

    # Add replay examples
    replay = []
    for ex in REPLAY_EXAMPLES:
        replay.append(format_example(
            pattern=ex["pattern"],
            buggy=ex["buggy"],
            fixed=ex["fixed"],
            error=ex["error"],
            is_replay=True,
        ))

    print(f"Created {len(replay)} replay examples")

    # Combine: targeted + replay
    all_examples = targeted + replay

    with open(OUTPUT_FILE, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nTotal: {len(all_examples)} examples")
    print(f"  Targeted (failures): {len(targeted)}")
    print(f"  Replay (passing): {len(replay)}")
    print(f"\nWritten to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
