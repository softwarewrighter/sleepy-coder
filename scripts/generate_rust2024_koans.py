#!/usr/bin/env python3
"""Generate training data for Rust 2024 edition features the base model doesn't know."""

import json
from pathlib import Path

# Features the base model (trained ~2023) likely doesn't know well:
# 1. Format string capture: println!("{x}") instead of println!("{}", x)
# 2. Let-chains: if let Some(x) = opt && x > 0
# 3. Let-else: let Some(x) = opt else { return; }
# 4. Modern clippy lints (uninlined_format_args, etc.)
# 5. Newer Result/Option methods

RUST_2024_EXAMPLES = [
    # Format string capture (clippy::uninlined_format_args)
    {
        "buggy": 'fn greet(name: &str) { println!("Hello, {}", name); }',
        "fixed": 'fn greet(name: &str) { println!("Hello, {name}"); }',
        "task_id": "r24_fmt_001",
        "category": "format_strings"
    },
    {
        "buggy": 'fn log_value(x: i32, y: i32) { println!("x={}, y={}", x, y); }',
        "fixed": 'fn log_value(x: i32, y: i32) { println!("x={x}, y={y}"); }',
        "task_id": "r24_fmt_002",
        "category": "format_strings"
    },
    {
        "buggy": 'fn debug(msg: &str, code: u32) { eprintln!("Error {}: {}", code, msg); }',
        "fixed": 'fn debug(msg: &str, code: u32) { eprintln!("Error {code}: {msg}"); }',
        "task_id": "r24_fmt_003",
        "category": "format_strings"
    },
    {
        "buggy": 'fn format_point(x: f64, y: f64) -> String { format!("({}, {})", x, y) }',
        "fixed": 'fn format_point(x: f64, y: f64) -> String { format!("({x}, {y})") }',
        "task_id": "r24_fmt_004",
        "category": "format_strings"
    },
    {
        "buggy": 'fn write_log(level: &str, msg: &str) { println!("[{}] {}", level, msg); }',
        "fixed": 'fn write_log(level: &str, msg: &str) { println!("[{level}] {msg}"); }',
        "task_id": "r24_fmt_005",
        "category": "format_strings"
    },
    {
        "buggy": 'fn panic_msg(reason: &str) { panic!("Fatal: {}", reason); }',
        "fixed": 'fn panic_msg(reason: &str) { panic!("Fatal: {reason}"); }',
        "task_id": "r24_fmt_006",
        "category": "format_strings"
    },
    {
        "buggy": 'fn assert_eq_msg(a: i32, b: i32) { assert_eq!(a, b, "Expected {} but got {}", a, b); }',
        "fixed": 'fn assert_eq_msg(a: i32, b: i32) { assert_eq!(a, b, "Expected {a} but got {b}"); }',
        "task_id": "r24_fmt_007",
        "category": "format_strings"
    },
    {
        "buggy": 'fn write_file(path: &str) -> std::io::Result<()> { std::fs::write(path, format!("path={}", path)) }',
        "fixed": 'fn write_file(path: &str) -> std::io::Result<()> { std::fs::write(path, format!("path={path}")) }',
        "task_id": "r24_fmt_008",
        "category": "format_strings"
    },

    # Let-chains (stabilized in Rust 1.76)
    {
        "buggy": 'fn check_positive(opt: Option<i32>) -> bool { if let Some(x) = opt { if x > 0 { return true; } } false }',
        "fixed": 'fn check_positive(opt: Option<i32>) -> bool { if let Some(x) = opt && x > 0 { return true; } false }',
        "task_id": "r24_letchain_001",
        "category": "let_chains"
    },
    {
        "buggy": 'fn get_valid(opt: Option<String>) -> Option<String> { if let Some(s) = opt { if !s.is_empty() { return Some(s); } } None }',
        "fixed": 'fn get_valid(opt: Option<String>) -> Option<String> { if let Some(s) = opt && !s.is_empty() { return Some(s); } None }',
        "task_id": "r24_letchain_002",
        "category": "let_chains"
    },
    {
        "buggy": 'fn check_bounds(opt: Option<usize>, max: usize) -> bool { if let Some(n) = opt { if n < max { return true; } } false }',
        "fixed": 'fn check_bounds(opt: Option<usize>, max: usize) -> bool { if let Some(n) = opt && n < max { return true; } false }',
        "task_id": "r24_letchain_003",
        "category": "let_chains"
    },
    {
        "buggy": 'fn validate(res: Result<i32, &str>) -> bool { if let Ok(n) = res { if n >= 0 { return true; } } false }',
        "fixed": 'fn validate(res: Result<i32, &str>) -> bool { if let Ok(n) = res && n >= 0 { return true; } false }',
        "task_id": "r24_letchain_004",
        "category": "let_chains"
    },

    # Let-else (stabilized in Rust 1.65, but model may not use it well)
    {
        "buggy": 'fn unwrap_or_return(opt: Option<i32>) -> i32 { match opt { Some(x) => x, None => return 0 } }',
        "fixed": 'fn unwrap_or_return(opt: Option<i32>) -> i32 { let Some(x) = opt else { return 0; }; x }',
        "task_id": "r24_letelse_001",
        "category": "let_else"
    },
    {
        "buggy": 'fn parse_or_default(s: &str) -> i32 { match s.parse() { Ok(n) => n, Err(_) => return -1 } }',
        "fixed": 'fn parse_or_default(s: &str) -> i32 { let Ok(n) = s.parse() else { return -1; }; n }',
        "task_id": "r24_letelse_002",
        "category": "let_else"
    },
    {
        "buggy": 'fn get_first(v: &[i32]) -> i32 { match v.first() { Some(&x) => x, None => return 0 } }',
        "fixed": 'fn get_first(v: &[i32]) -> i32 { let Some(&x) = v.first() else { return 0; }; x }',
        "task_id": "r24_letelse_003",
        "category": "let_else"
    },

    # Modern Option/Result methods (is_some_and, is_ok_and, etc. - Rust 1.70+)
    {
        "buggy": 'fn has_positive(opt: Option<i32>) -> bool { opt.map(|x| x > 0).unwrap_or(false) }',
        "fixed": 'fn has_positive(opt: Option<i32>) -> bool { opt.is_some_and(|x| x > 0) }',
        "task_id": "r24_opt_001",
        "category": "modern_methods"
    },
    {
        "buggy": 'fn is_valid_result(res: Result<i32, &str>) -> bool { res.map(|x| x >= 0).unwrap_or(false) }',
        "fixed": 'fn is_valid_result(res: Result<i32, &str>) -> bool { res.is_ok_and(|x| x >= 0) }',
        "task_id": "r24_opt_002",
        "category": "modern_methods"
    },
    {
        "buggy": 'fn is_error_match(res: Result<(), &str>) -> bool { res.err().map(|e| e.contains("fail")).unwrap_or(false) }',
        "fixed": 'fn is_error_match(res: Result<(), &str>) -> bool { res.is_err_and(|e| e.contains("fail")) }',
        "task_id": "r24_opt_003",
        "category": "modern_methods"
    },
    {
        "buggy": 'fn is_none_or_zero(opt: Option<i32>) -> bool { opt.map(|x| x == 0).unwrap_or(true) }',
        "fixed": 'fn is_none_or_zero(opt: Option<i32>) -> bool { opt.is_none_or(|x| x == 0) }',
        "task_id": "r24_opt_004",
        "category": "modern_methods"
    },

    # Clippy: use of iter().cloned() vs copied()
    {
        "buggy": 'fn double_all(v: &[i32]) -> Vec<i32> { v.iter().cloned().map(|x| x * 2).collect() }',
        "fixed": 'fn double_all(v: &[i32]) -> Vec<i32> { v.iter().copied().map(|x| x * 2).collect() }',
        "task_id": "r24_clippy_001",
        "category": "clippy_modern"
    },
    {
        "buggy": 'fn sum_all(v: &[i64]) -> i64 { v.iter().cloned().sum() }',
        "fixed": 'fn sum_all(v: &[i64]) -> i64 { v.iter().copied().sum() }',
        "task_id": "r24_clippy_002",
        "category": "clippy_modern"
    },

    # Clippy: manual_flatten
    {
        "buggy": 'fn get_somes(v: Vec<Option<i32>>) -> Vec<i32> { v.into_iter().filter(|x| x.is_some()).map(|x| x.unwrap()).collect() }',
        "fixed": 'fn get_somes(v: Vec<Option<i32>>) -> Vec<i32> { v.into_iter().flatten().collect() }',
        "task_id": "r24_clippy_003",
        "category": "clippy_modern"
    },

    # Clippy: manual_map / manual_ok_or
    {
        "buggy": 'fn add_one(opt: Option<i32>) -> Option<i32> { match opt { Some(x) => Some(x + 1), None => None } }',
        "fixed": 'fn add_one(opt: Option<i32>) -> Option<i32> { opt.map(|x| x + 1) }',
        "task_id": "r24_clippy_004",
        "category": "clippy_modern"
    },

    # Clippy: needless_collect
    {
        "buggy": 'fn sum_squares(v: &[i32]) -> i32 { v.iter().map(|x| x * x).collect::<Vec<_>>().into_iter().sum() }',
        "fixed": 'fn sum_squares(v: &[i32]) -> i32 { v.iter().map(|x| x * x).sum() }',
        "task_id": "r24_clippy_005",
        "category": "clippy_modern"
    },

    # Default::default() vs type inference
    {
        "buggy": 'fn new_vec() -> Vec<i32> { let v: Vec<i32> = Default::default(); v }',
        "fixed": 'fn new_vec() -> Vec<i32> { Vec::new() }',
        "task_id": "r24_clippy_006",
        "category": "clippy_modern"
    },

    # Clippy: manual_filter_map
    {
        "buggy": 'fn parse_numbers(v: Vec<&str>) -> Vec<i32> { v.into_iter().filter_map(|s| s.parse().ok()).collect() }',
        "fixed": 'fn parse_numbers(v: Vec<&str>) -> Vec<i32> { v.into_iter().filter_map(|s| s.parse().ok()).collect() }',
        "task_id": "r24_clippy_007",
        "category": "clippy_modern"
    },

    # Clippy: match_like_matches_macro
    {
        "buggy": 'fn is_digit(c: char) -> bool { match c { \'0\'..=\'9\' => true, _ => false } }',
        "fixed": 'fn is_digit(c: char) -> bool { matches!(c, \'0\'..=\'9\') }',
        "task_id": "r24_clippy_008",
        "category": "clippy_modern"
    },

    # More format strings with expressions
    {
        "buggy": 'fn show_calc(a: i32, b: i32) { println!("{} + {} = {}", a, b, a + b); }',
        "fixed": 'fn show_calc(a: i32, b: i32) { let sum = a + b; println!("{a} + {b} = {sum}"); }',
        "task_id": "r24_fmt_009",
        "category": "format_strings"
    },
    {
        "buggy": 'fn log_len(v: &Vec<i32>) { println!("Length: {}", v.len()); }',
        "fixed": 'fn log_len(v: &[i32]) { println!("Length: {}", v.len()); }',
        "task_id": "r24_clippy_009",
        "category": "clippy_modern"
    },
]


def format_example(ex):
    """Format an example into SFT training format."""
    return {
        "instruction": "You are a Rust expert. Fix the following code that has a compilation error or clippy warning.\nReturn ONLY the fixed Rust code without any explanation or markdown formatting.",
        "input": f"## Buggy Code:\n```rust\n{ex['buggy']}\n```\n\n## Compiler Error:\nUnknown error\n\n## Fixed Code:",
        "output": ex['fixed'],
        "task_id": ex['task_id'],
        "category": ex['category']
    }


def main():
    output_dir = Path(__file__).parent.parent / "data" / "sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the examples
    examples = [format_example(ex) for ex in RUST_2024_EXAMPLES]

    # Save to file
    output_file = output_dir / "rust2024_koans.jsonl"
    with open(output_file, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Generated {len(examples)} Rust 2024 koans:")

    # Count by category
    categories = {}
    for ex in examples:
        cat = ex['category']
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    print(f"\nSaved to {output_file}")

    # Also create a combined file with replay data
    replay_file = output_dir / "heavy_replay.jsonl"
    combined_file = output_dir / "rust2024_with_replay.jsonl"

    combined = list(examples)
    if replay_file.exists():
        with open(replay_file) as f:
            for line in f:
                if line.strip():
                    combined.append(json.loads(line))

    with open(combined_file, 'w') as f:
        for ex in combined:
            f.write(json.dumps(ex) + '\n')

    print(f"\nCombined dataset: {len(combined)} examples")
    print(f"  New: {len(examples)}")
    print(f"  Replay: {len(combined) - len(examples)}")
    print(f"Saved to {combined_file}")


if __name__ == "__main__":
    main()
