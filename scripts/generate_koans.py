#!/usr/bin/env python3
"""
Generate additional Rust koans for training.

Creates variations of existing patterns to expand the dataset.
"""

import json
from pathlib import Path

# ResultHandling koans - variations on common patterns
RESULT_HANDLING_KOANS = [
    # unwrap_or vs unwrap_or_else variations
    {
        "id": "rh_gen_001",
        "name": "Use unwrap_or_else for function call",
        "family": "ResultHandling",
        "buggy": 'fn get_value(opt: Option<String>) -> String { opt.unwrap_or(compute_default()) } fn compute_default() -> String { "default".to_string() }',
        "fixed": 'fn get_value(opt: Option<String>) -> String { opt.unwrap_or_else(|| compute_default()) } fn compute_default() -> String { "default".to_string() }',
    },
    {
        "id": "rh_gen_002",
        "name": "Use unwrap_or_else for expensive Vec creation",
        "family": "ResultHandling",
        "buggy": 'fn get_items(opt: Option<Vec<i32>>) -> Vec<i32> { opt.unwrap_or(vec![1,2,3,4,5]) }',
        "fixed": 'fn get_items(opt: Option<Vec<i32>>) -> Vec<i32> { opt.unwrap_or_else(|| vec![1,2,3,4,5]) }',
    },
    # ? operator patterns
    {
        "id": "rh_gen_003",
        "name": "Add Result return type for ? operator",
        "family": "ResultHandling",
        "buggy": 'fn parse_and_add(a: &str, b: &str) -> i32 { let x: i32 = a.parse()?; let y: i32 = b.parse()?; x + y }',
        "fixed": 'fn parse_and_add(a: &str, b: &str) -> Result<i32, std::num::ParseIntError> { let x: i32 = a.parse()?; let y: i32 = b.parse()?; Ok(x + y) }',
    },
    {
        "id": "rh_gen_004",
        "name": "Convert unwrap to ? with proper return",
        "family": "ResultHandling",
        "buggy": 'fn read_config(path: &str) -> String { std::fs::read_to_string(path).unwrap() }',
        "fixed": 'fn read_config(path: &str) -> Result<String, std::io::Error> { std::fs::read_to_string(path) }',
    },
    # map/and_then patterns
    {
        "id": "rh_gen_005",
        "name": "Use and_then for chaining Options",
        "family": "ResultHandling",
        "buggy": 'fn get_nested(opt: Option<Option<i32>>) -> Option<i32> { opt.map(|inner| inner).flatten() }',
        "fixed": 'fn get_nested(opt: Option<Option<i32>>) -> Option<i32> { opt.and_then(|inner| inner) }',
    },
    {
        "id": "rh_gen_006",
        "name": "Use ok_or for Option to Result",
        "family": "ResultHandling",
        "buggy": 'fn require_value(opt: Option<i32>) -> Result<i32, &\'static str> { match opt { Some(v) => Ok(v), None => Err("missing value") } }',
        "fixed": 'fn require_value(opt: Option<i32>) -> Result<i32, &\'static str> { opt.ok_or("missing value") }',
    },
]

# TraitBounds koans
TRAIT_BOUNDS_KOANS = [
    {
        "id": "tb_gen_001",
        "name": "Add Clone for Vec::clone",
        "family": "TraitBounds",
        "buggy": 'struct Container<T> { items: Vec<T> } impl<T> Container<T> { fn duplicate(&self) -> Self { Container { items: self.items.clone() } } }',
        "fixed": 'struct Container<T: Clone> { items: Vec<T> } impl<T: Clone> Container<T> { fn duplicate(&self) -> Self { Container { items: self.items.clone() } } }',
    },
    {
        "id": "tb_gen_002",
        "name": "Add Debug for println!",
        "family": "TraitBounds",
        "buggy": 'struct Point { x: i32, y: i32 } fn main() { let p = Point { x: 1, y: 2 }; println!("{:?}", p); }',
        "fixed": '#[derive(Debug)] struct Point { x: i32, y: i32 } fn main() { let p = Point { x: 1, y: 2 }; println!("{:?}", p); }',
    },
    {
        "id": "tb_gen_003",
        "name": "Add PartialEq for comparison",
        "family": "TraitBounds",
        "buggy": 'struct Item { id: u32 } fn same(a: &Item, b: &Item) -> bool { a == b }',
        "fixed": '#[derive(PartialEq)] struct Item { id: u32 } fn same(a: &Item, b: &Item) -> bool { a == b }',
    },
    {
        "id": "tb_gen_004",
        "name": "Add Hash for HashSet",
        "family": "TraitBounds",
        "buggy": 'use std::collections::HashSet; struct Key { id: u32 } fn main() { let mut set: HashSet<Key> = HashSet::new(); set.insert(Key { id: 1 }); }',
        "fixed": 'use std::collections::HashSet; #[derive(Hash, PartialEq, Eq)] struct Key { id: u32 } fn main() { let mut set: HashSet<Key> = HashSet::new(); set.insert(Key { id: 1 }); }',
    },
    {
        "id": "tb_gen_005",
        "name": "Add Default for struct initialization",
        "family": "TraitBounds",
        "buggy": 'struct Config { timeout: u32, retries: u32 } fn main() { let cfg: Config = Default::default(); }',
        "fixed": '#[derive(Default)] struct Config { timeout: u32, retries: u32 } fn main() { let cfg: Config = Default::default(); }',
    },
    {
        "id": "tb_gen_006",
        "name": "Use Arc instead of Rc for Send",
        "family": "TraitBounds",
        "buggy": 'use std::rc::Rc; fn spawn_task<T: Send + \'static>(val: T) { std::thread::spawn(move || { drop(val); }); } fn main() { let data = Rc::new(42); spawn_task(data); }',
        "fixed": 'use std::sync::Arc; fn spawn_task<T: Send + \'static>(val: T) { std::thread::spawn(move || { drop(val); }); } fn main() { let data = Arc::new(42); spawn_task(data); }',
    },
]

# BorrowChecker koans
BORROW_CHECKER_KOANS = [
    {
        "id": "bc_gen_001",
        "name": "Clone to avoid move",
        "family": "BorrowChecker",
        "buggy": 'fn main() { let s = String::from("hello"); let a = s; let b = s; }',
        "fixed": 'fn main() { let s = String::from("hello"); let a = s.clone(); let b = s; }',
    },
    {
        "id": "bc_gen_002",
        "name": "Use reference instead of move",
        "family": "BorrowChecker",
        "buggy": 'fn print_len(s: String) { println!("{}", s.len()); } fn main() { let s = String::from("hello"); print_len(s); print_len(s); }',
        "fixed": 'fn print_len(s: &String) { println!("{}", s.len()); } fn main() { let s = String::from("hello"); print_len(&s); print_len(&s); }',
    },
    {
        "id": "bc_gen_003",
        "name": "Split borrows with separate scopes",
        "family": "BorrowChecker",
        "buggy": 'fn main() { let mut v = vec![1, 2, 3]; let first = &v[0]; v.push(4); println!("{}", first); }',
        "fixed": 'fn main() { let mut v = vec![1, 2, 3]; { let first = &v[0]; println!("{}", first); } v.push(4); }',
    },
    {
        "id": "bc_gen_004",
        "name": "Use to_string for owned String",
        "family": "BorrowChecker",
        "buggy": 'fn get_greeting() -> &str { let s = String::from("hello"); &s }',
        "fixed": 'fn get_greeting() -> String { let s = String::from("hello"); s }',
    },
    {
        "id": "bc_gen_005",
        "name": "Take ownership in closure",
        "family": "BorrowChecker",
        "buggy": 'fn main() { let name = String::from("Alice"); let greet = || println!("Hello, {}", name); std::thread::spawn(greet).join(); }',
        "fixed": 'fn main() { let name = String::from("Alice"); let greet = move || println!("Hello, {}", name); std::thread::spawn(greet).join(); }',
    },
    {
        "id": "bc_gen_006",
        "name": "Return owned value instead of reference",
        "family": "BorrowChecker",
        "buggy": 'fn concat(a: &str, b: &str) -> &str { let result = format!("{}{}", a, b); &result }',
        "fixed": 'fn concat(a: &str, b: &str) -> String { format!("{}{}", a, b) }',
    },
]

def convert_to_sft_format(koan):
    """Convert koan to SFT training format."""
    return {
        "instruction": f"Fix the following Rust code. The issue is: {koan['name']}",
        "input": koan["buggy"],
        "output": koan["fixed"],
    }

def main():
    output_dir = Path("/home/mike/github/softwarewrighter/sleepy-coder/data/sft")

    all_koans = RESULT_HANDLING_KOANS + TRAIT_BOUNDS_KOANS + BORROW_CHECKER_KOANS

    print(f"Generated koans:")
    print(f"  ResultHandling: {len(RESULT_HANDLING_KOANS)}")
    print(f"  TraitBounds: {len(TRAIT_BOUNDS_KOANS)}")
    print(f"  BorrowChecker: {len(BORROW_CHECKER_KOANS)}")
    print(f"  Total: {len(all_koans)}")

    # Save as SFT data
    sft_data = [convert_to_sft_format(k) for k in all_koans]

    with open(output_dir / "generated_koans.jsonl", "w") as f:
        for item in sft_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved to {output_dir / 'generated_koans.jsonl'}")

    # Combine with existing training data
    existing = []
    if (output_dir / "train.jsonl").exists():
        with open(output_dir / "train.jsonl") as f:
            existing = [json.loads(line) for line in f if line.strip()]

    combined = existing + sft_data

    # Add replay (3x original)
    combined_with_replay = combined + existing * 2

    import random
    random.seed(42)
    random.shuffle(combined_with_replay)

    with open(output_dir / "expanded.jsonl", "w") as f:
        for item in combined_with_replay:
            f.write(json.dumps(item) + "\n")

    print(f"Expanded dataset: {len(combined_with_replay)} examples")
    print(f"Saved to {output_dir / 'expanded.jsonl'}")

if __name__ == "__main__":
    main()
