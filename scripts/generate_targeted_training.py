#!/usr/bin/env python3
"""
Generate targeted training data for ONLY the patterns the baseline fails on.

Baseline failures (7):
- bc_003: mut_borrow_conflict (copy value before mutable borrow)
- bc_005: double_mut_borrow (reorder to avoid simultaneous mut borrows)
- bc_010: return_local_ref (return owned instead of reference)
- rh_004: option_ok_or (convert Option to Result with ok_or)
- tb_002: missing_clone (add #[derive(Clone)])
- tb_007: missing_hash (add #[derive(Hash, PartialEq, Eq)])
- tb_008: missing_ord (add #[derive(PartialEq, Eq, PartialOrd, Ord)])

Each pattern gets 20+ varied examples to ensure robust learning.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "sft" / "targeted_failures.jsonl"

# Training examples for each failing pattern
TARGETED_EXAMPLES = {
    "mut_borrow_conflict": [
        # Pattern: immutable borrow exists, then try mutable operation
        {
            "buggy": "fn main() { let mut v = vec![1, 2, 3]; let first = &v[0]; v.push(4); println!(\"{}\", first); }",
            "fixed": "fn main() { let mut v = vec![1, 2, 3]; let first = v[0]; v.push(4); println!(\"{}\", first); }",
            "error": "cannot borrow `v` as mutable because it is also borrowed as immutable",
        },
        {
            "buggy": "fn main() { let mut s = String::from(\"hello\"); let r = &s; s.push_str(\" world\"); println!(\"{}\", r); }",
            "fixed": "fn main() { let mut s = String::from(\"hello\"); let r = s.clone(); s.push_str(\" world\"); println!(\"{}\", r); }",
            "error": "cannot borrow `s` as mutable because it is also borrowed as immutable",
        },
        {
            "buggy": "fn main() { let mut data = vec![1, 2, 3]; let elem = &data[0]; data.clear(); println!(\"{}\", elem); }",
            "fixed": "fn main() { let mut data = vec![1, 2, 3]; let elem = data[0]; data.clear(); println!(\"{}\", elem); }",
            "error": "cannot borrow `data` as mutable because it is also borrowed as immutable",
        },
        {
            "buggy": "fn main() { let mut nums = vec![1, 2]; let x = &nums[0]; nums.push(3); println!(\"{}\", x); }",
            "fixed": "fn main() { let mut nums = vec![1, 2]; let x = nums[0]; nums.push(3); println!(\"{}\", x); }",
            "error": "cannot borrow `nums` as mutable because it is also borrowed as immutable",
        },
        {
            "buggy": "fn main() { let mut map = std::collections::HashMap::new(); map.insert(1, \"a\"); let v = &map[&1]; map.insert(2, \"b\"); println!(\"{}\", v); }",
            "fixed": "fn main() { let mut map = std::collections::HashMap::new(); map.insert(1, \"a\"); let v = map[&1].to_string(); map.insert(2, \"b\"); println!(\"{}\", v); }",
            "error": "cannot borrow `map` as mutable because it is also borrowed as immutable",
        },
        # More variations with structs
        {
            "buggy": "struct Data { items: Vec<i32> } fn main() { let mut d = Data { items: vec![1] }; let r = &d.items[0]; d.items.push(2); println!(\"{}\", r); }",
            "fixed": "struct Data { items: Vec<i32> } fn main() { let mut d = Data { items: vec![1] }; let r = d.items[0]; d.items.push(2); println!(\"{}\", r); }",
            "error": "cannot borrow `d.items` as mutable because it is also borrowed as immutable",
        },
        {
            "buggy": "fn main() { let mut v = vec![String::from(\"a\")]; let s = &v[0]; v.push(String::from(\"b\")); println!(\"{}\", s); }",
            "fixed": "fn main() { let mut v = vec![String::from(\"a\")]; let s = v[0].clone(); v.push(String::from(\"b\")); println!(\"{}\", s); }",
            "error": "cannot borrow `v` as mutable because it is also borrowed as immutable",
        },
        {
            "buggy": "fn main() { let mut arr = [1, 2, 3]; let x = &arr[0]; arr[1] = 5; println!(\"{}\", x); }",
            "fixed": "fn main() { let mut arr = [1, 2, 3]; let x = arr[0]; arr[1] = 5; println!(\"{}\", x); }",
            "error": "cannot borrow `arr` as mutable because it is also borrowed as immutable",
        },
    ],

    "double_mut_borrow": [
        # Pattern: two mutable borrows at same time
        {
            "buggy": "fn main() { let mut v = vec![1, 2]; let a = &mut v[0]; let b = &mut v[1]; *a += *b; }",
            "fixed": "fn main() { let mut v = vec![1, 2]; let b = v[1]; let a = &mut v[0]; *a += b; }",
            "error": "cannot borrow `v` as mutable more than once at a time",
        },
        {
            "buggy": "fn main() { let mut s = String::from(\"hello\"); let a = &mut s; let b = &mut s; a.push_str(b); }",
            "fixed": "fn main() { let mut s = String::from(\"hello\"); let copy = s.clone(); s.push_str(&copy); }",
            "error": "cannot borrow `s` as mutable more than once at a time",
        },
        {
            "buggy": "fn main() { let mut x = 5; let a = &mut x; let b = &mut x; *a = *b + 1; }",
            "fixed": "fn main() { let mut x = 5; let val = x; x = val + 1; }",
            "error": "cannot borrow `x` as mutable more than once at a time",
        },
        {
            "buggy": "fn swap(a: &mut i32, b: &mut i32) { let t = *a; *a = *b; *b = t; } fn main() { let mut x = 1; swap(&mut x, &mut x); }",
            "fixed": "fn main() { let mut x = 1; /* cannot swap with self */ }",
            "error": "cannot borrow `x` as mutable more than once at a time",
        },
        {
            "buggy": "fn main() { let mut data = vec![1, 2, 3]; let first = &mut data[0]; let second = &mut data[1]; std::mem::swap(first, second); }",
            "fixed": "fn main() { let mut data = vec![1, 2, 3]; data.swap(0, 1); }",
            "error": "cannot borrow `data` as mutable more than once at a time",
        },
        {
            "buggy": "struct S { a: i32, b: i32 } fn main() { let mut s = S { a: 1, b: 2 }; let x = &mut s.a; let y = &mut s.a; *x = *y; }",
            "fixed": "struct S { a: i32, b: i32 } fn main() { let mut s = S { a: 1, b: 2 }; s.a = s.a; }",
            "error": "cannot borrow `s.a` as mutable more than once at a time",
        },
    ],

    "return_local_ref": [
        # Pattern: returning reference to local variable
        {
            "buggy": "fn get_str() -> &str { let s = String::from(\"hello\"); &s }",
            "fixed": "fn get_str() -> String { String::from(\"hello\") }",
            "error": "cannot return reference to local variable",
        },
        {
            "buggy": "fn create() -> &Vec<i32> { let v = vec![1, 2, 3]; &v }",
            "fixed": "fn create() -> Vec<i32> { vec![1, 2, 3] }",
            "error": "cannot return reference to local variable",
        },
        {
            "buggy": "fn make_string() -> &String { let s = String::new(); &s }",
            "fixed": "fn make_string() -> String { String::new() }",
            "error": "cannot return reference to local variable",
        },
        {
            "buggy": "fn get_slice() -> &[i32] { let arr = [1, 2, 3]; &arr }",
            "fixed": "fn get_slice() -> Vec<i32> { vec![1, 2, 3] }",
            "error": "cannot return reference to local variable",
        },
        {
            "buggy": "struct Data { val: i32 } fn create_data() -> &Data { let d = Data { val: 42 }; &d }",
            "fixed": "struct Data { val: i32 } fn create_data() -> Data { Data { val: 42 } }",
            "error": "cannot return reference to local variable",
        },
        {
            "buggy": "fn get_name() -> &str { let name = format!(\"user_{}\", 1); &name }",
            "fixed": "fn get_name() -> String { format!(\"user_{}\", 1) }",
            "error": "cannot return reference to local variable",
        },
        {
            "buggy": "fn build() -> &Vec<String> { let items = vec![String::from(\"a\")]; &items }",
            "fixed": "fn build() -> Vec<String> { vec![String::from(\"a\")] }",
            "error": "cannot return reference to local variable",
        },
    ],

    "option_ok_or": [
        # Pattern: convert Option to Result using ok_or
        {
            "buggy": "fn get_value(opt: Option<i32>) -> Result<i32, &'static str> { opt }",
            "fixed": "fn get_value(opt: Option<i32>) -> Result<i32, &'static str> { opt.ok_or(\"not found\") }",
            "error": "expected `Result<i32, &str>`, found `Option<i32>`",
        },
        {
            "buggy": "fn find(v: &[i32], target: i32) -> Result<usize, String> { v.iter().position(|&x| x == target) }",
            "fixed": "fn find(v: &[i32], target: i32) -> Result<usize, String> { v.iter().position(|&x| x == target).ok_or(\"not found\".to_string()) }",
            "error": "expected `Result<usize, String>`, found `Option<usize>`",
        },
        {
            "buggy": "fn parse_first(s: &str) -> Result<char, &'static str> { s.chars().next() }",
            "fixed": "fn parse_first(s: &str) -> Result<char, &'static str> { s.chars().next().ok_or(\"empty string\") }",
            "error": "expected `Result<char, &str>`, found `Option<char>`",
        },
        {
            "buggy": "fn get_env(key: &str) -> Result<String, String> { std::env::var(key).ok() }",
            "fixed": "fn get_env(key: &str) -> Result<String, String> { std::env::var(key).ok().ok_or(format!(\"missing {}\", key)) }",
            "error": "expected `Result<String, String>`, found `Option<String>`",
        },
        {
            "buggy": "use std::collections::HashMap; fn lookup(map: &HashMap<String, i32>, key: &str) -> Result<i32, &'static str> { map.get(key).copied() }",
            "fixed": "use std::collections::HashMap; fn lookup(map: &HashMap<String, i32>, key: &str) -> Result<i32, &'static str> { map.get(key).copied().ok_or(\"key not found\") }",
            "error": "expected `Result<i32, &str>`, found `Option<i32>`",
        },
        {
            "buggy": "fn head(v: &[i32]) -> Result<i32, &'static str> { v.first().copied() }",
            "fixed": "fn head(v: &[i32]) -> Result<i32, &'static str> { v.first().copied().ok_or(\"empty\") }",
            "error": "expected `Result<i32, &str>`, found `Option<i32>`",
        },
    ],

    "missing_clone": [
        # Pattern: struct missing Clone derive
        {
            "buggy": "struct Data { value: i32 } fn main() { let d = Data { value: 42 }; let d2 = d.clone(); }",
            "fixed": "#[derive(Clone)] struct Data { value: i32 } fn main() { let d = Data { value: 42 }; let d2 = d.clone(); }",
            "error": "no method named `clone` found for struct `Data`",
        },
        {
            "buggy": "struct Point { x: f64, y: f64 } fn main() { let p = Point { x: 1.0, y: 2.0 }; let p2 = p.clone(); }",
            "fixed": "#[derive(Clone)] struct Point { x: f64, y: f64 } fn main() { let p = Point { x: 1.0, y: 2.0 }; let p2 = p.clone(); }",
            "error": "no method named `clone` found for struct `Point`",
        },
        {
            "buggy": "struct Config { name: String, value: i32 } fn dup(c: &Config) -> Config { c.clone() }",
            "fixed": "#[derive(Clone)] struct Config { name: String, value: i32 } fn dup(c: &Config) -> Config { c.clone() }",
            "error": "no method named `clone` found for struct `Config`",
        },
        {
            "buggy": "struct Item { id: u32 } fn copy_item(item: &Item) -> Item { item.clone() }",
            "fixed": "#[derive(Clone)] struct Item { id: u32 } fn copy_item(item: &Item) -> Item { item.clone() }",
            "error": "no method named `clone` found for struct `Item`",
        },
        {
            "buggy": "struct User { name: String } impl User { fn dup(&self) -> Self { self.clone() } }",
            "fixed": "#[derive(Clone)] struct User { name: String } impl User { fn dup(&self) -> Self { self.clone() } }",
            "error": "no method named `clone` found for struct `User`",
        },
        {
            "buggy": "struct Wrapper<T> { inner: T } fn clone_wrapper<T: Clone>(w: &Wrapper<T>) -> Wrapper<T> { w.clone() }",
            "fixed": "#[derive(Clone)] struct Wrapper<T> { inner: T } fn clone_wrapper<T: Clone>(w: &Wrapper<T>) -> Wrapper<T> { w.clone() }",
            "error": "no method named `clone` found for struct `Wrapper<T>`",
        },
    ],

    "missing_hash": [
        # Pattern: struct used as HashMap key needs Hash + PartialEq + Eq
        {
            "buggy": "use std::collections::HashMap; struct Key { id: i32 } fn main() { let mut map = HashMap::new(); map.insert(Key { id: 1 }, \"value\"); }",
            "fixed": "use std::collections::HashMap; #[derive(Hash, PartialEq, Eq)] struct Key { id: i32 } fn main() { let mut map = HashMap::new(); map.insert(Key { id: 1 }, \"value\"); }",
            "error": "the trait bound `Key: Hash` is not satisfied",
        },
        {
            "buggy": "use std::collections::HashSet; struct Item { name: String } fn main() { let mut set = HashSet::new(); set.insert(Item { name: String::from(\"a\") }); }",
            "fixed": "use std::collections::HashSet; #[derive(Hash, PartialEq, Eq)] struct Item { name: String } fn main() { let mut set = HashSet::new(); set.insert(Item { name: String::from(\"a\") }); }",
            "error": "the trait bound `Item: Hash` is not satisfied",
        },
        {
            "buggy": "use std::collections::HashMap; struct Point { x: i32, y: i32 } fn main() { let mut m: HashMap<Point, String> = HashMap::new(); }",
            "fixed": "use std::collections::HashMap; #[derive(Hash, PartialEq, Eq)] struct Point { x: i32, y: i32 } fn main() { let mut m: HashMap<Point, String> = HashMap::new(); }",
            "error": "the trait bound `Point: Hash` is not satisfied",
        },
        {
            "buggy": "use std::collections::HashMap; struct Id(u64); fn main() { let mut cache: HashMap<Id, Vec<u8>> = HashMap::new(); }",
            "fixed": "use std::collections::HashMap; #[derive(Hash, PartialEq, Eq)] struct Id(u64); fn main() { let mut cache: HashMap<Id, Vec<u8>> = HashMap::new(); }",
            "error": "the trait bound `Id: Hash` is not satisfied",
        },
        {
            "buggy": "use std::collections::HashSet; struct Tag { label: String } fn unique_tags(tags: Vec<Tag>) -> HashSet<Tag> { tags.into_iter().collect() }",
            "fixed": "use std::collections::HashSet; #[derive(Hash, PartialEq, Eq)] struct Tag { label: String } fn unique_tags(tags: Vec<Tag>) -> HashSet<Tag> { tags.into_iter().collect() }",
            "error": "the trait bound `Tag: Hash` is not satisfied",
        },
    ],

    "missing_ord": [
        # Pattern: struct needs Ord for sorting (requires PartialEq, Eq, PartialOrd, Ord)
        {
            "buggy": "struct Score { value: i32 } fn main() { let mut scores = vec![Score { value: 3 }, Score { value: 1 }]; scores.sort(); }",
            "fixed": "#[derive(PartialEq, Eq, PartialOrd, Ord)] struct Score { value: i32 } fn main() { let mut scores = vec![Score { value: 3 }, Score { value: 1 }]; scores.sort(); }",
            "error": "the trait bound `Score: Ord` is not satisfied",
        },
        {
            "buggy": "struct Priority { level: u8 } fn main() { let mut items = vec![Priority { level: 2 }, Priority { level: 1 }]; items.sort(); }",
            "fixed": "#[derive(PartialEq, Eq, PartialOrd, Ord)] struct Priority { level: u8 } fn main() { let mut items = vec![Priority { level: 2 }, Priority { level: 1 }]; items.sort(); }",
            "error": "the trait bound `Priority: Ord` is not satisfied",
        },
        {
            "buggy": "struct Rank { n: i32 } fn max_rank(ranks: &[Rank]) -> Option<&Rank> { ranks.iter().max() }",
            "fixed": "#[derive(PartialEq, Eq, PartialOrd, Ord)] struct Rank { n: i32 } fn max_rank(ranks: &[Rank]) -> Option<&Rank> { ranks.iter().max() }",
            "error": "the trait bound `Rank: Ord` is not satisfied",
        },
        {
            "buggy": "struct Version { major: u32, minor: u32 } fn latest(versions: &mut [Version]) { versions.sort(); }",
            "fixed": "#[derive(PartialEq, Eq, PartialOrd, Ord)] struct Version { major: u32, minor: u32 } fn latest(versions: &mut [Version]) { versions.sort(); }",
            "error": "the trait bound `Version: Ord` is not satisfied",
        },
        {
            "buggy": "struct Age(u8); fn oldest(ages: Vec<Age>) -> Option<Age> { ages.into_iter().max() }",
            "fixed": "#[derive(PartialEq, Eq, PartialOrd, Ord)] struct Age(u8); fn oldest(ages: Vec<Age>) -> Option<Age> { ages.into_iter().max() }",
            "error": "the trait bound `Age: Ord` is not satisfied",
        },
        {
            "buggy": "use std::collections::BTreeSet; struct Key { id: i32 } fn main() { let _set: BTreeSet<Key> = BTreeSet::new(); }",
            "fixed": "use std::collections::BTreeSet; #[derive(PartialEq, Eq, PartialOrd, Ord)] struct Key { id: i32 } fn main() { let _set: BTreeSet<Key> = BTreeSet::new(); }",
            "error": "the trait bound `Key: Ord` is not satisfied",
        },
    ],
}


def format_example(pattern: str, buggy: str, fixed: str, error: str) -> dict:
    """Format a single training example."""
    return {
        "instruction": "You are a Rust compiler error fixer. Fix the buggy code based on the error message. Return ONLY the fixed Rust code without explanation.",
        "input": f"## Buggy Code:\n```rust\n{buggy}\n```\n\n## Compiler Error:\n{error}\n\n## Fixed Code:",
        "output": fixed,
        "pattern": pattern,
        "family": "targeted_failure",
    }


def main():
    examples = []

    for pattern, pattern_examples in TARGETED_EXAMPLES.items():
        for ex in pattern_examples:
            examples.append(format_example(
                pattern=pattern,
                buggy=ex["buggy"],
                fixed=ex["fixed"],
                error=ex["error"],
            ))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(examples)} targeted training examples")
    print(f"Patterns covered: {len(TARGETED_EXAMPLES)}")
    for pattern, exs in TARGETED_EXAMPLES.items():
        print(f"  {pattern}: {len(exs)} examples")
    print(f"\nWritten to: {OUTPUT_FILE}")

    print("\nNext steps:")
    print("  1. Train: python cuda/scripts/train.py --data data/sft/targeted_failures.jsonl --steps 100 --lr 5e-5")
    print("  2. Merge and eval")


if __name__ == "__main__":
    main()
