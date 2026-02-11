#!/usr/bin/env python3
"""
Generate targeted training data for the 7 persistently failing tasks.

These tasks require specific patterns that our previous data didn't capture well.
"""

import json
from pathlib import Path

SYSTEM_PROMPT = """You are a Rust compiler error fixing assistant. Given buggy Rust code with a compiler error, provide the corrected code that fixes the error. Only output the corrected code, no explanations."""


def make_example(instruction: str, buggy: str, fixed: str) -> dict:
    return {
        "instruction": f"{instruction}\n\nBuggy code:\n```rust\n{buggy}\n```",
        "output": f"```rust\n{fixed}\n```"
    }


# ============================================================================
# bc_003: Fix mutable borrow while immutable borrow exists
# Pattern: Copy the value instead of borrowing, or restructure code
# ============================================================================
BC_003_EXAMPLES = [
    # Basic vector index - copy instead of borrow
    make_example(
        "Fix: cannot borrow `v` as mutable because it is also borrowed as immutable",
        "fn main() { let mut v = vec![1, 2, 3]; let first = &v[0]; v.push(4); println!(\"{}\", first); }",
        "fn main() { let mut v = vec![1, 2, 3]; let first = v[0]; v.push(4); println!(\"{}\", first); }"
    ),
    # With different types
    make_example(
        "Fix: cannot borrow `data` as mutable because it is also borrowed as immutable",
        "fn main() { let mut data = vec![10, 20, 30]; let x = &data[1]; data.push(40); println!(\"{}\", x); }",
        "fn main() { let mut data = vec![10, 20, 30]; let x = data[1]; data.push(40); println!(\"{}\", x); }"
    ),
    # Clone approach
    make_example(
        "Fix: cannot borrow `items` as mutable because it is also borrowed as immutable",
        "fn main() { let mut items = vec![String::from(\"a\")]; let first = &items[0]; items.push(String::from(\"b\")); println!(\"{}\", first); }",
        "fn main() { let mut items = vec![String::from(\"a\")]; let first = items[0].clone(); items.push(String::from(\"b\")); println!(\"{}\", first); }"
    ),
    # Restructure to use value before mutation
    make_example(
        "Fix: cannot borrow `nums` as mutable because it is also borrowed as immutable",
        "fn main() { let mut nums = vec![1, 2]; let sum: i32 = nums.iter().sum(); nums.push(sum); }",
        "fn main() { let mut nums = vec![1, 2]; let sum: i32 = nums.iter().sum(); nums.push(sum); }"  # This one is actually fine, let's fix it
    ),
    # Use len() before push
    make_example(
        "Fix: cannot borrow `v` as mutable because it is also borrowed as immutable",
        "fn main() { let mut v = vec![1, 2, 3]; let len = &v.len(); v.push(4); println!(\"{}\", len); }",
        "fn main() { let mut v = vec![1, 2, 3]; let len = v.len(); v.push(4); println!(\"{}\", len); }"
    ),
    # Get last element
    make_example(
        "Fix: cannot borrow `arr` as mutable because it is also borrowed as immutable",
        "fn main() { let mut arr = vec![5, 10, 15]; let last = &arr[arr.len()-1]; arr.push(20); println!(\"{}\", last); }",
        "fn main() { let mut arr = vec![5, 10, 15]; let last = arr[arr.len()-1]; arr.push(20); println!(\"{}\", last); }"
    ),
    # With struct field
    make_example(
        "Fix: cannot borrow `s.items` as mutable because it is also borrowed as immutable",
        "struct S { items: Vec<i32> } fn main() { let mut s = S { items: vec![1] }; let x = &s.items[0]; s.items.push(2); println!(\"{}\", x); }",
        "struct S { items: Vec<i32> } fn main() { let mut s = S { items: vec![1] }; let x = s.items[0]; s.items.push(2); println!(\"{}\", x); }"
    ),
    # Different fix: complete operation before mutating
    make_example(
        "Fix: cannot borrow `buffer` as mutable because it is also borrowed as immutable",
        "fn main() { let mut buffer = vec![1, 2, 3]; for &x in &buffer { buffer.push(x * 2); } }",
        "fn main() { let mut buffer = vec![1, 2, 3]; let doubled: Vec<_> = buffer.iter().map(|&x| x * 2).collect(); buffer.extend(doubled); }"
    ),
]

# ============================================================================
# bc_005: Fix double mutable borrow
# Pattern: Sequence the borrows - use one, drop it, then use another
# ============================================================================
BC_005_EXAMPLES = [
    # Basic - sequence the operations
    make_example(
        "Fix: cannot borrow `s` as mutable more than once at a time",
        "fn main() { let mut s = String::from(\"hello\"); let r1 = &mut s; let r2 = &mut s; println!(\"{} {}\", r1, r2); }",
        "fn main() { let mut s = String::from(\"hello\"); let r1 = &mut s; println!(\"{}\", r1); let r2 = &mut s; println!(\"{}\", r2); }"
    ),
    # With push operations
    make_example(
        "Fix: cannot borrow `v` as mutable more than once at a time",
        "fn main() { let mut v = Vec::new(); let a = &mut v; let b = &mut v; a.push(1); b.push(2); }",
        "fn main() { let mut v = Vec::new(); let a = &mut v; a.push(1); let b = &mut v; b.push(2); }"
    ),
    # Use scope to limit borrow
    make_example(
        "Fix: cannot borrow `data` as mutable more than once at a time",
        "fn main() { let mut data = vec![1, 2, 3]; let x = &mut data; let y = &mut data; x.push(4); y.push(5); }",
        "fn main() { let mut data = vec![1, 2, 3]; { let x = &mut data; x.push(4); } { let y = &mut data; y.push(5); } }"
    ),
    # String manipulation
    make_example(
        "Fix: cannot borrow `text` as mutable more than once at a time",
        "fn main() { let mut text = String::from(\"hi\"); let a = &mut text; let b = &mut text; a.push('!'); b.push('?'); }",
        "fn main() { let mut text = String::from(\"hi\"); let a = &mut text; a.push('!'); let b = &mut text; b.push('?'); }"
    ),
    # With function calls
    make_example(
        "Fix: cannot borrow `nums` as mutable more than once at a time",
        "fn add(v: &mut Vec<i32>, x: i32) { v.push(x); } fn main() { let mut nums = vec![]; let a = &mut nums; let b = &mut nums; add(a, 1); add(b, 2); }",
        "fn add(v: &mut Vec<i32>, x: i32) { v.push(x); } fn main() { let mut nums = vec![]; add(&mut nums, 1); add(&mut nums, 2); }"
    ),
    # Swap operation
    make_example(
        "Fix: cannot borrow `arr` as mutable more than once at a time",
        "fn main() { let mut arr = [1, 2]; let a = &mut arr[0]; let b = &mut arr[1]; std::mem::swap(a, b); }",
        "fn main() { let mut arr = [1, 2]; arr.swap(0, 1); }"
    ),
    # Clear and refill
    make_example(
        "Fix: cannot borrow `list` as mutable more than once at a time",
        "fn main() { let mut list = vec![1, 2]; let a = &mut list; let b = &mut list; a.clear(); b.push(3); }",
        "fn main() { let mut list = vec![1, 2]; list.clear(); list.push(3); }"
    ),
]

# ============================================================================
# bc_010: Fix returning reference to local
# Pattern: Return owned value instead of reference
# ============================================================================
BC_010_EXAMPLES = [
    # Basic String
    make_example(
        "Fix: returns a reference to data owned by the current function",
        "fn get_str() -> &str { let s = String::from(\"hello\"); &s }",
        "fn get_str() -> String { String::from(\"hello\") }"
    ),
    # With parameter
    make_example(
        "Fix: returns a reference to data owned by the current function",
        "fn make_greeting(name: &str) -> &str { let s = format!(\"Hello, {}!\", name); &s }",
        "fn make_greeting(name: &str) -> String { format!(\"Hello, {}!\", name) }"
    ),
    # Vec reference
    make_example(
        "Fix: returns a reference to data owned by the current function",
        "fn get_vec() -> &Vec<i32> { let v = vec![1, 2, 3]; &v }",
        "fn get_vec() -> Vec<i32> { vec![1, 2, 3] }"
    ),
    # Slice from local vec
    make_example(
        "Fix: returns a reference to data owned by the current function",
        "fn get_slice() -> &[i32] { let v = vec![1, 2, 3]; &v[..] }",
        "fn get_slice() -> Vec<i32> { vec![1, 2, 3] }"
    ),
    # Struct field
    make_example(
        "Fix: returns a reference to data owned by the current function",
        "struct Data { value: String } fn get_value() -> &str { let d = Data { value: String::from(\"test\") }; &d.value }",
        "struct Data { value: String } fn get_value() -> String { let d = Data { value: String::from(\"test\") }; d.value }"
    ),
    # Concatenation
    make_example(
        "Fix: returns a reference to data owned by the current function",
        "fn concat(a: &str, b: &str) -> &str { let result = format!(\"{}{}\", a, b); &result }",
        "fn concat(a: &str, b: &str) -> String { format!(\"{}{}\", a, b) }"
    ),
    # Box reference
    make_example(
        "Fix: returns a reference to data owned by the current function",
        "fn make_box() -> &i32 { let b = Box::new(42); &*b }",
        "fn make_box() -> Box<i32> { Box::new(42) }"
    ),
    # Array to slice
    make_example(
        "Fix: returns a reference to data owned by the current function",
        "fn get_arr() -> &[u8] { let arr = [1u8, 2, 3]; &arr }",
        "fn get_arr() -> Vec<u8> { vec![1u8, 2, 3] }"
    ),
]

# ============================================================================
# tb_002: Add missing Clone trait
# Pattern: Add #[derive(Clone)] to struct
# ============================================================================
TB_002_EXAMPLES = [
    # Basic struct
    make_example(
        "Fix: no method named `clone` found; the trait `Clone` is not implemented",
        "struct Data { value: i32 } fn main() { let d = Data { value: 42 }; let d2 = d.clone(); }",
        "#[derive(Clone)] struct Data { value: i32 } fn main() { let d = Data { value: 42 }; let d2 = d.clone(); }"
    ),
    # Struct with String field
    make_example(
        "Fix: no method named `clone` found; the trait `Clone` is not implemented",
        "struct Person { name: String, age: u32 } fn main() { let p = Person { name: String::from(\"Alice\"), age: 30 }; let p2 = p.clone(); }",
        "#[derive(Clone)] struct Person { name: String, age: u32 } fn main() { let p = Person { name: String::from(\"Alice\"), age: 30 }; let p2 = p.clone(); }"
    ),
    # With Vec field
    make_example(
        "Fix: no method named `clone` found; the trait `Clone` is not implemented",
        "struct Container { items: Vec<i32> } fn main() { let c = Container { items: vec![1, 2] }; let c2 = c.clone(); }",
        "#[derive(Clone)] struct Container { items: Vec<i32> } fn main() { let c = Container { items: vec![1, 2] }; let c2 = c.clone(); }"
    ),
    # Tuple struct
    make_example(
        "Fix: no method named `clone` found; the trait `Clone` is not implemented",
        "struct Point(i32, i32); fn main() { let p = Point(1, 2); let p2 = p.clone(); }",
        "#[derive(Clone)] struct Point(i32, i32); fn main() { let p = Point(1, 2); let p2 = p.clone(); }"
    ),
    # In function
    make_example(
        "Fix: no method named `clone` found; the trait `Clone` is not implemented",
        "struct Config { debug: bool } fn copy_config(c: &Config) -> Config { c.clone() }",
        "#[derive(Clone)] struct Config { debug: bool } fn copy_config(c: &Config) -> Config { c.clone() }"
    ),
    # Multiple derives needed
    make_example(
        "Fix: no method named `clone` found; the trait `Clone` is not implemented",
        "#[derive(Debug)] struct Item { id: u64 } fn main() { let i = Item { id: 1 }; let i2 = i.clone(); }",
        "#[derive(Debug, Clone)] struct Item { id: u64 } fn main() { let i = Item { id: 1 }; let i2 = i.clone(); }"
    ),
    # Clone in vec
    make_example(
        "Fix: no method named `clone` found; the trait `Clone` is not implemented",
        "struct Entry { key: String } fn main() { let entries = vec![Entry { key: String::from(\"a\") }]; let copy = entries.clone(); }",
        "#[derive(Clone)] struct Entry { key: String } fn main() { let entries = vec![Entry { key: String::from(\"a\") }]; let copy = entries.clone(); }"
    ),
]

# ============================================================================
# tb_007: Add Hash trait for HashMap key
# Pattern: Add #[derive(Hash, PartialEq, Eq)] to struct used as key
# ============================================================================
TB_007_EXAMPLES = [
    # Basic HashMap key
    make_example(
        "Fix: the trait `Hash` is not implemented for `Key`",
        "use std::collections::HashMap; struct Key { id: i32 } fn main() { let mut map = HashMap::new(); map.insert(Key { id: 1 }, \"value\"); }",
        "use std::collections::HashMap; #[derive(Hash, PartialEq, Eq)] struct Key { id: i32 } fn main() { let mut map = HashMap::new(); map.insert(Key { id: 1 }, \"value\"); }"
    ),
    # With String field
    make_example(
        "Fix: the trait `Hash` is not implemented for `UserId`",
        "use std::collections::HashMap; struct UserId { name: String } fn main() { let mut users = HashMap::new(); users.insert(UserId { name: String::from(\"alice\") }, 100); }",
        "use std::collections::HashMap; #[derive(Hash, PartialEq, Eq)] struct UserId { name: String } fn main() { let mut users = HashMap::new(); users.insert(UserId { name: String::from(\"alice\") }, 100); }"
    ),
    # HashSet
    make_example(
        "Fix: the trait `Hash` is not implemented for `Item`",
        "use std::collections::HashSet; struct Item { code: u32 } fn main() { let mut set = HashSet::new(); set.insert(Item { code: 42 }); }",
        "use std::collections::HashSet; #[derive(Hash, PartialEq, Eq)] struct Item { code: u32 } fn main() { let mut set = HashSet::new(); set.insert(Item { code: 42 }); }"
    ),
    # Tuple struct as key
    make_example(
        "Fix: the trait `Hash` is not implemented for `Coord`",
        "use std::collections::HashMap; struct Coord(i32, i32); fn main() { let mut grid = HashMap::new(); grid.insert(Coord(0, 0), true); }",
        "use std::collections::HashMap; #[derive(Hash, PartialEq, Eq)] struct Coord(i32, i32); fn main() { let mut grid = HashMap::new(); grid.insert(Coord(0, 0), true); }"
    ),
    # Already has some derives
    make_example(
        "Fix: the trait `Hash` is not implemented for `Id`",
        "use std::collections::HashMap; #[derive(Debug, Clone)] struct Id { val: i64 } fn main() { let mut m = HashMap::new(); m.insert(Id { val: 1 }, \"x\"); }",
        "use std::collections::HashMap; #[derive(Debug, Clone, Hash, PartialEq, Eq)] struct Id { val: i64 } fn main() { let mut m = HashMap::new(); m.insert(Id { val: 1 }, \"x\"); }"
    ),
    # Multiple fields
    make_example(
        "Fix: the trait `Hash` is not implemented for `CacheKey`",
        "use std::collections::HashMap; struct CacheKey { prefix: String, id: u32 } fn main() { let mut cache = HashMap::new(); cache.insert(CacheKey { prefix: String::from(\"user\"), id: 1 }, vec![]); }",
        "use std::collections::HashMap; #[derive(Hash, PartialEq, Eq)] struct CacheKey { prefix: String, id: u32 } fn main() { let mut cache = HashMap::new(); cache.insert(CacheKey { prefix: String::from(\"user\"), id: 1 }, vec![]); }"
    ),
]

# ============================================================================
# tb_008: Add Ord trait for sorting
# Pattern: Add #[derive(PartialEq, Eq, PartialOrd, Ord)] for sort()
# ============================================================================
TB_008_EXAMPLES = [
    # Basic sort
    make_example(
        "Fix: the trait `Ord` is not implemented for `Score`",
        "struct Score { value: i32 } fn main() { let mut scores = vec![Score { value: 3 }, Score { value: 1 }]; scores.sort(); }",
        "#[derive(PartialEq, Eq, PartialOrd, Ord)] struct Score { value: i32 } fn main() { let mut scores = vec![Score { value: 3 }, Score { value: 1 }]; scores.sort(); }"
    ),
    # With multiple fields
    make_example(
        "Fix: the trait `Ord` is not implemented for `Entry`",
        "struct Entry { priority: u32, name: String } fn main() { let mut entries = vec![Entry { priority: 2, name: String::from(\"b\") }]; entries.sort(); }",
        "#[derive(PartialEq, Eq, PartialOrd, Ord)] struct Entry { priority: u32, name: String } fn main() { let mut entries = vec![Entry { priority: 2, name: String::from(\"b\") }]; entries.sort(); }"
    ),
    # BinaryHeap
    make_example(
        "Fix: the trait `Ord` is not implemented for `Task`",
        "use std::collections::BinaryHeap; struct Task { priority: i32 } fn main() { let mut heap = BinaryHeap::new(); heap.push(Task { priority: 5 }); }",
        "use std::collections::BinaryHeap; #[derive(PartialEq, Eq, PartialOrd, Ord)] struct Task { priority: i32 } fn main() { let mut heap = BinaryHeap::new(); heap.push(Task { priority: 5 }); }"
    ),
    # BTreeSet
    make_example(
        "Fix: the trait `Ord` is not implemented for `Key`",
        "use std::collections::BTreeSet; struct Key { id: i32 } fn main() { let mut set = BTreeSet::new(); set.insert(Key { id: 1 }); }",
        "use std::collections::BTreeSet; #[derive(PartialEq, Eq, PartialOrd, Ord)] struct Key { id: i32 } fn main() { let mut set = BTreeSet::new(); set.insert(Key { id: 1 }); }"
    ),
    # sort_by alternative shown but derive is simpler
    make_example(
        "Fix: the trait `Ord` is not implemented for `Item`",
        "struct Item { weight: u32 } fn main() { let mut items = vec![Item { weight: 10 }]; items.sort(); }",
        "#[derive(PartialEq, Eq, PartialOrd, Ord)] struct Item { weight: u32 } fn main() { let mut items = vec![Item { weight: 10 }]; items.sort(); }"
    ),
    # Tuple struct
    make_example(
        "Fix: the trait `Ord` is not implemented for `Rank`",
        "struct Rank(u32); fn main() { let mut ranks = vec![Rank(3), Rank(1), Rank(2)]; ranks.sort(); }",
        "#[derive(PartialEq, Eq, PartialOrd, Ord)] struct Rank(u32); fn main() { let mut ranks = vec![Rank(3), Rank(1), Rank(2)]; ranks.sort(); }"
    ),
    # max/min operations
    make_example(
        "Fix: the trait `Ord` is not implemented for `Value`",
        "struct Value { n: i32 } fn main() { let vals = vec![Value { n: 5 }, Value { n: 2 }]; let m = vals.iter().max(); }",
        "#[derive(PartialEq, Eq, PartialOrd, Ord)] struct Value { n: i32 } fn main() { let vals = vec![Value { n: 5 }, Value { n: 2 }]; let m = vals.iter().max(); }"
    ),
]

# ============================================================================
# rh_004: Use ok_or to convert Option to Result
# Pattern: Option -> Result conversion with ok_or/ok_or_else
# ============================================================================
RH_004_EXAMPLES = [
    # Basic ok_or
    make_example(
        "Fix: expected `Result<i32, &str>`, found `Option<i32>`",
        "fn parse_first(v: &[&str]) -> Result<i32, &str> { v.first().map(|s| s.parse().unwrap()) }",
        "fn parse_first(v: &[&str]) -> Result<i32, &str> { v.first().ok_or(\"empty\")?.parse().map_err(|_| \"parse error\") }"
    ),
    # Simple conversion
    make_example(
        "Fix: expected `Result<_, _>`, found `Option<_>`",
        "fn get_value(opt: Option<i32>) -> Result<i32, &'static str> { opt }",
        "fn get_value(opt: Option<i32>) -> Result<i32, &'static str> { opt.ok_or(\"no value\") }"
    ),
    # With map
    make_example(
        "Fix: expected `Result<String, &str>`, found `Option<String>`",
        "fn get_name(data: Option<&str>) -> Result<String, &str> { data.map(|s| s.to_uppercase()) }",
        "fn get_name(data: Option<&str>) -> Result<String, &str> { data.map(|s| s.to_uppercase()).ok_or(\"missing\") }"
    ),
    # Chain with ?
    make_example(
        "Fix: the `?` operator can only be applied to values that implement `Try`",
        "fn find_user(id: u32) -> Result<String, &'static str> { let users = vec![(1, \"alice\")]; users.iter().find(|(i, _)| *i == id).map(|(_, n)| n.to_string()) }",
        "fn find_user(id: u32) -> Result<String, &'static str> { let users = vec![(1, \"alice\")]; users.iter().find(|(i, _)| *i == id).map(|(_, n)| n.to_string()).ok_or(\"not found\") }"
    ),
    # HashMap get
    make_example(
        "Fix: expected `Result<&i32, &str>`, found `Option<&i32>`",
        "use std::collections::HashMap; fn lookup(m: &HashMap<String, i32>, k: &str) -> Result<&i32, &str> { m.get(k) }",
        "use std::collections::HashMap; fn lookup(m: &HashMap<String, i32>, k: &str) -> Result<&i32, &str> { m.get(k).ok_or(\"key not found\") }"
    ),
    # Vec get
    make_example(
        "Fix: expected `Result<&T, E>`, found `Option<&T>`",
        "fn get_item<T>(v: &[T], idx: usize) -> Result<&T, &'static str> { v.get(idx) }",
        "fn get_item<T>(v: &[T], idx: usize) -> Result<&T, &'static str> { v.get(idx).ok_or(\"index out of bounds\") }"
    ),
    # parse with ok_or_else
    make_example(
        "Fix: expected `Result<_, String>`, found `Option<_>`",
        "fn parse_env(key: &str) -> Result<String, String> { std::env::var(key).ok().filter(|s| !s.is_empty()) }",
        "fn parse_env(key: &str) -> Result<String, String> { std::env::var(key).ok().filter(|s| !s.is_empty()).ok_or_else(|| format!(\"missing or empty: {}\", key)) }"
    ),
    # first() with ok_or
    make_example(
        "Fix: expected `Result<&i32, &str>`, found `Option<&i32>`",
        "fn head(v: &[i32]) -> Result<&i32, &str> { v.first() }",
        "fn head(v: &[i32]) -> Result<&i32, &str> { v.first().ok_or(\"empty slice\") }"
    ),
]


def main():
    output_dir = Path("data/sft/targeted_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_examples = []

    datasets = [
        ("bc_003_immut_mut_borrow", BC_003_EXAMPLES),
        ("bc_005_double_mut_borrow", BC_005_EXAMPLES),
        ("bc_010_return_local_ref", BC_010_EXAMPLES),
        ("tb_002_missing_clone", TB_002_EXAMPLES),
        ("tb_007_missing_hash", TB_007_EXAMPLES),
        ("tb_008_missing_ord", TB_008_EXAMPLES),
        ("rh_004_option_to_result", RH_004_EXAMPLES),
    ]

    for name, examples in datasets:
        output_file = output_dir / f"{name}.jsonl"
        with open(output_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(examples)} examples to {output_file.name}")
        all_examples.extend(examples)

    # Combined file
    all_file = output_dir / "all_targeted_v2.jsonl"
    with open(all_file, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nTotal: {len(all_examples)} examples in {all_file.name}")


if __name__ == "__main__":
    main()
