#!/usr/bin/env python3
"""
Generate SFT training data for Rust 2024 edition patterns.

These patterns represent concepts that Qwen2.5-Coder was NOT trained on
(training cutoff before Rust 1.85 / Feb 2025). This is truly new information
that the model cannot learn from its pretraining data.

Also generates training data for sw-checklist coding standards
(Software Wrighter LLC conventions).

Usage:
    python scripts/generate_rust2024_data.py
"""

import json
from pathlib import Path

OUTPUT_DIR = Path("data/sft/rust2024")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_examples(filename: str, examples: list[dict]):
    """Write examples to JSONL file."""
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} examples to {path}")


# ============================================================
# Pattern 1: Match ergonomics - redundant ref
# ============================================================
match_ergonomics_ref = [
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """fn get_name(names: &[String]) -> &str {
    match names.first() {
        Some(ref name) => name.as_str(),
        None => "unknown",
    }
}

fn main() {
    let names = vec!["Alice".to_string(), "Bob".to_string()];
    println!("{}", get_name(&names));
}""",
        "output": """fn get_name(names: &[String]) -> &str {
    match names.first() {
        Some(name) => name.as_str(),
        None => "unknown",
    }
}

fn main() {
    let names = vec!["Alice".to_string(), "Bob".to_string()];
    println!("{}", get_name(&names));
}""",
        "pattern": "match_ergonomics_redundant_ref",
        "error_hint": "binding modifier `ref` is redundant because the default binding mode is already `ref`",
    },
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """fn process_values(data: &Vec<(String, i32)>) {
    for &(ref key, value) in data {
        println!("{}: {}", key, value);
    }
}

fn main() {
    let data = vec![("a".to_string(), 1), ("b".to_string(), 2)];
    process_values(&data);
}""",
        "output": """fn process_values(data: &Vec<(String, i32)>) {
    for (key, value) in data {
        println!("{}: {}", key, value);
    }
}

fn main() {
    let data = vec![("a".to_string(), 1), ("b".to_string(), 2)];
    process_values(&data);
}""",
        "pattern": "match_ergonomics_redundant_ref",
        "error_hint": "binding modifier `ref` is redundant because the default binding mode is already `ref`",
    },
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """fn find_max(values: &[i32]) -> Option<i32> {
    let mut max = None;
    for &ref val in values {
        match max {
            Some(ref current) if current >= val => {}
            _ => max = Some(*val),
        }
    }
    max
}

fn main() {
    let nums = vec![3, 1, 4, 1, 5, 9];
    println!("{:?}", find_max(&nums));
}""",
        "output": """fn find_max(values: &[i32]) -> Option<i32> {
    let mut max = None;
    for val in values {
        match max {
            Some(current) if current >= *val => {}
            _ => max = Some(*val),
        }
    }
    max
}

fn main() {
    let nums = vec![3, 1, 4, 1, 5, 9];
    println!("{:?}", find_max(&nums));
}""",
        "pattern": "match_ergonomics_redundant_ref",
        "error_hint": "binding modifier `ref` is redundant because the default binding mode is already `ref`",
    },
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """use std::collections::HashMap;

fn get_or_default<'a>(map: &'a HashMap<String, String>, key: &str) -> &'a str {
    match map.get(key) {
        Some(ref value) => value.as_str(),
        None => "default",
    }
}

fn main() {
    let mut map = HashMap::new();
    map.insert("key".to_string(), "value".to_string());
    println!("{}", get_or_default(&map, "key"));
}""",
        "output": """use std::collections::HashMap;

fn get_or_default<'a>(map: &'a HashMap<String, String>, key: &str) -> &'a str {
    match map.get(key) {
        Some(value) => value.as_str(),
        None => "default",
    }
}

fn main() {
    let mut map = HashMap::new();
    map.insert("key".to_string(), "value".to_string());
    println!("{}", get_or_default(&map, "key"));
}""",
        "pattern": "match_ergonomics_redundant_ref",
        "error_hint": "binding modifier `ref` is redundant because the default binding mode is already `ref`",
    },
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """enum Command {
    Move { x: i32, y: i32 },
    Print(String),
    Quit,
}

fn handle(cmd: &Command) {
    match cmd {
        Command::Move { ref x, ref y } => println!("Moving to ({}, {})", x, y),
        Command::Print(ref msg) => println!("{}", msg),
        Command::Quit => println!("Quitting"),
    }
}

fn main() {
    handle(&Command::Print("hello".to_string()));
}""",
        "output": """enum Command {
    Move { x: i32, y: i32 },
    Print(String),
    Quit,
}

fn handle(cmd: &Command) {
    match cmd {
        Command::Move { x, y } => println!("Moving to ({}, {})", x, y),
        Command::Print(msg) => println!("{}", msg),
        Command::Quit => println!("Quitting"),
    }
}

fn main() {
    handle(&Command::Print("hello".to_string()));
}""",
        "pattern": "match_ergonomics_redundant_ref",
        "error_hint": "binding modifier `ref` is redundant because the default binding mode is already `ref`",
    },
]

# ============================================================
# Pattern 2: unsafe extern blocks
# ============================================================
unsafe_extern = [
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """extern "C" {
    fn abs(input: i32) -> i32;
    fn strlen(s: *const u8) -> usize;
}

fn main() {
    let result = unsafe { abs(-42) };
    println!("abs(-42) = {}", result);
}""",
        "output": """unsafe extern "C" {
    fn abs(input: i32) -> i32;
    fn strlen(s: *const u8) -> usize;
}

fn main() {
    let result = unsafe { abs(-42) };
    println!("abs(-42) = {}", result);
}""",
        "pattern": "unsafe_extern_block",
        "error_hint": "extern blocks must be unsafe",
    },
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
    fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
}

fn main() {
    unsafe {
        let ptr = malloc(256);
        if !ptr.is_null() {
            free(ptr);
        }
    }
}""",
        "output": """unsafe extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
    fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
}

fn main() {
    unsafe {
        let ptr = malloc(256);
        if !ptr.is_null() {
            free(ptr);
        }
    }
}""",
        "pattern": "unsafe_extern_block",
        "error_hint": "extern blocks must be unsafe",
    },
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """extern "C" {
    fn getenv(name: *const i8) -> *const i8;
}

extern "C" fn callback(x: i32) -> i32 {
    x * 2
}

fn main() {
    println!("callback(21) = {}", callback(21));
}""",
        "output": """unsafe extern "C" {
    fn getenv(name: *const i8) -> *const i8;
}

extern "C" fn callback(x: i32) -> i32 {
    x * 2
}

fn main() {
    println!("callback(21) = {}", callback(21));
}""",
        "pattern": "unsafe_extern_block",
        "error_hint": "extern blocks must be unsafe",
    },
]

# ============================================================
# Pattern 3: unsafe attributes (#[no_mangle] -> #[unsafe(no_mangle)])
# ============================================================
unsafe_attributes = [
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """#[no_mangle]
pub extern "C" fn rust_multiply(a: i32, b: i32) -> i32 {
    a * b
}

fn main() {
    println!("{}", rust_multiply(6, 7));
}""",
        "output": """#[unsafe(no_mangle)]
pub extern "C" fn rust_multiply(a: i32, b: i32) -> i32 {
    a * b
}

fn main() {
    println!("{}", rust_multiply(6, 7));
}""",
        "pattern": "unsafe_attributes",
        "error_hint": "unsafe attribute used without unsafe",
    },
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """#[export_name = "custom_add"]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[link_section = ".custom_data"]
pub static VERSION: u32 = 1;

fn main() {
    println!("{}", add(1, 2));
}""",
        "output": """#[unsafe(export_name = "custom_add")]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[unsafe(link_section = ".custom_data")]
pub static VERSION: u32 = 1;

fn main() {
    println!("{}", add(1, 2));
}""",
        "pattern": "unsafe_attributes",
        "error_hint": "unsafe attribute used without unsafe",
    },
]

# ============================================================
# Pattern 4: env::set_var now unsafe
# ============================================================
env_set_var_unsafe = [
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """use std::env;

fn setup_environment() {
    env::set_var("RUST_LOG", "debug");
    env::set_var("APP_MODE", "development");
}

fn main() {
    setup_environment();
    println!("RUST_LOG={}", env::var("RUST_LOG").unwrap_or_default());
}""",
        "output": """use std::env;

fn setup_environment() {
    unsafe {
        env::set_var("RUST_LOG", "debug");
        env::set_var("APP_MODE", "development");
    }
}

fn main() {
    setup_environment();
    println!("RUST_LOG={}", env::var("RUST_LOG").unwrap_or_default());
}""",
        "pattern": "env_var_unsafe",
        "error_hint": "call to unsafe function `std::env::set_var` is unsafe and requires unsafe block",
    },
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """fn main() {
    std::env::set_var("MY_VAR", "hello");
    let val = std::env::var("MY_VAR").unwrap();
    println!("{}", val);
    std::env::remove_var("MY_VAR");
}""",
        "output": """fn main() {
    unsafe {
        std::env::set_var("MY_VAR", "hello");
    }
    let val = std::env::var("MY_VAR").unwrap();
    println!("{}", val);
    unsafe {
        std::env::remove_var("MY_VAR");
    }
}""",
        "pattern": "env_var_unsafe",
        "error_hint": "call to unsafe function `std::env::set_var` is unsafe and requires unsafe block",
    },
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        std::env::set_var("TEST_DB_URL", "sqlite::memory:");
    }

    fn teardown() {
        std::env::remove_var("TEST_DB_URL");
    }

    #[test]
    fn test_connection() {
        setup();
        let url = std::env::var("TEST_DB_URL").unwrap();
        assert_eq!(url, "sqlite::memory:");
        teardown();
    }
}

fn main() {}""",
        "output": """#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        unsafe {
            std::env::set_var("TEST_DB_URL", "sqlite::memory:");
        }
    }

    fn teardown() {
        unsafe {
            std::env::remove_var("TEST_DB_URL");
        }
    }

    #[test]
    fn test_connection() {
        setup();
        let url = std::env::var("TEST_DB_URL").unwrap();
        assert_eq!(url, "sqlite::memory:");
        teardown();
    }
}

fn main() {}""",
        "pattern": "env_var_unsafe",
        "error_hint": "call to unsafe function `std::env::set_var` is unsafe and requires unsafe block",
    },
]

# ============================================================
# Pattern 5: unsafe_op_in_unsafe_fn
# ============================================================
unsafe_op_in_unsafe_fn = [
    {
        "instruction": "Fix the Rust 2024 edition compilation warning (treated as error).",
        "input": """unsafe fn get_unchecked(slice: &[i32], index: usize) -> i32 {
    *slice.get_unchecked(index)
}

fn main() {
    let data = [10, 20, 30, 40, 50];
    let val = unsafe { get_unchecked(&data, 2) };
    println!("{}", val);
}""",
        "output": """unsafe fn get_unchecked(slice: &[i32], index: usize) -> i32 {
    unsafe { *slice.get_unchecked(index) }
}

fn main() {
    let data = [10, 20, 30, 40, 50];
    let val = unsafe { get_unchecked(&data, 2) };
    println!("{}", val);
}""",
        "pattern": "unsafe_op_in_unsafe_fn",
        "error_hint": "call to unsafe function is unsafe and requires unsafe block",
    },
    {
        "instruction": "Fix the Rust 2024 edition compilation warning (treated as error).",
        "input": """use std::ptr;

unsafe fn swap_raw<T>(a: *mut T, b: *mut T) {
    let tmp = ptr::read(a);
    ptr::write(a, ptr::read(b));
    ptr::write(b, tmp);
}

fn main() {
    let mut x = 1;
    let mut y = 2;
    unsafe { swap_raw(&mut x, &mut y) };
    println!("x={}, y={}", x, y);
}""",
        "output": """use std::ptr;

unsafe fn swap_raw<T>(a: *mut T, b: *mut T) {
    unsafe {
        let tmp = ptr::read(a);
        ptr::write(a, ptr::read(b));
        ptr::write(b, tmp);
    }
}

fn main() {
    let mut x = 1;
    let mut y = 2;
    unsafe { swap_raw(&mut x, &mut y) };
    println!("x={}, y={}", x, y);
}""",
        "pattern": "unsafe_op_in_unsafe_fn",
        "error_hint": "call to unsafe function is unsafe and requires unsafe block",
    },
]

# ============================================================
# Pattern 6: gen keyword reserved (rand crate)
# ============================================================
gen_keyword = [
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """fn main() {
    let gen = 42;
    println!("Generation: {}", gen);
}""",
        "output": """fn main() {
    let generation = 42;
    println!("Generation: {}", generation);
}""",
        "pattern": "gen_keyword_reserved",
        "error_hint": "expected identifier, found keyword `gen`",
    },
    {
        "instruction": "Fix the Rust 2024 edition compilation error.",
        "input": """struct Config {
    gen: u32,
    name: String,
}

fn main() {
    let config = Config { gen: 1, name: "test".to_string() };
    println!("Gen {}: {}", config.gen, config.name);
}""",
        "output": """struct Config {
    generation: u32,
    name: String,
}

fn main() {
    let config = Config { generation: 1, name: "test".to_string() };
    println!("Gen {}: {}", config.generation, config.name);
}""",
        "pattern": "gen_keyword_reserved",
        "error_hint": "expected identifier, found keyword `gen`",
    },
]

# ============================================================
# Pattern 7: Let chains (idiomatic 2024 style)
# ============================================================
let_chains = [
    {
        "instruction": "Refactor this Rust code to use let chains (Rust 2024 idiom).",
        "input": """fn process(a: Option<i32>, b: Option<i32>) -> Option<i32> {
    if let Some(x) = a {
        if let Some(y) = b {
            if x > 0 && y > 0 {
                return Some(x + y);
            }
        }
    }
    None
}

fn main() {
    println!("{:?}", process(Some(3), Some(4)));
    println!("{:?}", process(Some(-1), Some(4)));
    println!("{:?}", process(None, Some(4)));
}""",
        "output": """fn process(a: Option<i32>, b: Option<i32>) -> Option<i32> {
    if let Some(x) = a && let Some(y) = b && x > 0 && y > 0 {
        return Some(x + y);
    }
    None
}

fn main() {
    println!("{:?}", process(Some(3), Some(4)));
    println!("{:?}", process(Some(-1), Some(4)));
    println!("{:?}", process(None, Some(4)));
}""",
        "pattern": "let_chains",
        "error_hint": "Nested if-let can be simplified with let chains in Rust 2024",
    },
    {
        "instruction": "Refactor this Rust code to use let chains (Rust 2024 idiom).",
        "input": """use std::collections::HashMap;

fn lookup(map: &HashMap<String, Vec<String>>, key: &str, idx: usize) -> Option<String> {
    if let Some(values) = map.get(key) {
        if let Some(item) = values.get(idx) {
            if !item.is_empty() {
                return Some(item.clone());
            }
        }
    }
    None
}

fn main() {
    let mut map = HashMap::new();
    map.insert("greetings".to_string(), vec!["hello".to_string(), "hi".to_string()]);
    println!("{:?}", lookup(&map, "greetings", 0));
}""",
        "output": """use std::collections::HashMap;

fn lookup(map: &HashMap<String, Vec<String>>, key: &str, idx: usize) -> Option<String> {
    if let Some(values) = map.get(key) && let Some(item) = values.get(idx) && !item.is_empty() {
        return Some(item.clone());
    }
    None
}

fn main() {
    let mut map = HashMap::new();
    map.insert("greetings".to_string(), vec!["hello".to_string(), "hi".to_string()]);
    println!("{:?}", lookup(&map, "greetings", 0));
}""",
        "pattern": "let_chains",
        "error_hint": "Nested if-let can be simplified with let chains in Rust 2024",
    },
]

# ============================================================
# SW-Checklist patterns: Function LOC
# ============================================================
function_loc = [
    {
        "instruction": "Refactor this Rust function to be under 50 lines (sw-checklist requirement).",
        "input": """fn process_records(records: Vec<Record>) -> Result<Summary, Error> {
    let mut total = 0;
    let mut count = 0;
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut by_category: HashMap<String, Vec<Record>> = HashMap::new();

    for record in &records {
        if record.value < 0 {
            errors.push(format!("Negative value in record {}", record.id));
            continue;
        }
        if record.value == 0 {
            warnings.push(format!("Zero value in record {}", record.id));
        }
        total += record.value;
        count += 1;
        by_category
            .entry(record.category.clone())
            .or_default()
            .push(record.clone());
    }

    let average = if count > 0 { total / count } else { 0 };

    let mut category_summaries = Vec::new();
    for (cat, cat_records) in &by_category {
        let cat_total: i64 = cat_records.iter().map(|r| r.value).sum();
        let cat_avg = cat_total / cat_records.len() as i64;
        let cat_max = cat_records.iter().map(|r| r.value).max().unwrap_or(0);
        let cat_min = cat_records.iter().map(|r| r.value).min().unwrap_or(0);
        category_summaries.push(CategorySummary {
            name: cat.clone(),
            count: cat_records.len(),
            total: cat_total,
            average: cat_avg,
            max: cat_max,
            min: cat_min,
        });
    }

    category_summaries.sort_by(|a, b| b.total.cmp(&a.total));

    if !errors.is_empty() {
        return Err(Error::ValidationErrors(errors));
    }

    Ok(Summary {
        total,
        count,
        average,
        warnings,
        categories: category_summaries,
    })
}""",
        "output": """fn validate_records(records: &[Record]) -> Result<(), Vec<String>> {
    let errors: Vec<String> = records
        .iter()
        .filter(|r| r.value < 0)
        .map(|r| format!("Negative value in record {}", r.id))
        .collect();
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

fn collect_warnings(records: &[Record]) -> Vec<String> {
    records
        .iter()
        .filter(|r| r.value == 0)
        .map(|r| format!("Zero value in record {}", r.id))
        .collect()
}

fn summarize_category(name: &str, records: &[Record]) -> CategorySummary {
    let total: i64 = records.iter().map(|r| r.value).sum();
    CategorySummary {
        name: name.to_string(),
        count: records.len(),
        total,
        average: total / records.len().max(1) as i64,
        max: records.iter().map(|r| r.value).max().unwrap_or(0),
        min: records.iter().map(|r| r.value).min().unwrap_or(0),
    }
}

fn process_records(records: Vec<Record>) -> Result<Summary, Error> {
    if let Err(errors) = validate_records(&records) {
        return Err(Error::ValidationErrors(errors));
    }

    let valid: Vec<_> = records.iter().filter(|r| r.value >= 0).collect();
    let total: i64 = valid.iter().map(|r| r.value).sum();
    let count = valid.len();

    let mut by_cat: HashMap<String, Vec<Record>> = HashMap::new();
    for r in &valid {
        by_cat.entry(r.category.clone()).or_default().push((*r).clone());
    }

    let mut categories: Vec<_> = by_cat
        .iter()
        .map(|(name, recs)| summarize_category(name, recs))
        .collect();
    categories.sort_by(|a, b| b.total.cmp(&a.total));

    Ok(Summary {
        total,
        count,
        average: if count > 0 { total / count as i64 } else { 0 },
        warnings: collect_warnings(&records),
        categories,
    })
}""",
        "pattern": "function_loc_refactor",
        "error_hint": "Function has too many lines (max 50). Split into smaller focused functions.",
    },
]

# ============================================================
# SW-Checklist patterns: Module function count
# ============================================================
module_function_count = [
    {
        "instruction": "This module has too many functions (max 7 per module). Refactor by splitting into submodules.",
        "input": """// lib.rs - 15 functions in one file
pub fn create_user() { /* ... */ }
pub fn delete_user() { /* ... */ }
pub fn update_user() { /* ... */ }
pub fn get_user() { /* ... */ }
pub fn list_users() { /* ... */ }
pub fn create_post() { /* ... */ }
pub fn delete_post() { /* ... */ }
pub fn update_post() { /* ... */ }
pub fn get_post() { /* ... */ }
pub fn list_posts() { /* ... */ }
pub fn create_comment() { /* ... */ }
pub fn delete_comment() { /* ... */ }
pub fn get_comment() { /* ... */ }
pub fn list_comments() { /* ... */ }
pub fn search() { /* ... */ }""",
        "output": """// lib.rs - split into submodules
mod users;
mod posts;
mod comments;
mod search;

pub use users::{create_user, delete_user, update_user, get_user, list_users};
pub use posts::{create_post, delete_post, update_post, get_post, list_posts};
pub use comments::{create_comment, delete_comment, get_comment, list_comments};
pub use search::search;

// users.rs
// pub fn create_user() { /* ... */ }
// pub fn delete_user() { /* ... */ }
// pub fn update_user() { /* ... */ }
// pub fn get_user() { /* ... */ }
// pub fn list_users() { /* ... */ }

// posts.rs
// pub fn create_post() { /* ... */ }
// ...etc""",
        "pattern": "module_function_count",
        "error_hint": "Module has too many functions (max 7). Split into submodules.",
    },
]

# ============================================================
# Write all training data
# ============================================================
if __name__ == "__main__":
    write_examples("match_ergonomics_ref.jsonl", match_ergonomics_ref)
    write_examples("unsafe_extern.jsonl", unsafe_extern)
    write_examples("unsafe_attributes.jsonl", unsafe_attributes)
    write_examples("env_set_var_unsafe.jsonl", env_set_var_unsafe)
    write_examples("unsafe_op_in_unsafe_fn.jsonl", unsafe_op_in_unsafe_fn)
    write_examples("gen_keyword.jsonl", gen_keyword)
    write_examples("let_chains.jsonl", let_chains)
    write_examples("function_loc.jsonl", function_loc)
    write_examples("module_function_count.jsonl", module_function_count)

    # Also write a combined file
    all_examples = (
        match_ergonomics_ref
        + unsafe_extern
        + unsafe_extern
        + unsafe_attributes
        + env_set_var_unsafe
        + unsafe_op_in_unsafe_fn
        + gen_keyword
        + let_chains
        + function_loc
        + module_function_count
    )
    write_examples("all_rust2024.jsonl", all_examples)

    print(f"\nTotal: {len(all_examples)} examples across all patterns")
    print("\nBreakdown:")
    print(f"  match_ergonomics_ref: {len(match_ergonomics_ref)}")
    print(f"  unsafe_extern: {len(unsafe_extern)}")
    print(f"  unsafe_attributes: {len(unsafe_attributes)}")
    print(f"  env_set_var_unsafe: {len(env_set_var_unsafe)}")
    print(f"  unsafe_op_in_unsafe_fn: {len(unsafe_op_in_unsafe_fn)}")
    print(f"  gen_keyword: {len(gen_keyword)}")
    print(f"  let_chains: {len(let_chains)}")
    print(f"  function_loc (sw-checklist): {len(function_loc)}")
    print(f"  module_function_count (sw-checklist): {len(module_function_count)}")
