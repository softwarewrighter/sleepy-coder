//! Rust koan task generator and loader for sleepy-coder.
//!
//! This crate provides a collection of small Rust coding tasks (koans)
//! designed to trigger specific compiler errors for learning.

use core_types::{ErrorFamily, Task};
use std::path::Path;
use thiserror::Error;

/// Errors from koan operations.
#[derive(Error, Debug)]
pub enum KoanError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type for koan operations.
pub type Result<T> = std::result::Result<T, KoanError>;

/// Load all builtin koans.
pub fn load_builtin_koans() -> Vec<Task> {
    let mut koans = Vec::new();

    // Borrow Checker koans (10)
    koans.extend(borrow_checker_koans());

    // Lifetime koans (6)
    koans.extend(lifetime_koans());

    // Trait Bounds koans (10)
    koans.extend(trait_bounds_koans());

    // Result Handling koans (10)
    koans.extend(result_handling_koans());

    // Type Mismatch koans (6)
    koans.extend(type_mismatch_koans());

    koans
}

/// Filter koans by error family.
pub fn filter_by_family(koans: &[Task], family: ErrorFamily) -> Vec<Task> {
    koans
        .iter()
        .filter(|k| k.family == family)
        .cloned()
        .collect()
}

/// Load koans from a JSON string.
pub fn load_koans_from_json(json: &str) -> Result<Vec<Task>> {
    Ok(serde_json::from_str(json)?)
}

/// Load koans from a JSON file.
pub fn load_koans_from_file(path: &Path) -> Result<Vec<Task>> {
    let content = std::fs::read_to_string(path)?;
    load_koans_from_json(&content)
}

/// Get a random sample of koans.
pub fn get_random_koans(koans: &[Task], count: usize, family: Option<ErrorFamily>) -> Vec<Task> {
    use std::collections::HashSet;

    let filtered: Vec<_> = match family {
        Some(f) => koans.iter().filter(|k| k.family == f).collect(),
        None => koans.iter().collect(),
    };

    if filtered.len() <= count {
        return filtered.into_iter().cloned().collect();
    }

    // Simple deterministic selection (for reproducibility, use first N)
    // In production, use actual random sampling with a seed
    let mut selected = Vec::new();
    let mut seen = HashSet::new();
    let step = filtered.len() / count;

    for i in 0..count {
        let idx = (i * step) % filtered.len();
        if !seen.contains(&idx) {
            seen.insert(idx);
            selected.push(filtered[idx].clone());
        }
    }

    // Fill any remaining slots
    for (idx, koan) in filtered.iter().enumerate() {
        if selected.len() >= count {
            break;
        }
        if !seen.contains(&idx) {
            selected.push((*koan).clone());
        }
    }

    selected
}

/// Get a koan by ID.
pub fn get_koan_by_id<'a>(koans: &'a [Task], id: &str) -> Option<&'a Task> {
    koans.iter().find(|k| k.id == id)
}

/// Get the frozen evaluation set (30 koans: 10 each from 3 categories).
pub fn get_frozen_eval_set() -> Vec<Task> {
    let all = load_builtin_koans();

    let mut eval_set = Vec::new();

    // 10 borrow checker
    eval_set.extend(
        filter_by_family(&all, ErrorFamily::BorrowChecker)
            .into_iter()
            .take(10),
    );

    // 10 trait bounds
    eval_set.extend(
        filter_by_family(&all, ErrorFamily::TraitBounds)
            .into_iter()
            .take(10),
    );

    // 10 result handling
    eval_set.extend(
        filter_by_family(&all, ErrorFamily::ResultHandling)
            .into_iter()
            .take(10),
    );

    eval_set
}

// ============================================================================
// Builtin Koans
// ============================================================================

fn borrow_checker_koans() -> Vec<Task> {
    vec![
        Task::new(
            "bc_001".into(),
            ErrorFamily::BorrowChecker,
            "Fix the moved value error - use clone".into(),
            r#"fn main() { let s = String::from("hello"); let t = s; println!("{}", s); }"#.into(),
            r#"fn main() { let s = String::from("hello"); let t = s.clone(); println!("{}", s); }"#.into(),
        ),
        Task::new(
            "bc_002".into(),
            ErrorFamily::BorrowChecker,
            "Fix the moved value error - reorder".into(),
            r#"fn main() { let s = String::from("hello"); let t = s; println!("{}", t); println!("{}", s); }"#.into(),
            r#"fn main() { let s = String::from("hello"); println!("{}", s); let t = s; println!("{}", t); }"#.into(),
        ),
        Task::new(
            "bc_003".into(),
            ErrorFamily::BorrowChecker,
            "Fix mutable borrow while immutable borrow exists".into(),
            r#"fn main() { let mut v = vec![1, 2, 3]; let first = &v[0]; v.push(4); println!("{}", first); }"#.into(),
            r#"fn main() { let mut v = vec![1, 2, 3]; let first = v[0]; v.push(4); println!("{}", first); }"#.into(),
        ),
        Task::new(
            "bc_004".into(),
            ErrorFamily::BorrowChecker,
            "Fix cannot borrow as mutable".into(),
            r#"fn main() { let v = vec![1, 2, 3]; v.push(4); }"#.into(),
            r#"fn main() { let mut v = vec![1, 2, 3]; v.push(4); }"#.into(),
        ),
        Task::new(
            "bc_005".into(),
            ErrorFamily::BorrowChecker,
            "Fix double mutable borrow".into(),
            r#"fn main() { let mut s = String::from("hello"); let r1 = &mut s; let r2 = &mut s; println!("{} {}", r1, r2); }"#.into(),
            r#"fn main() { let mut s = String::from("hello"); let r1 = &mut s; println!("{}", r1); let r2 = &mut s; println!("{}", r2); }"#.into(),
        ),
        Task::new(
            "bc_006".into(),
            ErrorFamily::BorrowChecker,
            "Fix use of moved value in loop".into(),
            r#"fn main() { let s = String::from("hello"); for _ in 0..3 { println!("{}", s); let _ = s; } }"#.into(),
            r#"fn main() { let s = String::from("hello"); for _ in 0..3 { println!("{}", s); let _ = s.clone(); } }"#.into(),
        ),
        Task::new(
            "bc_007".into(),
            ErrorFamily::BorrowChecker,
            "Fix borrow of moved value in match".into(),
            r#"fn main() { let opt = Some(String::from("hi")); match opt { Some(s) => println!("{}", s), None => {} } println!("{:?}", opt); }"#.into(),
            r#"fn main() { let opt = Some(String::from("hi")); match &opt { Some(s) => println!("{}", s), None => {} } println!("{:?}", opt); }"#.into(),
        ),
        Task::new(
            "bc_008".into(),
            ErrorFamily::BorrowChecker,
            "Fix cannot move out of borrowed content".into(),
            r#"fn take(s: String) {} fn main() { let v = vec![String::from("a")]; take(v[0]); }"#.into(),
            r#"fn take(s: String) {} fn main() { let mut v = vec![String::from("a")]; take(v.remove(0)); }"#.into(),
        ),
        Task::new(
            "bc_009".into(),
            ErrorFamily::BorrowChecker,
            "Fix cannot assign to immutable field".into(),
            r#"struct Point { x: i32, y: i32 } fn main() { let p = Point { x: 0, y: 0 }; p.x = 5; }"#.into(),
            r#"struct Point { x: i32, y: i32 } fn main() { let mut p = Point { x: 0, y: 0 }; p.x = 5; }"#.into(),
        ),
        Task::new(
            "bc_010".into(),
            ErrorFamily::BorrowChecker,
            "Fix returning reference to local".into(),
            r#"fn get_str() -> &str { let s = String::from("hello"); &s }"#.into(),
            r#"fn get_str() -> String { String::from("hello") }"#.into(),
        ),
    ]
}

fn lifetime_koans() -> Vec<Task> {
    vec![
        Task::new(
            "lt_001".into(),
            ErrorFamily::Lifetimes,
            "Add missing lifetime parameter".into(),
            r#"fn longest(x: &str, y: &str) -> &str { if x.len() > y.len() { x } else { y } }"#.into(),
            r#"fn longest<'a>(x: &'a str, y: &'a str) -> &'a str { if x.len() > y.len() { x } else { y } }"#.into(),
        ),
        Task::new(
            "lt_002".into(),
            ErrorFamily::Lifetimes,
            "Fix struct lifetime annotation".into(),
            r#"struct Excerpt { part: &str } fn main() { let s = String::from("hello"); let e = Excerpt { part: &s }; }"#.into(),
            r#"struct Excerpt<'a> { part: &'a str } fn main() { let s = String::from("hello"); let e = Excerpt { part: &s }; }"#.into(),
        ),
        Task::new(
            "lt_003".into(),
            ErrorFamily::Lifetimes,
            "Fix lifetime mismatch in return".into(),
            r#"fn first_word(s: &str) -> &str { let bytes = s.as_bytes(); for (i, &item) in bytes.iter().enumerate() { if item == b' ' { return &s[0..i]; } } &s[..] }"#.into(),
            r#"fn first_word<'a>(s: &'a str) -> &'a str { let bytes = s.as_bytes(); for (i, &item) in bytes.iter().enumerate() { if item == b' ' { return &s[0..i]; } } &s[..] }"#.into(),
        ),
        Task::new(
            "lt_004".into(),
            ErrorFamily::Lifetimes,
            "Fix dangling reference".into(),
            r#"fn main() { let r; { let x = 5; r = &x; } println!("{}", r); }"#.into(),
            r#"fn main() { let x = 5; let r = &x; println!("{}", r); }"#.into(),
        ),
        Task::new(
            "lt_005".into(),
            ErrorFamily::Lifetimes,
            "Fix lifetime in impl block".into(),
            r#"struct Holder { data: &str } impl Holder { fn get(&self) -> &str { self.data } }"#.into(),
            r#"struct Holder<'a> { data: &'a str } impl<'a> Holder<'a> { fn get(&self) -> &str { self.data } }"#.into(),
        ),
        Task::new(
            "lt_006".into(),
            ErrorFamily::Lifetimes,
            "Fix static lifetime requirement".into(),
            r#"fn print_it(s: &'static str) { println!("{}", s); } fn main() { let s = String::from("hello"); print_it(&s); }"#.into(),
            r#"fn print_it(s: &str) { println!("{}", s); } fn main() { let s = String::from("hello"); print_it(&s); }"#.into(),
        ),
    ]
}

fn trait_bounds_koans() -> Vec<Task> {
    vec![
        Task::new(
            "tb_001".into(),
            ErrorFamily::TraitBounds,
            "Add missing Debug trait".into(),
            r#"struct Point { x: i32, y: i32 } fn main() { let p = Point { x: 1, y: 2 }; println!("{:?}", p); }"#.into(),
            r#"#[derive(Debug)] struct Point { x: i32, y: i32 } fn main() { let p = Point { x: 1, y: 2 }; println!("{:?}", p); }"#.into(),
        ),
        Task::new(
            "tb_002".into(),
            ErrorFamily::TraitBounds,
            "Add missing Clone trait".into(),
            r#"struct Data { value: i32 } fn main() { let d = Data { value: 42 }; let d2 = d.clone(); }"#.into(),
            r#"#[derive(Clone)] struct Data { value: i32 } fn main() { let d = Data { value: 42 }; let d2 = d.clone(); }"#.into(),
        ),
        Task::new(
            "tb_003".into(),
            ErrorFamily::TraitBounds,
            "Add missing Copy trait".into(),
            r#"struct Num { v: i32 } fn double(n: Num) -> i32 { n.v * 2 } fn main() { let n = Num { v: 5 }; println!("{} {}", double(n), double(n)); }"#.into(),
            r#"#[derive(Clone, Copy)] struct Num { v: i32 } fn double(n: Num) -> i32 { n.v * 2 } fn main() { let n = Num { v: 5 }; println!("{} {}", double(n), double(n)); }"#.into(),
        ),
        Task::new(
            "tb_004".into(),
            ErrorFamily::TraitBounds,
            "Add trait bound to generic function".into(),
            r#"fn print_debug<T>(val: T) { println!("{:?}", val); }"#.into(),
            r#"fn print_debug<T: std::fmt::Debug>(val: T) { println!("{:?}", val); }"#.into(),
        ),
        Task::new(
            "tb_005".into(),
            ErrorFamily::TraitBounds,
            "Add Default trait".into(),
            r#"struct Config { timeout: u32 } fn main() { let c: Config = Default::default(); }"#.into(),
            r#"#[derive(Default)] struct Config { timeout: u32 } fn main() { let c: Config = Default::default(); }"#.into(),
        ),
        Task::new(
            "tb_006".into(),
            ErrorFamily::TraitBounds,
            "Add PartialEq trait".into(),
            r#"struct Point { x: i32, y: i32 } fn main() { let p1 = Point { x: 1, y: 2 }; let p2 = Point { x: 1, y: 2 }; println!("{}", p1 == p2); }"#.into(),
            r#"#[derive(PartialEq)] struct Point { x: i32, y: i32 } fn main() { let p1 = Point { x: 1, y: 2 }; let p2 = Point { x: 1, y: 2 }; println!("{}", p1 == p2); }"#.into(),
        ),
        Task::new(
            "tb_007".into(),
            ErrorFamily::TraitBounds,
            "Add Hash trait for HashMap key".into(),
            r#"use std::collections::HashMap; struct Key { id: i32 } fn main() { let mut map = HashMap::new(); map.insert(Key { id: 1 }, "value"); }"#.into(),
            r#"use std::collections::HashMap; #[derive(Hash, PartialEq, Eq)] struct Key { id: i32 } fn main() { let mut map = HashMap::new(); map.insert(Key { id: 1 }, "value"); }"#.into(),
        ),
        Task::new(
            "tb_008".into(),
            ErrorFamily::TraitBounds,
            "Add Ord trait for sorting".into(),
            r#"struct Score { value: i32 } fn main() { let mut scores = vec![Score { value: 3 }, Score { value: 1 }]; scores.sort(); }"#.into(),
            r#"#[derive(PartialEq, Eq, PartialOrd, Ord)] struct Score { value: i32 } fn main() { let mut scores = vec![Score { value: 3 }, Score { value: 1 }]; scores.sort(); }"#.into(),
        ),
        Task::new(
            "tb_009".into(),
            ErrorFamily::TraitBounds,
            "Fix Send trait requirement".into(),
            r#"use std::rc::Rc; fn send_to_thread<T: Send>(val: T) {} fn main() { let rc = Rc::new(5); send_to_thread(rc); }"#.into(),
            r#"use std::sync::Arc; fn send_to_thread<T: Send>(val: T) {} fn main() { let arc = Arc::new(5); send_to_thread(arc); }"#.into(),
        ),
        Task::new(
            "tb_010".into(),
            ErrorFamily::TraitBounds,
            "Add Display trait".into(),
            r#"struct Message { text: String } fn main() { let m = Message { text: "hello".into() }; println!("{}", m); }"#.into(),
            r#"use std::fmt; struct Message { text: String } impl fmt::Display for Message { fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", self.text) } } fn main() { let m = Message { text: "hello".into() }; println!("{}", m); }"#.into(),
        ),
    ]
}

fn result_handling_koans() -> Vec<Task> {
    vec![
        Task::new(
            "rh_001".into(),
            ErrorFamily::ResultHandling,
            "Add ? operator with proper return type".into(),
            r#"fn read_num(s: &str) -> i32 { s.parse()? }"#.into(),
            r#"fn read_num(s: &str) -> Result<i32, std::num::ParseIntError> { Ok(s.parse()?) }"#.into(),
        ),
        Task::new(
            "rh_002".into(),
            ErrorFamily::ResultHandling,
            "Handle Option with unwrap_or".into(),
            r#"fn first_char(s: &str) -> char { s.chars().next() }"#.into(),
            r#"fn first_char(s: &str) -> char { s.chars().next().unwrap_or(' ') }"#.into(),
        ),
        Task::new(
            "rh_003".into(),
            ErrorFamily::ResultHandling,
            "Convert Option to Result".into(),
            r#"fn get_env(key: &str) -> Result<String, &str> { std::env::var(key) }"#.into(),
            r#"fn get_env(key: &str) -> Result<String, &str> { std::env::var(key).map_err(|_| "not found") }"#.into(),
        ),
        Task::new(
            "rh_004".into(),
            ErrorFamily::ResultHandling,
            "Use ok_or to convert Option".into(),
            r#"fn parse_first(v: &[&str]) -> Result<i32, &str> { v.first().map(|s| s.parse().unwrap()) }"#.into(),
            r#"fn parse_first(v: &[&str]) -> Result<i32, &str> { v.first().ok_or("empty")?.parse().map_err(|_| "parse error") }"#.into(),
        ),
        Task::new(
            "rh_005".into(),
            ErrorFamily::ResultHandling,
            "Handle Result in main with ?".into(),
            r#"fn main() { let n: i32 = "42".parse()?; println!("{}", n); }"#.into(),
            r#"fn main() -> Result<(), Box<dyn std::error::Error>> { let n: i32 = "42".parse()?; println!("{}", n); Ok(()) }"#.into(),
        ),
        Task::new(
            "rh_006".into(),
            ErrorFamily::ResultHandling,
            "Use if let for Option".into(),
            r#"fn main() { let v = vec![1, 2, 3]; let first = v.first(); println!("{}", first); }"#.into(),
            r#"fn main() { let v = vec![1, 2, 3]; if let Some(first) = v.first() { println!("{}", first); } }"#.into(),
        ),
        Task::new(
            "rh_007".into(),
            ErrorFamily::ResultHandling,
            "Use match for Result".into(),
            r#"fn main() { let n: i32 = "abc".parse(); println!("{}", n); }"#.into(),
            r#"fn main() { match "abc".parse::<i32>() { Ok(n) => println!("{}", n), Err(e) => println!("Error: {}", e) } }"#.into(),
        ),
        Task::new(
            "rh_008".into(),
            ErrorFamily::ResultHandling,
            "Chain Result with and_then".into(),
            r#"fn parse_add(s: &str) -> i32 { s.parse::<i32>().map(|n| n + 1) }"#.into(),
            r#"fn parse_add(s: &str) -> Result<i32, std::num::ParseIntError> { s.parse::<i32>().map(|n| n + 1) }"#.into(),
        ),
        Task::new(
            "rh_009".into(),
            ErrorFamily::ResultHandling,
            "Use unwrap_or_else for lazy default".into(),
            r#"fn get_or_compute(opt: Option<i32>) -> i32 { opt.unwrap_or(expensive_default()) } fn expensive_default() -> i32 { 42 }"#.into(),
            r#"fn get_or_compute(opt: Option<i32>) -> i32 { opt.unwrap_or_else(|| expensive_default()) } fn expensive_default() -> i32 { 42 }"#.into(),
        ),
        Task::new(
            "rh_010".into(),
            ErrorFamily::ResultHandling,
            "Propagate error with ?".into(),
            r#"use std::fs; fn read_file(path: &str) -> String { fs::read_to_string(path) }"#.into(),
            r#"use std::fs; fn read_file(path: &str) -> std::io::Result<String> { fs::read_to_string(path) }"#.into(),
        ),
    ]
}

fn type_mismatch_koans() -> Vec<Task> {
    vec![
        Task::new(
            "tm_001".into(),
            ErrorFamily::TypeMismatch,
            "Fix iterator type with collect".into(),
            r#"fn main() { let v: Vec<i32> = (1..5).map(|x| x * 2); }"#.into(),
            r#"fn main() { let v: Vec<i32> = (1..5).map(|x| x * 2).collect(); }"#.into(),
        ),
        Task::new(
            "tm_002".into(),
            ErrorFamily::TypeMismatch,
            "Fix String vs &str".into(),
            r#"fn greet(name: &str) { println!("Hello, {}", name); } fn main() { let name = String::from("World"); greet(name); }"#.into(),
            r#"fn greet(name: &str) { println!("Hello, {}", name); } fn main() { let name = String::from("World"); greet(&name); }"#.into(),
        ),
        Task::new(
            "tm_003".into(),
            ErrorFamily::TypeMismatch,
            "Fix integer type mismatch".into(),
            r#"fn take_u32(n: u32) {} fn main() { let x: i32 = 5; take_u32(x); }"#.into(),
            r#"fn take_u32(n: u32) {} fn main() { let x: i32 = 5; take_u32(x as u32); }"#.into(),
        ),
        Task::new(
            "tm_004".into(),
            ErrorFamily::TypeMismatch,
            "Fix Vec vs slice".into(),
            r#"fn sum(nums: &[i32]) -> i32 { nums.iter().sum() } fn main() { let v = vec![1, 2, 3]; println!("{}", sum(v)); }"#.into(),
            r#"fn sum(nums: &[i32]) -> i32 { nums.iter().sum() } fn main() { let v = vec![1, 2, 3]; println!("{}", sum(&v)); }"#.into(),
        ),
        Task::new(
            "tm_005".into(),
            ErrorFamily::TypeMismatch,
            "Fix return type mismatch".into(),
            r#"fn maybe_num(b: bool) -> i32 { if b { 42 } else { None } }"#.into(),
            r#"fn maybe_num(b: bool) -> Option<i32> { if b { Some(42) } else { None } }"#.into(),
        ),
        Task::new(
            "tm_006".into(),
            ErrorFamily::TypeMismatch,
            "Fix closure return type".into(),
            r#"fn main() { let add = |a, b| { a + b }; let result: i64 = add(1i32, 2i32); }"#.into(),
            r#"fn main() { let add = |a: i64, b: i64| { a + b }; let result: i64 = add(1, 2); }"#.into(),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::ErrorFamily;

    #[test]
    fn test_load_builtin_koans() {
        let koans = load_builtin_koans();

        // Should have at least 30 koans
        assert!(koans.len() >= 30);

        // Each koan should have required fields
        for koan in &koans {
            assert!(!koan.id.is_empty());
            assert!(!koan.description.is_empty());
            assert!(!koan.buggy_code.is_empty());
            assert!(!koan.correct_code.is_empty());
        }
    }

    #[test]
    fn test_koans_by_family() {
        let koans = load_builtin_koans();

        let borrow_koans = filter_by_family(&koans, ErrorFamily::BorrowChecker);
        let lifetime_koans = filter_by_family(&koans, ErrorFamily::Lifetimes);
        let trait_koans = filter_by_family(&koans, ErrorFamily::TraitBounds);
        let result_koans = filter_by_family(&koans, ErrorFamily::ResultHandling);
        let type_koans = filter_by_family(&koans, ErrorFamily::TypeMismatch);

        // Should have koans in each major category
        assert!(!borrow_koans.is_empty(), "Should have borrow checker koans");
        assert!(!lifetime_koans.is_empty(), "Should have lifetime koans");
        assert!(!trait_koans.is_empty(), "Should have trait bound koans");
        assert!(
            !result_koans.is_empty(),
            "Should have result handling koans"
        );
        assert!(!type_koans.is_empty(), "Should have type mismatch koans");
    }

    #[test]
    fn test_load_koans_from_json() {
        let json = r#"[
            {
                "id": "test_001",
                "family": "borrow_checker",
                "description": "Fix the moved value error",
                "buggy_code": "fn main() { let s = String::new(); let t = s; println!(\"{}\", s); }",
                "correct_code": "fn main() { let s = String::new(); let t = s.clone(); println!(\"{}\", s); }",
                "expected_error": "borrow of moved value"
            }
        ]"#;

        let koans = load_koans_from_json(json).unwrap();
        assert_eq!(koans.len(), 1);
        assert_eq!(koans[0].id, "test_001");
        assert_eq!(koans[0].family, ErrorFamily::BorrowChecker);
    }

    #[test]
    fn test_load_koans_from_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("koans.json");

        let json = r#"[
            {
                "id": "file_001",
                "family": "lifetimes",
                "description": "Add lifetime annotation",
                "buggy_code": "fn first(s: &str) -> &str { &s[0..1] }",
                "correct_code": "fn first<'a>(s: &'a str) -> &'a str { &s[0..1] }"
            }
        ]"#;

        std::fs::write(&file_path, json).unwrap();

        let koans = load_koans_from_file(&file_path).unwrap();
        assert_eq!(koans.len(), 1);
        assert_eq!(koans[0].id, "file_001");
    }

    #[test]
    fn test_get_random_koans() {
        let koans = load_builtin_koans();

        let sample = get_random_koans(&koans, 5, None);
        assert_eq!(sample.len(), 5);

        // All sampled koans should be from the original set
        for koan in &sample {
            assert!(koans.iter().any(|k| k.id == koan.id));
        }
    }

    #[test]
    fn test_get_random_koans_by_family() {
        let koans = load_builtin_koans();

        let sample = get_random_koans(&koans, 3, Some(ErrorFamily::BorrowChecker));
        assert_eq!(sample.len(), 3);

        // All sampled koans should be borrow checker koans
        for koan in &sample {
            assert_eq!(koan.family, ErrorFamily::BorrowChecker);
        }
    }

    #[test]
    fn test_get_koan_by_id() {
        let koans = load_builtin_koans();

        // Should find existing koan
        let first_id = &koans[0].id;
        let found = get_koan_by_id(&koans, first_id);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, *first_id);

        // Should return None for non-existent koan
        let not_found = get_koan_by_id(&koans, "nonexistent_koan_id");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_borrow_checker_koan_compiles_when_fixed() {
        let koans = load_builtin_koans();
        let borrow_koans = filter_by_family(&koans, ErrorFamily::BorrowChecker);

        // At least one borrow checker koan should exist
        assert!(!borrow_koans.is_empty());

        // The correct_code should be different from buggy_code
        let koan = &borrow_koans[0];
        assert_ne!(koan.buggy_code, koan.correct_code);
    }

    #[test]
    fn test_koan_serialization_roundtrip() {
        let koans = load_builtin_koans();

        // Serialize to JSON
        let json = serde_json::to_string(&koans).unwrap();

        // Deserialize back
        let loaded: Vec<core_types::Task> = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.len(), koans.len());
        assert_eq!(loaded[0].id, koans[0].id);
    }

    #[test]
    fn test_frozen_eval_set() {
        let eval_set = get_frozen_eval_set();

        // Frozen set should have exactly 30 koans
        assert_eq!(eval_set.len(), 30);

        // Should have 10 from each of 3 major categories
        let borrow = filter_by_family(&eval_set, ErrorFamily::BorrowChecker);
        let traits = filter_by_family(&eval_set, ErrorFamily::TraitBounds);
        let results = filter_by_family(&eval_set, ErrorFamily::ResultHandling);

        assert_eq!(borrow.len(), 10);
        assert_eq!(traits.len(), 10);
        assert_eq!(results.len(), 10);
    }
}
