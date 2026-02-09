#!/usr/bin/env python3
"""
Generate training data aligned with the frozen eval set.

The key insight: our training data (yew, axum, sqlx, cli) doesn't overlap
with our eval data (borrow_checker, trait_bounds, result_handling).
This script generates training examples that teach the same patterns
tested in the eval set -- covering ALL 30 eval koans, not just a subset.

Additionally generates:
- Clippy suppression anti-patterns (teach "fix, don't suppress")
- Rust 2024 edition clippy patterns (format strings, let-chains, etc.)
- Code complexity patterns (function decomposition, module splitting)

Previous versions only covered ~10/30 patterns, which is likely why
cycles 9-13 plateaued at 73.3% and couldn't beat the 76.7% baseline.

Usage:
    python scripts/generate_eval_aligned_koans.py
    python scripts/generate_eval_aligned_koans.py --variants 20 --output data/sft/eval_aligned.jsonl
    python scripts/generate_eval_aligned_koans.py --no-supplemental  # eval-aligned only
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
# BORROW CHECKER PATTERNS - All 10 eval koans (bc_001 through bc_010)
# =============================================================================

BORROW_CHECKER_TEMPLATES = [
    # bc_001: Use after move - clone fix
    {
        "pattern": "use_after_move_clone",
        "error_hint": "borrow of moved value: use .clone() before the move",
        "templates": [
            {
                "buggy": 'fn main() {{ let {v} = String::from("{val}"); let {v2} = {v}; println!("{{}}", {v}); }}',
                "fixed": 'fn main() {{ let {v} = String::from("{val}"); let {v2} = {v}.clone(); println!("{{}}", {v}); }}',
                "vars": [
                    {"v": "s", "v2": "t", "val": "hello"},
                    {"v": "name", "v2": "copy", "val": "world"},
                    {"v": "data", "v2": "backup", "val": "test"},
                    {"v": "msg", "v2": "saved", "val": "foo"},
                ],
            },
            {
                "buggy": 'fn main() {{ let {v} = vec![1, 2, 3]; let {v2} = {v}; println!("{{:?}}", {v}); }}',
                "fixed": 'fn main() {{ let {v} = vec![1, 2, 3]; let {v2} = {v}.clone(); println!("{{:?}}", {v}); }}',
                "vars": [
                    {"v": "v", "v2": "w"},
                    {"v": "nums", "v2": "copy"},
                    {"v": "items", "v2": "backup"},
                ],
            },
        ],
    },
    # bc_002: Use after move - reorder fix
    {
        "pattern": "use_after_move_reorder",
        "error_hint": "borrow of moved value: reorder statements so use comes before move",
        "templates": [
            {
                "buggy": 'fn main() {{ let {v} = String::from("{val}"); let {v2} = {v}; println!("{{}}", {v2}); println!("{{}}", {v}); }}',
                "fixed": 'fn main() {{ let {v} = String::from("{val}"); println!("{{}}", {v}); let {v2} = {v}; println!("{{}}", {v2}); }}',
                "vars": [
                    {"v": "s", "v2": "t", "val": "hello"},
                    {"v": "msg", "v2": "taken", "val": "world"},
                    {"v": "text", "v2": "moved", "val": "data"},
                ],
            },
        ],
    },
    # bc_003: Mutable borrow while immutable borrow exists - copy value
    {
        "pattern": "mut_borrow_conflict",
        "error_hint": "cannot borrow as mutable because it is also borrowed as immutable",
        "templates": [
            {
                "buggy": 'fn main() {{ let mut {v} = vec![1, 2, 3]; let {r} = &{v}[0]; {v}.push(4); println!("{{}}", {r}); }}',
                "fixed": 'fn main() {{ let mut {v} = vec![1, 2, 3]; let {r} = {v}[0]; {v}.push(4); println!("{{}}", {r}); }}',
                "vars": [
                    {"v": "v", "r": "first"},
                    {"v": "nums", "r": "head"},
                    {"v": "data", "r": "elem"},
                ],
            },
            {
                "buggy": 'fn main() {{ let mut {v} = vec!["a".to_string(), "b".to_string()]; let {r} = &{v}; {v}.push("c".to_string()); println!("{{:?}}", {r}); }}',
                "fixed": 'fn main() {{ let mut {v} = vec!["a".to_string(), "b".to_string()]; let {r} = {v}.clone(); {v}.push("c".to_string()); println!("{{:?}}", {r}); }}',
                "vars": [
                    {"v": "v", "r": "snapshot"},
                    {"v": "items", "r": "saved"},
                    {"v": "words", "r": "copy"},
                ],
            },
        ],
    },
    # bc_004: Cannot borrow as mutable (missing mut)
    {
        "pattern": "missing_mut",
        "error_hint": "cannot borrow as mutable, as it is not declared as mutable",
        "templates": [
            {
                "buggy": "fn main() {{ let {v} = vec![1, 2, 3]; {v}.push(4); }}",
                "fixed": "fn main() {{ let mut {v} = vec![1, 2, 3]; {v}.push(4); }}",
                "vars": [{"v": "v"}, {"v": "nums"}, {"v": "items"}, {"v": "data"}],
            },
            {
                "buggy": 'fn main() {{ let {v} = String::new(); {v}.push_str("hello"); }}',
                "fixed": 'fn main() {{ let mut {v} = String::new(); {v}.push_str("hello"); }}',
                "vars": [{"v": "s"}, {"v": "text"}, {"v": "msg"}, {"v": "buf"}],
            },
            {
                "buggy": "fn main() {{ let {v} = std::collections::HashMap::new(); {v}.insert(1, 2); }}",
                "fixed": "fn main() {{ let mut {v} = std::collections::HashMap::new(); {v}.insert(1, 2); }}",
                "vars": [{"v": "map"}, {"v": "table"}, {"v": "cache"}],
            },
        ],
    },
    # bc_005: Double mutable borrow - split lifetimes
    {
        "pattern": "double_mut_borrow",
        "error_hint": "cannot borrow as mutable more than once at a time",
        "templates": [
            {
                "buggy": 'fn main() {{ let mut {v} = String::from("hello"); let {r1} = &mut {v}; let {r2} = &mut {v}; println!("{{}} {{}}", {r1}, {r2}); }}',
                "fixed": 'fn main() {{ let mut {v} = String::from("hello"); let {r1} = &mut {v}; println!("{{}}", {r1}); let {r2} = &mut {v}; println!("{{}}", {r2}); }}',
                "vars": [
                    {"v": "s", "r1": "r1", "r2": "r2"},
                    {"v": "text", "r1": "a", "r2": "b"},
                    {"v": "data", "r1": "x", "r2": "y"},
                ],
            },
        ],
    },
    # bc_006: Use of moved value in loop - clone in loop body
    {
        "pattern": "move_in_loop",
        "error_hint": "use of moved value in loop: clone inside the loop body",
        "templates": [
            {
                "buggy": 'fn main() {{ let {v} = String::from("hello"); for _ in 0..3 {{ println!("{{}}", {v}); let _ = {v}; }} }}',
                "fixed": 'fn main() {{ let {v} = String::from("hello"); for _ in 0..3 {{ println!("{{}}", {v}); let _ = {v}.clone(); }} }}',
                "vars": [{"v": "s"}, {"v": "msg"}, {"v": "data"}],
            },
            {
                "buggy": "fn main() {{ let {v} = vec![1, 2, 3]; for _ in 0..2 {{ let _ = {v}; }} }}",
                "fixed": "fn main() {{ let {v} = vec![1, 2, 3]; for _ in 0..2 {{ let _ = {v}.clone(); }} }}",
                "vars": [{"v": "v"}, {"v": "nums"}, {"v": "items"}],
            },
        ],
    },
    # bc_007: Borrow of moved value in match - use reference in match
    {
        "pattern": "match_move",
        "error_hint": "borrow of moved value in match: match on reference instead",
        "templates": [
            {
                "buggy": 'fn main() {{ let {v} = Some(String::from("hi")); match {v} {{ Some({inner}) => println!("{{}}", {inner}), None => {{}} }} println!("{{:?}}", {v}); }}',
                "fixed": 'fn main() {{ let {v} = Some(String::from("hi")); match &{v} {{ Some({inner}) => println!("{{}}", {inner}), None => {{}} }} println!("{{:?}}", {v}); }}',
                "vars": [
                    {"v": "opt", "inner": "s"},
                    {"v": "maybe", "inner": "val"},
                    {"v": "item", "inner": "x"},
                ],
            },
        ],
    },
    # bc_008: Cannot move out of indexed content
    {
        "pattern": "move_out_of_index",
        "error_hint": "cannot move out of index of Vec: use .remove() or .clone()",
        "templates": [
            {
                "buggy": 'fn take({param}: String) {{}} fn main() {{ let mut {v} = vec![String::from("a")]; take({v}[0]); }}',
                "fixed": 'fn take({param}: String) {{}} fn main() {{ let mut {v} = vec![String::from("a")]; take({v}.remove(0)); }}',
                "vars": [
                    {"param": "s", "v": "v"},
                    {"param": "item", "v": "items"},
                    {"param": "val", "v": "data"},
                ],
            },
            {
                "buggy": 'fn main() {{ let {v} = vec![String::from("a"), String::from("b")]; let {out} = {v}[0]; }}',
                "fixed": 'fn main() {{ let {v} = vec![String::from("a"), String::from("b")]; let {out} = {v}[0].clone(); }}',
                "vars": [
                    {"v": "v", "out": "first"},
                    {"v": "items", "out": "item"},
                    {"v": "data", "out": "elem"},
                ],
            },
        ],
    },
    # bc_009: Cannot assign to immutable field
    {
        "pattern": "immutable_field",
        "error_hint": "cannot assign to field of immutable binding: add mut",
        "templates": [
            {
                "buggy": "struct {name} {{ {field}: i32 }} fn main() {{ let {v} = {name} {{ {field}: 0 }}; {v}.{field} = 5; }}",
                "fixed": "struct {name} {{ {field}: i32 }} fn main() {{ let mut {v} = {name} {{ {field}: 0 }}; {v}.{field} = 5; }}",
                "vars": [
                    {"name": "Point", "field": "x", "v": "p"},
                    {"name": "Rect", "field": "width", "v": "r"},
                    {"name": "Counter", "field": "count", "v": "c"},
                ],
            },
        ],
    },
    # bc_010: Returning reference to local variable
    {
        "pattern": "return_local_ref",
        "error_hint": "cannot return reference to local variable: return owned type instead",
        "templates": [
            {
                "buggy": 'fn get_str() -> &str {{ let s = String::from("hello"); &s }}',
                "fixed": 'fn get_str() -> String {{ String::from("hello") }}',
                "vars": [{}],
            },
            {
                "buggy": "fn make_vec() -> &Vec<i32> {{ let v = vec![1, 2, 3]; &v }}",
                "fixed": "fn make_vec() -> Vec<i32> {{ vec![1, 2, 3] }}",
                "vars": [{}],
            },
            {
                "buggy": 'fn create() -> &String {{ let s = String::from("test"); &s }}',
                "fixed": 'fn create() -> String {{ String::from("test") }}',
                "vars": [{}],
            },
        ],
    },
]

# =============================================================================
# TRAIT BOUNDS PATTERNS - All 10 eval koans (tb_001 through tb_010)
# =============================================================================

TRAIT_BOUNDS_TEMPLATES = [
    # tb_001: Missing Debug derive
    {
        "pattern": "missing_debug",
        "error_hint": "doesn't implement `Debug`: add #[derive(Debug)]",
        "templates": [
            {
                "buggy": "struct {name} {{ {field}: {typ} }} fn main() {{ let x = {name} {{ {field}: {val} }}; println!(\"{{:?}}\", x); }}",
                "fixed": "#[derive(Debug)] struct {name} {{ {field}: {typ} }} fn main() {{ let x = {name} {{ {field}: {val} }}; println!(\"{{:?}}\", x); }}",
                "vars": [
                    {"name": "Point", "field": "x", "typ": "i32", "val": "5"},
                    {"name": "User", "field": "id", "typ": "u32", "val": "1"},
                    {"name": "Config", "field": "value", "typ": "String", "val": 'String::from("test")'},
                    {"name": "Item", "field": "count", "typ": "usize", "val": "10"},
                ],
            },
        ],
    },
    # tb_002: Missing Clone derive
    {
        "pattern": "missing_clone",
        "error_hint": "no method named `clone` found: add #[derive(Clone)]",
        "templates": [
            {
                "buggy": "struct {name} {{ {field}: String }} fn main() {{ let a = {name} {{ {field}: String::from(\"x\") }}; let b = a.clone(); }}",
                "fixed": "#[derive(Clone)] struct {name} {{ {field}: String }} fn main() {{ let a = {name} {{ {field}: String::from(\"x\") }}; let b = a.clone(); }}",
                "vars": [
                    {"name": "Data", "field": "value"},
                    {"name": "Item", "field": "name"},
                    {"name": "Entry", "field": "key"},
                    {"name": "Record", "field": "text"},
                ],
            },
            {
                "buggy": "struct {name} {{ {field}: i32 }} fn main() {{ let a = {name} {{ {field}: 42 }}; let b = a.clone(); }}",
                "fixed": "#[derive(Clone)] struct {name} {{ {field}: i32 }} fn main() {{ let a = {name} {{ {field}: 42 }}; let b = a.clone(); }}",
                "vars": [
                    {"name": "Score", "field": "value"},
                    {"name": "Count", "field": "n"},
                ],
            },
        ],
    },
    # tb_003: Missing Copy trait (requires Clone too)
    {
        "pattern": "missing_copy",
        "error_hint": "use of moved value: add #[derive(Clone, Copy)]",
        "templates": [
            {
                "buggy": "struct {name} {{ {field}: i32 }} fn double(n: {name}) -> i32 {{ n.{field} * 2 }} fn main() {{ let n = {name} {{ {field}: 5 }}; println!(\"{{}} {{}}\", double(n), double(n)); }}",
                "fixed": "#[derive(Clone, Copy)] struct {name} {{ {field}: i32 }} fn double(n: {name}) -> i32 {{ n.{field} * 2 }} fn main() {{ let n = {name} {{ {field}: 5 }}; println!(\"{{}} {{}}\", double(n), double(n)); }}",
                "vars": [
                    {"name": "Num", "field": "v"},
                    {"name": "Coord", "field": "x"},
                    {"name": "Amount", "field": "qty"},
                ],
            },
        ],
    },
    # tb_004: Missing trait bound on generic function
    {
        "pattern": "missing_generic_bound",
        "error_hint": "doesn't implement `Debug`: add trait bound T: Debug",
        "templates": [
            {
                "buggy": "fn print_it<T>(val: T) {{ println!(\"{{:?}}\", val); }}",
                "fixed": "fn print_it<T: std::fmt::Debug>(val: T) {{ println!(\"{{:?}}\", val); }}",
                "vars": [{}],
            },
            {
                "buggy": "fn clone_it<T>(val: T) -> T {{ val.clone() }}",
                "fixed": "fn clone_it<T: Clone>(val: T) -> T {{ val.clone() }}",
                "vars": [{}],
            },
            {
                "buggy": "fn compare<T>(a: T, b: T) -> bool {{ a == b }}",
                "fixed": "fn compare<T: PartialEq>(a: T, b: T) -> bool {{ a == b }}",
                "vars": [{}],
            },
        ],
    },
    # tb_005: Missing Default derive
    {
        "pattern": "missing_default",
        "error_hint": "doesn't implement `Default`: add #[derive(Default)]",
        "templates": [
            {
                "buggy": "struct {name} {{ {field}: u32 }} fn main() {{ let c: {name} = Default::default(); }}",
                "fixed": "#[derive(Default)] struct {name} {{ {field}: u32 }} fn main() {{ let c: {name} = Default::default(); }}",
                "vars": [
                    {"name": "Config", "field": "timeout"},
                    {"name": "Settings", "field": "retries"},
                    {"name": "Options", "field": "limit"},
                ],
            },
        ],
    },
    # tb_006: Missing PartialEq derive
    {
        "pattern": "missing_partial_eq",
        "error_hint": "binary operation `==` cannot be applied: add #[derive(PartialEq)]",
        "templates": [
            {
                "buggy": "struct {name} {{ {f1}: i32, {f2}: i32 }} fn main() {{ let a = {name} {{ {f1}: 1, {f2}: 2 }}; let b = {name} {{ {f1}: 1, {f2}: 2 }}; println!(\"{{}}\", a == b); }}",
                "fixed": "#[derive(PartialEq)] struct {name} {{ {f1}: i32, {f2}: i32 }} fn main() {{ let a = {name} {{ {f1}: 1, {f2}: 2 }}; let b = {name} {{ {f1}: 1, {f2}: 2 }}; println!(\"{{}}\", a == b); }}",
                "vars": [
                    {"name": "Point", "f1": "x", "f2": "y"},
                    {"name": "Vec2", "f1": "a", "f2": "b"},
                    {"name": "Pair", "f1": "first", "f2": "second"},
                ],
            },
        ],
    },
    # tb_007: Missing Hash + PartialEq + Eq for HashMap key
    {
        "pattern": "missing_hash",
        "error_hint": "the trait `Hash` is not implemented: add #[derive(Hash, PartialEq, Eq)]",
        "templates": [
            {
                "buggy": "use std::collections::HashMap; struct {name} {{ {field}: i32 }} fn main() {{ let mut map = HashMap::new(); map.insert({name} {{ {field}: 1 }}, \"value\"); }}",
                "fixed": "use std::collections::HashMap; #[derive(Hash, PartialEq, Eq)] struct {name} {{ {field}: i32 }} fn main() {{ let mut map = HashMap::new(); map.insert({name} {{ {field}: 1 }}, \"value\"); }}",
                "vars": [
                    {"name": "Key", "field": "id"},
                    {"name": "Tag", "field": "code"},
                    {"name": "Index", "field": "pos"},
                ],
            },
        ],
    },
    # tb_008: Missing Ord for sorting
    {
        "pattern": "missing_ord",
        "error_hint": "the trait `Ord` is not implemented: add #[derive(PartialEq, Eq, PartialOrd, Ord)]",
        "templates": [
            {
                "buggy": "struct {name} {{ {field}: i32 }} fn main() {{ let mut items = vec![{name} {{ {field}: 3 }}, {name} {{ {field}: 1 }}]; items.sort(); }}",
                "fixed": "#[derive(PartialEq, Eq, PartialOrd, Ord)] struct {name} {{ {field}: i32 }} fn main() {{ let mut items = vec![{name} {{ {field}: 3 }}, {name} {{ {field}: 1 }}]; items.sort(); }}",
                "vars": [
                    {"name": "Score", "field": "value"},
                    {"name": "Priority", "field": "level"},
                    {"name": "Rank", "field": "pos"},
                ],
            },
        ],
    },
    # tb_009: Send trait - Rc not Send, use Arc
    {
        "pattern": "rc_not_send",
        "error_hint": "`Rc` cannot be sent between threads safely: use Arc instead",
        "templates": [
            {
                "buggy": "use std::rc::Rc; fn send_to_thread<T: Send>(val: T) {{}} fn main() {{ let {v} = Rc::new(5); send_to_thread({v}); }}",
                "fixed": "use std::sync::Arc; fn send_to_thread<T: Send>(val: T) {{}} fn main() {{ let {v} = Arc::new(5); send_to_thread({v}); }}",
                "vars": [{"v": "rc"}, {"v": "shared"}, {"v": "ptr"}],
            },
        ],
    },
    # tb_010: Missing Display trait impl
    {
        "pattern": "missing_display",
        "error_hint": "doesn't implement `std::fmt::Display`: implement Display trait",
        "templates": [
            {
                "buggy": 'struct {name} {{ {field}: String }} fn main() {{ let m = {name} {{ {field}: "hello".into() }}; println!("{{}}", m); }}',
                "fixed": 'use std::fmt; struct {name} {{ {field}: String }} impl fmt::Display for {name} {{ fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {{ write!(f, "{{}}", self.{field}) }} }} fn main() {{ let m = {name} {{ {field}: "hello".into() }}; println!("{{}}", m); }}',
                "vars": [
                    {"name": "Message", "field": "text"},
                    {"name": "Label", "field": "value"},
                    {"name": "Status", "field": "msg"},
                ],
            },
        ],
    },
]

# =============================================================================
# RESULT HANDLING PATTERNS - All 10 eval koans (rh_001 through rh_010)
# =============================================================================

RESULT_HANDLING_TEMPLATES = [
    # rh_001: ? operator needs proper return type
    {
        "pattern": "question_mark_return_type",
        "error_hint": "the `?` operator can only be used in a function that returns `Result`",
        "templates": [
            {
                "buggy": "fn read_num(s: &str) -> i32 {{ s.parse()? }}",
                "fixed": "fn read_num(s: &str) -> Result<i32, std::num::ParseIntError> {{ Ok(s.parse()?) }}",
                "vars": [{}],
            },
            {
                "buggy": 'fn read_file(path: &str) -> String {{ std::fs::read_to_string(path)? }}',
                "fixed": 'fn read_file(path: &str) -> Result<String, std::io::Error> {{ Ok(std::fs::read_to_string(path)?) }}',
                "vars": [{}],
            },
        ],
    },
    # rh_002: Option needs unwrap_or
    {
        "pattern": "option_unwrap_or",
        "error_hint": "mismatched types: expected char, found Option<char>: use unwrap_or",
        "templates": [
            {
                "buggy": "fn first_char(s: &str) -> char {{ s.chars().next() }}",
                "fixed": "fn first_char(s: &str) -> char {{ s.chars().next().unwrap_or(' ') }}",
                "vars": [{}],
            },
            {
                "buggy": "fn last_elem(v: &[i32]) -> i32 {{ *v.last() }}",
                "fixed": "fn last_elem(v: &[i32]) -> i32 {{ *v.last().unwrap_or(&0) }}",
                "vars": [{}],
            },
        ],
    },
    # rh_003: Convert Option to Result with map_err
    {
        "pattern": "option_to_result_map_err",
        "error_hint": "mismatched types: expected Result, use map_err to convert",
        "templates": [
            {
                "buggy": 'fn get_env(key: &str) -> Result<String, &str> {{ std::env::var(key) }}',
                "fixed": 'fn get_env(key: &str) -> Result<String, &str> {{ std::env::var(key).map_err(|_| "not found") }}',
                "vars": [{}],
            },
            {
                "buggy": 'fn parse_int(s: &str) -> Result<i32, &str> {{ s.parse() }}',
                "fixed": 'fn parse_int(s: &str) -> Result<i32, &str> {{ s.parse().map_err(|_| "parse error") }}',
                "vars": [{}],
            },
        ],
    },
    # rh_004: Use ok_or to convert Option to Result
    {
        "pattern": "option_ok_or",
        "error_hint": "mismatched types: use ok_or to convert Option to Result",
        "templates": [
            {
                "buggy": 'fn parse_first(v: &[&str]) -> Result<i32, &str> {{ v.first().map(|s| s.parse().unwrap()) }}',
                "fixed": 'fn parse_first(v: &[&str]) -> Result<i32, &str> {{ v.first().ok_or("empty")?.parse().map_err(|_| "parse error") }}',
                "vars": [{}],
            },
            {
                "buggy": 'fn find_item(v: &[i32], target: i32) -> Result<i32, &str> {{ v.iter().find(|&&x| x == target).copied() }}',
                "fixed": 'fn find_item(v: &[i32], target: i32) -> Result<i32, &str> {{ v.iter().find(|&&x| x == target).copied().ok_or("not found") }}',
                "vars": [{}],
            },
        ],
    },
    # rh_005: Handle Result in main with ? (need return type on main)
    {
        "pattern": "main_question_mark",
        "error_hint": "the `?` operator in main needs `fn main() -> Result<...>`",
        "templates": [
            {
                "buggy": 'fn main() {{ let n: i32 = "42".parse()?; println!("{{}}", n); }}',
                "fixed": 'fn main() -> Result<(), Box<dyn std::error::Error>> {{ let n: i32 = "42".parse()?; println!("{{}}", n); Ok(()) }}',
                "vars": [{}],
            },
            {
                "buggy": 'fn main() {{ let content = std::fs::read_to_string("data.txt")?; println!("{{}}", content); }}',
                "fixed": 'fn main() -> Result<(), Box<dyn std::error::Error>> {{ let content = std::fs::read_to_string("data.txt")?; println!("{{}}", content); Ok(()) }}',
                "vars": [{}],
            },
        ],
    },
    # rh_006: Use if let for Option
    {
        "pattern": "if_let_option",
        "error_hint": "mismatched types: expected integer, found Option: use if let",
        "templates": [
            {
                "buggy": 'fn main() {{ let {v} = vec![1, 2, 3]; let first = {v}.first(); println!("{{}}", first); }}',
                "fixed": 'fn main() {{ let {v} = vec![1, 2, 3]; if let Some(first) = {v}.first() {{ println!("{{}}", first); }} }}',
                "vars": [{"v": "v"}, {"v": "nums"}, {"v": "items"}],
            },
            {
                "buggy": 'fn main() {{ let {v} = std::collections::HashMap::from([("key", 1)]); let val = {v}.get("key"); println!("{{}}", val); }}',
                "fixed": 'fn main() {{ let {v} = std::collections::HashMap::from([("key", 1)]); if let Some(val) = {v}.get("key") {{ println!("{{}}", val); }} }}',
                "vars": [{"v": "map"}, {"v": "table"}, {"v": "cache"}],
            },
        ],
    },
    # rh_007: Use match for Result
    {
        "pattern": "match_result",
        "error_hint": "mismatched types: expected integer, found Result: use match",
        "templates": [
            {
                "buggy": 'fn main() {{ let n: i32 = "abc".parse(); println!("{{}}", n); }}',
                "fixed": 'fn main() {{ match "abc".parse::<i32>() {{ Ok(n) => println!("{{}}", n), Err(e) => println!("Error: {{}}", e) }} }}',
                "vars": [{}],
            },
            {
                "buggy": 'fn main() {{ let content: String = std::fs::read_to_string("file.txt"); println!("{{}}", content); }}',
                "fixed": 'fn main() {{ match std::fs::read_to_string("file.txt") {{ Ok(content) => println!("{{}}", content), Err(e) => println!("Error: {{}}", e) }} }}',
                "vars": [{}],
            },
        ],
    },
    # rh_008: Chain Result with map (return type mismatch)
    {
        "pattern": "result_map_return_type",
        "error_hint": "mismatched types: function returns Result but body doesn't",
        "templates": [
            {
                "buggy": "fn parse_add(s: &str) -> i32 {{ s.parse::<i32>().map(|n| n + 1) }}",
                "fixed": "fn parse_add(s: &str) -> Result<i32, std::num::ParseIntError> {{ s.parse::<i32>().map(|n| n + 1) }}",
                "vars": [{}],
            },
            {
                "buggy": "fn double_parse(s: &str) -> i32 {{ s.parse::<i32>().map(|n| n * 2) }}",
                "fixed": "fn double_parse(s: &str) -> Result<i32, std::num::ParseIntError> {{ s.parse::<i32>().map(|n| n * 2) }}",
                "vars": [{}],
            },
        ],
    },
    # rh_009: Use unwrap_or_else for lazy default
    {
        "pattern": "unwrap_or_else",
        "error_hint": "unnecessary eager evaluation: use unwrap_or_else for lazy default",
        "templates": [
            {
                "buggy": "fn get_or_compute(opt: Option<i32>) -> i32 {{ opt.unwrap_or(expensive_default()) }} fn expensive_default() -> i32 {{ 42 }}",
                "fixed": "fn get_or_compute(opt: Option<i32>) -> i32 {{ opt.unwrap_or_else(|| expensive_default()) }} fn expensive_default() -> i32 {{ 42 }}",
                "vars": [{}],
            },
            {
                "buggy": "fn get_or_make(opt: Option<String>) -> String {{ opt.unwrap_or(generate()) }} fn generate() -> String {{ String::from(\"default\") }}",
                "fixed": "fn get_or_make(opt: Option<String>) -> String {{ opt.unwrap_or_else(generate) }} fn generate() -> String {{ String::from(\"default\") }}",
                "vars": [{}],
            },
        ],
    },
    # rh_010: Propagate error with ? (return type needs Result)
    {
        "pattern": "propagate_error",
        "error_hint": "mismatched types: expected String, found Result: propagate with Result return type",
        "templates": [
            {
                "buggy": "use std::fs; fn read_file(path: &str) -> String {{ fs::read_to_string(path) }}",
                "fixed": "use std::fs; fn read_file(path: &str) -> std::io::Result<String> {{ fs::read_to_string(path) }}",
                "vars": [{}],
            },
            {
                "buggy": "use std::fs; fn read_bytes(path: &str) -> Vec<u8> {{ fs::read(path) }}",
                "fixed": "use std::fs; fn read_bytes(path: &str) -> std::io::Result<Vec<u8>> {{ fs::read(path) }}",
                "vars": [{}],
            },
        ],
    },
]


# =============================================================================
# CLIPPY SUPPRESSION ANTI-PATTERNS
# Teach the model to fix the actual issue instead of adding #[allow(...)]
# =============================================================================

CLIPPY_SUPPRESSION_TEMPLATES = [
    # Anti-pattern: suppressing needless_collect
    {
        "pattern": "no_suppress_needless_collect",
        "error_hint": "clippy::needless_collect: remove the unnecessary .collect() instead of suppressing the lint",
        "templates": [
            {
                "buggy": "#[allow(clippy::needless_collect)] fn sum_doubled(v: &[i32]) -> i32 {{ let doubled: Vec<i32> = v.iter().map(|x| x * 2).collect(); doubled.into_iter().sum() }}",
                "fixed": "fn sum_doubled(v: &[i32]) -> i32 {{ v.iter().map(|x| x * 2).sum() }}",
                "vars": [{}],
            },
            {
                "buggy": "#[allow(clippy::needless_collect)] fn count_positive(v: &[i32]) -> usize {{ let pos: Vec<&i32> = v.iter().filter(|x| **x > 0).collect(); pos.len() }}",
                "fixed": "fn count_positive(v: &[i32]) -> usize {{ v.iter().filter(|x| **x > 0).count() }}",
                "vars": [{}],
            },
        ],
    },
    # Anti-pattern: suppressing uninlined_format_args
    {
        "pattern": "no_suppress_format_args",
        "error_hint": "clippy::uninlined_format_args: use inline format args instead of suppressing",
        "templates": [
            {
                "buggy": '#[allow(clippy::uninlined_format_args)] fn greet(name: &str) {{ println!("Hello, {{}}", name); }}',
                "fixed": 'fn greet(name: &str) {{ println!("Hello, {{name}}"); }}',
                "vars": [{}],
            },
            {
                "buggy": '#[allow(clippy::uninlined_format_args)] fn log(level: &str, msg: &str) {{ eprintln!("[{{}}] {{}}", level, msg); }}',
                "fixed": 'fn log(level: &str, msg: &str) {{ eprintln!("[{{level}}] {{msg}}"); }}',
                "vars": [{}],
            },
        ],
    },
    # Anti-pattern: suppressing manual_map
    {
        "pattern": "no_suppress_manual_map",
        "error_hint": "clippy::manual_map: use .map() instead of suppressing the lint",
        "templates": [
            {
                "buggy": "#[allow(clippy::manual_map)] fn inc(opt: Option<i32>) -> Option<i32> {{ match opt {{ Some(x) => Some(x + 1), None => None }} }}",
                "fixed": "fn inc(opt: Option<i32>) -> Option<i32> {{ opt.map(|x| x + 1) }}",
                "vars": [{}],
            },
            {
                "buggy": '#[allow(clippy::manual_map)] fn to_upper(opt: Option<String>) -> Option<String> {{ match opt {{ Some(s) => Some(s.to_uppercase()), None => None }} }}',
                "fixed": "fn to_upper(opt: Option<String>) -> Option<String> {{ opt.map(|s| s.to_uppercase()) }}",
                "vars": [{}],
            },
        ],
    },
    # Anti-pattern: suppressing match_like_matches_macro
    {
        "pattern": "no_suppress_matches",
        "error_hint": "clippy::match_like_matches_macro: use matches!() instead of suppressing",
        "templates": [
            {
                "buggy": "#[allow(clippy::match_like_matches_macro)] fn is_vowel(c: char) -> bool {{ match c {{ 'a' | 'e' | 'i' | 'o' | 'u' => true, _ => false }} }}",
                "fixed": "fn is_vowel(c: char) -> bool {{ matches!(c, 'a' | 'e' | 'i' | 'o' | 'u') }}",
                "vars": [{}],
            },
        ],
    },
    # Anti-pattern: suppressing clone_on_ref_ptr
    {
        "pattern": "no_suppress_clone_ref",
        "error_hint": "clippy::clone_on_ref_ptr: use Arc::clone() for clarity instead of suppressing",
        "templates": [
            {
                "buggy": "#[allow(clippy::clone_on_ref_ptr)] fn share(a: &std::sync::Arc<String>) -> std::sync::Arc<String> {{ a.clone() }}",
                "fixed": "fn share(a: &std::sync::Arc<String>) -> std::sync::Arc<String> {{ std::sync::Arc::clone(a) }}",
                "vars": [{}],
            },
        ],
    },
    # Anti-pattern: suppressing clippy::all or blanket suppression
    {
        "pattern": "no_blanket_suppress",
        "error_hint": "never use blanket #[allow(clippy::all)]: fix the specific issues instead",
        "templates": [
            {
                "buggy": "#![allow(clippy::all)] fn add(a: i32, b: i32) -> i32 {{ let result: i32 = a + b; return result; }}",
                "fixed": "fn add(a: i32, b: i32) -> i32 {{ a + b }}",
                "vars": [{}],
            },
        ],
    },
    # Anti-pattern: suppressing unused_must_use
    {
        "pattern": "no_suppress_must_use",
        "error_hint": "don't suppress unused Result: handle the error properly",
        "templates": [
            {
                "buggy": '#[allow(unused_must_use)] fn write_file(path: &str) {{ std::fs::write(path, "data"); }}',
                "fixed": 'fn write_file(path: &str) {{ let _ = std::fs::write(path, "data"); }}',
                "vars": [{}],
            },
            {
                "buggy": '#[allow(unused_must_use)] fn remove(path: &str) {{ std::fs::remove_file(path); }}',
                "fixed": 'fn remove(path: &str) -> std::io::Result<()> {{ std::fs::remove_file(path) }}',
                "vars": [{}],
            },
        ],
    },
    # Anti-pattern: suppressing dead_code instead of removing it
    {
        "pattern": "no_suppress_dead_code",
        "error_hint": "don't suppress dead_code: remove unused functions instead",
        "templates": [
            {
                "buggy": "#[allow(dead_code)] fn unused_helper() -> i32 {{ 42 }} fn main() {{ println!(\"hello\"); }}",
                "fixed": "fn main() {{ println!(\"hello\"); }}",
                "vars": [{}],
            },
        ],
    },
]

# =============================================================================
# RUST 2024 EDITION / MODERN CLIPPY PATTERNS
# Patterns that LLMs trained on pre-2024 Rust get wrong
# =============================================================================

RUST_2024_TEMPLATES = [
    # Format string capture (clippy::uninlined_format_args)
    {
        "pattern": "inline_format_args",
        "error_hint": "clippy::uninlined_format_args: use inline format variables",
        "templates": [
            {
                "buggy": 'fn greet({v}: &str) {{ println!("Hello, {{}}", {v}); }}',
                "fixed": 'fn greet({v}: &str) {{ println!("Hello, {{{v}}}"); }}',
                "vars": [{"v": "name"}, {"v": "user"}, {"v": "who"}],
            },
            {
                "buggy": 'fn log({v}: &str, {v2}: u32) {{ eprintln!("[{{}}] {{}}", {v2}, {v}); }}',
                "fixed": 'fn log({v}: &str, {v2}: u32) {{ eprintln!("[{{{v2}}}] {{{v}}}"); }}',
                "vars": [{"v": "msg", "v2": "code"}, {"v": "text", "v2": "level"}],
            },
            {
                "buggy": 'fn format_point({v}: f64, {v2}: f64) -> String {{ format!("({{}}, {{}})", {v}, {v2}) }}',
                "fixed": 'fn format_point({v}: f64, {v2}: f64) -> String {{ format!("({{{v}}}, {{{v2}}})") }}',
                "vars": [{"v": "x", "v2": "y"}, {"v": "lat", "v2": "lon"}],
            },
        ],
    },
    # iter().cloned() vs copied() for Copy types
    {
        "pattern": "iter_copied",
        "error_hint": "clippy::cloned_instead_of_copied: use .copied() for Copy types",
        "templates": [
            {
                "buggy": "fn double_all(v: &[i32]) -> Vec<i32> {{ v.iter().cloned().map(|x| x * 2).collect() }}",
                "fixed": "fn double_all(v: &[i32]) -> Vec<i32> {{ v.iter().copied().map(|x| x * 2).collect() }}",
                "vars": [{}],
            },
            {
                "buggy": "fn sum_all(v: &[i64]) -> i64 {{ v.iter().cloned().sum() }}",
                "fixed": "fn sum_all(v: &[i64]) -> i64 {{ v.iter().copied().sum() }}",
                "vars": [{}],
            },
        ],
    },
    # matches! macro
    {
        "pattern": "use_matches_macro",
        "error_hint": "clippy::match_like_matches_macro: use matches!() macro",
        "templates": [
            {
                "buggy": "fn is_digit(c: char) -> bool {{ match c {{ '0'..='9' => true, _ => false }} }}",
                "fixed": "fn is_digit(c: char) -> bool {{ matches!(c, '0'..='9') }}",
                "vars": [{}],
            },
            {
                "buggy": "fn is_ok(r: &Result<i32, &str>) -> bool {{ match r {{ Ok(_) => true, _ => false }} }}",
                "fixed": "fn is_ok(r: &Result<i32, &str>) -> bool {{ matches!(r, Ok(_)) }}",
                "vars": [{}],
            },
        ],
    },
    # manual_flatten
    {
        "pattern": "use_flatten",
        "error_hint": "clippy::manual_flatten: use .flatten() instead of filter+unwrap",
        "templates": [
            {
                "buggy": "fn get_somes(v: Vec<Option<i32>>) -> Vec<i32> {{ v.into_iter().filter(|x| x.is_some()).map(|x| x.unwrap()).collect() }}",
                "fixed": "fn get_somes(v: Vec<Option<i32>>) -> Vec<i32> {{ v.into_iter().flatten().collect() }}",
                "vars": [{}],
            },
        ],
    },
    # needless_collect
    {
        "pattern": "remove_needless_collect",
        "error_hint": "clippy::needless_collect: remove unnecessary intermediate collect()",
        "templates": [
            {
                "buggy": "fn sum_squares(v: &[i32]) -> i32 {{ v.iter().map(|x| x * x).collect::<Vec<_>>().into_iter().sum() }}",
                "fixed": "fn sum_squares(v: &[i32]) -> i32 {{ v.iter().map(|x| x * x).sum() }}",
                "vars": [{}],
            },
        ],
    },
    # Let-chains (stabilized in Rust 1.76, edition 2024)
    {
        "pattern": "let_chains",
        "error_hint": "use let-chains to combine if-let with conditions",
        "templates": [
            {
                "buggy": "fn check(opt: Option<i32>) -> bool {{ if let Some(x) = opt {{ if x > 0 {{ return true; }} }} false }}",
                "fixed": "fn check(opt: Option<i32>) -> bool {{ if let Some(x) = opt && x > 0 {{ return true; }} false }}",
                "vars": [{}],
            },
            {
                "buggy": "fn get_valid(opt: Option<String>) -> Option<String> {{ if let Some(s) = opt {{ if !s.is_empty() {{ return Some(s); }} }} None }}",
                "fixed": "fn get_valid(opt: Option<String>) -> Option<String> {{ if let Some(s) = opt && !s.is_empty() {{ return Some(s); }} None }}",
                "vars": [{}],
            },
        ],
    },
    # Let-else
    {
        "pattern": "let_else",
        "error_hint": "use let-else pattern for early returns on None/Err",
        "templates": [
            {
                "buggy": "fn unwrap_or_ret(opt: Option<i32>) -> i32 {{ match opt {{ Some(x) => x, None => return 0 }} }}",
                "fixed": "fn unwrap_or_ret(opt: Option<i32>) -> i32 {{ let Some(x) = opt else {{ return 0; }}; x }}",
                "vars": [{}],
            },
            {
                "buggy": "fn parse_or_default(s: &str) -> i32 {{ match s.parse() {{ Ok(n) => n, Err(_) => return -1 }} }}",
                "fixed": "fn parse_or_default(s: &str) -> i32 {{ let Ok(n) = s.parse() else {{ return -1; }}; n }}",
                "vars": [{}],
            },
        ],
    },
    # Modern Option/Result methods
    {
        "pattern": "is_some_and",
        "error_hint": "use is_some_and/is_ok_and instead of map+unwrap_or",
        "templates": [
            {
                "buggy": "fn has_positive(opt: Option<i32>) -> bool {{ opt.map(|x| x > 0).unwrap_or(false) }}",
                "fixed": "fn has_positive(opt: Option<i32>) -> bool {{ opt.is_some_and(|x| x > 0) }}",
                "vars": [{}],
            },
            {
                "buggy": 'fn is_valid(res: Result<i32, &str>) -> bool {{ res.map(|x| x >= 0).unwrap_or(false) }}',
                "fixed": "fn is_valid(res: Result<i32, &str>) -> bool {{ res.is_ok_and(|x| x >= 0) }}",
                "vars": [{}],
            },
        ],
    },
    # &Vec<T> -> &[T] in function params
    {
        "pattern": "slice_param",
        "error_hint": "clippy::ptr_arg: use &[T] instead of &Vec<T> in function parameters",
        "templates": [
            {
                "buggy": "fn sum(v: &Vec<i32>) -> i32 {{ v.iter().sum() }}",
                "fixed": "fn sum(v: &[i32]) -> i32 {{ v.iter().sum() }}",
                "vars": [{}],
            },
            {
                "buggy": 'fn join(v: &Vec<String>) -> String {{ v.join(", ") }}',
                "fixed": 'fn join(v: &[String]) -> String {{ v.join(", ") }}',
                "vars": [{}],
            },
        ],
    },
]

# =============================================================================
# CODE COMPLEXITY PATTERNS
# Teach decomposition and module structure (sw-checker compliance)
# =============================================================================

COMPLEXITY_TEMPLATES = [
    # Too many lines per function -> decompose
    {
        "pattern": "decompose_long_function",
        "error_hint": "function too long: decompose into smaller focused functions",
        "templates": [
            {
                "buggy": "fn process(data: &str) -> Result<String, Error> {{ let trimmed = data.trim(); if trimmed.is_empty() {{ return Err(Error::Empty); }} let parsed: i32 = trimmed.parse().map_err(|_| Error::Parse)?; let validated = if parsed < 0 {{ return Err(Error::Negative); }} else {{ parsed }}; let result = validated * 2 + 10; Ok(format!(\"result: {{}}\", result)) }}",
                "fixed": "fn parse_input(data: &str) -> Result<i32, Error> {{ let trimmed = data.trim(); if trimmed.is_empty() {{ return Err(Error::Empty); }} trimmed.parse().map_err(|_| Error::Parse) }} fn validate(n: i32) -> Result<i32, Error> {{ if n < 0 {{ Err(Error::Negative) }} else {{ Ok(n) }} }} fn process(data: &str) -> Result<String, Error> {{ let parsed = parse_input(data)?; let validated = validate(parsed)?; Ok(format!(\"result: {{}}\", validated * 2 + 10)) }}",
                "vars": [{}],
            },
            {
                "buggy": "fn handle_request(req: Request) -> Response {{ let auth = req.headers.get(\"Authorization\"); if auth.is_none() {{ return Response::unauthorized(); }} let token = auth.unwrap(); if !verify_token(token) {{ return Response::forbidden(); }} let body = req.body(); let data: Data = serde_json::from_str(body).unwrap(); let result = db::save(&data); match result {{ Ok(_) => Response::ok(), Err(e) => Response::error(e) }} }}",
                "fixed": "fn authenticate(req: &Request) -> Result<&str, Response> {{ req.headers.get(\"Authorization\").filter(|t| verify_token(t)).ok_or_else(Response::unauthorized) }} fn parse_body(req: &Request) -> Result<Data, Response> {{ serde_json::from_str(req.body()).map_err(|_| Response::bad_request()) }} fn handle_request(req: Request) -> Response {{ let _token = match authenticate(&req) {{ Ok(t) => t, Err(r) => return r }}; match parse_body(&req).and_then(|data| db::save(&data).map_err(|e| Response::error(e))) {{ Ok(_) => Response::ok(), Err(r) => r }} }}",
                "vars": [{}],
            },
        ],
    },
    # Too many functions per module -> split into submodules
    {
        "pattern": "split_large_module",
        "error_hint": "too many functions in module: split into focused submodules",
        "templates": [
            {
                "buggy": "// lib.rs with too many functions\npub fn create_user() {{}} pub fn get_user() {{}} pub fn update_user() {{}} pub fn delete_user() {{}} pub fn list_users() {{}} pub fn create_post() {{}} pub fn get_post() {{}} pub fn update_post() {{}} pub fn delete_post() {{}} pub fn list_posts() {{}} pub fn create_comment() {{}} pub fn get_comment() {{}}",
                "fixed": "// lib.rs\npub mod users;\npub mod posts;\npub mod comments;\n\n// users.rs\npub fn create() {{}} pub fn get() {{}} pub fn update() {{}} pub fn delete() {{}} pub fn list() {{}}\n\n// posts.rs\npub fn create() {{}} pub fn get() {{}} pub fn update() {{}} pub fn delete() {{}} pub fn list() {{}}\n\n// comments.rs\npub fn create() {{}} pub fn get() {{}}",
                "vars": [{}],
            },
        ],
    },
    # Monolithic main.rs -> separate modules into files
    {
        "pattern": "extract_modules_to_files",
        "error_hint": "file too long: extract inline modules into separate files",
        "templates": [
            {
                "buggy": "// main.rs - everything inline\nmod config {{ pub struct Config {{ pub db_url: String }} impl Config {{ pub fn load() -> Self {{ todo!() }} }} }}\nmod db {{ pub fn connect(_url: &str) {{}} pub fn query() {{}} }}\nmod handlers {{ pub fn index() {{}} pub fn health() {{}} }}\nfn main() {{}}",
                "fixed": "// main.rs\nmod config;\nmod db;\nmod handlers;\n\nfn main() {{}}\n\n// config.rs\npub struct Config {{ pub db_url: String }}\nimpl Config {{ pub fn load() -> Self {{ todo!() }} }}\n\n// db.rs\npub fn connect(_url: &str) {{}}\npub fn query() {{}}\n\n// handlers.rs\npub fn index() {{}}\npub fn health() {{}}",
                "vars": [{}],
            },
        ],
    },
    # God struct -> decompose into smaller types
    {
        "pattern": "decompose_god_struct",
        "error_hint": "struct has too many fields: decompose into smaller focused types",
        "templates": [
            {
                "buggy": "struct App {{ db_host: String, db_port: u16, db_name: String, cache_host: String, cache_port: u16, log_level: String, log_file: String, server_host: String, server_port: u16 }}",
                "fixed": "struct DbConfig {{ host: String, port: u16, name: String }}\nstruct CacheConfig {{ host: String, port: u16 }}\nstruct LogConfig {{ level: String, file: String }}\nstruct ServerConfig {{ host: String, port: u16 }}\nstruct App {{ db: DbConfig, cache: CacheConfig, log: LogConfig, server: ServerConfig }}",
                "vars": [{}],
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
        error_hint = pattern_group.get("error_hint", f"Fix the {pattern.replace('_', ' ')} error")

        for template in pattern_group["templates"]:
            vars_list = template["vars"]

            # Generate n variants by cycling through var dicts
            for i in range(min(n_per_template, len(vars_list) * 3)):
                var_map = vars_list[i % len(vars_list)]

                try:
                    buggy = template["buggy"].format(**var_map)
                    fixed = template["fixed"].format(**var_map)
                except (KeyError, IndexError):
                    # Fallback: template has no vars (empty dict)
                    buggy = template["buggy"]
                    fixed = template["fixed"]

                koan = Koan(
                    task_id=f"{family}_{pattern}_{koan_id:03d}",
                    family=family,
                    pattern=pattern,
                    buggy_code=buggy,
                    fixed_code=fixed,
                    error_hint=error_hint,
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
    parser.add_argument("--no-supplemental", action="store_true",
                        help="Only generate eval-aligned data (skip clippy/complexity/rust2024)")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=== Generating Eval-Aligned Training Data ===")
    print(f"Variants per template: {args.variants}")
    print(f"Supplemental data: {'disabled' if args.no_supplemental else 'enabled'}")
    print()

    all_koans = []

    # --- Core eval-aligned patterns (30 frozen eval koans) ---
    bc_koans = generate_variants(BORROW_CHECKER_TEMPLATES, "borrow_checker", args.variants)
    print(f"Borrow checker: {len(bc_koans)} koans ({len(BORROW_CHECKER_TEMPLATES)} patterns)")
    all_koans.extend(bc_koans)

    tb_koans = generate_variants(TRAIT_BOUNDS_TEMPLATES, "trait_bounds", args.variants)
    print(f"Trait bounds:    {len(tb_koans)} koans ({len(TRAIT_BOUNDS_TEMPLATES)} patterns)")
    all_koans.extend(tb_koans)

    rh_koans = generate_variants(RESULT_HANDLING_TEMPLATES, "result_handling", args.variants)
    print(f"Result handling: {len(rh_koans)} koans ({len(RESULT_HANDLING_TEMPLATES)} patterns)")
    all_koans.extend(rh_koans)

    bc_patterns = {k.pattern for k in bc_koans}
    tb_patterns = {k.pattern for k in tb_koans}
    rh_patterns = {k.pattern for k in rh_koans}
    print(f"\nEval coverage: BC={len(bc_patterns)}/10  TB={len(tb_patterns)}/10  RH={len(rh_patterns)}/10")

    # --- Supplemental patterns (clippy, rust 2024, complexity) ---
    if not args.no_supplemental:
        print()

        cs_koans = generate_variants(CLIPPY_SUPPRESSION_TEMPLATES, "clippy_suppression", args.variants)
        print(f"Clippy anti-suppress: {len(cs_koans)} koans ({len(CLIPPY_SUPPRESSION_TEMPLATES)} patterns)")
        all_koans.extend(cs_koans)

        r24_koans = generate_variants(RUST_2024_TEMPLATES, "rust_2024", args.variants)
        print(f"Rust 2024 edition:    {len(r24_koans)} koans ({len(RUST_2024_TEMPLATES)} patterns)")
        all_koans.extend(r24_koans)

        cx_koans = generate_variants(COMPLEXITY_TEMPLATES, "complexity", args.variants)
        print(f"Code complexity:      {len(cx_koans)} koans ({len(COMPLEXITY_TEMPLATES)} patterns)")
        all_koans.extend(cx_koans)

    # --- Also merge in any manual corrections ---
    corrections_path = Path("data/sft/corrections.jsonl")
    corrections_count = 0
    if corrections_path.exists():
        with open(corrections_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # Convert correction entry to Koan for unified handling
                    koan = Koan(
                        task_id=entry.get("task_id", f"correction_{corrections_count:03d}"),
                        family=entry.get("family", "correction"),
                        pattern=entry.get("pattern", "manual_correction"),
                        buggy_code=entry.get("buggy", entry.get("input", "")),
                        fixed_code=entry.get("fixed", entry.get("output", "")),
                        error_hint=entry.get("error_hint", "Fix the issue"),
                    )
                    all_koans.append(koan)
                    corrections_count += 1
        print(f"Manual corrections:   {corrections_count} (from {corrections_path})")

    print(f"\nTotal: {len(all_koans)} training examples")

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
    print("  2. Merge: python cuda/scripts/merge.py --adapter runs/adapters/<timestamp>/adapter --model-name sleepy-coder-v14")
    print("  3. Eval:  ./rust/target/release/sleepy-coder eval --cycle 14 --model sleepy-coder-v14")
    print()
    print("To add manual corrections, append JSONL entries to data/sft/corrections.jsonl:")
    print('  {"task_id":"fix_001","family":"clippy","pattern":"specific_issue",')
    print('   "buggy":"<code with issue>","fixed":"<corrected code>","error_hint":"<description>"}')


if __name__ == "__main__":
    main()
