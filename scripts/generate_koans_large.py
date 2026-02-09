#!/usr/bin/env python3
"""
Generate 200+ diverse Rust koans for training.

Creates many variations of common Rust patterns to build a diverse training set
that enables generalization rather than memorization.
"""

import json
import random
from pathlib import Path

# Template-based koan generation
RESULT_HANDLING_TEMPLATES = [
    # unwrap_or vs unwrap_or_else
    {
        "template": "unwrap_or_else_{type}",
        "types": ["String", "Vec<i32>", "HashMap<String, i32>", "Box<dyn Error>"],
        "buggy": 'fn get_{name}(opt: Option<{type}>) -> {type} {{ opt.unwrap_or({default}) }}',
        "fixed": 'fn get_{name}(opt: Option<{type}>) -> {type} {{ opt.unwrap_or_else(|| {default}) }}',
        "defaults": {
            "String": '"default".to_string()',
            "Vec<i32>": "vec![1, 2, 3]",
            "HashMap<String, i32>": "HashMap::new()",
            "Box<dyn Error>": 'Box::new(std::io::Error::new(std::io::ErrorKind::Other, "error"))',
        }
    },
    # ? operator with Result
    {
        "template": "question_mark_{op}",
        "ops": ["parse", "read", "write", "connect"],
        "buggy_templates": {
            "parse": 'fn parse_{name}(s: &str) -> i32 {{ s.parse()? }}',
            "read": 'fn read_{name}(path: &str) -> String {{ std::fs::read_to_string(path)? }}',
            "write": 'fn write_{name}(path: &str, data: &str) {{ std::fs::write(path, data)? }}',
            "connect": 'fn connect_{name}(addr: &str) -> std::net::TcpStream {{ std::net::TcpStream::connect(addr)? }}',
        },
        "fixed_templates": {
            "parse": 'fn parse_{name}(s: &str) -> Result<i32, std::num::ParseIntError> {{ Ok(s.parse()?) }}',
            "read": 'fn read_{name}(path: &str) -> Result<String, std::io::Error> {{ std::fs::read_to_string(path) }}',
            "write": 'fn write_{name}(path: &str, data: &str) -> Result<(), std::io::Error> {{ std::fs::write(path, data) }}',
            "connect": 'fn connect_{name}(addr: &str) -> Result<std::net::TcpStream, std::io::Error> {{ std::net::TcpStream::connect(addr) }}',
        }
    },
    # map vs and_then
    {
        "template": "and_then_{variant}",
        "variants": ["nested_option", "chain_result", "flat_map"],
        "buggy_templates": {
            "nested_option": 'fn flatten_{name}(opt: Option<Option<i32>>) -> Option<i32> {{ opt.map(|x| x).flatten() }}',
            "chain_result": 'fn chain_{name}(r: Result<String, E>) -> Result<i32, E> {{ r.map(|s| s.parse().ok()).flatten() }}',
            "flat_map": 'fn get_{name}(v: Vec<Option<i32>>) -> Vec<i32> {{ v.iter().map(|x| x).flatten().collect() }}',
        },
        "fixed_templates": {
            "nested_option": 'fn flatten_{name}(opt: Option<Option<i32>>) -> Option<i32> {{ opt.and_then(|x| x) }}',
            "chain_result": 'fn chain_{name}(r: Result<String, E>) -> Result<i32, E> {{ r.and_then(|s| s.parse().map_err(|_| todo!())) }}',
            "flat_map": 'fn get_{name}(v: Vec<Option<i32>>) -> Vec<i32> {{ v.into_iter().filter_map(|x| x).collect() }}',
        }
    },
    # ok_or patterns
    {
        "template": "ok_or_{variant}",
        "variants": ["static_err", "dynamic_err", "custom_err"],
        "buggy_templates": {
            "static_err": 'fn require_{name}(opt: Option<i32>) -> Result<i32, &\'static str> {{ match opt {{ Some(v) => Ok(v), None => Err("missing") }} }}',
            "dynamic_err": 'fn require_{name}(opt: Option<String>, msg: &str) -> Result<String, String> {{ match opt {{ Some(v) => Ok(v), None => Err(msg.to_string()) }} }}',
            "custom_err": 'fn require_{name}(opt: Option<i32>) -> Result<i32, MyError> {{ match opt {{ Some(v) => Ok(v), None => Err(MyError::NotFound) }} }}',
        },
        "fixed_templates": {
            "static_err": 'fn require_{name}(opt: Option<i32>) -> Result<i32, &\'static str> {{ opt.ok_or("missing") }}',
            "dynamic_err": 'fn require_{name}(opt: Option<String>, msg: &str) -> Result<String, String> {{ opt.ok_or_else(|| msg.to_string()) }}',
            "custom_err": 'fn require_{name}(opt: Option<i32>) -> Result<i32, MyError> {{ opt.ok_or(MyError::NotFound) }}',
        }
    },
]

TRAIT_BOUNDS_TEMPLATES = [
    # derive macros
    {
        "template": "derive_{trait}",
        "traits": ["Debug", "Clone", "PartialEq", "Eq", "Hash", "Default", "PartialOrd", "Ord"],
        "use_cases": {
            "Debug": ('println!("{{:?}}", {var});', '#[derive(Debug)]'),
            "Clone": ('let copy = {var}.clone();', '#[derive(Clone)]'),
            "PartialEq": ('if {var} == other {{ }}', '#[derive(PartialEq)]'),
            "Eq": ('set.insert({var});', '#[derive(PartialEq, Eq)]'),
            "Hash": ('map.insert({var}, value);', '#[derive(Hash, PartialEq, Eq)]'),
            "Default": ('let d: {type} = Default::default();', '#[derive(Default)]'),
            "PartialOrd": ('if {var} < other {{ }}', '#[derive(PartialOrd, PartialEq)]'),
            "Ord": ('{vec}.sort();', '#[derive(Ord, PartialOrd, PartialEq, Eq)]'),
        }
    },
    # Send/Sync
    {
        "template": "send_sync_{variant}",
        "variants": ["rc_to_arc", "refcell_to_mutex", "cell_to_atomic"],
        "buggy_templates": {
            "rc_to_arc": 'use std::rc::Rc; fn spawn_{name}<T: Send>(val: Rc<T>) {{ std::thread::spawn(move || drop(val)); }}',
            "refcell_to_mutex": 'use std::cell::RefCell; fn share_{name}<T: Send>(val: RefCell<T>) {{ std::thread::spawn(move || drop(val)); }}',
            "cell_to_atomic": 'use std::cell::Cell; fn share_{name}(val: Cell<i32>) {{ std::thread::spawn(move || drop(val)); }}',
        },
        "fixed_templates": {
            "rc_to_arc": 'use std::sync::Arc; fn spawn_{name}<T: Send + Sync>(val: Arc<T>) {{ std::thread::spawn(move || drop(val)); }}',
            "refcell_to_mutex": 'use std::sync::Mutex; fn share_{name}<T: Send>(val: Mutex<T>) {{ std::thread::spawn(move || drop(val)); }}',
            "cell_to_atomic": 'use std::sync::atomic::AtomicI32; fn share_{name}(val: AtomicI32) {{ std::thread::spawn(move || drop(val)); }}',
        }
    },
    # Generic bounds
    {
        "template": "generic_bound_{op}",
        "ops": ["clone", "display", "from_str", "add", "iterator"],
        "buggy_templates": {
            "clone": 'fn duplicate<T>(val: T) -> (T, T) {{ (val.clone(), val) }}',
            "display": 'fn print<T>(val: T) {{ println!("{{}}", val); }}',
            "from_str": 'fn parse<T>(s: &str) -> T {{ s.parse().unwrap() }}',
            "add": 'fn sum<T>(a: T, b: T) -> T {{ a + b }}',
            "iterator": 'fn first<T, I>(iter: I) -> Option<T> {{ iter.into_iter().next() }}',
        },
        "fixed_templates": {
            "clone": 'fn duplicate<T: Clone>(val: T) -> (T, T) {{ (val.clone(), val.clone()) }}',
            "display": 'fn print<T: std::fmt::Display>(val: T) {{ println!("{{}}", val); }}',
            "from_str": 'fn parse<T: std::str::FromStr>(s: &str) -> Result<T, T::Err> {{ s.parse() }}',
            "add": 'fn sum<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {{ a + b }}',
            "iterator": 'fn first<T, I: IntoIterator<Item = T>>(iter: I) -> Option<T> {{ iter.into_iter().next() }}',
        }
    },
]

BORROW_CHECKER_TEMPLATES = [
    # Move vs clone
    {
        "template": "move_clone_{variant}",
        "variants": ["string", "vec", "box", "custom"],
        "buggy_templates": {
            "string": 'fn use_{name}() {{ let s = String::from("hello"); let a = s; let b = s; }}',
            "vec": 'fn use_{name}() {{ let v = vec![1,2,3]; let a = v; let b = v; }}',
            "box": 'fn use_{name}() {{ let b = Box::new(42); let a = b; let c = b; }}',
            "custom": 'fn use_{name}() {{ let x = MyStruct::new(); process(x); process(x); }}',
        },
        "fixed_templates": {
            "string": 'fn use_{name}() {{ let s = String::from("hello"); let a = s.clone(); let b = s; }}',
            "vec": 'fn use_{name}() {{ let v = vec![1,2,3]; let a = v.clone(); let b = v; }}',
            "box": 'fn use_{name}() {{ let b = Box::new(42); let a = b.clone(); let c = b; }}',
            "custom": 'fn use_{name}() {{ let x = MyStruct::new(); process(x.clone()); process(x); }}',
        }
    },
    # Reference vs ownership
    {
        "template": "ref_ownership_{variant}",
        "variants": ["fn_param", "return_ref", "struct_field"],
        "buggy_templates": {
            "fn_param": 'fn print_{name}(s: String) {{ println!("{{}}", s); }} fn main() {{ let s = String::from("hi"); print_{name}(s); print_{name}(s); }}',
            "return_ref": 'fn get_{name}() -> &str {{ let s = String::from("hello"); &s }}',
            "struct_field": 'struct Container {{ data: &str }} fn new_{name}() -> Container {{ let s = String::from("data"); Container {{ data: &s }} }}',
        },
        "fixed_templates": {
            "fn_param": 'fn print_{name}(s: &str) {{ println!("{{}}", s); }} fn main() {{ let s = String::from("hi"); print_{name}(&s); print_{name}(&s); }}',
            "return_ref": 'fn get_{name}() -> String {{ String::from("hello") }}',
            "struct_field": 'struct Container {{ data: String }} fn new_{name}() -> Container {{ Container {{ data: String::from("data") }} }}',
        }
    },
    # Mutable borrow conflicts
    {
        "template": "mut_borrow_{variant}",
        "variants": ["vec_push", "hashmap_get", "split_borrow"],
        "buggy_templates": {
            "vec_push": 'fn modify_{name}() {{ let mut v = vec![1,2,3]; let first = &v[0]; v.push(4); println!("{{}}", first); }}',
            "hashmap_get": 'fn modify_{name}() {{ let mut m = HashMap::new(); let val = m.get(&key); m.insert(key, value); println!("{{:?}}", val); }}',
            "split_borrow": 'fn swap_{name}(v: &mut Vec<i32>) {{ let a = &mut v[0]; let b = &mut v[1]; std::mem::swap(a, b); }}',
        },
        "fixed_templates": {
            "vec_push": 'fn modify_{name}() {{ let mut v = vec![1,2,3]; {{ let first = &v[0]; println!("{{}}", first); }} v.push(4); }}',
            "hashmap_get": 'fn modify_{name}() {{ let mut m = HashMap::new(); let val = m.get(&key).cloned(); m.insert(key, value); println!("{{:?}}", val); }}',
            "split_borrow": 'fn swap_{name}(v: &mut Vec<i32>) {{ let (left, right) = v.split_at_mut(1); std::mem::swap(&mut left[0], &mut right[0]); }}',
        }
    },
    # Closure captures
    {
        "template": "closure_{variant}",
        "variants": ["move_keyword", "ref_capture", "mut_capture"],
        "buggy_templates": {
            "move_keyword": 'fn spawn_{name}() {{ let s = String::from("hi"); std::thread::spawn(|| println!("{{}}", s)); }}',
            "ref_capture": 'fn iter_{name}() {{ let data = vec![1,2,3]; let sum: i32 = data.iter().map(|x| x + data.len() as i32).sum(); }}',
            "mut_capture": 'fn count_{name}() {{ let mut n = 0; let inc = || n += 1; inc(); inc(); }}',
        },
        "fixed_templates": {
            "move_keyword": 'fn spawn_{name}() {{ let s = String::from("hi"); std::thread::spawn(move || println!("{{}}", s)); }}',
            "ref_capture": 'fn iter_{name}() {{ let data = vec![1,2,3]; let len = data.len(); let sum: i32 = data.iter().map(|x| x + len as i32).sum(); }}',
            "mut_capture": 'fn count_{name}() {{ let mut n = 0; let mut inc = || n += 1; inc(); inc(); }}',
        }
    },
]

def generate_names():
    """Generate random variable/function names."""
    prefixes = ["data", "value", "item", "result", "config", "state", "info", "record", "entry", "node"]
    suffixes = ["", "_v2", "_new", "_alt", "_impl"]
    return [p + s for p in prefixes for s in suffixes]

def generate_koans():
    """Generate all koans from templates."""
    koans = []
    names = generate_names()
    random.seed(42)

    # Result Handling koans
    for template in RESULT_HANDLING_TEMPLATES:
        if "types" in template:
            for t in template["types"]:
                name = random.choice(names)
                default = template["defaults"].get(t, "Default::default()")
                koans.append({
                    "id": f"rh_gen_{len(koans):03d}",
                    "family": "ResultHandling",
                    "name": f"Use unwrap_or_else for {t}",
                    "buggy": template["buggy"].format(name=name, type=t, default=default),
                    "fixed": template["fixed"].format(name=name, type=t, default=default),
                })
        elif "ops" in template:
            for op in template["ops"]:
                name = random.choice(names)
                koans.append({
                    "id": f"rh_gen_{len(koans):03d}",
                    "family": "ResultHandling",
                    "name": f"Add proper Result type for {op}",
                    "buggy": template["buggy_templates"][op].format(name=name),
                    "fixed": template["fixed_templates"][op].format(name=name),
                })
        elif "variants" in template:
            for var in template["variants"]:
                name = random.choice(names)
                koans.append({
                    "id": f"rh_gen_{len(koans):03d}",
                    "family": "ResultHandling",
                    "name": f"Fix {var} pattern",
                    "buggy": template["buggy_templates"][var].format(name=name),
                    "fixed": template["fixed_templates"][var].format(name=name),
                })

    # Trait Bounds koans
    for template in TRAIT_BOUNDS_TEMPLATES:
        if "traits" in template:
            for trait in template["traits"]:
                name = random.choice(names)
                use_case, derive = template["use_cases"][trait]
                koans.append({
                    "id": f"tb_gen_{len(koans):03d}",
                    "family": "TraitBounds",
                    "name": f"Add {trait} derive for {use_case[:20]}...",
                    "buggy": f"struct {name.title()} {{ value: i32 }} fn main() {{ let {name} = {name.title()} {{ value: 1 }}; {use_case.format(var=name, type=name.title(), vec='v')} }}",
                    "fixed": f"{derive} struct {name.title()} {{ value: i32 }} fn main() {{ let {name} = {name.title()} {{ value: 1 }}; {use_case.format(var=name, type=name.title(), vec='v')} }}",
                })
        elif "variants" in template:
            for var in template["variants"]:
                name = random.choice(names)
                koans.append({
                    "id": f"tb_gen_{len(koans):03d}",
                    "family": "TraitBounds",
                    "name": f"Fix {var} for Send/Sync",
                    "buggy": template["buggy_templates"][var].format(name=name),
                    "fixed": template["fixed_templates"][var].format(name=name),
                })
        elif "ops" in template:
            for op in template["ops"]:
                koans.append({
                    "id": f"tb_gen_{len(koans):03d}",
                    "family": "TraitBounds",
                    "name": f"Add generic bound for {op}",
                    "buggy": template["buggy_templates"][op],
                    "fixed": template["fixed_templates"][op],
                })

    # Borrow Checker koans
    for template in BORROW_CHECKER_TEMPLATES:
        for var in template["variants"]:
            name = random.choice(names)
            koans.append({
                "id": f"bc_gen_{len(koans):03d}",
                "family": "BorrowChecker",
                "name": f"Fix {template['template'].replace('_{variant}', '')} {var}",
                "buggy": template["buggy_templates"][var].format(name=name),
                "fixed": template["fixed_templates"][var].format(name=name),
            })

    return koans

def convert_to_sft(koan):
    """Convert koan to SFT training format."""
    return {
        "instruction": f"Fix the following Rust compilation error. The issue is: {koan['name']}",
        "input": koan["buggy"],
        "output": koan["fixed"],
    }

def main():
    output_dir = Path("/home/mike/github/softwarewrighter/sleepy-coder/data/sft")

    koans = generate_koans()
    print(f"Generated {len(koans)} koans:")

    # Count by family
    families = {}
    for k in koans:
        f = k["family"]
        families[f] = families.get(f, 0) + 1
    for f, c in families.items():
        print(f"  {f}: {c}")

    # Convert to SFT format
    sft_data = [convert_to_sft(k) for k in koans]

    # Save generated koans
    with open(output_dir / "generated_large.jsonl", "w") as f:
        for item in sft_data:
            f.write(json.dumps(item) + "\n")
    print(f"\nSaved {len(sft_data)} examples to generated_large.jsonl")

    # Load existing training data
    existing = []
    if (output_dir / "train.jsonl").exists():
        with open(output_dir / "train.jsonl") as f:
            existing = [json.loads(line) for line in f if line.strip()]

    # Combine: new data + 3x replay of original
    combined = sft_data + existing * 3
    random.shuffle(combined)

    with open(output_dir / "large_with_replay.jsonl", "w") as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")

    print(f"Combined dataset: {len(combined)} examples")
    print(f"  New: {len(sft_data)}")
    print(f"  Replay: {len(existing) * 3}")
    print(f"Saved to large_with_replay.jsonl")

if __name__ == "__main__":
    main()
