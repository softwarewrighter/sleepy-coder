# Training Data Quality: Clippy, Complexity, and Feedback Loops

*Document created: 2026-02-09*
*Status: Active — updated each sleep cycle*

## Overview

This document describes three improvements to the sleepy-coder training data
pipeline that address gaps identified during cycle 9–13 plateau analysis:

1. **Clippy suppression anti-patterns** — teach "fix, don't suppress"
2. **Rust 2024 edition patterns** — correct stale idioms from pre-2024 LLMs
3. **Code complexity patterns** — sw-checker compliance training
4. **Corrections feedback loop** — route external tool output into sleep cycles

These supplement the core eval-aligned data (30 frozen koans covering
BorrowChecker, TraitBounds, ResultHandling) with training signal for
code quality concerns that go beyond compilation errors.

## Problem Statement

Three gaps were identified in the training data:

### Gap 1: LLMs Suppress Clippy Instead of Fixing

Models trained on pre-2024 Rust corpora learn to silence warnings with
`#[allow(clippy::...)]` or even `#![allow(clippy::all)]` rather than fixing
the underlying issue. Since `scripts/lint.sh` runs `cargo clippy -- -D warnings`,
these suppressions hide problems that later fail in CI.

### Gap 2: Stale Rust Idioms

The base model (Qwen2.5-Coder-1.5B-Instruct) was trained on code where:
- `println!("{}", x)` is idiomatic (should be `println!("{x}")`)
- `iter().cloned()` is used on Copy types (should be `copied()`)
- Nested `if let` + `if` (should be let-chains)
- `match ... true/false` (should be `matches!()` macro)

Edition 2024 clippy flags all of these.

### Gap 3: No Complexity Training

sw-checker enforces:
- Maximum lines per function
- Maximum functions per module
- Maximum modules per crate
- Maximum crates per workspace

The training data had only 6 style examples (in `data/sft/domains/style_metrics.jsonl`)
and none were included in the eval-aligned pipeline.

### Gap 4: No Feedback Path

When sw-checker or clippy rejected model output, that signal was lost.
The sleep cycle only learned from its own episodes — there was no mechanism
to inject external corrections.

## Solution

### Training Data Generation

All new patterns are integrated into `scripts/generate_eval_aligned_koans.py`:

```bash
# Full generation (eval-aligned + supplemental)
python scripts/generate_eval_aligned_koans.py

# Eval-aligned only (original 30 patterns)
python scripts/generate_eval_aligned_koans.py --no-supplemental
```

Output breakdown (default settings, `--variants 10`):

| Category | Patterns | Examples | Family |
|---|---|---|---|
| BorrowChecker (eval) | 10 | 147 | `borrow_checker` |
| TraitBounds (eval) | 10 | 98 | `trait_bounds` |
| ResultHandling (eval) | 10 | 72 | `result_handling` |
| Clippy anti-suppress | 8 | 36 | `clippy_suppression` |
| Rust 2024 edition | 9 | 63 | `rust_2024` |
| Code complexity | 4 | 15 | `complexity` |
| **Total** | **51** | **431** | |

### Clippy Anti-Suppression Patterns (8 patterns)

Each pattern shows `#[allow(clippy::X)]` as the buggy code and the actual
fix as the correct code:

| Pattern | Clippy Lint | Fix |
|---|---|---|
| `no_suppress_needless_collect` | `needless_collect` | Remove intermediate `.collect()` |
| `no_suppress_format_args` | `uninlined_format_args` | Use `println!("{x}")` |
| `no_suppress_manual_map` | `manual_map` | Use `.map()` |
| `no_suppress_matches` | `match_like_matches_macro` | Use `matches!()` |
| `no_suppress_clone_ref` | `clone_on_ref_ptr` | Use `Arc::clone()` |
| `no_blanket_suppress` | `clippy::all` | Fix each issue individually |
| `no_suppress_must_use` | `unused_must_use` | Handle the `Result` |
| `no_suppress_dead_code` | `dead_code` | Remove unused code |

### Rust 2024 Edition Patterns (9 patterns)

| Pattern | What It Teaches |
|---|---|
| `inline_format_args` | `println!("{x}")` instead of `println!("{}", x)` |
| `iter_copied` | `.copied()` instead of `.cloned()` for Copy types |
| `use_matches_macro` | `matches!(c, '0'..='9')` instead of match-true-false |
| `use_flatten` | `.flatten()` instead of filter-is_some-map-unwrap |
| `remove_needless_collect` | Drop unnecessary `.collect::<Vec<_>>()` |
| `let_chains` | `if let Some(x) = opt && x > 0` |
| `let_else` | `let Some(x) = opt else { return 0; };` |
| `is_some_and` | `opt.is_some_and(\|x\| x > 0)` |
| `slice_param` | `&[T]` instead of `&Vec<T>` in function params |

### Code Complexity Patterns (4 patterns)

| Pattern | sw-checker Rule | Fix |
|---|---|---|
| `decompose_long_function` | Lines per function | Extract into smaller focused functions |
| `split_large_module` | Functions per module | Split into submodules |
| `extract_modules_to_files` | Lines per file | Move inline `mod {}` to separate files |
| `decompose_god_struct` | Fields per struct | Break into focused sub-structs |

## Corrections Feedback Loop

### How It Works

```
sw-checker / clippy / review
        │
        ▼
data/sft/corrections.jsonl     ◄── Append one entry per issue
        │
        ├──► generate_eval_aligned_koans.py   (merged into training JSONL)
        │
        └──► prepare_training_data.py         (3x weighted in mixed dataset)
                │
                ▼
        data/sft/mixed.jsonl    ──► train.py  ──► next sleep cycle
```

### Adding a Correction

When sw-checker or clippy catches something the model gets wrong, append
a JSONL entry to `data/sft/corrections.jsonl`:

**Short format** (buggy/fixed):
```json
{"task_id": "swck_001", "family": "complexity", "pattern": "long_function",
 "buggy": "fn do_everything() { /* 80 lines */ }",
 "fixed": "fn parse() {} fn validate() {} fn execute() {}",
 "error_hint": "sw-checker: function exceeds 40 lines",
 "source": "sw-checker"}
```

**Full SFT format** (instruction/input/output):
```json
{"instruction": "You are a Rust expert...",
 "input": "## Buggy Code:\n```rust\n...\n```\n\n## Compiler Error:\n...",
 "output": "...",
 "task_id": "review_001",
 "source": "code-review"}
```

### Automating Corrections

To routinely capture sw-checker feedback:

```bash
# 1. Run sw-checker on model output, capture failures
sw-checker --check target/model_output.rs 2>&1 | tee /tmp/swck.log

# 2. For each failure, create a correction entry
# (This could be scripted — parse sw-checker output, pair with model's
#  input, and append to corrections.jsonl)

# 3. Next sleep cycle picks it up automatically
python scripts/generate_eval_aligned_koans.py
python scripts/prepare_training_data.py --corrections data/sft/corrections.jsonl
python cuda/scripts/train.py --data data/sft/mixed.jsonl --steps 100
```

### Weight in Training

Corrections are treated as high-signal data:

- In `generate_eval_aligned_koans.py`: merged directly into the output JSONL
- In `prepare_training_data.py`: **3x copies** added to the mixed dataset

This ensures the model sees corrections frequently during training without
needing thousands of examples.

## Prepare Script Usage

The `prepare_training_data.py` script now supports three new flags:

```bash
# Default: auto-loads corrections from standard location
python scripts/prepare_training_data.py

# Explicit corrections path
python scripts/prepare_training_data.py \
  --corrections data/sft/corrections.jsonl

# Include additional supplemental JSONL files
python scripts/prepare_training_data.py \
  --supplemental data/sft/rust2024_koans.jsonl data/sft/domains/style_metrics.jsonl

# Full pipeline example
python scripts/prepare_training_data.py \
  --corrections data/sft/corrections.jsonl \
  --supplemental data/sft/rust2024_koans.jsonl \
  --output data/sft/mixed.jsonl
```

## Recommendations

### Short Term

1. **Run a training cycle with the full 431-example dataset** to see if
   supplemental data hurts eval pass rate (risk: complexity/clippy patterns
   dilute the eval-aligned signal).

2. **Compare `--no-supplemental` (317 examples) vs full (431)** — if the
   supplemental data hurts eval, keep it separate and only mix via
   `prepare_training_data.py --supplemental`.

3. **Seed `corrections.jsonl`** with the top 5 most common clippy failures
   from recent model output.

### Medium Term

4. **Automate the sw-checker → corrections pipeline**: add a post-eval hook
   that runs sw-checker on model output and auto-generates correction entries.

5. **Add eval metrics for clippy compliance**: track what % of model output
   passes `cargo clippy -- -D warnings` in addition to compilation pass rate.

6. **Expand complexity patterns**: the current 4 patterns with 15 examples
   are minimal. Add more variants with realistic code to make the signal
   stronger.

### Long Term

7. **Separate eval sets**: create a frozen clippy/complexity eval set
   alongside the existing compilation eval set. This gives a measurable
   signal for code quality improvements.

8. **Weighted training phases**: train on eval-aligned data first (for
   compilation), then fine-tune on supplemental data (for style) with a
   lower learning rate to avoid forgetting.

## Files Changed

| File | Change |
|---|---|
| `scripts/generate_eval_aligned_koans.py` | Added 3 template sections + corrections intake + `--no-supplemental` flag |
| `scripts/prepare_training_data.py` | Added `load_corrections()`, `--corrections`, `--supplemental` flags, 3x weighting |
| `data/sft/corrections.jsonl` | Created empty file (tracked in git, not gitignored) |
| `.gitignore` | Added exception for `corrections.jsonl` |
