# Next Steps: Breaking the 73.3% Ceiling

**Date**: 2026-02-09
**Author**: Claude Opus 4.5
**Status**: Recommendations

---

## Executive Summary

After 13 training cycles, the best result achieved is **73.3%** — still 3.4% below the baseline of 76.7%. Despite implementing the Share algorithm properly (SVD-based subspace extraction, coefficient-only training), we cannot improve because **the training data doesn't overlap with the evaluation tasks**.

---

## Current State

### Training History

| Cycle | Approach | Pass Rate | Delta |
|-------|----------|-----------|-------|
| C0 | Baseline (no training) | **76.7%** | — |
| C1 | Naive LoRA on failures | 60.0% | -16.7% |
| C2-3 | Added replay buffer | 66-70% | -7 to -10% |
| C9-10 | Minimal steps (20) | 73.3% | -3.4% |
| C11 | Expanded data (112 ex) | 70.0% | -6.7% |
| C12 | Rust 2024 koans | 73.3% | -3.4% |
| C13 | Share (proper SVD) | 73.3% | -3.4% |

### The 73.3% Ceiling

We hit 73.3% three times (C9, C10, C12, C13) with different approaches. This suggests:
1. The ceiling is **not** due to the training algorithm
2. The ceiling is due to **what** we're training on

---

## Root Cause: Domain Mismatch

### Training Data vs Eval Data

| Training Domains | Eval Domains |
|-----------------|--------------|
| yew_wasm (WASM/frontend) | borrow_checker |
| axum_server (web servers) | trait_bounds |
| sqlx_db (database) | result_handling |
| cli_clap (CLI parsing) | |
| refactoring | |
| style_metrics | |

**There is zero overlap.** Training on web frameworks doesn't help fix borrow checker errors.

### Why This Matters

The Share paper works because each adapter learns a **distinct skill** that contributes to a shared subspace. Our adapters learn skills the eval set doesn't test.

```
Share assumption:  Adapter_i teaches skill_i, eval tests skill_i
Our reality:       Adapter_i teaches skill_i, eval tests skill_j
                   → No transfer possible
```

---

## Recommendations

### Option 1: Generate Training Data That Matches Eval (Recommended)

Create training examples that teach the **same patterns** tested in the eval set.

**Implementation:**
```bash
python scripts/generate_eval_aligned_koans.py
```

This script should:
1. Analyze the 30 frozen eval koans
2. Generate 100+ similar (but not identical) training examples
3. Cover the same error patterns: move semantics, lifetime annotations, trait bounds, Result/Option handling

**Expected outcome:** Training should directly improve eval performance since we're teaching what's being tested.

### Option 2: Expand Eval Set to Match Training

Add new eval tasks that test the domains we trained on.

**Pros:**
- Uses existing training data
- Tests real-world scenarios (web, db, CLI)

**Cons:**
- Moves the goalposts
- Original eval set was designed to test core Rust patterns

### Option 3: Self-Synthesized Rehearsal (SSR)

Use the base model itself to generate training data.

**From ACL 2024 research:**
> "LLMs can generate synthetic instances for rehearsal, achieving superior performance while being more data-efficient."

**Implementation:**
```bash
python scripts/generate_ssr_data.py
```

1. For each eval koan pattern, ask the base model to generate 10 variants
2. Filter for quality (must compile, must have the target error)
3. Use base model to generate fixes
4. Train on this synthetic data

### Option 4: Direct Preference Optimization (DPO)

Switch from SFT to preference-based learning.

**Why:**
- SFT memorizes specific input→output mappings
- DPO learns to prefer good outputs over bad ones
- Better generalization according to research

**Implementation:**
```bash
python scripts/train_dpo.py --pairs data/sft/preference_pairs.jsonl
```

---

## Recommended Action Plan

### Phase 1: Align Training Data (This Week)

1. **Analyze eval koans** — Extract the specific error patterns being tested
2. **Generate aligned training data** — 100+ examples per error family
3. **Retrain with aligned data** — Use existing Share infrastructure
4. **Evaluate** — Should see improvement if hypothesis is correct

### Phase 2: SSR Augmentation (If Phase 1 Works)

1. Use base model to generate more variants
2. Scale to 500+ training examples
3. Retrain and measure

### Phase 3: DPO (If SFT Plateaus)

1. Create preference pairs from eval attempts
2. Implement DPO training loop
3. Compare to SFT results

---

## Scripts to Create

### 1. `scripts/generate_eval_aligned_koans.py`

Generates training data that matches eval patterns.

```python
# Pseudocode
def generate_aligned_koans():
    # 1. Load frozen eval set
    eval_koans = load_frozen_eval_set()

    # 2. Extract error patterns
    patterns = {
        'borrow_checker': extract_bc_patterns(eval_koans),
        'trait_bounds': extract_tb_patterns(eval_koans),
        'result_handling': extract_rh_patterns(eval_koans),
    }

    # 3. Generate variants for each pattern
    training_data = []
    for family, family_patterns in patterns.items():
        for pattern in family_patterns:
            variants = generate_variants(pattern, n=10)
            training_data.extend(variants)

    # 4. Save as training JSONL
    save_jsonl(training_data, 'data/sft/eval_aligned.jsonl')
```

### 2. `scripts/generate_ssr_data.py`

Uses base model to synthesize training data.

### 3. `scripts/train_dpo.py`

Implements Direct Preference Optimization.

### 4. `scripts/analyze_eval_coverage.py`

Measures overlap between training and eval domains.

---

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Pass Rate | 73.3% | ≥76.7% (baseline) | ≥85% |
| Regression | -3.4% | 0% | +10% |
| Training/Eval Overlap | 0% | ≥50% | ≥80% |

---

## Key Insight

> **The model isn't failing to learn. It's learning the wrong things.**
>
> Share, replay, conservative hyperparameters — none of these matter if we're teaching web framework patterns and testing borrow checker fixes.

The fix is simple: **teach what you test**.

---

## References

- [Share Paper (arXiv:2602.06043)](https://arxiv.org/abs/2602.06043)
- [Self-Synthesized Rehearsal (ACL 2024)](https://aclanthology.org/2024.acl-long.77/)
- [DPO Paper (arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Course Correction Analysis](./course-correction.md)
- [Training Changes Log](./changes.md)
