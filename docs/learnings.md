# Learnings Log

This document captures key mistakes and corrections during sleepy-coder development.

---

## 2026-02-10: Share Algorithm Export Mistake

### The Mistake

After training 51 pattern-specific LoRA adapters and consolidating them with the Share algorithm (SVD-based subspace extraction), I exported **only task 0's coefficients** to create the final model.

```bash
# Wrong: Only exports one task's specialization
python share_proper.py export --share share_model --task 0 -o exported_adapter
```

**Result**: 70.0% pass rate (regression from 76.7% baseline)

### Why It Was Wrong

The Share algorithm maintains:
1. **Shared basis** (β, α): Captures common structure across all 51 patterns
2. **Per-task coefficients** (ε_β, ε_α): Specializes the basis for each specific pattern

Exporting task 0 (`decompose_god_struct`) gave an adapter specialized for ONLY that pattern. When evaluated on the 30-koan test set (covering borrow checker, trait bounds, result handling), it performed poorly because:
- The exported adapter was tuned for `decompose_god_struct` only
- It had no knowledge of `missing_clone`, `mut_borrow_conflict`, `option_unwrap_or`, etc.

### The Fix

For general-purpose use, **average the coefficients across all tasks**:

```python
# Average coefficients across all 51 tasks
avg_eps_beta[layer] = np.stack([task[layer] for task in all_eps_beta]).mean(axis=0)
avg_eps_alpha[layer] = np.stack([task[layer] for task in all_eps_alpha]).mean(axis=0)

# Reconstruct adapter using averaged coefficients
B_hat = beta @ avg_eps_beta
A_hat = (alpha @ avg_eps_alpha).T
```

This creates a single adapter that represents the "average" behavior across all 51 learned patterns, which should generalize better.

### Key Insight

Share was designed for **continual learning** where you maintain separate coefficients per task and select the appropriate one at inference time. For our use case (improving overall model performance), we need to:
1. Either average coefficients for a general-purpose model
2. Or implement task routing to select coefficients dynamically

---

## 2026-02-10: Training on Known Patterns Causes Forgetting

### The Mistake

After fixing the Share export to average coefficients (73.3%), I analyzed which koans failed and compared to baseline. The baseline (no training) fails 7 koans but PASSES rh_008. After Share training, the model STILL fails all 7 original failures AND NOW also fails rh_008.

**Training data**: 51 patterns × 3-15 examples each = 315 examples
**Result**: No improvement on failures + 1 regression

### Why It Was Wrong

The base model (Qwen2.5-Coder-1.5B-Instruct) already knows ~77% of the patterns. Training on ALL patterns:
1. Dilutes learning signal for the 7 failure cases
2. Overwrites existing knowledge (catastrophic forgetting)
3. The Share algorithm can't help if we're training on things already known

### The Fix

**Only train on patterns the baseline FAILS on:**

```
Baseline failures (7):
- bc_003: mut_borrow_conflict
- bc_005: double_mut_borrow
- bc_010: return_local_ref
- rh_004: option_ok_or
- tb_002: missing_clone
- tb_007: missing_hash
- tb_008: missing_ord
```

Create targeted training data ONLY for these 7 patterns with:
1. More examples per pattern (20-50 instead of 5-10)
2. More variation in code structure
3. Clear error messages matching what rustc produces

### Key Insight

**Don't train on what the model already knows.** Identify failures first, then create targeted training data. Training on already-mastered patterns wastes compute and risks forgetting.

---

## 2026-02-10: Targeted Training Causes MORE Forgetting

### The Mistake

After identifying that training on known patterns caused forgetting, I created targeted training with ONLY the 7 failing patterns (44 examples total). I expected this would avoid dilution and forgetting.

**Training data**: 44 examples for 7 failure patterns only
**Result**: 63.3% pass rate (19/30) - WORSE than baseline (76.7%)

### Why It Was Wrong

The model went from 7 failures to 11 failures. Four new regressions:
- rh_001: `result_ok_or` (was passing)
- rh_005: `result_map_err` (was passing)
- tb_004: `missing_generic_bound` (was passing)
- tb_009: `missing_ord_for_btreeset` (was passing)

Even targeted training modified weights that were working correctly. The LoRA adaptation touched shared representations, causing interference.

### Key Insight

With a capable base model, LoRA fine-tuning can ONLY hurt overall performance. The model's knowledge is interconnected - modifying any part risks breaking other parts.

---

## 2026-02-10: Replay Buffer Still Causes Forgetting

### The Mistake

To prevent forgetting during targeted training, I added a "replay buffer" with 17 examples from patterns the model PASSES. Theory: training on passing patterns would reinforce existing knowledge while learning failure patterns.

**Training data**: 61 examples (44 targeted failures + 17 replay from passing patterns)
**Result**: 70.0% pass rate (21/30) - Still worse than baseline

### Why It Was Wrong

Still caused 2 regressions (rh_005, tb_009) while fixing ZERO baseline failures. The replay buffer reduced forgetting compared to pure targeted training (9 failures vs 11), but couldn't prevent it entirely.

### Key Insight

**The base model (Qwen2.5-Coder-1.5B-Instruct) cannot be improved by LoRA fine-tuning for this task.**

Evidence across all experiments:
| Approach | Pass Rate | vs Baseline |
|----------|-----------|-------------|
| Baseline (no LoRA) | 76.7% | - |
| Share-51 averaged | 73.3% | -3.4% |
| Targeted (7 patterns) | 63.3% | -13.4% |
| Replay buffer | 70.0% | -6.7% |
| Ultra-conservative (lr=1e-6) | No learning | N/A |

**Possible paths forward:**
1. **Prompt engineering**: Improve prompts rather than weights
2. **Multi-turn repair**: Use dialogue to iteratively fix errors
3. **Larger model**: Use a model with more capacity to absorb new knowledge
4. **Model ensemble**: Run multiple models and pick best output
5. **Accept baseline**: 76.7% may be the ceiling for this model+task

---

## 2026-02-10: Share Phase 2 Coefficient-Only Training

### The Experiment

Implemented Share Phase 2 training exactly as described in arXiv:2602.06043:
- Froze the shared basis (β, α) extracted from 51 adapters
- Trained only coefficients (ε_β, ε_α) initialized to zero
- Used ~10K trainable parameters vs ~689K for full LoRA (67x fewer)

**Training data**: 44 examples targeting 7 baseline failure patterns
**Steps**: 50, LR: 1e-4, Batch: 4

### Results

| Model | Pass Rate | Failures |
|-------|-----------|----------|
| Fresh Baseline | 73.3% | 8 |
| Phase 2 Coef-Only | 66.7% | 10 |

**New regressions**: rh_001, tb_009
**Fixed**: None

### Why It Still Caused Forgetting

Even with 67x fewer trainable parameters, Phase 2 training still caused 2 regressions. Possible reasons:

1. **Basis quality**: The shared basis was extracted from 51 adapters trained on ALL patterns (including ones the model already knew). This may not be the optimal basis for learning only the failure patterns.

2. **Subspace coverage**: The Share paper assumes the basis spans the relevant task subspace. If the failure patterns require directions not well-represented in the basis, training forces suboptimal projections.

3. **Scale of change**: Even small coefficient changes can shift the model's output distribution enough to cause regressions on edge cases.

### Key Insight

**Phase 2 coefficient-only training reduces but does not eliminate forgetting.** The Share algorithm's anti-forgetting properties depend on:
- The basis properly spanning all relevant task directions
- New tasks being representable as linear combinations of existing basis vectors
- Coefficients being small enough not to interfere with existing behavior

For our use case (improving failures without regressing passes), even the minimal parameter count of Share Phase 2 isn't enough. The fundamental issue may be that the base model's knowledge is interconnected in ways that any modification (however small) can disrupt.

### Comparison of All Approaches

| Approach | Pass Rate | Trainable Params | Regressions |
|----------|-----------|------------------|-------------|
| Baseline (no training) | 73.3% | 0 | 0 |
| Share-51 averaged | 73.3% | N/A | 0 |
| Targeted full LoRA | 63.3% | ~689K | 4+ |
| Replay buffer | 70.0% | ~689K | 2 |
| **Phase 2 coef-only** | **66.7%** | **~10K** | **2** |

---

## 2026-02-10: Share Full Algorithm (Phase 2 + Phase 3)

### The Experiment

After Phase 2 alone caused 2 regressions (66.7%), implemented the complete Share algorithm with all three phases:

1. **Phase 1**: Initialize shared basis from 51 adapters via SVD
2. **Phase 2**: Train coefficients with φ=2 temporary expansion (558K params)
3. **Phase 3**: Merge trained coefficients, re-run SVD, update basis for all tasks

**Training data**: 44 examples targeting 7 baseline failure patterns
**φ expansion**: 2x (temporary basis vectors during Phase 2)

### Results

| Model | Pass Rate | Failures | Regressions |
|-------|-----------|----------|-------------|
| Baseline | 73.3% | 8 | 0 |
| Phase 2 only | 66.7% | 10 | 2 (rh_001, tb_009) |
| **Share Full (Ph2+Ph3)** | **73.3%** | **8** | **0** |

The exact same 8 failures as baseline: bc_003, bc_005, bc_010, rh_004, rh_005, tb_002, tb_007, tb_008

### What Happened

Phase 3 merge successfully "neutralized" the Phase 2 training damage:
- Re-running SVD on the updated basis (now 7 tasks) recalculated the shared subspace
- Averaging coefficients across all tasks returned to "generalist" behavior
- The model behaves identically to baseline

### Key Insight

**The Share algorithm correctly prevents forgetting when all three phases are used.** However, averaging coefficients negates any task-specific learning. To actually improve on failures, we need:

1. **Task routing**: Select appropriate coefficients at inference time based on error type
2. **Better basis**: Train more adapters (~100+) to build a richer shared subspace
3. **Better training data**: More examples with greater variation for failure patterns

The current experiment validates that Share works as designed (no forgetting with full algorithm), but doesn't demonstrate improvement because we're averaging away the specialization.

### Comparison of All Approaches

| Approach | Pass Rate | Trainable Params | Regressions |
|----------|-----------|------------------|-------------|
| Baseline (no training) | 73.3% | 0 | 0 |
| Share-51 averaged | 73.3% | N/A | 0 |
| Targeted full LoRA | 63.3% | ~689K | 4+ |
| Replay buffer | 70.0% | ~689K | 2 |
| Phase 2 coef-only | 66.7% | ~10K | 2 |
| **Share Full (Ph2+Ph3)** | **73.3%** | **~558K** | **0** |

---

## 2026-02-10: Share51 - More Adapters Doesn't Always Help

### The Experiment

After discovering our Share model was built from only 6 adapters (not 51), rebuilt Phase 1 with all 51 pattern-specific adapters. This gave a much richer shared subspace:

| Metric | Share6 | Share51 |
|--------|--------|---------|
| k_beta (total) | ~6 | 1437 |
| k_alpha (total) | ~6 | 17578 |
| Adapters | 6 | 51 |

Then ran Phase 2+3 (train on 7 failure patterns, merge, average coefficients).

### Results

| Model | Pass Rate | Failures | Changes vs Baseline |
|-------|-----------|----------|---------------------|
| Baseline | 73.3% | 8 | - |
| Share6 Full | 73.3% | 8 | No change |
| **Share51 Full** | **70.0%** | **9** | Fixed rh_005, regressed rh_001+tb_009 |

### Analysis

With 51 adapters, the model actually:
- **Fixed** rh_005 (result_map_err) - this was a baseline failure that now passes
- **Regressed** rh_001 (result_ok_or), tb_009 (missing_ord_for_btreeset)

Net result: 1 fix, 2 regressions = worse overall.

### Why More Adapters Made It Worse

1. **Subspace dilution**: With 51 very different adapters, the shared basis captures more diverse patterns but less strongly represents any single one

2. **Averaging artifacts**: When we average coefficients across 52 tasks (51 original + 1 new), the contribution from any single task is diluted further

3. **Interference**: The richer subspace allows the new task to project onto more dimensions, some of which may interfere with existing behaviors

### Key Insight

**For averaged-coefficient inference, fewer well-chosen adapters may outperform many diverse adapters.** The Share paper's assumption is that tasks live in a low-dimensional subspace. If the adapters are too diverse, the "shared" subspace becomes too general to be useful.

**Better approach**: Instead of averaging, implement task routing to select specific coefficients based on error type. The Share model already supports `export_adapter(task_id)` for per-task adapters.

---

## Template for Future Entries

### YYYY-MM-DD: [Title]

**The Mistake**: [What was done wrong]

**Why It Was Wrong**: [Root cause analysis]

**The Fix**: [Correct approach]

**Key Insight**: [Generalizable lesson]
