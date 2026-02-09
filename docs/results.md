# Results: Sleepy Coder Training Progress

## Baseline (Before Training)

### Model: qwen2.5-coder:1.5b

| Metric | Value | Notes |
|--------|-------|-------|
| Pass Rate (5 borrow koans) | 80% (4/5) | Initial test run |
| Pass Rate (30 frozen eval) | 66.7% (20/30) | Full baseline |
| Median Steps to Green | 2.0 | Most solved in 2 attempts |
| Date | 2026-02-08 | Phase 1 MVP complete |

### Initial Quick Test (5 borrow koans)

```
[1/5] bc_001 ... PASS (attempts: 2)
[2/5] bc_003 ... FAIL (max attempts reached)
[3/5] bc_005 ... PASS (attempts: 2)
[4/5] bc_007 ... PASS (attempts: 2)
[5/5] bc_009 ... PASS (attempts: 2)
```

---

## Training Cycles

### Cycle 0: Baseline Evaluation (Frozen Eval Set)

| Metric | Run 1 | Run 2 | Notes |
|--------|-------|-------|-------|
| Tasks Evaluated | 30 | 30 | |
| Pass Rate | 66.7% (20/30) | 76.7% (23/30) | Variance due to LLM non-determinism |
| Failed | 10 | 7 | |
| Repeat Error Rate | N/A | N/A | First cycle |
| Median Steps to Green | 2.0 | 2.0 | Consistent |

**Error Signatures:**
- `max_attempts_exceeded`: 7-10 tasks (varies by run)

**Note**: Model outputs are non-deterministic, so baseline results may vary ~10% between runs.

### Cycle 1: First Training (Quick Test - 50 Steps)

**Training Config:**
- Steps: 50 (quick validation run)
- Training samples: 23 episodes
- Duration: ~25 min on Mac M-series
- Loss: 2.35 → 0.72 (69% reduction)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Pass Rate | 66.7% | 66.7% | 0% |
| Median Steps | 2.0 | 2.0 | 0 |
| Repeat Error Rate | N/A | TBD | TBD |

**Analysis:** No improvement with 50 training steps. This is expected - the model needs more training data and iterations. The loss dropped significantly (69%) suggesting the model is learning, but not enough to generalize to the eval set yet.

---

## Frozen Eval Set Performance

The frozen eval set contains 30 koans (10 borrow_checker, 10 trait_bounds, 10 result_handling).

| Cycle | Pass Rate | Passed | Failed | Regression? |
|-------|-----------|--------|--------|-------------|
| 0 | 66.7% | 20 | 10 | N/A (baseline) |
| 1 | 66.7% | 20 | 10 | No change |
| 2 | TBD | TBD | TBD | TBD |

---

## Error Signature Analysis

Most common error signatures across cycles:

| Signature | Cycle 0 | Cycle 1 | Trend |
|-----------|---------|---------|-------|
| max_attempts_exceeded | 10 | 10 | No change |

---

## Training Metrics

| Cycle | Steps | Initial Loss | Final Loss | Duration | Examples |
|-------|-------|--------------|------------|----------|----------|
| 1 | 50 | 2.35 | 0.72 | 25 min | 23 |
| 2 | TBD | TBD | TBD | TBD | TBD |

### Loss Curve (Cycle 1)
```
Loss │
2.3  │ ●●●
2.1  │    ●
2.0  │     ●
1.7  │      ●
1.6  │       ●
1.2  │        ●
1.0  │         ●
0.7  │          ●  ← Final
     └─────────────────
       5  10 15 20 25 30 35 40 45 50  Steps
```

---

## Visualizations

Plots will be saved to `viz/` directory:
- `pass_rate_by_cycle.png` - Pass rate improvement
- `steps_to_green_by_cycle.png` - Efficiency improvement
- `repeat_error_rate_by_cycle.png` - Learning from mistakes
- `error_distribution.png` - Error type breakdown
- `before_after_comparison.png` - Before/after comparison

---

## Key Observations

### Baseline Analysis

1. **Pass Rate**: 66.7% (20/30) on frozen eval set
2. **Efficiency**: Most successful fixes in 2 attempts
3. **Failure Mode**: All 10 failures hit max attempts limit (5)
4. **Target**: Improve pass rate to >90% while maintaining low steps-to-green

### Next Steps

1. ✅ Export failed episodes for training data
2. ✅ Create SFT dataset from successful fixes
3. ✅ Train LoRA adapter on the data (50 steps, quick test)
4. ✅ Re-evaluate on frozen set to measure improvement (no change yet)
5. **Train longer** (500-1000 steps) to see measurable improvement

---

## Cost Estimates for Further Progress

### Mac M-series (Current Setup)
| Training Steps | Duration | Expected Result |
|----------------|----------|-----------------|
| 50 (done) | 25 min | No change |
| 500 | ~4 hours | Potentially 5-10% improvement |
| 1000 | ~8 hours | Potentially 10-20% improvement |
| 5000 | ~40 hours | Target: 80%+ pass rate |

### NVIDIA GPU (Linux Workstation)
Estimated 5-10x speedup vs Mac:
| Training Steps | Duration | Improvement |
|----------------|----------|-------------|
| 500 | ~25-50 min | 5-10% |
| 1000 | ~50-100 min | 10-20% |
| 5000 | ~4-8 hours | Target: 80%+ |

### Validation Decision Point

**When to move to CUDA:**
- If 500 Mac steps (4 hours) show ANY improvement → validate on Mac, then scale on CUDA
- If 500 Mac steps show NO improvement → investigate data quality before investing more compute

### Key Variables for Success

1. **Training Data Quality**: 23 examples is very small. Need more diverse examples.
2. **Training Steps**: 50 is too few for generalization. Need 500+ minimum.
3. **Learning Rate**: Current 2e-4 may need tuning.
4. **Data Diversity**: Currently only training on failed episodes from baseline. Need to add synthetic examples or more varied tasks.
