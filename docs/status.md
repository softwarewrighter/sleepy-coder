# Project Status: Sleepy Coder

## Current Phase: Prompt Engineering (Success!)

**Last Updated**: 2026-02-17

---

## Key Result: Prompt Engineering Succeeds Where LoRA Failed

After LoRA fine-tuning proved ineffective, **prompt engineering improved pass rate from 73.3% to 83.3% average** (up to 86.7% peak).

Error-specific hints added to the prompt guide the model to fix:
- Borrow checker errors (mutable/immutable conflicts)
- Trait bound errors (Hash, Ord with full trait hierarchies)
- Result/Option conversions

See [rust/crates/agent/src/lib.rs](../rust/crates/agent/src/lib.rs) `get_error_hints()` function.

---

## Experiment Summary

| Approach | Pass Rate | vs Baseline | Regressions |
|----------|-----------|-------------|-------------|
| Baseline (no training) | 73.3% | — | 0 |
| Naive LoRA (C1) | 60.0% | -13.3% | Many |
| Share-51 averaged | 73.3% | 0% | 0 |
| Targeted full LoRA (7 patterns) | 63.3% | -10% | 4+ |
| Replay buffer | 70.0% | -3.3% | 2 |
| Phase 2 coef-only (10K params) | 66.7% | -6.6% | 2 |
| Share Full (Ph2+Ph3) | 73.3% | 0% | 0 |
| **Prompt Engineering** | **83.3%** | **+10%** | **~0** |
| Share Routed (Exp 1) | 43.3%* | -6.7%* | 3 |

### Key Insights

1. **Training on known patterns causes forgetting** — The base model already knows ~77% of patterns. Training dilutes signal and overwrites knowledge.

2. **Even targeted training causes regressions** — Training only on the 7 failure patterns (44 examples) caused 4 new regressions.

3. **Replay buffers reduce but don't prevent forgetting** — Adding 17 "passing" examples still caused 2 regressions.

4. **Share prevents forgetting but doesn't improve** — The full 3-phase algorithm returns to baseline. Averaging coefficients negates specialization.

5. **More adapters ≠ better** — 51 adapters performed worse than 6 due to subspace dilution when averaging.

6. **The model's knowledge is interconnected** — Modifying any weights (even 10K params) risks breaking other capabilities.

7. **Routing is worse than averaging** — See Experiment 1 (2026-02-17) below. The Share paper's routing strategy doesn't help for this model+task. Applying task-specific coefficients actively hurts trait_bounds (40% → 20%).

---

## Experiment 1: Routing vs Averaging vs Baseline (2026-02-17)

Tested the Share paper's core claim: route to task-specific coefficients instead of averaging them.

**Setup**: Single-shot inference (no multi-attempt loop), plain prompt (no hints/examples), Python eval via HuggingFace bf16 model with direct weight modification. Fixed the critical bug in `RoutedShareInference` where LoRA weights were stored as buffers but never intercepted the forward pass.

### Results

| Strategy | Pass Rate | BC (10) | TB (10) | RH (10) |
|----------|-----------|---------|---------|---------|
| Baseline | **50.0%** (15/30) | 70% | 40% | 40% |
| Averaged | 50.0% (15/30) | 70% | 40% | 40% |
| Routed | 43.3% (13/30) | 70% | **20%** | 40% |

### Per-Koan Differences (Routed vs Baseline)

| Koan | Baseline | Routed | Pattern | Direction |
|------|----------|--------|---------|-----------|
| rh_002 | FAIL | PASS | (none) | Improved |
| rh_008 | PASS | FAIL | result_map_err | **Regressed** |
| tb_005 | PASS | FAIL | (none) | **Regressed** |
| tb_009 | PASS | FAIL | (none) | **Regressed** |

### Analysis

1. **Averaging = baseline** — Coefficient averaging produces effectively zero net delta, confirming prior results.
2. **Routing hurts** — Applying specific coefficients (e.g., `result_map_err`, `missing_hash`, `missing_ord`) actively degrades performance. The trait_bounds family dropped from 40% to 20%.
3. **Pattern misrouting** — Some borrow checker errors (bc_001, bc_002, bc_008) were incorrectly routed to `missing_clone` due to overlapping regex patterns. This didn't cause regressions (the base model handles them regardless) but shows the routing logic needs refinement.
4. **Base model is sufficient** — The koans the model can solve, it solves without LoRA help. The koans it can't solve, LoRA doesn't help either.
5. **50% vs 73.3%** — The lower pass rate vs previous Rust eval results is expected: single attempt (vs 5), no error hints, and HuggingFace bf16 model vs Ollama q4_K_M.

### Conclusion

Routing to task-specific Share coefficients does not improve performance and causes regressions. Experiments 2 (coefficient training) and 3 (sequential learning) are deprioritized since the underlying coefficients don't add value. **Prompt engineering remains the most effective approach** (83.3% with hints vs 50% without).

Results saved to `runs/experiments/routing_vs_averaging/`.

---

## Overall Progress

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 0: Setup | Complete | 100% |
| Phase 1: MVP | Complete | 100% |
| Phase 2: PaCT | Complete (Negative Result) | 100% |
| **Phase 2b: Prompt Engineering** | **Complete (Success!)** | **100%** |
| Phase 3: Production | Unblocked | 0% |

---

## Phase 2: PaCT Results

### What We Built
- [x] CUDA training pipeline (QLoRA, Flash Attention)
- [x] GGUF export and Ollama integration
- [x] Share algorithm (proper SVD-based implementation)
- [x] 51 pattern-specific LoRA adapters
- [x] Phase 2 coefficient-only training
- [x] Phase 3 basis merging
- [x] Extensive training data (431+ examples across 50+ patterns)

### What We Learned
- [x] LoRA fine-tuning causes catastrophic forgetting on capable base models
- [x] Share algorithm prevents forgetting but can't improve when averaging
- [x] The base model's 73-77% pass rate is likely the ceiling
- [x] Task routing (not averaging) may be needed for Share to help
- [x] Task routing tested (Exp 1, 2026-02-17): routing actively hurts (43.3% vs 50% baseline)

---

## Phase 1: MVP (Complete)

- [x] Core Types (core_types) - 12 tests
- [x] Episode Capture (capture) - 9 tests
- [x] Sandbox (sandbox) - 9 tests
- [x] Rust Koans Tasks (tasks_rust_koans) - 10 tests, 42 builtin koans
- [x] LLM Client (agent crate) - OllamaClient
- [x] Agent Loop (agent crate) - RED-patch-GREEN loop, 13 tests
- [x] Evaluation harness (eval crate) - 12 tests
- [x] CLI (cli crate) - 7 tests

**Total: 72 tests passing**

---

## Possible Paths Forward

Since LoRA fine-tuning is not viable, alternative approaches:

1. **Prompt Engineering** ✅ **IMPLEMENTED - SUCCESS**
   - Added error-specific hints to prompts
   - Pass rate improved from 73.3% → 83.3% average
   - No weight modification needed

2. **Multi-Turn Repair**
   - Use dialogue to iteratively fix errors
   - Agent asks clarifying questions
   - Leverages base model's reasoning

3. **Larger Model**
   - Try 7B or 14B parameter models
   - More capacity to absorb new knowledge without forgetting
   - Higher compute cost

4. **Task Routing with Share** -- **TESTED (Negative Result)**
   - Routing to task-specific coefficients: 43.3% (worse than baseline 50%)
   - Coefficients actively degrade trait_bounds performance
   - See Experiment 1 results above

5. **Model Ensemble**
   - Run multiple models, pick best output
   - Or use voting/consensus

6. **Accept Baseline**
   - 73-77% may be acceptable for this model+task
   - Focus on other improvements (UX, speed, reliability)

---

## Metrics

| Metric | Baseline | Best Achieved | Target |
|--------|----------|---------------|--------|
| Pass Rate | 73.3% | **86.7%** | ≥ 76.7% |
| Median Steps | 2.0 | 2.0 | ≤ 2.0 |
| Regressions | 0 | ~0 | 0 |

**Conclusion**: Target exceeded via prompt engineering! LoRA failed but hints work.

---

## Training Cycle History

| Cycle | Date | Approach | Pass Rate | Notes |
|-------|------|----------|-----------|-------|
| C0 | 02-08 | Baseline | 76.7% | Initial target |
| C1 | 02-08 | Naive LoRA | 60.0% | Catastrophic forgetting |
| C9-10 | 02-09 | Minimal (20 steps) | 73.3% | Best fine-tuned |
| C12 | 02-09 | Rust 2024 | 73.3% | Same ceiling |
| C13 | 02-09 | Share (6 adapters) | 73.3% | Proper implementation |
| C14+ | 02-10 | Share (51 adapters) | 70.0-73.3% | More adapters = worse |
| Final | 02-10 | Share Full (Ph2+Ph3) | 73.3% | Prevents forgetting, no improvement |
| **C100-102** | **02-12** | **Prompt Engineering** | **80-86.7%** | **Error-specific hints work!** |
| Exp 1 | 02-17 | Share Routing vs Averaging | 43-50%* | Routing hurts, averaging neutral |

---

## Documentation

- [x] [README.md](../README.md) - Project overview with diagrams
- [x] [CLAUDE.md](../CLAUDE.md) - AI agent context
- [x] [learnings.md](./learnings.md) - Detailed experiment logs
- [x] [story.md](./story.md) - Project narrative
- [x] [next-steps.md](./next-steps.md) - Analysis and recommendations
- [x] [course-correction.md](./course-correction.md) - Share implementation details
- [x] [changes.md](./changes.md) - Training history

---

## Links

- Research Papers:
  - [Share LoRA Subspaces (arXiv:2602.06043)](https://arxiv.org/abs/2602.06043)
  - [UWSH (arXiv:2512.05117)](https://arxiv.org/abs/2512.05117)
- Project:
  - [GitHub Repository](https://github.com/softwarewrighter/sleepy-coder)
